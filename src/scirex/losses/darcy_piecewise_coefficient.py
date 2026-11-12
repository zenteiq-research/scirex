# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Piecewise-constant Darcy coefficient sampling, a grid-aligned T3 mesh
builder, and a direct FEM solve -- the shared infrastructure for a
Galerkin-loss operator benchmark on the Kovachki et al. (JMLR 2023, Sec.
6.2) Darcy dataset (a(x) in {3, 12}, thresholded Gaussian random field,
f=1), independent of `scirex.losses.galerkin` and
`scirex.experimental.geometry.mesh_io` (only imports from them -- this file
never edits either, since both are under active review elsewhere).

Grid-aligned mesh, on purpose
------------------------------
The sampled coefficient is genuinely discontinuous, not smoothly varying
like this repo's other variable-coefficient benchmarks (`kappa(x)`,
`kappa(x,y)`). `build_grid_aligned_t3_mesh` builds a T3 mesh whose
element edges coincide exactly with the coefficient grid's cell
boundaries (each cell cut into 2 triangles), so every material interface
falls exactly on element edges and every triangle sits entirely inside
one constant-coefficient cell -- the standard, textbook-correct way to
handle an interface problem with low-order elements, and the reason T3
(not T6) is used here: a prior jump-coefficient investigation in this
repo (RBF-FD, strong-form) found geometric *curvature*, not the jump
itself, was the dominant source of error -- straight, mesh-aligned
interfaces already converged well. This construction has no curved
geometry anywhere.

`kappa_fn`/`source_fn` here reuse `T3Galerkin`
unmodified: `params` is simply the *entire* per-instance coefficient
grid (not a small parameter vector like `(A, f)` elsewhere in this
repo), and `kappa_fn` does a nearest-cell lookup into it. Since
`T3Galerkin`/`BatchedVariableCoefficientGalerkinResidualLossT3` only
ever thread `params` through to these two functions as an opaque
pytree, this needs no changes to that (in-review) module.

Direct FEM solve, not a manufactured solution
-----------------------------------------------
Unlike this repo's other variable-coefficient benchmarks, there is no
closed-form solution to derive an exact source from -- `f=1` is simply
given. `assemble_and_solve` instead solves `K u = F` by
assembling both directly from `T3Galerkin`'s closed-form per-element
geometry (a T3 element's gradient is constant per triangle, so this needs
no quadrature loop or differentiation) -- giving trustworthy `(a, u)`
reference pairs for zero-shot evaluation, entirely in JAX/numpy, no
PyTorch anywhere in this pipeline. See that function's own docstring for
why this isn't built on `jax.jacfwd` despite the PDE being linear
(a real, measured performance trap, not just a style choice).
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import idctn
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from scirex.losses.galerkin import T3Galerkin
from scirex.losses.galerkin_q4 import (
    _N_AT_QUAD as _Q4_N_AT_QUAD,
)
from scirex.losses.galerkin_q4 import (
    _QUAD_WEIGHTS as _Q4_QUAD_WEIGHTS,
)
from scirex.losses.galerkin_q4 import Q4Galerkin


def sample_piecewise_coefficient(
    key: jax.Array,
    resolution: int,
    tau: float = 3.0,
    alpha: float = 2.0,
    lo: float = 3.0,
    hi: float = 12.0,
    base_resolution: int = 421,
) -> np.ndarray:
    """One `a(x) in {lo, hi}` sample, matching Kovachki et al. (JMLR
    2023, Sec. 6.2): a Gaussian random field `g ~ N(0, C)`,
    `C = (-Delta + tau^2 I)^{-alpha}` with Neumann BC (cosine
    eigenbasis, sampled via a 2D inverse discrete cosine transform),
    thresholded at zero. `tau=3, alpha=2` matches the paper's own
    `C = (-Delta + 9I)^{-2}`.

    Sampled at `base_resolution` (421, the paper's own native FD
    resolution) and *downsampled* to `resolution` by nearest-index
    selection when the two differ -- matching the paper's own convention
    ("all other resolutions are downsampled from the 421x421 dataset
    rather than solving independently at each resolution"), *not*
    sampling the GRF directly at `resolution`. These are not the same
    distribution: direct sampling at a low resolution only ever includes
    frequency modes up to `resolution-1`, while downsampling a
    `base_resolution` draw folds in aliasing from the higher modes
    genuinely present in that finer field.

    Returns
    -------
    a_grid : np.ndarray, shape (resolution, resolution)
        One constant value per grid *cell* (not per node) -- cell (i, j)
        covers ``[i/resolution, (i+1)/resolution] x [j/resolution,
        (j+1)/resolution]``, matched by `build_grid_aligned_t3_mesh`
        and `kappa_fn`.
    """
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)

    k = np.arange(base_resolution)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    eigs = (np.pi**2 * (kx**2 + ky**2) + tau**2) ** (-alpha)

    coef = rng.standard_normal((base_resolution, base_resolution)) * np.sqrt(eigs)
    # Zero the DC (k=0,0) mode before the IDCT: without this, every sample
    # gets its own random, uniform, sign-random shift added to the whole
    # field before thresholding, occasionally biasing an entire sample
    # toward one phase almost everywhere. Measured directly: over 300
    # samples, the fraction-of-area-at-hi statistic had std=0.32 (range
    # 0.0-1.0, i.e. some samples were an entirely constant field, not a
    # two-phase medium at all) without this line, vs std=0.06 with it --
    # matching a reference implementation of this same benchmark that
    # zeros this mode explicitly for exactly this reason.
    coef[0, 0] = 0.0
    g = idctn(coef, type=2, norm="ortho")
    a_base = np.where(g >= 0, hi, lo).astype(np.float32)

    if resolution == base_resolution:
        return a_base
    idx = np.linspace(0, base_resolution - 1, resolution).round().astype(int)
    return a_base[np.ix_(idx, idx)]


def build_grid_aligned_t3_mesh(
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A T3 mesh over the unit square whose element edges exactly match a
    `resolution x resolution` grid of cells -- each cell cut into 2
    triangles along its own diagonal. Pure numpy, no gmsh dependency.

    Returns
    -------
    xy : np.ndarray, shape ((resolution+1)**2, 2)
        Node coordinates, one per grid corner.
    tri : np.ndarray, shape (2*resolution**2, 3), int
        Triangle connectivity, 0-indexed.
    boundary_mask : np.ndarray, shape ((resolution+1)**2,), bool
    """
    r = resolution
    xs = np.linspace(0.0, 1.0, r + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    def idx(i: int, j: int) -> int:
        return i * (r + 1) + j

    tri_list: list[list[int]] = []
    for i in range(r):
        for j in range(r):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tri_list.append([a, b, c])
            tri_list.append([a, c, d])
    tri = np.array(tri_list, dtype=np.int32)

    boundary_mask = (xy[:, 0] == 0.0) | (xy[:, 0] == 1.0)
    boundary_mask |= (xy[:, 1] == 0.0) | (xy[:, 1] == 1.0)
    return xy, tri, boundary_mask


def build_grid_aligned_t6_mesh(
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T6 (quadratic triangle) counterpart of
    `build_grid_aligned_t3_mesh`: elevates that same grid-aligned T3
    mesh by adding one de-duplicated mid-edge node per shared T3 edge, in
    gmsh's T6 node order (3 corners, then mid(1,2), mid(2,3), mid(3,1)) --
    the same construction this repo's own T6 test suite uses to build a
    T6 mesh without a gmsh dependency. Every added mid-edge node sits
    exactly on a straight T3 edge (a chord midpoint), so the elements
    stay affine -- no curved geometry is introduced, consistent with this
    module's use of straight-sided elements throughout.

    Returns
    -------
    xy : np.ndarray, shape (N, 2)
        Corner nodes (matching `build_grid_aligned_t3_mesh`'s xy,
        same order) followed by the newly added mid-edge nodes.
    tri : np.ndarray, shape (2*resolution**2, 6), int
    boundary_mask : np.ndarray, shape (N,), bool
    """
    xy3, tri3, boundary_mask3 = build_grid_aligned_t3_mesh(resolution)

    edge_count: dict[tuple[int, int], int] = {}
    for a, b, c in tri3:
        for i, j in ((a, b), (b, c), (c, a)):
            key = (int(i), int(j)) if i < j else (int(j), int(i))
            edge_count[key] = edge_count.get(key, 0) + 1

    nodes = list(xy3)
    edge_to_mid: dict[tuple[int, int], int] = {}
    boundary_extra: list[bool] = []

    def get_mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in edge_to_mid:
            edge_to_mid[key] = len(nodes)
            nodes.append((xy3[a] + xy3[b]) / 2.0)
            boundary_extra.append(edge_count[key] == 1)
        return edge_to_mid[key]

    tri6_list: list[list[int]] = []
    for a, b, c in tri3:
        tri6_list.append([a, b, c, get_mid(a, b), get_mid(b, c), get_mid(c, a)])

    xy6 = np.array(nodes, dtype=np.float32)
    tri6 = np.array(tri6_list, dtype=np.int32)
    boundary_mask6 = np.concatenate(
        [boundary_mask3, np.array(boundary_extra, dtype=bool)]
    )
    return xy6, tri6, boundary_mask6


def kappa_fn(a_grid: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """`a_grid` (the *entire* per-instance coefficient grid, playing the
    role `params` plays elsewhere in this repo) + a physical point ->
    nearest-cell lookup. Exact (not an approximation) because every
    quadrature point `T3Galerkin` evaluates this at lies strictly
    inside one grid cell, by construction of
    `build_grid_aligned_t3_mesh`.
    """
    resolution = a_grid.shape[0]
    i = jnp.clip(jnp.floor(point[0] * resolution).astype(jnp.int32), 0, resolution - 1)
    j = jnp.clip(jnp.floor(point[1] * resolution).astype(jnp.int32), 0, resolution - 1)
    return a_grid[i, j]


def source_fn(a_grid: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """The paper's PDE is `-div(a*grad(u)) = f`, `f(x) = 1` everywhere
    -- i.e. `div(a*grad(u)) = -f = -1`. This module's (and the rest of
    this repo's) convention is the strong form ``div(kappa*grad(u)) -
    source = 0``, so `source = div(a*grad(u)) = -1`, not `+1` --
    getting this backwards silently solves `+div(a*grad(u))=f` instead
    (same magnitude, sign-flipped `u`, confirmed by comparing against
    the expected positive "pressure bump" shape for `f>0`, `u=0` on
    the boundary). `a_grid` is unused, kept only for the same
    `(params, point)` calling convention as `kappa_fn`.
    """
    del a_grid, point
    return jnp.asarray(-1.0)


_TRI_QUAD_WEIGHTS = jnp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
_TRI_QUAD_BARY = jnp.array(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ]
)


@partial(jax.jit, static_argnums=(0, 1))
def _evaluate_kappa_and_load(
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    quad_pts: jnp.ndarray,
    area: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One fused, jit-compiled evaluation of `kappa`/`source` at every
    triangle's 3-point quadrature rule -- the only genuinely JAX-heavy
    part of `assemble_and_solve`. Calling `kappa_fn_`/
    `source_fn_` this way (rather than eagerly, one quadrature point at
    a time) matters: this repo's `kappa_fn` does a dynamic array lookup
    (`a_grid[i, j]`), and eager (un-jitted) dispatch of that lookup
    thousands of times (once per triangle per quadrature point) turned
    out to be dramatically slower than letting XLA compile and fuse it
    once. `kappa_fn_`/`source_fn_` are static arguments (plain Python
    functions, not traced arrays) so this recompiles only when the
    function *object* changes, not on every call with the same one.

    Returns
    -------
    kappa_elem : array, shape (T,)
    local_f : array, shape (T, 3)
        Matches `t3_weak_residual`'s own barycentric-weighted load-vector
        formula exactly (not a flat 1/3-average of the 3 quadrature
        values), verified once against it via a `jax.jacfwd` cross-check.
    """
    kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn_(params, pt))(pts))(
        quad_pts
    )  # (3, T)
    kappa_elem = jnp.sum(kappa_q * _TRI_QUAD_WEIGHTS[:, None], axis=0)  # (T,)

    source_q = jax.vmap(lambda pts: jax.vmap(lambda pt: source_fn_(params, pt))(pts))(
        quad_pts
    )  # (3, T)
    local_f = area[:, None] * jnp.einsum(
        "q,qt,qi->ti", _TRI_QUAD_WEIGHTS, source_q, _TRI_QUAD_BARY
    )  # (T, 3)
    return kappa_elem, local_f


def assemble_and_solve(
    galerkin: T3Galerkin,
    params: jnp.ndarray,
    interior_mask: np.ndarray,
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = kappa_fn,
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = source_fn,
) -> np.ndarray:
    """Direct FEM solve of `div(kappa*grad(u)) - source = 0` with
    homogeneous Dirichlet BC (`u=0` on the boundary), by assembling the
    global stiffness matrix/load vector from `galerkin`'s own
    closed-form per-element pieces (T3-specific: a linear element's
    gradient is constant per triangle, so both assemble without any
    quadrature loop or differentiation). No manufactured solution needed
    -- this *is* the ground truth.

    An earlier version of this function tried computing `K`/`F` via
    `jax.jacfwd` on `t3_weak_residual` (exploiting that the residual
    is affine in `u`) instead of hand-deriving them, specifically to
    reuse the residual's own already-correct formula rather than risk
    re-deriving it by hand. That was a real correctness win -- it caught
    a bug in this function's own load-vector assembly (a flat average of
    the 3 quadrature source values, instead of the correct
    barycentric-weighted quadrature `t3_weak_residual` actually uses,
    the same kind of under-integration `galerkin.py`'s own module
    docstring describes fixing for `kappa`, here applied to the source
    term instead). But calling it *un-jitted* was orders of magnitude
    slower than necessary (confirmed directly: un-jitted, effectively
    unbounded; jitted, 0.28s at resolution 32) -- and neither `K` nor
    `F` actually needs differentiating through `kappa_fn`/`source_fn`
    at all (only `galerkin`'s own precomputed geometry, `b`/`c`/
    `area`), so this version drops `jax.jacfwd` entirely in favor of
    the closed-form stiffness algebra plus one small `jax.jit`-compiled
    call for the `kappa`/`source` quadrature evaluation -- the only
    part that actually needs JAX at all here. (A large `nvidia-smi`
    memory reading while debugging this turned out to be a red herring:
    JAX's default GPU memory pool pre-allocates a large, roughly-fixed
    fraction of total device memory on first use regardless of what's
    being computed -- it wasn't evidence of a leak or an
    inefficiently-large intermediate array, just normal JAX behavior that
    happened to coincide with the real, separate slowness problem.)

    Homogeneous BC only needs boundary *rows* fixed to the identity (not
    the fuller row+column elimination an inhomogeneous BC would need):
    since `u=0` on the boundary, a boundary column's contribution to any
    other row is already exactly zero regardless of that entry's value.
    """
    interior_mask = np.asarray(interior_mask)
    g = galerkin
    n = g.n_nodes

    # kappa/source at every quadrature point are the only genuinely
    # JAX-heavy part of this function (they call the user-supplied
    # kappa_fn_/source_fn_, here a grid lookup) -- computed once, fused,
    # inside a single jit-compiled call (kappa_fn_/source_fn_ marked
    # static since they're plain Python functions, not traced arrays).
    # Calling per-element JAX ops like this un-jitted (eager dispatch of
    # one small op at a time) was the actual source of a severe slowdown
    # caught while building this -- see the docstring above.
    kappa_elem_j, local_f_j = _evaluate_kappa_and_load(
        kappa_fn_, source_fn_, params, g.quad_pts, g.area
    )
    kappa_elem = np.asarray(kappa_elem_j)  # (T,)
    local_f = np.asarray(local_f_j)  # (T, 3)
    b, c, area = np.asarray(g.b), np.asarray(g.c), np.asarray(g.area)

    # Local 3x3 stiffness: K_e[p,q] = kappa * (b_p*b_q + c_p*c_q) / (4*Area)
    # -- a closed form since a T3 element's gradient is constant per
    # triangle, no quadrature/differentiation needed.
    local_k = (
        kappa_elem[:, None, None]
        * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
        / (4.0 * area[:, None, None])
    )  # (T, 3, 3)
    tri = np.asarray(g.tri)
    rows = np.repeat(tri, 3, axis=1).reshape(-1)
    cols = np.tile(tri, (1, 3)).reshape(-1)
    # A T3 stiffness matrix is extremely sparse (~9 nonzeros/row, from a
    # node's handful of incident triangles) -- assembling it as a sparse
    # matrix and using a sparse solver, not np.linalg.solve, matters in
    # practice, not just in principle: a dense solve of the *same* 1681x1681
    # system measured 43s here, entirely wasted on a matrix that's >99%
    # zeros; the sparse solve below is near-instant at the same size.
    K = coo_matrix((local_k.reshape(-1), (rows, cols)), shape=(n, n)).tocsr()
    F = np.zeros(n)
    np.add.at(F, tri.reshape(-1), local_f.reshape(-1))

    boundary = ~interior_mask
    K = K.tolil()
    K[boundary, :] = 0.0
    K[boundary, boundary] = 1.0
    K = K.tocsr()
    F[boundary] = 0.0

    # R(u) = K@u + F = 0 (see galerkin.py's sign-convention docstring),
    # so the linear solve is K@u = -F, not K@u = F.
    return spsolve(K, -F)


def energy_functional(
    galerkin: T3Galerkin,
    u: jnp.ndarray,
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """`J(u) = 0.5*int(kappa*|grad(u)|^2) dOmega + int(source*u) dOmega`
    -- the variational (Euler-Lagrange) characterization of
    `div(kappa*grad(u)) - source = 0`: minimizing this over trial fields
    already satisfying the Dirichlet BC (via the boundary ansatz, same as
    everywhere else in this repo) recovers exactly the weak solution, a
    *different, mathematically equivalent* characterization of the same
    physics as `t3_weak_residual`'s squared-residual approach -- not
    extra physics.

    Used here instead of the squared weak-form residual because it is
    well-scaled in a way the squared residual is not: at `u=0`,
    `t3_weak_residual`'s per-node values are just the load vector
    `F` itself, whose entries shrink with element size (``~source *
    area/3``), so `mean(R_i^2)` collapses toward the same tiny
    (~1e-6 to 1e-7) magnitude regardless of mesh density or whether `u`
    is actually correct -- confirmed directly: at resolution 32,
    `loss(u=0)` and `loss(u_true)` at an *actual trained model's*
    scale were numerically indistinguishable, both ~1e-6. `J` has no
    such issue: `J(0) = 0` exactly, while `J(u_true) = -0.5*E_grad`
    (a direct consequence of the weak form's own energy identity,
    `E_grad = int(source*u) dOmega` at the true solution) -- a gap on
    the order of `1e-3` to `1e-2` for this problem, five orders of
    magnitude larger than the squared residual's discrimination at the
    same resolution. `J` is also coercive/strictly convex for
    `kappa` bounded away from zero (a classical result, via a Poincare
    inequality bounding the linear term by the quadratic one for large
    `||grad(u)||`), so it needs no energy-floor-style collapse
    mitigation the way the squared residual did -- `u=0` is not even a
    local minimum of `J`, since the linear term's gradient at `u=0`
    is exactly `-E_grad(u_true)` in the direction of the true solution,
    not zero.

    `int(source*u) dOmega` is computed per element as ``area *
    mean_q(source(x_q)) * mean(u1, u2, u3)`` -- exact for `u` piecewise
    linear (T3) and `source` piecewise constant per element (this
    module's grid-aligned convention), reusing the same 3-point
    quadrature rule already used for `source`/`kappa` elsewhere in
    this repo's Galerkin losses.
    """
    g = galerkin
    grad_term = 0.5 * g.grad_energy(u, kappa_fn_, params)

    quad_pts = g.quadrature_points()  # (3, T, 2)
    source_q = jax.vmap(lambda pts: jax.vmap(lambda pt: source_fn_(params, pt))(pts))(
        quad_pts
    )  # (3, T)
    source_elem = jnp.mean(source_q, axis=0)  # (T,)
    u1, u2, u3 = u[g.n1], u[g.n2], u[g.n3]
    u_mean = (u1 + u2 + u3) / 3.0
    source_term = jnp.sum(g.area * source_elem * u_mean)

    return grad_term + source_term


class BatchedEnergyFunctionalLoss:
    """Minimizes `J(u)` (see `energy_functional`) directly, mean
    over the instance batch -- the variational alternative to
    `BatchedVariableCoefficientGalerkinResidualLossT3`'s
    squared-residual loss. No `interior_mask` argument: `J` is
    evaluated over the whole mesh, relying on the boundary ansatz (not a
    masked residual) to enforce the Dirichlet BC, matching the
    variational problem's own trial space (`H^1_0`).
    """

    def __init__(
        self,
        galerkin: T3Galerkin,
        kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> None:
        self.galerkin = galerkin
        self.kappa_fn = kappa_fn_
        self.source_fn = source_fn_

    def __call__(self, u_batch: jnp.ndarray, params_batch: jnp.ndarray) -> jnp.ndarray:
        j_vals = jax.vmap(
            lambda u, p: energy_functional(
                self.galerkin, u, self.kappa_fn, self.source_fn, p
            )
        )(u_batch, params_batch)
        return jnp.mean(j_vals)


# -- Q4 (bilinear quadrilateral) counterparts -------------------------------
#
# A Q4 element maps 1:1 onto one grid cell -- no diagonal split needed
# (unlike T3, which must cut each cell into 2 triangles, an arbitrary choice
# of diagonal). Since this benchmark's coefficient is genuinely piecewise
# constant *per grid cell*, a Q4 mesh represents every material interface
# exactly on element edges with no cell-splitting ambiguity at all, making
# it a more natural fit than T3 for this specific problem -- worth
# comparing directly against the T3 results above.


def build_grid_aligned_q4_mesh(
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q4 counterpart of `build_grid_aligned_t3_mesh`: one
    quadrilateral per grid cell, same node coordinates/ordering/count as
    the T3 mesh (so `kappa_fn`/`source_fn` and
    `sample_piecewise_coefficient` are reused unmodified).

    Returns
    -------
    xy : np.ndarray, shape ((resolution+1)**2, 2)
    quad : np.ndarray, shape (resolution**2, 4), int
        0-indexed, CCW (bottom-left, bottom-right, top-right, top-left) --
        the node order `Q4Galerkin`
        expects.
    boundary_mask : np.ndarray, shape ((resolution+1)**2,), bool
    """
    r = resolution
    xs = np.linspace(0.0, 1.0, r + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    def idx(i: int, j: int) -> int:
        return i * (r + 1) + j

    quad = []
    for i in range(r):
        for j in range(r):
            quad.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    quad_arr = np.array(quad, dtype=np.int32)

    boundary_mask = (xy[:, 0] == 0.0) | (xy[:, 0] == 1.0)
    boundary_mask |= (xy[:, 1] == 0.0) | (xy[:, 1] == 1.0)
    return xy, quad_arr, boundary_mask


@partial(jax.jit, static_argnums=(0, 1))
def _evaluate_local_kf_q4(
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    quad_phys_pts: jnp.ndarray,
    dN_dx: jnp.ndarray,
    dN_dy: jnp.ndarray,
    det_j: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Local 4x4 stiffness and 4-vector load per element, Q4 analogue of
    `_evaluate_kappa_and_load`. Unlike T3 (constant gradient per
    element -> closed-form 3x3 stiffness), a Q4 element's shape-function
    gradients vary within the element (isoparametric map), so the local
    stiffness genuinely needs the quadrature sum here, not a closed form.

    Returns
    -------
    local_k : array, shape (T, 4, 4)
    local_f : array, shape (T, 4)
    """
    kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn_(params, pt))(pts))(
        quad_phys_pts
    )  # (Q, T)
    source_q = jax.vmap(lambda pts: jax.vmap(lambda pt: source_fn_(params, pt))(pts))(
        quad_phys_pts
    )  # (Q, T)
    w = _Q4_QUAD_WEIGHTS[:, None] * jnp.abs(det_j)  # (Q, T)
    local_k = jnp.einsum("qt,qtp,qtr->tpr", w * kappa_q, dN_dx, dN_dx) + jnp.einsum(
        "qt,qtp,qtr->tpr", w * kappa_q, dN_dy, dN_dy
    )  # (T, 4, 4)
    local_f = jnp.einsum("qt,qp->tp", w * source_q, _Q4_N_AT_QUAD)  # (T, 4)
    return local_k, local_f


def assemble_and_solve_q4(
    galerkin: Q4Galerkin,
    params: jnp.ndarray,
    interior_mask: np.ndarray,
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = kappa_fn,
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = source_fn,
) -> np.ndarray:
    """Q4 analogue of `assemble_and_solve`: direct FEM solve of
    `div(kappa*grad(u)) - source = 0` with homogeneous Dirichlet BC, by
    assembling the global stiffness/load from `galerkin`'s own
    isoparametric per-element quadrature. Same sign convention (``K@u =
    -F``) and boundary-row-only elimination as the T3 version -- see that
    function's docstring for the derivation.
    """
    interior_mask = np.asarray(interior_mask)
    g = galerkin
    n = g.n_nodes

    dN_dx, dN_dy, det_j = g.physical_grad_and_jacobian()
    local_k_j, local_f_j = _evaluate_local_kf_q4(
        kappa_fn_,
        source_fn_,
        params,
        g.quadrature_points_physical(),
        dN_dx,
        dN_dy,
        det_j,
    )
    local_k = np.asarray(local_k_j)  # (T, 4, 4)
    local_f = np.asarray(local_f_j)  # (T, 4)
    quad = np.asarray(g.quad)  # (T, 4)

    rows = np.repeat(quad, 4, axis=1).reshape(-1)
    cols = np.tile(quad, (1, 4)).reshape(-1)
    K = coo_matrix((local_k.reshape(-1), (rows, cols)), shape=(n, n)).tocsr()
    F = np.zeros(n)
    np.add.at(F, quad.reshape(-1), local_f.reshape(-1))

    boundary = ~interior_mask
    K = K.tolil()
    K[boundary, :] = 0.0
    K[boundary, boundary] = 1.0
    K = K.tocsr()
    F[boundary] = 0.0

    return spsolve(K, -F)


def energy_functional_q4(
    galerkin: Q4Galerkin,
    u: jnp.ndarray,
    kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Q4 analogue of `energy_functional` -- same
    `J(u) = 0.5*int(kappa*|grad(u)|^2) + int(source*u)` functional, same
    sign convention, only the element type (and therefore the quadrature
    used for both terms) differs.
    """
    g = galerkin
    grad_term = 0.5 * g.grad_energy(u, kappa_fn_, params)

    quad_phys_pts = g.quadrature_points_physical()  # (Q, T, 2)
    source_q = jax.vmap(lambda pts: jax.vmap(lambda pt: source_fn_(params, pt))(pts))(
        quad_phys_pts
    )  # (Q, T)
    u_elem = u[g.quad]  # (T, 4)
    u_at_quad = jnp.einsum("qp,tp->qt", _Q4_N_AT_QUAD, u_elem)  # (Q, T)
    w = _Q4_QUAD_WEIGHTS[:, None] * jnp.abs(g.det_j)  # (Q, T)
    source_term = jnp.sum(w * source_q * u_at_quad)

    return grad_term + source_term


class BatchedEnergyFunctionalLossQ4:
    """Q4 analogue of `BatchedEnergyFunctionalLoss`."""

    def __init__(
        self,
        galerkin: Q4Galerkin,
        kappa_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        source_fn_: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    ) -> None:
        self.galerkin = galerkin
        self.kappa_fn = kappa_fn_
        self.source_fn = source_fn_

    def __call__(self, u_batch: jnp.ndarray, params_batch: jnp.ndarray) -> jnp.ndarray:
        j_vals = jax.vmap(
            lambda u, p: energy_functional_q4(
                self.galerkin, u, self.kappa_fn, self.source_fn, p
            )
        )(u_batch, params_batch)
        return jnp.mean(j_vals)
