# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Modular Galerkin (weak-form) gradient machinery on a T3 (linear
triangle) mesh, plus a batched variable-coefficient residual loss built on
top of it: a generic, problem-agnostic geometry/gradient class
(`T3Galerkin`) and a family of residual-loss classes built on it for
one specific PDE family (here, `div(kappa * grad(u)) = f`). A different
weak form (e.g. linear elasticity's strain energy, or the magnetostatic
interface condition) is a different, problem-specific class built on the
same `T3Galerkin`, not a change to it.

Mesh-agnostic to *where* the mesh came from -- generated via
`scirex.experimental.geometry.gmsh_mesh` or loaded via
`scirex.experimental.geometry.mesh_io` -- this module only ever consumes plain
`(xy, tri)` arrays and has no `gmsh`/`meshio` dependency itself, so it
runs on a training server that has neither installed.

Operator learning, not single-instance fitting
------------------------------------------------
`BatchedVariableCoefficientGalerkinResidualLossT3` takes a *batch*
of instance parameters per call (``(u_batch, coords, params_batch,
source_batch)``), not a single fixed field -- one training run learns an
entire family of `kappa` fields, not one instance. The mesh (here, the
T3 element geometry) is shared and built once; only the per-instance
`kappa`/source vary per training step. This loss and mesh layer is
itself agnostic to what produced `u_batch` -- an operator-learning
backbone trained across the batch, or a single-instance fit -- it only
ever consumes the predicted field and the mesh.

`kappa_fn`/exact-source convention
--------------------------------------
`kappa_fn(params, point) -> scalar` and a per-instance `source_batch`
computed exactly (e.g. via `jax.jacfwd` on a manufactured solution) are
the two functions a training script defines for itself and passes in at
construction. Keeping the same two-function convention across every
variable-coefficient benchmark script means the manufactured solution,
sampling, evaluation, and plotting logic can be shared verbatim across
scripts; only the residual class (and the mesh) changes between them.

Sign convention (read before touching the residual assembly)
-----------------------------------------------------------------
This targets the strong form `div(kappa * grad(u)) - source = 0` (the
convention every variable-coefficient benchmark script in this repo
already uses), *not* the
`-div(kappa * grad(u)) = f` convention some Galerkin/FEM references use
(a leading-sign flip, equivalent to `source = -f`). Multiplying our
strong form by a test function `N_i` and integrating by parts (Green's
first identity, dropping the boundary-flux term -- valid where `N_i` is
zero on the Dirichlet boundary, which every PINO script here already
arranges via its own boundary ansatz applied to `u` before this loss
ever sees it) gives::

    R_i = int kappa * grad(u).grad(N_i) dOmega + int source * N_i dOmega

i.e. a **plus** between the two terms -- flip it and this residual would
silently solve the wrong PDE, converging smoothly to a self-consistent but
incorrect field (a weak-form analogue of getting a strong-form residual's
own source sign backwards). `t3_weak_residual` is unit-tested
against a manufactured solution's own exact source specifically to catch
this class of bug.

Quadrature: evaluate the real functions, not nodal averages
------------------------------------------------------------
`kappa` and `source` are evaluated at a 3-point, degree-2-accurate
symmetric quadrature rule per triangle (`_TRI_QUAD_BARY`), each
point evaluated through the actual `kappa_fn`/`source_fn` closed forms
-- not approximated by averaging the 3 nodal values (a 1-point,
degree-1-accurate rule equivalent to a single centroid evaluation, an
earlier version of this module used). The difference matters in practice:
a coefficient family that is smooth in one direction but varies in *both*
`x` and `y` within a single element (e.g. ``kappa(x,y) =
2+A*sin(2*pi*f1*x)*sin(2*pi*f2*y)``) has real curvature a 1-point rule
under-resolves, even though the very same manufactured solution's own
`u` field -- linear-element-approximated identically either way -- is
no harder to represent than a comparable 1D-varying-coefficient case. That
asymmetry (2D-varying coefficient families scoring meaningfully worse than
1D-varying ones at matched mesh resolution) is exactly the signature of
under-integrated element coefficients, not under-resolved element
geometry, which is why the fix is a better quadrature rule rather than a
finer mesh or a higher-order element.

Since `kappa_fn`/`source_fn` are known analytically everywhere (they
are the same closed-form Python functions a PINO script already uses to
score zero-shot error), evaluating them at 3 interior points per triangle
costs three function calls, not three new *approximations* -- there is no
accuracy given up anywhere else to buy this.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

_TRI_QUAD_BARY = jnp.array(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ]
)
_TRI_QUAD_WEIGHTS = jnp.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def element_scatter_add(
    tri: jnp.ndarray, n_nodes: int, elem_contrib: jnp.ndarray
) -> jnp.ndarray:
    """Scatter/accumulate per-triangle-per-local-node contributions into a
    nodal array -- the FEM assembly step every element family (T3, T6, and
    future ones) performs identically, parameterized only by the local-node
    count (`tri.shape[-1]`, `elem_contrib.shape[-1]`). Shared rather
    than reimplemented per element type so an assembly fix only needs to
    land once.
    """
    return (
        jnp.zeros((n_nodes,), dtype=elem_contrib.dtype)
        .at[tri.reshape(-1)]
        .add(elem_contrib.reshape(-1))
    )


class T3Galerkin:
    """Generic linear (T3) triangular-element Galerkin gradient machinery,
    precomputed once for a fixed mesh and shared by every instance in a
    training batch, via known closed-form shape-function gradients on a
    triangulation.

    A T3 element's shape functions are linear, so the gradient of a
    nodal field's piecewise-linear interpolant is *constant per triangle*
    -- `element_gradient` and `element_areas` reflect that with
    a leading `(T,)` shape throughout, `T` the triangle count.

    Parameters
    ----------
    xy : array, shape (N, 2)
        Node coordinates.
    tri : array, shape (T, 3), int
        Triangle connectivity, 0-indexed.
    """

    def __init__(self, xy: jnp.ndarray, tri: jnp.ndarray) -> None:
        xy = jnp.asarray(xy)
        tri = jnp.asarray(tri, dtype=jnp.int32)
        if tri.shape[-1] != 3:
            raise ValueError(
                f"T3Galerkin expects 3-node triangle connectivity (T, 3), "
                f"got tri.shape={tri.shape} -- for a 6-node (T6) mesh, use "
                f"scirex.experimental.losses.galerkin_t6.T6Galerkin instead."
            )
        self.xy = xy
        self.tri = tri
        self.n_nodes = xy.shape[0]
        self.n_elements = tri.shape[0]

        n1, n2, n3 = tri[:, 0], tri[:, 1], tri[:, 2]
        self.n1, self.n2, self.n3 = n1, n2, n3
        x1, x2, x3 = xy[n1], xy[n2], xy[n3]

        det = (x2[:, 0] - x1[:, 0]) * (x3[:, 1] - x1[:, 1]) - (x2[:, 1] - x1[:, 1]) * (
            x3[:, 0] - x1[:, 0]
        )
        # det scales as length^2, so a fixed absolute floor is only
        # meaningful for one particular choice of units/mesh scale: a mesh
        # in kilometers (or just a coarser one, area ~ length^2) can have
        # every det comfortably above 1e-12 while still containing a
        # genuinely degenerate (near-collinear) triangle relative to its
        # own size. Compare against each triangle's own longest edge
        # squared instead, which is scale-invariant.
        e12_sq = (x2[:, 0] - x1[:, 0]) ** 2 + (x2[:, 1] - x1[:, 1]) ** 2
        e23_sq = (x3[:, 0] - x2[:, 0]) ** 2 + (x3[:, 1] - x2[:, 1]) ** 2
        e31_sq = (x1[:, 0] - x3[:, 0]) ** 2 + (x1[:, 1] - x3[:, 1]) ** 2
        edge_scale_sq = jnp.maximum(jnp.maximum(e12_sq, e23_sq), e31_sq)
        # + 1e-12 as a floor for the all-coincident-points case, where
        # edge_scale_sq is itself exactly 0 and a purely relative test
        # (0 < 0) would never fire.
        if jnp.any(jnp.abs(det) < 1e-8 * edge_scale_sq + 1e-12):
            raise ValueError(
                "T3Galerkin: one or more triangles are degenerate (near-zero "
                "area relative to their own edge length) -- check the "
                "mesh's node ordering/connectivity."
            )
        area = 0.5 * jnp.abs(det)
        self.area = area

        b1, b2, b3 = x2[:, 1] - x3[:, 1], x3[:, 1] - x1[:, 1], x1[:, 1] - x2[:, 1]
        c1, c2, c3 = x3[:, 0] - x2[:, 0], x1[:, 0] - x3[:, 0], x2[:, 0] - x1[:, 0]
        self.b = jnp.stack([b1, b2, b3], axis=-1)  # (T, 3)
        self.c = jnp.stack([c1, c2, c3], axis=-1)  # (T, 3)
        self.inv2a = 1.0 / (2.0 * area)
        self.quad_pts = self._compute_quadrature_points()

    def element_gradient(self, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """`u`, shape (N,), a nodal field -> `(du_dx, du_dy)`, each
        shape (T,): the constant gradient of `u`'s T3 (piecewise-linear)
        interpolant over each triangle.
        """
        u1, u2, u3 = u[self.n1], u[self.n2], u[self.n3]
        du_dx = self.inv2a * (self.b[:, 0] * u1 + self.b[:, 1] * u2 + self.b[:, 2] * u3)
        du_dy = self.inv2a * (self.c[:, 0] * u1 + self.c[:, 1] * u2 + self.c[:, 2] * u3)
        return du_dx, du_dy

    def _compute_quadrature_points(self) -> jnp.ndarray:
        x1, x2, x3 = self.xy[self.n1], self.xy[self.n2], self.xy[self.n3]
        return jnp.stack(
            [bary[0] * x1 + bary[1] * x2 + bary[2] * x3 for bary in _TRI_QUAD_BARY],
            axis=0,
        )

    def quadrature_points(self) -> jnp.ndarray:
        """Physical coordinates of the 3-point/degree-2 quadrature rule's
        points (see the module docstring's "Quadrature" section), shape
        `(3, T, 2)` -- exact for a field varying up to quadratic order
        within an element, unlike a single centroid-value point (exact
        only for a field constant per element). Precomputed once in
        `__init__` since it depends only on the mesh, not on any
        per-instance field.
        """
        return self.quad_pts

    def scatter_to_nodes(self, elem_contrib: jnp.ndarray) -> jnp.ndarray:
        """`elem_contrib`, shape (T, 3) -- one value per triangle per
        local node (0, 1, 2) -- scattered/accumulated into a nodal array,
        shape (N,). This is the assembly step: summing every triangle's
        contribution to each test function `N_i` it touches.
        """
        return element_scatter_add(self.tri, self.n_nodes, elem_contrib)

    def element_kappa(
        self,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """`kappa_fn` evaluated at the 3-point quadrature rule and
        combined into one effective per-element value, shape `(T,)` --
        the quantity `grad_energy` and `t3_weak_residual` both
        need, factored out since both use it identically.
        """
        quad_pts = self.quadrature_points()  # (3, T, 2)
        kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn(params, pt))(pts))(
            quad_pts
        )  # (3, T)
        return jnp.sum(kappa_q * _TRI_QUAD_WEIGHTS[:, None], axis=0)

    def grad_energy(
        self,
        u: jnp.ndarray,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scalar `int kappa * |grad(u)|^2 dOmega`, `kappa` evaluated
        via `element_kappa`'s quadrature rule, exact quadrature for
        the piecewise-linear `u`'s constant-per-element gradient.
        """
        du_dx, du_dy = self.element_gradient(u)
        kappa_elem = self.element_kappa(kappa_fn, params)
        return jnp.sum(kappa_elem * (du_dx**2 + du_dy**2) * self.area)


def t3_weak_residual(
    galerkin: T3Galerkin,
    u: jnp.ndarray,
    kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Per-node Galerkin weak-form residual, shape (N,), for
    `div(kappa * grad(u)) - source = 0` (see the module docstring's sign
    derivation -- this is a **plus** between the two element terms, unlike
    the `-div(kappa*grad(u))=f` convention's minus).

    Parameters
    ----------
    galerkin : T3Galerkin
    u : array, shape (N,)
        Nodal field, evaluated at `galerkin.xy`.
    kappa_fn, source_fn : callable
        `(params, point) -> scalar`, evaluated at this element's
        3-point quadrature rule (see the module docstring's "Quadrature"
        section) -- not sampled only at nodes.
    params : array
        This instance's parameter vector, passed through to `kappa_fn`/
        `source_fn`.
    """
    g = galerkin
    du_dx, du_dy = g.element_gradient(u)
    kappa_elem = g.element_kappa(kappa_fn, params)

    quad_pts = g.quadrature_points()  # (3, T, 2)
    source_q = jax.vmap(lambda pts: jax.vmap(lambda pt: source_fn(params, pt))(pts))(
        quad_pts
    )  # (3, T)

    grad_dot = kappa_elem[:, None] * (
        du_dx[:, None] * g.b * 0.5 + du_dy[:, None] * g.c * 0.5
    )
    # int source * N_i dOmega, T3: N_i at quad point q is exactly that
    # point's own barycentric coordinate i (T3's shape functions *are* the
    # barycentric coordinates), so this is
    # area * sum_q weight_q * source(x_q) * bary_{q,i} -- not
    # area/3 * mean_q(source(x_q)), which silently assumes N_i = 1/3 at
    # every quadrature point (see the module docstring's "Quadrature"
    # section: that flat weighting is exactly the under-integration this
    # section argues against, just applied to the source term instead of
    # kappa).
    src = g.area[:, None] * jnp.einsum(
        "q,qt,qi->ti", _TRI_QUAD_WEIGHTS, source_q, _TRI_QUAD_BARY
    )
    elem_contrib = grad_dot + src  # (T, 3) -- see module docstring: '+', not '-'
    return g.scatter_to_nodes(elem_contrib)


class BatchedVariableCoefficientGalerkinResidualLossT3:
    """A batched training signal for `div(kappa * grad(u)) = source` on
    a fixed T3 mesh shared across every instance, using known linear
    shape-function gradients to assemble the weak-form residual.

    Parameters
    ----------
    galerkin : T3Galerkin
        Built once over the mesh shared by every instance in the batch.
    kappa_fn : callable
        `(params, point) -> scalar`, evaluated at each element's 3-point
        quadrature rule (see the module docstring's "Quadrature" section).
    source_fn : callable
        `(params, point) -> scalar`, the manufactured source (e.g. a
        training script's own `exact_source_point`), evaluated the same
        way as `kappa_fn`. This is a function supplied once at
        construction, not a precomputed nodal array passed to every call
        -- the Galerkin residual needs the source at quadrature points,
        not just at nodes, and since the source is already a known closed
        form, evaluating it directly is simpler than precomputing and
        threading a nodal array through.
    interior_mask : array, shape (N,)
        Only interior (non-Dirichlet-boundary) nodes contribute to the
        residual -- boundary nodes get their supervision from the model's
        own boundary ansatz upstream, not from this residual.
    energy_floor_weight, energy_floor : float, optional
        Mitigates a known failure mode of residual-squared physics losses:
        such a loss alone can be minimized by a near-zero-amplitude field
        that trivially
        satisfies the residual without solving the PDE. Adding
        `energy_floor_weight * mean(relu(energy_floor - E_grad)^2)`,
        where `E_grad = int kappa*|grad(u)|^2` (which shrinks
        monotonically toward zero as `u -> 0`, unlike the plain
        residual), penalizes that collapse. Both default to `0.0`
        (disabled) -- `energy_floor` has no principled default and should
        be calibrated from a reference solution's own `E_grad` at the
        mesh resolution being trained at, only if collapse is actually
        observed.
    """

    def __init__(
        self,
        galerkin: T3Galerkin,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        interior_mask: jnp.ndarray,
        energy_floor_weight: float = 0.0,
        energy_floor: float = 0.0,
    ) -> None:
        if not bool(jnp.any(interior_mask)):
            raise ValueError(
                "BatchedVariableCoefficientGalerkinResidualLossT3: "
                "interior_mask has no True entries -- every node is on the "
                "boundary, so the residual mean would divide by zero. Check "
                "the mesh's boundary detection."
            )
        self.galerkin = galerkin
        self.kappa_fn = kappa_fn
        self.source_fn = source_fn
        self.interior_mask = interior_mask
        self.energy_floor_weight = energy_floor_weight
        self.energy_floor = energy_floor

    def __call__(
        self,
        u_batch: jnp.ndarray,
        params_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        """`mean_b mean_i(weak_residual_{b,i}^2)` over the instance batch
        and interior nodes.

        Parameters
        ----------
        u_batch : array, shape (B, N)
            The model's predicted field for every instance in the batch,
            already evaluated at `self.galerkin.xy` (and already
            boundary-ansatz'd). There is deliberately no separate `xy`
            argument to keep in sync with it: the mesh is fixed at
            construction, and asserting `xy is self.galerkin.xy` would
            not actually catch a mismatch under `jax.jit` (a traced
            argument is a distinct Tracer object from the concrete array
            it closes over, even when the caller passes the same array),
            so an unused parameter here would buy no real safety.
        params_batch : array, shape (B, P)
            Per-instance parameter vectors passed to `kappa_fn`/
            `source_fn`.
        """
        g = self.galerkin

        residual_batch = jax.vmap(
            lambda u, params: t3_weak_residual(
                g, u, self.kappa_fn, self.source_fn, params
            )
        )(u_batch, params_batch)  # (B, N)

        masked = jnp.where(self.interior_mask[None, :], residual_batch, 0.0)
        n_instances = params_batch.shape[0]
        loss = jnp.sum(masked**2) / (jnp.sum(self.interior_mask) * n_instances)

        if self.energy_floor_weight != 0.0:
            e_grad = jax.vmap(
                lambda u, params: g.grad_energy(u, self.kappa_fn, params)
            )(u_batch, params_batch)
            floor_penalty = jnp.mean(jax.nn.relu(self.energy_floor - e_grad) ** 2)
            loss = loss + self.energy_floor_weight * floor_penalty
        return loss
