# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Cross-check that T3 and T6 agree on a shared manufactured problem.

Each of ``test_galerkin.py``/``test_galerkin_t6.py`` only checks its
own element family against itself, which is exactly how the T3 load-vector
under-integration bug (T3's source term evaluated at 3 quadrature points but
then flattened to a uniform 1/3 weight per node, instead of the point's own
linear shape-function value) went undetected: T3's own small-residual test
used a tolerance loose enough to pass with the wrong per-node split, since
the *element total* was preserved exactly (only the split among the three
nodes was wrong). T6's analogous load vector already weighted correctly, so
comparing the two against each other -- not just each against a loose
absolute threshold -- is what would have caught it.

T6 is second-order accurate and T3 is first-order, so their residuals are
not expected to be numerically equal even when both are correct; this checks
that T3's mean interior residual stays within a small multiple of T6's
(rather than the two being close to equal), calibrated against the actual
gap this bug produced (see the module docstring reference above).
"""

import jax
import jax.numpy as jnp
import numpy as np

from scirex.experimental.losses.galerkin_t6 import T6Galerkin, t6_weak_residual
from scirex.losses.galerkin import T3Galerkin, t3_weak_residual


def _regular_grid_mesh_t3(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit square [0,1]^2, an n x n regular grid cut into two triangles
    per cell."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=-1)

    def idx(i: int, j: int) -> int:
        return i * n + j

    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tri.append([a, b, c])
            tri.append([a, c, d])
    return xy, np.array(tri, dtype=np.int32)


def _elevate_to_t6(
    xy3: np.ndarray, tri3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add one de-duplicated mid-edge node per shared T3 edge, in gmsh's
    T6 node order -- gives a T6 mesh occupying the exact same triangles as
    ``(xy3, tri3)``, so the two element families are compared on the same
    geometry rather than an unrelated mesh."""
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

    tri6 = []
    for a, b, c in tri3:
        tri6.append([a, b, c, get_mid(a, b), get_mid(b, c), get_mid(c, a)])

    xy6 = np.array(nodes)
    tri6 = np.array(tri6, dtype=np.int32)
    corner_boundary = (
        (xy3[:, 0] == 0) | (xy3[:, 0] == 1) | (xy3[:, 1] == 0) | (xy3[:, 1] == 1)
    )
    boundary_mask6 = np.concatenate(
        [corner_boundary, np.array(boundary_extra, dtype=bool)]
    )
    return xy6, tri6, boundary_mask6


def _kappa_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    del params, point
    return jnp.array(2.0)


def _u_exact_point(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    del params
    x, y = point[0], point[1]
    # Higher frequency than the sign-convention tests' sin(pi x)sin(pi y) --
    # deliberately so the source term has real within-element curvature,
    # the condition under which the fixed bug's ~20% per-node error showed
    # up (see the reviewed commit's numeric check).
    return jnp.sin(3.0 * jnp.pi * x) * jnp.sin(3.0 * jnp.pi * y)


def _source_fn_exact(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """div(kappa * grad(u_exact)) via autodiff on the true flux -- exact
    for whatever kappa/u_exact above are, no hand-differentiation to get
    wrong."""

    def flux(q: jnp.ndarray) -> jnp.ndarray:
        return _kappa_fn_const(params, q) * jax.grad(
            lambda qq: _u_exact_point(params, qq)
        )(q)

    return jnp.trace(jax.jacfwd(flux)(point))


def test_t3_residual_stays_within_a_small_multiple_of_t6s() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=15)
    xy6, tri6, boundary_mask6 = _elevate_to_t6(xy3, tri3)
    boundary_mask3 = (
        (xy3[:, 0] == 0) | (xy3[:, 0] == 1) | (xy3[:, 1] == 0) | (xy3[:, 1] == 1)
    )
    params = jnp.array([0.0])

    g3 = T3Galerkin(xy3, tri3)
    g6 = T6Galerkin(xy6, tri6)
    u3 = jnp.asarray(
        [_u_exact_point(params, jnp.asarray(p)) for p in xy3], dtype=jnp.float32
    )
    u6 = jnp.asarray(
        [_u_exact_point(params, jnp.asarray(p)) for p in xy6], dtype=jnp.float32
    )

    r3 = np.asarray(t3_weak_residual(g3, u3, _kappa_fn_const, _source_fn_exact, params))
    r6 = np.asarray(t6_weak_residual(g6, u6, _kappa_fn_const, _source_fn_exact, params))

    mean_r3 = np.abs(r3[~boundary_mask3]).mean()
    mean_r6 = np.abs(r6[~boundary_mask6]).mean()

    # T6 is second-order accurate and T3 first-order, so T3's residual is
    # expected to be larger, not equal -- but on this mesh a correct T3
    # stays under ~6x T6's; the under-integrated load vector pushed it past
    # ~10x (see module docstring).
    assert mean_r3 < 8.0 * mean_r6
