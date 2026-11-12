# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Correctness properties for T6, roughly in order of directness:

* :meth:`T6Galerkin.element_gradient_at_quad` must reproduce the gradient
  of a *quadratic* nodal field exactly (to floating-point precision) -- T6
  shape functions are themselves quadratic (degree-2 complete), so this
  has no discretization error at all, the direct analogue of T3's
  linear-field-exactness test.
* :func:`t6_weak_residual`, given the exact manufactured solution and its
  exact source, must be small at interior nodes -- same sign-convention
  check as the T3 module, ported to confirm the T6 assembly didn't
  introduce a new sign bug.
* the batched loss must match a manual single-instance computation.
* a degenerate element (near-zero Jacobian determinant) must be rejected
  at construction, mirroring T3's own degenerate-triangle guard, rather
  than silently producing inf/nan gradients later.
* the energy-floor amplitude-collapse mitigation must be available on T6
  and behave the same way it does on T3.

No gmsh dependency: a T6 mesh is built here by hand from a regular grid
(add a de-duplicated mid-edge node per T3 edge), since gmsh is not
expected to be installed on the machine running these tests (see
gmsh_mesh.py's own docstring).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.losses.galerkin_t6 import (
    BatchedVariableCoefficientGalerkinResidualLossT6,
    T6Galerkin,
    t6_weak_residual,
)


def _regular_grid_mesh_t3(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit square [0,1]^2, an n x n regular grid cut into two triangles
    per cell. Returns (xy, tri) -- no boundary mask needed here, computed
    separately once mid-edge nodes exist."""
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
    T6 node order (3 corners, then mid(1,2), mid(2,3), mid(3,1)).
    Boundary status of a mid-edge node matches its edge's own boundary
    status (an edge shared by exactly one triangle is a boundary edge),
    not just "both endpoints happen to be boundary nodes" -- a diagonal
    between two hull corners is not itself a boundary edge."""
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


def test_element_gradient_is_exact_for_a_quadratic_field() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=6)
    xy6, tri6, _ = _elevate_to_t6(xy3, tri3)
    galerkin = T6Galerkin(xy6, tri6)

    # u(x, y) = 2x^2 - 3xy + y^2 + 4x - y + 1 -> grad(u) = (4x - 3y + 4, -3x + 2y - 1).
    def u_exact(p: np.ndarray) -> float:
        x, y = p
        return 2 * x**2 - 3 * x * y + y**2 + 4 * x - y + 1

    def grad_exact(p: np.ndarray) -> tuple[float, float]:
        x, y = p
        return 4 * x - 3 * y + 4, -3 * x + 2 * y - 1

    u = np.array([u_exact(p) for p in xy6])
    du_dx, du_dy = galerkin.element_gradient_at_quad(jnp.asarray(u, dtype=jnp.float32))
    du_dx, du_dy = np.asarray(du_dx), np.asarray(du_dy)  # (Q, T)

    # Physical coordinates of each quadrature point (valid since this mesh
    # is straight-sided: the isoparametric map reduces to the affine map
    # of the 3 corners -- see the module docstring).
    from scirex.experimental.losses.galerkin_t6 import _QUAD_POINTS

    for q in range(_QUAD_POINTS.shape[0]):
        xi, eta = float(_QUAD_POINTS[q, 0]), float(_QUAD_POINTS[q, 1])
        l1, l2, l3 = 1.0 - xi - eta, xi, eta
        x1, x2, x3 = xy6[tri6[:, 0]], xy6[tri6[:, 1]], xy6[tri6[:, 2]]
        phys_pts = l1 * x1 + l2 * x2 + l3 * x3  # (T, 2)
        expected = np.array([grad_exact(p) for p in phys_pts])
        np.testing.assert_allclose(du_dx[q], expected[:, 0], atol=1e-4)
        np.testing.assert_allclose(du_dy[q], expected[:, 1], atol=1e-4)


def _kappa_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    del point
    return params[0]


def _u_exact_fn(point: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * point[0]) * jnp.sin(jnp.pi * point[1])


def _source_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    kappa = _kappa_fn_const(params, point)
    return -2.0 * jnp.pi**2 * kappa * _u_exact_fn(point)


def test_weak_residual_is_small_for_the_exact_manufactured_solution() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=15)
    xy6, tri6, boundary_mask = _elevate_to_t6(xy3, tri3)
    galerkin = T6Galerkin(xy6, tri6)
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy6])
    params = jnp.array([2.0])

    residual = t6_weak_residual(
        galerkin,
        jnp.asarray(u_exact, dtype=jnp.float32),
        _kappa_fn_const,
        _source_fn_const,
        params,
    )
    interior = ~boundary_mask
    interior_residual = np.asarray(residual)[interior]

    # T6 is higher-order than T3, so this should be small even at a
    # coarser mesh than T3's own equivalent test needed.
    assert np.abs(interior_residual).mean() < 0.01


@pytest.mark.needs_x64
def test_batched_loss_matches_manual_single_instance_computation() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=10)
    xy6, tri6, boundary_mask = _elevate_to_t6(xy3, tri3)
    galerkin = T6Galerkin(xy6, tri6)
    interior_mask = jnp.asarray(~boundary_mask)
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy6])

    loss_fn = BatchedVariableCoefficientGalerkinResidualLossT6(
        galerkin, _kappa_fn_const, _source_fn_const, interior_mask
    )

    batch_size = 4
    u_batch = jnp.tile(
        jnp.asarray(u_exact, dtype=jnp.float32)[None, :], (batch_size, 1)
    )
    params_batch = jnp.tile(jnp.array([2.0]), (batch_size, 1))

    loss = loss_fn(u_batch, params_batch)

    manual_residual = t6_weak_residual(
        galerkin,
        jnp.asarray(u_exact, dtype=jnp.float32),
        _kappa_fn_const,
        _source_fn_const,
        jnp.asarray([2.0], dtype=jnp.float32),
    )
    manual_masked = jnp.where(interior_mask, manual_residual, 0.0)
    manual_loss = jnp.sum(manual_masked**2) / jnp.sum(interior_mask)

    # A fixed value, not tolerances.rtol: the loss is a sum of squared near-zero residuals
    # (about 5e-7), so float64 rounding of the residual alone gives a 2e-6 relative gap.
    np.testing.assert_allclose(np.asarray(loss), np.asarray(manual_loss), rtol=1e-5)


def test_degenerate_element_is_rejected_at_construction() -> None:
    """Two corners collapsed onto the same point -> zero Jacobian
    determinant at every quadrature point. T3Galerkin raises on the
    analogous zero-area case; T6Galerkin must too, not silently produce
    inf/nan gradients the first time a caller divides by det_j."""
    xy_degenerate = np.array(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]
    )
    tri_degenerate = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)

    with pytest.raises(ValueError, match="degenerate"):
        T6Galerkin(xy_degenerate, tri_degenerate)


def test_degenerate_element_is_rejected_even_on_a_large_scale_mesh() -> None:
    """Same reasoning as T3Galerkin's analogous test: a fixed absolute
    Jacobian-determinant floor only catches degeneracy at one particular
    coordinate scale. This element's corner triangle has the same
    near-collinear-sliver shape as that test (raw |det| far above what a
    1e-12 absolute floor would flag), just elevated to T6 with exact
    mid-edge nodes."""
    scale = 1e5
    c0, c1, c2 = (
        np.array([0.0, 0.0]),
        np.array([1.0 * scale, 0.0]),
        np.array([0.5 * scale, 1e-9 * scale]),
    )
    xy = np.array([c0, c1, c2, (c0 + c1) / 2.0, (c1 + c2) / 2.0, (c2 + c0) / 2.0])
    tri = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32)

    with pytest.raises(ValueError, match="degenerate"):
        T6Galerkin(xy, tri)


def test_t6galerkin_rejects_t3_shaped_connectivity() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=4)

    with pytest.raises(ValueError, match="6-node"):
        T6Galerkin(xy3, tri3)


def test_loss_rejects_interior_mask_with_no_interior_nodes() -> None:
    xy3, tri3 = _regular_grid_mesh_t3(n=3)
    xy6, tri6, _ = _elevate_to_t6(xy3, tri3)
    galerkin = T6Galerkin(xy6, tri6)
    all_boundary_mask = jnp.zeros((xy6.shape[0],), dtype=bool)

    with pytest.raises(ValueError, match="no True entries"):
        BatchedVariableCoefficientGalerkinResidualLossT6(
            galerkin, _kappa_fn_const, _source_fn_const, all_boundary_mask
        )


def test_energy_floor_penalizes_amplitude_collapse() -> None:
    """T6 analogue of the T3 module's own energy-floor test: a trivially
    zero field satisfies the plain residual, but the energy floor must
    penalize it whenever energy_floor > 0 -- this parameter didn't exist
    on the T6 loss class until this test was added."""
    xy3, tri3 = _regular_grid_mesh_t3(n=6)
    xy6, tri6, boundary_mask = _elevate_to_t6(xy3, tri3)
    galerkin = T6Galerkin(xy6, tri6)
    interior_mask = jnp.asarray(~boundary_mask)

    def source_fn_zero(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        del params, point
        return jnp.array(0.0)

    loss_no_floor = BatchedVariableCoefficientGalerkinResidualLossT6(
        galerkin,
        _kappa_fn_const,
        source_fn_zero,
        interior_mask,
        energy_floor_weight=0.0,
    )
    loss_with_floor = BatchedVariableCoefficientGalerkinResidualLossT6(
        galerkin,
        _kappa_fn_const,
        source_fn_zero,
        interior_mask,
        energy_floor_weight=1.0,
        energy_floor=1.0,
    )

    u_zero = jnp.zeros((1, xy6.shape[0]), dtype=jnp.float32)
    params_batch = jnp.array([[2.0]])

    assert float(loss_no_floor(u_zero, params_batch)) == 0.0
    assert float(loss_with_floor(u_zero, params_batch)) > 0.0
