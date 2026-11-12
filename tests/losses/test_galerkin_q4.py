# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Correctness properties for Q4, the bilinear-quadrilateral sibling of
:mod:`scirex.experimental.losses.galerkin_t6`'s ``T6Galerkin``:

* :meth:`Q4Galerkin.element_gradient_at_quad` must reproduce the gradient
  of a *bilinear* nodal field exactly (to floating-point precision) on a
  regular, axis-aligned grid -- Q4 shape functions are themselves
  bilinear, and the isoparametric map of an axis-aligned rectangle is
  affine, so there is no discretization error here at all, the direct
  analogue of T3's linear-field-exactness test.
* :func:`q4_weak_residual`, given the exact manufactured solution and its
  exact source, must be small at interior nodes -- same sign-convention
  check as the T3/T6 modules, confirming the Q4 assembly didn't
  introduce a new sign bug.
* the batched loss must match a manual single-instance computation.
* a degenerate element (near-zero Jacobian determinant, both a plain
  zero-area case and a large-scale sliver where a fixed absolute floor
  would miss it) must be rejected at construction.
* ``Q4Galerkin`` must reject connectivity that isn't 4-node.
* the empty-``interior_mask`` guard on the batched loss (this module's
  own explicit divide-by-zero check) must actually fire.
* the energy-floor amplitude-collapse mitigation must be available on Q4
  and behave the same way it does on T3/T6.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from scirex.losses.galerkin_q4 import (
    BatchedVariableCoefficientGalerkinResidualLossQ4,
    Q4Galerkin,
    q4_weak_residual,
)
from tests import tolerances


def _regular_grid_mesh_q4(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit square [0,1]^2, an n x n regular grid of axis-aligned cells,
    one Q4 element per cell. Node order per element: bottom-left,
    bottom-right, top-right, top-left (CCW), matching the module
    docstring's convention."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=-1)

    def idx(i: int, j: int) -> int:
        return i * n + j

    quad = []
    for i in range(n - 1):
        for j in range(n - 1):
            bl, br, tr, tl = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            quad.append([bl, br, tr, tl])
    return xy, np.array(quad, dtype=np.int32)


def test_element_gradient_is_exact_for_a_bilinear_field() -> None:
    xy, quad = _regular_grid_mesh_q4(n=6)
    galerkin = Q4Galerkin(xy, quad)

    # u(x, y) = 1 + 2x + 3y + 4xy -> grad(u) = (2 + 4y, 3 + 4x). A genuine
    # bilinear polynomial, exactly representable by Q4's bilinear basis on
    # every axis-aligned element (the isoparametric map is affine here),
    # so this has no discretization error.
    def u_exact(p: np.ndarray) -> float:
        x, y = p
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * y

    def grad_exact(p: np.ndarray) -> tuple[float, float]:
        x, y = p
        return 2.0 + 4.0 * y, 3.0 + 4.0 * x

    u = np.array([u_exact(p) for p in xy])
    du_dx, du_dy = galerkin.element_gradient_at_quad(jnp.asarray(u, dtype=jnp.float32))
    du_dx, du_dy = np.asarray(du_dx), np.asarray(du_dy)  # (Q, T)

    phys_pts = np.asarray(galerkin.quadrature_points_physical())  # (Q, T, 2)
    for q in range(phys_pts.shape[0]):
        expected = np.array([grad_exact(p) for p in phys_pts[q]])
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


def _boundary_mask(xy: np.ndarray) -> np.ndarray:
    return (xy[:, 0] == 0.0) | (xy[:, 0] == 1.0) | (xy[:, 1] == 0.0) | (xy[:, 1] == 1.0)


def test_weak_residual_is_small_for_the_exact_manufactured_solution() -> None:
    xy, quad = _regular_grid_mesh_q4(n=25)
    galerkin = Q4Galerkin(xy, quad)
    boundary_mask = _boundary_mask(xy)
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy])
    params = jnp.array([2.0])

    residual = q4_weak_residual(
        galerkin,
        jnp.asarray(u_exact, dtype=jnp.float32),
        _kappa_fn_const,
        _source_fn_const,
        params,
    )
    interior = ~boundary_mask
    interior_residual = np.asarray(residual)[interior]

    assert np.abs(interior_residual).mean() < 0.01


def test_batched_loss_matches_manual_single_instance_computation() -> None:
    xy, quad = _regular_grid_mesh_q4(n=10)
    galerkin = Q4Galerkin(xy, quad)
    boundary_mask = _boundary_mask(xy)
    interior_mask = jnp.asarray(~boundary_mask)
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy])

    loss_fn = BatchedVariableCoefficientGalerkinResidualLossQ4(
        galerkin, _kappa_fn_const, _source_fn_const, interior_mask
    )

    batch_size = 4
    u_batch = jnp.tile(
        jnp.asarray(u_exact, dtype=jnp.float32)[None, :], (batch_size, 1)
    )
    params_batch = jnp.tile(jnp.array([2.0]), (batch_size, 1))

    loss = loss_fn(u_batch, params_batch)

    manual_residual = q4_weak_residual(
        galerkin,
        jnp.asarray(u_exact, dtype=jnp.float32),
        _kappa_fn_const,
        _source_fn_const,
        jnp.asarray([2.0], dtype=jnp.float32),
    )
    manual_masked = jnp.where(interior_mask, manual_residual, 0.0)
    manual_loss = jnp.sum(manual_masked**2) / jnp.sum(interior_mask)

    np.testing.assert_allclose(
        np.asarray(loss), np.asarray(manual_loss), rtol=tolerances.rtol(1e-5)
    )


def test_degenerate_element_is_rejected_at_construction() -> None:
    """Two corners collapsed onto the same point (top-right = bottom-right,
    top-left = bottom-left) -> the "quad" is really a zero-area line
    segment, zero Jacobian determinant at every quadrature point."""
    xy_degenerate = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    quad_degenerate = np.array([[0, 1, 2, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="degenerate"):
        Q4Galerkin(xy_degenerate, quad_degenerate)


def test_degenerate_element_is_rejected_even_on_a_large_scale_mesh() -> None:
    """Same reasoning as T3Galerkin/T6Galerkin's analogous test: a fixed
    absolute Jacobian-determinant floor only catches degeneracy at one
    particular coordinate scale. This element is an extremely thin
    axis-aligned rectangle (scale in x, ~1e-9*scale in y) -- its raw
    |det_j| is not small in absolute terms at this scale, but is tiny
    relative to the element's own (x-direction) size, which is exactly
    what the relative floor is designed to catch."""
    scale = 1e5
    thin = 1e-9 * scale
    xy = np.array([[0.0, 0.0], [scale, 0.0], [scale, thin], [0.0, thin]])
    quad = np.array([[0, 1, 2, 3]], dtype=np.int32)

    with pytest.raises(ValueError, match="degenerate"):
        Q4Galerkin(xy, quad)


def test_q4galerkin_rejects_non_4_node_connectivity() -> None:
    xy, quad = _regular_grid_mesh_q4(n=4)
    tri_shaped = quad[:, :3]  # drop the 4th node -> looks like T3 connectivity

    with pytest.raises(ValueError, match="4-node"):
        Q4Galerkin(xy, tri_shaped)


def test_loss_rejects_interior_mask_with_no_interior_nodes() -> None:
    xy, quad = _regular_grid_mesh_q4(n=3)
    galerkin = Q4Galerkin(xy, quad)
    all_boundary_mask = jnp.zeros((xy.shape[0],), dtype=bool)

    with pytest.raises(ValueError, match="no True entries"):
        BatchedVariableCoefficientGalerkinResidualLossQ4(
            galerkin, _kappa_fn_const, _source_fn_const, all_boundary_mask
        )


def test_energy_floor_penalizes_amplitude_collapse() -> None:
    """Q4 analogue of the T3/T6 modules' own energy-floor test: a
    trivially zero field satisfies the plain residual, but the energy
    floor must penalize it whenever energy_floor > 0."""
    xy, quad = _regular_grid_mesh_q4(n=6)
    galerkin = Q4Galerkin(xy, quad)
    interior_mask = jnp.asarray(~_boundary_mask(xy))

    def source_fn_zero(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        del params, point
        return jnp.array(0.0)

    loss_no_floor = BatchedVariableCoefficientGalerkinResidualLossQ4(
        galerkin,
        _kappa_fn_const,
        source_fn_zero,
        interior_mask,
        energy_floor_weight=0.0,
    )
    loss_with_floor = BatchedVariableCoefficientGalerkinResidualLossQ4(
        galerkin,
        _kappa_fn_const,
        source_fn_zero,
        interior_mask,
        energy_floor_weight=1.0,
        energy_floor=1.0,
    )

    u_zero = jnp.zeros((1, xy.shape[0]), dtype=jnp.float32)
    params_batch = jnp.array([[2.0]])

    assert float(loss_no_floor(u_zero, params_batch)) == 0.0
    assert float(loss_with_floor(u_zero, params_batch)) > 0.0
