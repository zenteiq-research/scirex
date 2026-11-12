# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Correctness properties, roughly in order of how directly they check the
actual mathematics:

* :func:`T3Galerkin.element_gradient` must reproduce the gradient of a
  *linear* nodal field exactly (to floating-point precision) -- T3 shape
  functions are themselves linear, so this has no discretization error at
  all, unlike the manufactured-solution checks below.
* :func:`t3_weak_residual`, given the *exact* manufactured solution and its
  exact source, must be small at interior nodes -- this is the sign
  convention check the module docstring warns about: get the ``+``/``-``
  between the two element terms backwards, and this residual would not be
  small, it would be wrong by roughly twice the true term's own magnitude.
* the 3-point quadrature rule must actually beat a 1-point (centroid)
  rule on a 2D-varying coefficient field -- the concrete regression test
  for the bug this module's quadrature upgrade fixes (see the module
  docstring's "Quadrature" section): a real training run showed
  kappa(x,y) families scoring much worse than kappa(x)-only families at
  matched mesh resolution, traced to under-integrated element coefficients
  rather than under-resolved element geometry.
* :class:`T3Galerkin` must reject a T6-shaped ``(T, 6)`` connectivity array
  instead of silently truncating it to 3 columns, and the loss class must
  reject an ``interior_mask`` with no interior nodes instead of dividing by
  zero -- both were unvalidated when this module first landed.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.losses.galerkin import (
    BatchedVariableCoefficientGalerkinResidualLossT3,
    T3Galerkin,
    t3_weak_residual,
)
from tests import tolerances


def _regular_grid_mesh(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit square [0,1]^2, an n x n regular grid cut into two triangles
    per cell. Returns (xy, tri, boundary_mask)."""
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
    tri = np.array(tri, dtype=np.int32)

    boundary_mask = (
        (xy[:, 0] == 0) | (xy[:, 0] == 1) | (xy[:, 1] == 0) | (xy[:, 1] == 1)
    )
    return xy, tri, boundary_mask


def test_element_gradient_is_exact_for_a_linear_field() -> None:
    xy, tri, _ = _regular_grid_mesh(n=6)
    galerkin = T3Galerkin(xy, tri)

    # u(x, y) = 2x - 3y + 1 -> grad(u) = (2, -3) everywhere.
    u = 2.0 * xy[:, 0] - 3.0 * xy[:, 1] + 1.0
    du_dx, du_dy = galerkin.element_gradient(jnp.asarray(u, dtype=jnp.float32))

    np.testing.assert_allclose(np.asarray(du_dx), 2.0, atol=tolerances.atol(1e-5))
    np.testing.assert_allclose(np.asarray(du_dy), -3.0, atol=tolerances.atol(1e-5))


def _constant_kappa_problem(
    n: int, kappa_value: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """u = sin(pi x) sin(pi y) (zero on the unit square's boundary),
    div(kappa * grad(u)) = -2 pi^2 kappa u for constant kappa -- exact
    source in closed form, no autodiff needed for this simple case."""
    xy, tri, boundary_mask = _regular_grid_mesh(n)
    return xy, tri, boundary_mask, np.array([kappa_value])


def _kappa_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    del point
    return params[0]


def _u_exact(point: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * point[0]) * jnp.sin(jnp.pi * point[1])


def _source_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    kappa = _kappa_fn_const(params, point)
    return -2.0 * jnp.pi**2 * kappa * _u_exact(point)


def test_weak_residual_is_small_for_the_exact_manufactured_solution() -> None:
    xy, tri, boundary_mask, params = _constant_kappa_problem(n=25)
    galerkin = T3Galerkin(xy, tri)
    u_exact = np.asarray([_u_exact(p) for p in xy])

    residual = t3_weak_residual(
        galerkin,
        jnp.asarray(u_exact, dtype=jnp.float32),
        _kappa_fn_const,
        _source_fn_const,
        jnp.asarray(params, dtype=jnp.float32),
    )
    interior = np.asarray(~boundary_mask)
    interior_residual = np.asarray(residual)[interior]

    # Not exactly zero (a smooth, non-polynomial solution/source still has
    # some quadrature error even at 3 points), but small relative to the
    # element-scale magnitude of the two terms being subtracted -- see
    # test_flipped_sign_convention_is_much_larger below for the contrast.
    assert np.abs(interior_residual).mean() < 0.05


def test_flipped_sign_convention_is_much_larger() -> None:
    """Sanity check on the check above: at the exact solution, our
    correct-sign residual is canceling two comparable-magnitude terms
    (grad_dot ~ -src), so it is small purely from discretization error. The
    *wrong* sign convention (int kappa*gradu.gradN - int source*N, the
    -div(kappa*grad(u))=f convention some references use) instead *adds*
    two comparable-magnitude, same-signed quantities -- it should be much
    larger than the correct residual at the same mesh, not just nonzero,
    otherwise the small-residual test above would not actually be
    discriminating between the two conventions."""
    xy, tri, boundary_mask, params = _constant_kappa_problem(n=25)
    galerkin = T3Galerkin(xy, tri)
    g = galerkin
    u_exact = np.asarray([_u_exact(p) for p in xy])
    params_j = jnp.asarray(params, dtype=jnp.float32)

    du_dx, du_dy = g.element_gradient(jnp.asarray(u_exact, dtype=jnp.float32))
    kappa_elem = np.asarray(g.element_kappa(_kappa_fn_const, params_j))
    quad_pts = np.asarray(g.quadrature_points())
    source_q = np.stack(
        [
            np.array([_source_fn_const(params_j, jnp.asarray(pt)) for pt in pts])
            for pts in quad_pts
        ],
        axis=0,
    )
    source_elem = source_q.mean(axis=0)

    grad_dot = kappa_elem[:, None] * (
        np.asarray(du_dx)[:, None] * np.asarray(g.b) * 0.5
        + np.asarray(du_dy)[:, None] * np.asarray(g.c) * 0.5
    )
    src = source_elem[:, None] * np.asarray(g.area)[:, None] / 3.0
    wrong_sign_residual = np.asarray(g.scatter_to_nodes(jnp.asarray(grad_dot - src)))
    correct_sign_residual = np.asarray(
        t3_weak_residual(
            g,
            jnp.asarray(u_exact, dtype=jnp.float32),
            _kappa_fn_const,
            _source_fn_const,
            params_j,
        )
    )

    interior = np.asarray(~boundary_mask)
    wrong_mean = np.abs(wrong_sign_residual[interior]).mean()
    correct_mean = np.abs(correct_sign_residual[interior]).mean()
    assert wrong_mean > 10.0 * correct_mean


def test_batched_loss_matches_manual_single_instance_computation() -> None:
    xy, tri, boundary_mask, _params = _constant_kappa_problem(n=15, kappa_value=2.0)
    galerkin = T3Galerkin(xy, tri)
    interior_mask = jnp.asarray(~boundary_mask)
    u_exact = np.asarray([_u_exact(p) for p in xy])

    loss_fn = BatchedVariableCoefficientGalerkinResidualLossT3(
        galerkin, _kappa_fn_const, _source_fn_const, interior_mask
    )

    batch_size = 4
    u_batch = jnp.tile(
        jnp.asarray(u_exact, dtype=jnp.float32)[None, :], (batch_size, 1)
    )
    params_batch = jnp.tile(jnp.array([2.0]), (batch_size, 1))

    loss = loss_fn(u_batch, params_batch)

    manual_residual = t3_weak_residual(
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


def test_energy_floor_penalizes_amplitude_collapse() -> None:
    """A trivially-zero field satisfies the plain residual (0 - 0 = 0
    everywhere kappa*grad(0)=0 and source is also 0), but the energy floor
    must penalize it whenever energy_floor > 0."""
    xy, tri, boundary_mask = _regular_grid_mesh(n=10)
    galerkin = T3Galerkin(xy, tri)
    interior_mask = jnp.asarray(~boundary_mask)

    def source_fn_zero(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        del params, point
        return jnp.array(0.0)

    loss_no_floor = BatchedVariableCoefficientGalerkinResidualLossT3(
        galerkin,
        _kappa_fn_const,
        source_fn_zero,
        interior_mask,
        energy_floor_weight=0.0,
    )
    loss_with_floor = BatchedVariableCoefficientGalerkinResidualLossT3(
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


def _kappa_fn_2d(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """kappa(x,y) = 2 + A*sin(2*pi*f1*x)*sin(2*pi*f2*y) -- the family whose
    within-element curvature a 1-point quadrature rule under-resolves."""
    a, f1, f2 = params[0], params[1], params[2]
    return 2.0 + a * jnp.sin(2.0 * jnp.pi * f1 * point[0]) * jnp.sin(
        2.0 * jnp.pi * f2 * point[1]
    )


def _element_kappa_1point(
    galerkin: T3Galerkin,
    kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """The old (pre-fix) centroid/1-point rule: average kappa_fn's value at
    the 3 nodes -- reimplemented here (not imported) since the fixed
    module no longer exposes it, precisely because it should not be used."""
    g = galerkin
    x1, x2, x3 = g.xy[g.n1], g.xy[g.n2], g.xy[g.n3]
    k1 = jax.vmap(lambda p: kappa_fn(params, p))(x1)
    k2 = jax.vmap(lambda p: kappa_fn(params, p))(x2)
    k3 = jax.vmap(lambda p: kappa_fn(params, p))(x3)
    return (k1 + k2 + k3) / 3.0


def test_3point_quadrature_beats_1point_for_a_2d_varying_coefficient() -> None:
    """The actual regression test: on a coefficient field that varies in
    both x and y within an element, the 3-point quadrature rule's element
    kappa must be closer to the true element-average kappa (computed by a
    fine reference quadrature) than the old 1-point/nodal-average rule
    would have been -- confirming the fix addresses the diagnosed
    problem, not just a plausible-sounding guess."""
    xy, tri, _ = _regular_grid_mesh(n=8)  # deliberately coarse: h ~ 0.14
    galerkin = T3Galerkin(xy, tri)
    params = jnp.array(
        [0.7, 3.0, 3.0]
    )  # near the upper end of the PINO study's A/f range

    # High-order reference: a dense regular grid of points per triangle,
    # plain average -- converges to the true element-integral average of
    # kappa as the grid refines, independent of the 1-/3-point rules being
    # compared against it.
    def reference_element_kappa(tri_idx: int) -> float:
        x1, x2, x3 = (
            np.asarray(xy[tri[tri_idx, 0]]),
            np.asarray(xy[tri[tri_idx, 1]]),
            np.asarray(xy[tri[tri_idx, 2]]),
        )
        vals = []
        m = 12
        for i in range(m):
            for j in range(m - i):
                u_, v_ = i / m, j / m
                w = 1.0 - u_ - v_
                pt = w * x1 + u_ * x2 + v_ * x3
                vals.append(float(_kappa_fn_2d(params, jnp.asarray(pt))))
        return float(np.mean(vals))

    ref = np.array([reference_element_kappa(t) for t in range(tri.shape[0])])

    kappa_3pt = np.asarray(galerkin.element_kappa(_kappa_fn_2d, params))
    kappa_1pt = np.asarray(_element_kappa_1point(galerkin, _kappa_fn_2d, params))

    err_3pt = np.abs(kappa_3pt - ref).mean()
    err_1pt = np.abs(kappa_1pt - ref).mean()

    assert err_3pt < err_1pt


def test_t3galerkin_rejects_t6_shaped_connectivity() -> None:
    xy, tri, _ = _regular_grid_mesh(n=4)
    tri6 = np.concatenate([tri, tri[:, :1], tri[:, :1], tri[:, :1]], axis=1)
    assert tri6.shape[-1] == 6

    with pytest.raises(ValueError, match="3-node"):
        T3Galerkin(xy, tri6)


def test_loss_rejects_interior_mask_with_no_interior_nodes() -> None:
    xy, tri, _ = _regular_grid_mesh(n=3)
    galerkin = T3Galerkin(xy, tri)
    all_boundary_mask = jnp.zeros((xy.shape[0],), dtype=bool)

    with pytest.raises(ValueError, match="no True entries"):
        BatchedVariableCoefficientGalerkinResidualLossT3(
            galerkin, _kappa_fn_const, _source_fn_const, all_boundary_mask
        )


def test_all_coincident_points_are_rejected() -> None:
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    tri = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="degenerate"):
        T3Galerkin(xy, tri)


def test_degenerate_triangle_is_rejected_even_on_a_large_scale_mesh() -> None:
    """A fixed absolute area/determinant threshold (e.g. 1e-12) only ever
    catches degeneracy at one particular coordinate scale: a triangle whose
    raw determinant is far above such a floor can still be a near-collinear
    sliver *relative to its own size* once the mesh's coordinates are large
    (a bigger domain, different units, ...). This triangle's raw |det| is
    10.0 -- nowhere near degenerate by an absolute 1e-12 standard -- but its
    area is ~5e-10 of its own longest edge squared, an aspect ratio no valid
    element should have."""
    scale = 1e5
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0 * scale, 0.0],
            [0.5 * scale, 1e-9 * scale],  # almost exactly on the base edge
        ]
    )
    tri = np.array([[0, 1, 2]], dtype=np.int32)

    det = (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1]) - (xy[1, 1] - xy[0, 1]) * (
        xy[2, 0] - xy[0, 0]
    )
    assert abs(det) > 1e-12  # would have slipped past the old absolute check

    with pytest.raises(ValueError, match="degenerate"):
        T3Galerkin(xy, tri)
