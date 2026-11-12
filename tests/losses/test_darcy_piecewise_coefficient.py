# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Correctness properties for the piecewise-coefficient Darcy
infrastructure, roughly in order of how central they are to this
module's own stated design:

* the grid-aligned T3/Q4 mesh builders' entire reason to exist is that
  every element sits inside exactly one coefficient-grid cell --
  verified directly by evaluating :func:`kappa_fn` at every quadrature
  point of every element and checking it never varies within an
  element (the property the module docstring claims, not just a mesh
  shape/count smoke test).
* :func:`kappa_fn`'s nearest-cell lookup, including the boundary/clip
  edge cases (``point`` exactly at 0 or 1).
* :func:`source_fn` is the constant ``-1.0`` regardless of its inputs.
* :func:`sample_piecewise_coefficient` only ever returns ``{lo, hi}``
  values, at the requested shape, at both native and downsampled
  resolution.
* :func:`assemble_and_solve` (T3) and :func:`assemble_and_solve_q4` (Q4)
  against a shared manufactured solution -- the strongest test here,
  since it exercises the full assembly + sparse-solve pipeline
  (stiffness matrix, load vector, sign convention, boundary rows) on
  both element types at once, cross-checked against each other.
* :func:`energy_functional`/:func:`energy_functional_q4`'s stated energy
  identity, ``J(u_true) = -0.5 * E_grad(u_true)``, and their batched
  loss classes matching a manual single-instance computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.losses.darcy_piecewise_coefficient import (
    BatchedEnergyFunctionalLoss,
    BatchedEnergyFunctionalLossQ4,
    assemble_and_solve,
    assemble_and_solve_q4,
    build_grid_aligned_q4_mesh,
    build_grid_aligned_t3_mesh,
    energy_functional,
    energy_functional_q4,
    kappa_fn,
    sample_piecewise_coefficient,
    source_fn,
)
from scirex.losses.galerkin import T3Galerkin
from scirex.losses.galerkin_q4 import Q4Galerkin


def _checkerboard_grid(
    resolution: int, lo: float = 3.0, hi: float = 12.0
) -> np.ndarray:
    """A deterministic (non-random) 2-value coefficient grid -- enough
    variation to exercise the nearest-cell lookup without depending on
    :func:`sample_piecewise_coefficient`'s randomness."""
    i, j = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    return np.where((i + j) % 2 == 0, hi, lo).astype(np.float32)


def test_t3_mesh_is_grid_aligned_kappa_constant_within_every_element() -> None:
    resolution = 8
    a_grid = jnp.asarray(_checkerboard_grid(resolution))
    xy, tri, _ = build_grid_aligned_t3_mesh(resolution)
    galerkin = T3Galerkin(jnp.asarray(xy), jnp.asarray(tri))

    kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn(a_grid, pt))(pts))(
        galerkin.quad_pts
    )  # (3, T)
    # Every quadrature point of a given triangle must see the same kappa --
    # the entire point of a grid-aligned mesh (no element straddles a
    # material interface).
    np.testing.assert_array_equal(np.asarray(kappa_q[0]), np.asarray(kappa_q[1]))
    np.testing.assert_array_equal(np.asarray(kappa_q[0]), np.asarray(kappa_q[2]))


def test_q4_mesh_is_grid_aligned_kappa_constant_within_every_element() -> None:
    resolution = 8
    a_grid = jnp.asarray(_checkerboard_grid(resolution))
    xy, quad, _ = build_grid_aligned_q4_mesh(resolution)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))

    quad_phys_pts = galerkin.quadrature_points_physical()  # (Q, T, 2)
    kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn(a_grid, pt))(pts))(
        quad_phys_pts
    )  # (Q, T)
    for q in range(1, kappa_q.shape[0]):
        np.testing.assert_array_equal(np.asarray(kappa_q[0]), np.asarray(kappa_q[q]))


def test_t3_and_q4_meshes_agree_on_nodes_and_boundary_mask() -> None:
    """build_grid_aligned_q4_mesh's own docstring claims "same node
    coordinates/ordering/count" as the T3 builder -- verified directly,
    since kappa_fn/source_fn/sample_piecewise_coefficient are shared
    unmodified between the two element types on that assumption."""
    resolution = 6
    xy3, _, boundary3 = build_grid_aligned_t3_mesh(resolution)
    xy4, _, boundary4 = build_grid_aligned_q4_mesh(resolution)

    np.testing.assert_array_equal(xy3, xy4)
    np.testing.assert_array_equal(boundary3, boundary4)


def test_kappa_fn_nearest_cell_lookup() -> None:
    a_grid = jnp.asarray(_checkerboard_grid(4))  # cell (i, j) = hi if (i+j) even

    # Well inside cell (0, 0) -> hi.
    assert float(kappa_fn(a_grid, jnp.array([0.1, 0.1]))) == float(a_grid[0, 0])
    # Well inside cell (1, 0) -> lo.
    assert float(kappa_fn(a_grid, jnp.array([0.3, 0.1]))) == float(a_grid[1, 0])
    # Exactly at the far boundary (x=1, y=1) must clip into the last cell,
    # not index out of bounds.
    val = float(kappa_fn(a_grid, jnp.array([1.0, 1.0])))
    assert val == float(a_grid[3, 3])


def test_source_fn_is_constant_negative_one() -> None:
    a_grid = jnp.asarray(_checkerboard_grid(4))
    for point in (jnp.array([0.0, 0.0]), jnp.array([0.5, 0.7]), jnp.array([1.0, 1.0])):
        assert float(source_fn(a_grid, point)) == -1.0


def test_sample_piecewise_coefficient_shape_and_two_valued() -> None:
    key = jax.random.PRNGKey(0)
    lo, hi = 3.0, 12.0

    a_native = sample_piecewise_coefficient(
        key, resolution=32, lo=lo, hi=hi, base_resolution=32
    )
    assert a_native.shape == (32, 32)
    assert set(np.unique(a_native).tolist()) <= {lo, hi}

    a_downsampled = sample_piecewise_coefficient(
        key, resolution=16, lo=lo, hi=hi, base_resolution=32
    )
    assert a_downsampled.shape == (16, 16)
    assert set(np.unique(a_downsampled).tolist()) <= {lo, hi}


def _u_exact_fn(point: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(jnp.pi * point[0]) * jnp.sin(jnp.pi * point[1])


def _kappa_fn_const(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    del point
    return params[0]


def _source_fn_manufactured(params: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """For ``div(kappa*grad(u)) - source = 0`` (this module's convention,
    see :func:`source_fn`'s own docstring) with constant kappa and
    ``u = sin(pi x) sin(pi y)``: ``div(kappa*grad(u)) = -2*pi^2*kappa*u``,
    so ``source`` must equal that, not its negation."""
    kappa = _kappa_fn_const(params, point)
    return -2.0 * jnp.pi**2 * kappa * _u_exact_fn(point)


def test_assemble_and_solve_t3_matches_manufactured_solution() -> None:
    resolution = 30
    xy, tri, boundary_mask = build_grid_aligned_t3_mesh(resolution)
    galerkin = T3Galerkin(jnp.asarray(xy), jnp.asarray(tri))
    params = jnp.array([2.0])

    u = assemble_and_solve(
        galerkin,
        params,
        interior_mask=~boundary_mask,
        kappa_fn_=_kappa_fn_const,
        source_fn_=_source_fn_manufactured,
    )
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy])

    assert np.abs(u - u_exact).max() < 0.02


def test_assemble_and_solve_q4_matches_manufactured_solution() -> None:
    resolution = 30
    xy, quad, boundary_mask = build_grid_aligned_q4_mesh(resolution)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    params = jnp.array([2.0])

    u = assemble_and_solve_q4(
        galerkin,
        params,
        interior_mask=~boundary_mask,
        kappa_fn_=_kappa_fn_const,
        source_fn_=_source_fn_manufactured,
    )
    u_exact = np.array([_u_exact_fn(jnp.asarray(p)) for p in xy])

    assert np.abs(u - u_exact).max() < 0.02


def test_t3_and_q4_direct_solves_agree_on_the_same_problem() -> None:
    """Both element types share the same nodes (see the mesh-agreement
    test above), so their direct FEM solutions on the identical
    manufactured problem should agree closely with each other, not just
    each independently with the analytic solution."""
    resolution = 30
    xy3, tri, boundary3 = build_grid_aligned_t3_mesh(resolution)
    xy4, quad, boundary4 = build_grid_aligned_q4_mesh(resolution)
    params = jnp.array([2.0])

    u3 = assemble_and_solve(
        T3Galerkin(jnp.asarray(xy3), jnp.asarray(tri)),
        params,
        interior_mask=~boundary3,
        kappa_fn_=_kappa_fn_const,
        source_fn_=_source_fn_manufactured,
    )
    u4 = assemble_and_solve_q4(
        Q4Galerkin(jnp.asarray(xy4), jnp.asarray(quad)),
        params,
        interior_mask=~boundary4,
        kappa_fn_=_kappa_fn_const,
        source_fn_=_source_fn_manufactured,
    )

    assert np.abs(u3 - u4).max() < 0.02


def test_energy_functional_t3_matches_grad_energy_identity() -> None:
    """``J(u_true) = -0.5 * E_grad(u_true)`` -- the identity
    :func:`energy_functional`'s own docstring claims."""
    resolution = 30
    xy, tri, _ = build_grid_aligned_t3_mesh(resolution)
    galerkin = T3Galerkin(jnp.asarray(xy), jnp.asarray(tri))
    params = jnp.array([2.0])
    u_exact = jnp.asarray(
        np.array([_u_exact_fn(jnp.asarray(p)) for p in xy]), dtype=jnp.float32
    )

    j_val = float(
        energy_functional(
            galerkin, u_exact, _kappa_fn_const, _source_fn_manufactured, params
        )
    )
    e_grad = float(galerkin.grad_energy(u_exact, _kappa_fn_const, params))

    assert j_val == pytest.approx(-0.5 * e_grad, rel=0.05)


def test_energy_functional_q4_matches_grad_energy_identity() -> None:
    resolution = 30
    xy, quad, _ = build_grid_aligned_q4_mesh(resolution)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    params = jnp.array([2.0])
    u_exact = jnp.asarray(
        np.array([_u_exact_fn(jnp.asarray(p)) for p in xy]), dtype=jnp.float32
    )

    j_val = float(
        energy_functional_q4(
            galerkin, u_exact, _kappa_fn_const, _source_fn_manufactured, params
        )
    )
    e_grad = float(galerkin.grad_energy(u_exact, _kappa_fn_const, params))

    assert j_val == pytest.approx(-0.5 * e_grad, rel=0.05)


def test_batched_energy_functional_loss_t3_matches_manual_computation() -> None:
    resolution = 10
    xy, tri, _ = build_grid_aligned_t3_mesh(resolution)
    galerkin = T3Galerkin(jnp.asarray(xy), jnp.asarray(tri))
    u_exact = jnp.asarray(
        np.array([_u_exact_fn(jnp.asarray(p)) for p in xy]), dtype=jnp.float32
    )

    loss_fn = BatchedEnergyFunctionalLoss(
        galerkin, _kappa_fn_const, _source_fn_manufactured
    )
    batch_size = 3
    u_batch = jnp.tile(u_exact[None, :], (batch_size, 1))
    params_batch = jnp.tile(jnp.array([2.0]), (batch_size, 1))

    loss = float(loss_fn(u_batch, params_batch))
    manual = float(
        energy_functional(
            galerkin,
            u_exact,
            _kappa_fn_const,
            _source_fn_manufactured,
            jnp.array([2.0]),
        )
    )
    assert loss == pytest.approx(manual, rel=1e-5)


def test_batched_energy_functional_loss_q4_matches_manual_computation() -> None:
    resolution = 10
    xy, quad, _ = build_grid_aligned_q4_mesh(resolution)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    u_exact = jnp.asarray(
        np.array([_u_exact_fn(jnp.asarray(p)) for p in xy]), dtype=jnp.float32
    )

    loss_fn = BatchedEnergyFunctionalLossQ4(
        galerkin, _kappa_fn_const, _source_fn_manufactured
    )
    batch_size = 3
    u_batch = jnp.tile(u_exact[None, :], (batch_size, 1))
    params_batch = jnp.tile(jnp.array([2.0]), (batch_size, 1))

    loss = float(loss_fn(u_batch, params_batch))
    manual = float(
        energy_functional_q4(
            galerkin,
            u_exact,
            _kappa_fn_const,
            _source_fn_manufactured,
            jnp.array([2.0]),
        )
    )
    assert loss == pytest.approx(manual, rel=1e-5)
