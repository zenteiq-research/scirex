# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/darcy_physics/.

Everything runs on small synthetic fields on the CPU, so no dataset or GPU is
needed: the protocol shared with the data-driven benchmark and the committed
runs, the element-coefficient rules, the nodes-to-sample-points map, the
model-input / cell split, the exact boundary factor, the label-free output
scale and the extra-coefficient pool; then, per model, that its script builds
the parameter-matched data-driven model and that the script's prediction /
loss wiring trains a tiny model on a small mesh.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    BatchedEnergyFunctionalLossQ4,
    build_grid_aligned_q4_mesh,
    kappa_fn,
    source_fn,
)
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.darcy._config import PROTOCOL as DATA_PROTOCOL  # noqa: E402
from scripts.benchmarks.darcy_physics import (  # noqa: E402
    train_fno,
    train_gino,
    train_transolver,
    train_wno,
)
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    FIELDS,
    HEADLINE_FIELDS,
    MODELS,
    PROTOCOL,
    run_dir,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    FE_PRECISION,
    cell_centre_interpolation,
    extra_coefficient_keys,
    node_to_cell_coefficient,
    sample_extra_coefficients,
    sin_boundary_factor,
    split_coefficient,
    torsion_output_scale,
)


def test_protocol_is_the_data_driven_split() -> None:
    for key, value in DATA_PROTOCOL.items():
        assert PROTOCOL[key] == value
    assert PROTOCOL["cell_mode"] == "majority"


@pytest.mark.parametrize("model", MODELS)
def test_every_published_run_is_committed_with_the_protocol(model: str) -> None:
    for n_fields in FIELDS:
        metrics = json.loads((run_dir(model, n_fields) / "metrics.json").read_text())
        assert metrics["n_train"] == n_fields
        # The first Transolver run (released fields only) predates this key.
        assert metrics.get("n_train_released", n_fields) == PROTOCOL["n_train"]
        assert metrics["n_test"] == PROTOCOL["n_test"]
        assert metrics["cell_mode"] == PROTOCOL["cell_mode"]
        assert metrics["seed"] == 0
    assert HEADLINE_FIELDS[model] in FIELDS


def _random_two_phase(n: int, side: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice([3.0, 12.0], size=(n, side, side))


def test_corner_matches_stride_then_drop_last_row_and_column() -> None:
    a = _random_two_phase(2, 21, seed=0)
    cells = node_to_cell_coefficient(a, resolution=4, mode="corner")
    np.testing.assert_array_equal(cells, a[:, ::5, ::5][:, :4, :4])


def test_mean_averages_every_node_the_cell_covers() -> None:
    a = _random_two_phase(2, 21, seed=1)
    cells = node_to_cell_coefficient(a, resolution=4, mode="mean")
    assert cells.shape == (2, 4, 4)
    # Cell (1, 2) spans native nodes 5..10 in x and 10..15 in y, inclusive.
    np.testing.assert_allclose(cells[1, 1, 2], a[1, 5:11, 10:16].mean())


def test_majority_picks_the_more_frequent_value() -> None:
    a = np.full((1, 11, 11), 3.0)
    a[0, :6, :6] = 12.0  # cell (0, 0) is all 12
    a[0, 5:, :4] = 12.0  # cell (1, 0): 26 of its 36 nodes are now 12
    cells = node_to_cell_coefficient(a, resolution=2, mode="majority")
    np.testing.assert_array_equal(cells[0], [[12.0, 3.0], [12.0, 3.0]])


def test_majority_keeps_the_two_coefficient_values() -> None:
    a = _random_two_phase(3, 21, seed=2)
    cells = node_to_cell_coefficient(a, resolution=4, mode="majority")
    assert set(np.unique(cells)) <= {3.0, 12.0}


def test_constant_field_is_unchanged_by_every_mode() -> None:
    a = np.full((1, 13, 13), 7.0)
    for mode in ("corner", "majority", "mean"):
        cells = node_to_cell_coefficient(a, resolution=4, mode=mode)
        np.testing.assert_array_equal(cells, np.full((1, 4, 4), 7.0))


def test_resolution_must_divide_the_native_grid() -> None:
    with pytest.raises(ValueError, match="equal cells"):
        node_to_cell_coefficient(np.zeros((1, 21, 21)), resolution=3, mode="mean")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        node_to_cell_coefficient(np.zeros((1, 21, 21)), resolution=4, mode="nearest")


def test_cell_centre_map_rows_are_linear_interpolation_weights() -> None:
    m = cell_centre_interpolation(n_side=85, subsample_rate=5)
    assert m.shape == (85, 85)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    assert (m >= 0).all() and ((m > 0).sum(axis=1) <= 2).all()


def test_cell_centre_map_reproduces_linear_fields_at_the_sample_points() -> None:
    m = cell_centre_interpolation(n_side=85, subsample_rate=5)
    nodes = np.linspace(0.0, 1.0, 85)
    points = (5 * np.arange(85) + 0.5) / 421
    np.testing.assert_allclose(m @ (2.0 * nodes - 1.0), 2.0 * points - 1.0, atol=1e-12)


def test_model_input_is_the_strided_node_field() -> None:
    a = np.random.default_rng(0).choice([3.0, 12.0], size=(3, 21, 21))
    model_input, _ = split_coefficient(a, subsample_rate=5, cell_mode="majority")
    # (21 - 1) // 5 + 1 == 5, same rule as 421 -> 85.
    np.testing.assert_array_equal(model_input, a[:, ::5, ::5])


def test_cells_come_from_the_native_grid_one_per_mesh_cell() -> None:
    a = np.random.default_rng(1).choice([3.0, 12.0], size=(3, 21, 21))
    model_input, cells = split_coefficient(a, subsample_rate=5, cell_mode="majority")
    assert cells.shape == (3, model_input.shape[1] - 1, model_input.shape[2] - 1)
    np.testing.assert_array_equal(
        cells, node_to_cell_coefficient(a, resolution=4, mode="majority")
    )


def test_boundary_factor_is_zero_on_the_boundary_and_positive_inside() -> None:
    xy, _, boundary_mask = build_grid_aligned_q4_mesh(8)
    factor = sin_boundary_factor(xy)
    np.testing.assert_allclose(factor[boundary_mask], 0.0, atol=1e-6)
    assert np.all(factor[~boundary_mask] > 0.0)
    assert np.isclose(factor.max(), 1.0)


def test_output_scale_is_the_torsion_function_over_the_effective_coefficient() -> None:
    xy, quad, boundary_mask = build_grid_aligned_q4_mesh(84)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    c, m = torsion_output_scale(galerkin, ~boundary_mask, 3.0, 12.0)
    # Mean / std over the 85 x 85 nodes of phi / 6, -Lap(phi) = 1 on the unit
    # square (max phi = 0.0737); no label information involved.
    assert abs(m - 0.00572) < 2e-5
    assert abs(c - 0.00367) < 2e-5


def test_extra_coefficient_pool_is_nested_and_seed_independent() -> None:
    small, large = extra_coefficient_keys(3), extra_coefficient_keys(7)
    np.testing.assert_array_equal(np.asarray(small), np.asarray(large[:3]))


def test_extra_coefficients_have_the_released_layout() -> None:
    model_input, cells = sample_extra_coefficients(
        2, n_native=21, subsample_rate=5, cell_mode="majority"
    )
    assert model_input.shape == (2, 5, 5) and cells.shape == (2, 4, 4)
    assert set(np.unique(cells)) <= {3.0, 12.0}


# -------------------------------------------------------------------------
# train_transolver.py


def _transolver_init(model, n_points: int, fx: jnp.ndarray):
    return model.init(
        jax.random.PRNGKey(0), x=jnp.zeros((1, n_points, 2), jnp.float32), fx=fx
    )


def test_script_model_has_the_benchmark_parameter_count_transolver() -> None:
    jax.config.update("jax_enable_x64", False)
    model = train_transolver.build_model(85)
    params = _transolver_init(model, 85 * 85, jnp.zeros((1, 85 * 85, 1), jnp.float32))
    assert sum(p.size for p in jax.tree_util.tree_leaves(params)) == 2_826_945


def test_tiny_model_trains_on_the_ritz_energy_with_an_exact_boundary_transolver() -> (
    None
):
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(0)
    native = rng.choice([3.0, 12.0], size=(4, 41, 41)).astype(np.float32)
    model_input, cells = split_coefficient(
        native, subsample_rate=5, cell_mode="majority"
    )
    n_side = model_input.shape[1]  # 9
    n_points = n_side * n_side
    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, 3.0, 12.0)
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    factor = jnp.asarray(sin_boundary_factor(xy), jnp.float32)

    model = train_transolver.build_model(
        n_side, n_hidden=16, n_layers=1, n_head=2, slice_num=4, ref=2
    )
    fx = jnp.asarray((model_input - 7.5) / 4.5).reshape(-1, n_points, 1)
    params = _transolver_init(model, n_points, fx[:1])
    predict_nodes = train_transolver.make_predict_nodes(
        model, n_points, scale, offset, factor
    )
    loss_fn = train_transolver.make_loss_fn(predict_nodes, physics_loss)
    cells_j = jnp.asarray(cells, jnp.float32)

    u = predict_nodes(params, fx)
    assert u.shape == (4, n_points)
    np.testing.assert_allclose(np.asarray(u)[:, boundary], 0.0, atol=1e-7)

    tx = optax.adam(1e-3)
    state = tx.init(params)
    step = jax.jit(jax.value_and_grad(loss_fn))
    first, _ = step(params, fx, cells_j)
    for _ in range(20):
        _, grads = step(params, fx, cells_j)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    last, _ = step(params, fx, cells_j)
    assert float(last) < float(first)


# -------------------------------------------------------------------------
# train_fno.py


def _fno_model_and_params():
    jax.config.update("jax_enable_x64", False)
    model = train_fno.build_model()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 1, 85, 85), jnp.float32))
    return model, params


def test_parameter_count_matches_the_data_driven_benchmark_fno() -> None:
    _, params = _fno_model_and_params()
    # Complex spectral weights count as two real parameters (#320: 2,836,281).
    assert count_model_params(params) == 2_836_281


def test_spectral_weights_keep_20_by_11_modes_on_the_s85_grid_fno() -> None:
    model, params = _fno_model_and_params()
    shapes = [
        leaf.shape
        for leaf in jax.tree_util.tree_leaves(params)
        if jnp.iscomplexobj(leaf)
    ]
    assert shapes, "expected complex spectral weights"
    # 20 modes in total along the first axis, 20 // 2 + 1 = 11 real-FFT modes
    # along the second, both below the s=85 grid's 43 real-FFT frequencies.
    assert all(s[-2:] == (20, 11) for s in shapes)
    out = model.apply(params, jnp.zeros((2, 1, 85, 85), jnp.float32))
    assert out.shape == (2, 1, 85, 85) and out.dtype == jnp.float32


def test_tiny_model_trains_on_the_ritz_energy_with_an_exact_boundary_fno() -> None:
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(0)
    native = rng.choice([3.0, 12.0], size=(4, 41, 41)).astype(np.float32)
    model_input, cells = split_coefficient(
        native, subsample_rate=5, cell_mode="majority"
    )
    n_side = model_input.shape[1]  # 9
    n_points = n_side * n_side
    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, 3.0, 12.0)
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    factor = jnp.asarray(sin_boundary_factor(xy), jnp.float32)

    model = train_fno.build_model(hidden_channels=8, modes=4, n_layers=1)
    a_enc = jnp.asarray((model_input - 7.5) / 4.5)[:, None]
    params = model.init(jax.random.PRNGKey(0), a_enc[:1])
    predict_nodes = train_fno.make_predict_nodes(model, scale, offset, factor)
    loss_fn = train_fno.make_loss_fn(predict_nodes, physics_loss)
    cells_j = jnp.asarray(cells, jnp.float32)

    u = predict_nodes(params, a_enc)
    assert u.shape == (4, n_points)
    np.testing.assert_allclose(np.asarray(u)[:, boundary], 0.0, atol=1e-7)

    tx = optax.adam(1e-3)
    state = tx.init(params)
    # Scaled by the interior node count, as in the script's default.
    scaled = float((~boundary).sum())
    step = jax.jit(jax.value_and_grad(lambda p, a, c: scaled * loss_fn(p, a, c)))
    first, _ = step(params, a_enc, cells_j)
    for _ in range(20):
        _, grads = step(params, a_enc, cells_j)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    last, _ = step(params, a_enc, cells_j)
    assert float(last) < float(first)


# -------------------------------------------------------------------------
# train_gino.py


def test_parameter_count_and_output_order_match_the_data_driven_benchmark_gino() -> (
    None
):
    jax.config.update("jax_enable_x64", False)
    n_side = 85
    model, kwargs, _, _ = train_gino.build_model(n_side)
    x = jnp.zeros((2, n_side * n_side, 1), jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x=x[:1], **kwargs)
    assert count_model_params(params) == 2_801_340
    out = model.apply(params, x=x, **kwargs)
    assert out.shape == (2, n_side * n_side, 1)
    # Output queries follow the data's row-major flatten order (x = i / 84).
    xy = kwargs["output_queries"][0]
    assert jnp.allclose(xy[1], jnp.array([0.0, 1.0 / (n_side - 1)]))


def test_tiny_model_trains_on_the_ritz_energy_with_an_exact_boundary_gino() -> None:
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(0)
    native = rng.choice([3.0, 12.0], size=(4, 41, 41)).astype(np.float32)
    model_input, cells = split_coefficient(
        native, subsample_rate=5, cell_mode="majority"
    )
    n_side = model_input.shape[1]  # 9
    n_points = n_side * n_side
    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, 3.0, 12.0)
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    factor = jnp.asarray(sin_boundary_factor(xy), jnp.float32)

    model, kwargs, _, _ = train_gino.build_model(
        n_side, hidden_channels=4, modes=2, n_layers=1
    )
    a_enc = jnp.asarray((model_input - 7.5) / 4.5).reshape(-1, n_points)
    params = model.init(jax.random.PRNGKey(0), x=a_enc[:1, :, None], **kwargs)
    predict_nodes = train_gino.make_predict_nodes(model, kwargs, scale, offset, factor)
    loss_fn = train_gino.make_loss_fn(predict_nodes, physics_loss)
    cells_j = jnp.asarray(cells, jnp.float32)

    u = predict_nodes(params, a_enc)
    assert u.shape == (4, n_points)
    np.testing.assert_allclose(np.asarray(u)[:, boundary], 0.0, atol=1e-7)

    tx = optax.adam(1e-3)
    state = tx.init(params)
    step = jax.jit(jax.value_and_grad(loss_fn))
    first, _ = step(params, a_enc, cells_j)
    for _ in range(20):
        _, grads = step(params, a_enc, cells_j)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    last, _ = step(params, a_enc, cells_j)
    assert float(last) < float(first)


# -------------------------------------------------------------------------
# train_wno.py


def test_parameter_count_and_shapes_match_the_data_driven_benchmark_wno() -> None:
    jax.config.update("jax_enable_x64", False)
    model = train_wno.build_model(85)
    x = jnp.zeros((2, 85, 85, 1), jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x[:1])
    assert sum(p.size for p in jax.tree_util.tree_leaves(params)) == 2_829_601
    out = model.apply(params, x)
    assert out.shape == (2, 85, 85, 1) and out.dtype == jnp.float32


def test_tiny_model_trains_on_the_ritz_energy_with_an_exact_boundary_wno() -> None:
    jax.config.update("jax_enable_x64", False)
    rng = np.random.default_rng(0)
    native = rng.choice([3.0, 12.0], size=(4, 41, 41)).astype(np.float32)
    model_input, cells = split_coefficient(
        native, subsample_rate=5, cell_mode="majority"
    )
    n_side = model_input.shape[1]  # 9
    n_points = n_side * n_side
    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, 3.0, 12.0)
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    factor = jnp.asarray(sin_boundary_factor(xy), jnp.float32)

    model = train_wno.build_model(
        n_side, hidden_channels=4, level=1, wavelet="db1", n_layers=1
    )
    a_enc = jnp.asarray((model_input - 7.5) / 4.5)[..., None]
    params = model.init(jax.random.PRNGKey(0), a_enc[:1])
    predict_nodes = train_wno.make_predict_nodes(model, scale, offset, factor)
    loss_fn = train_wno.make_loss_fn(predict_nodes, physics_loss)
    cells_j = jnp.asarray(cells, jnp.float32)

    u = predict_nodes(params, a_enc)
    assert u.shape == (4, n_points)
    np.testing.assert_allclose(np.asarray(u)[:, boundary], 0.0, atol=1e-7)

    tx = optax.adam(1e-3)
    state = tx.init(params)
    # Scaled by the interior node count, as in the script's default.
    scaled = float((~boundary).sum())
    step = jax.jit(jax.value_and_grad(lambda p, a, c: scaled * loss_fn(p, a, c)))
    first, _ = step(params, a_enc, cells_j)
    for _ in range(20):
        _, grads = step(params, a_enc, cells_j)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    last, _ = step(params, a_enc, cells_j)
    assert float(last) < float(first)
