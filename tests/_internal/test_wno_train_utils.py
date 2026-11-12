# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the WNO training utilities shared across the three
train_*_wno_jax.py scripts (src/scirex/_internal/wno_train_utils.py).

Covers:
  - create_train_state - TrainState construction from a dummy forward pass
  - resolve_dtype - all supported aliases, invalid-name error path
  - make_schedule - piecewise-constant LR schedule boundaries
  - plot_loss_curves - writes a PNG file, with/without evaluated_epochs
  - relative_l2_sum - correctness on known inputs
  - append_grid_nwc / append_grid_nhwc - shape and coordinate correctness
  - write_metrics - JSON payload correctness
"""

import json

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn

from scirex._internal.wno_train_utils import (
    append_grid_nhwc,
    append_grid_nwc,
    create_train_state,
    make_schedule,
    per_frame_relative_l2,
    plot_loss_curves,
    relative_l2_sum,
    resolve_dtype,
    write_metrics,
)


class _TinyDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.features)(x)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# create_train_state
# ---------------------------------------------------------------------------


def test_create_train_state_builds_working_state(rng):
    """create_train_state initialises params and an apply_fn that runs."""
    model = _TinyDense(features=4)
    tx = optax.adam(1e-3)
    state = create_train_state(rng=rng, model=model, input_shape=(1, 3), tx=tx)

    x = jnp.ones((2, 3))
    out = state.apply_fn({"params": state.params}, x)
    assert out.shape == (2, 4)


def test_create_train_state_respects_input_dtype(rng):
    """The dummy init input uses input_dtype, and params come out float32."""
    model = _TinyDense(features=2)
    tx = optax.sgd(1e-2)
    state = create_train_state(
        rng=rng, model=model, input_shape=(1, 5), tx=tx, input_dtype=jnp.float32
    )
    kernel = state.params["Dense_0"]["kernel"]
    assert kernel.shape == (5, 2)
    assert kernel.dtype == jnp.float32


def test_create_train_state_step_updates_params(rng):
    """A single apply_gradients step actually changes the parameters."""
    model = _TinyDense(features=1)
    tx = optax.sgd(1e-1)
    state = create_train_state(rng=rng, model=model, input_shape=(1, 2), tx=tx)
    x = jnp.ones((4, 2))
    y = jnp.zeros((4, 1))

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x)
        return jnp.mean((pred - y) ** 2)

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    old_kernel = state.params["Dense_0"]["kernel"]
    new_kernel = new_state.params["Dense_0"]["kernel"]
    assert not jnp.allclose(old_kernel, new_kernel)


# ---------------------------------------------------------------------------
# resolve_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("float32", jnp.float32),
        ("fp32", jnp.float32),
        ("bfloat16", jnp.bfloat16),
        ("bf16", jnp.bfloat16),
        ("float16", jnp.float16),
        ("fp16", jnp.float16),
        ("BFloat16", jnp.bfloat16),  # case-insensitive
        ("FP32", jnp.float32),
    ],
)
def test_resolve_dtype_supported_aliases(name, expected):
    """All documented dtype aliases resolve to the correct jnp dtype."""
    assert resolve_dtype(name) == expected


def test_resolve_dtype_unsupported_raises():
    """An unrecognised dtype string raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported dtype string"):
        resolve_dtype("float8")


# ---------------------------------------------------------------------------
# make_schedule
# ---------------------------------------------------------------------------


def test_make_schedule_initial_value():
    """The schedule starts at the configured learning rate."""
    schedule = make_schedule(
        learning_rate=1e-3,
        epochs=100,
        scheduler_step_size=50,
        scheduler_gamma=0.5,
        steps_per_epoch=10,
    )
    assert float(schedule(0)) == pytest.approx(1e-3)


def test_make_schedule_decays_at_boundary():
    """The schedule decays by scheduler_gamma after each scheduler_step_size epochs."""
    schedule = make_schedule(
        learning_rate=1e-3,
        epochs=150,
        scheduler_step_size=50,
        scheduler_gamma=0.5,
        steps_per_epoch=10,
    )
    assert float(schedule(0)) == pytest.approx(1e-3)
    assert float(schedule(500)) == pytest.approx(1e-3 * 0.5)
    assert float(schedule(1000)) == pytest.approx(1e-3 * 0.25)


def test_make_schedule_no_decay_when_epochs_below_step_size():
    """If epochs < scheduler_step_size, no decay boundaries exist at all."""
    schedule = make_schedule(
        learning_rate=2e-3,
        epochs=10,
        scheduler_step_size=50,
        scheduler_gamma=0.1,
        steps_per_epoch=5,
    )
    assert float(schedule(0)) == pytest.approx(2e-3)
    assert float(schedule(10_000)) == pytest.approx(2e-3)


# ---------------------------------------------------------------------------
# plot_loss_curves
# ---------------------------------------------------------------------------


def test_plot_loss_curves_writes_file(tmp_path):
    """plot_loss_curves writes a non-empty file to save_path."""
    history = {
        "train_rel_l2": [1.0, 0.5, 0.25],
        "test_rel_l2": [1.1, 0.6],
        "evaluated_epochs": [0, 2],
    }
    save_path = tmp_path / "loss.png"
    plot_loss_curves(history, str(save_path), "Test Convergence")
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_loss_curves_without_evaluated_epochs(tmp_path):
    """plot_loss_curves falls back to range(len(test_rel_l2)) when
    evaluated_epochs is absent from history."""
    history = {
        "train_rel_l2": [1.0, 0.5],
        "test_rel_l2": [1.1, 0.6],
    }
    save_path = tmp_path / "loss_no_eval_epochs.png"
    plot_loss_curves(history, str(save_path), "No Eval Epochs")
    assert save_path.exists()


# ---------------------------------------------------------------------------
# relative_l2_sum
# ---------------------------------------------------------------------------


def test_relative_l2_sum_zero_when_identical():
    """relative_l2_sum is zero when prediction equals target exactly."""
    y = jax.random.normal(jax.random.PRNGKey(1), (4, 8))
    assert float(relative_l2_sum(y, y)) == pytest.approx(0.0, abs=1e-6)


def test_relative_l2_sum_known_value():
    """relative_l2_sum matches a hand-computed value for a simple 1-sample case."""
    y_pred = jnp.array([[2.0, 0.0]])
    y_true = jnp.array([[0.0, 0.0]])
    # diff_norm = ||[2,0]|| = 2; true_norm = ||[0,0]|| = 0 -> denom uses +1e-8
    expected = 2.0 / (0.0 + 1e-8)
    assert float(relative_l2_sum(y_pred, y_true)) == pytest.approx(expected, rel=1e-3)


def test_relative_l2_sum_sums_over_batch():
    """relative_l2_sum sums (not means) the per-sample relative errors."""
    y_pred = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    y_true = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    single = float(relative_l2_sum(y_pred[:1], y_true[:1]))
    both = float(relative_l2_sum(y_pred, y_true))
    assert both == pytest.approx(2 * single, rel=1e-3)


def test_relative_l2_sum_layout_agnostic():
    """relative_l2_sum flattens arbitrary non-batch rank (NHWC-style)."""
    y_pred = jax.random.normal(jax.random.PRNGKey(2), (3, 4, 4, 2))
    y_true = jax.random.normal(jax.random.PRNGKey(3), (3, 4, 4, 2))
    result = relative_l2_sum(y_pred, y_true)
    assert jnp.isfinite(result)
    assert float(result) > 0.0


# ---------------------------------------------------------------------------
# per_frame_relative_l2
# ---------------------------------------------------------------------------


def test_per_frame_relative_l2_zero_when_identical():
    """per_frame_relative_l2 is zero when the rollout matches targets exactly."""
    pred_rollout = jax.random.normal(jax.random.PRNGKey(4), (3, 6))
    result = per_frame_relative_l2(
        pred_rollout, pred_rollout, n_rollout_steps=3, rollout_step=2
    )
    assert result == pytest.approx(0.0, abs=1e-6)


def test_per_frame_relative_l2_known_value():
    """per_frame_relative_l2 matches a hand-computed value for a 1-sample,
    2-step rollout."""
    # frame0 = [2, 0], frame1 = [0, 3]; targets are all zero.
    pred_rollout = jnp.array([[2.0, 0.0, 0.0, 3.0]])
    targets = jnp.zeros_like(pred_rollout)
    result = per_frame_relative_l2(
        pred_rollout, targets, n_rollout_steps=2, rollout_step=2
    )
    # Each frame's true_norm is 0, so the denominator falls back to 1e-8.
    expected = (2.0 / 1e-8 + 3.0 / 1e-8) / (1 * 2)
    assert result == pytest.approx(expected, rel=1e-3)


def test_per_frame_relative_l2_matches_manual_per_frame_average():
    """per_frame_relative_l2 agrees with a naive Python loop over frames,
    for a multi-batch, multi-step rollout - a regression check on the
    reshape/moveaxis bookkeeping."""
    batch, n_rollout_steps, rollout_step = 3, 4, 2
    total_channels = n_rollout_steps * rollout_step
    pred_rollout = jax.random.normal(jax.random.PRNGKey(5), (batch, total_channels))
    targets = jax.random.normal(jax.random.PRNGKey(6), (batch, total_channels))

    result = per_frame_relative_l2(pred_rollout, targets, n_rollout_steps, rollout_step)

    manual_sum = 0.0
    for step in range(n_rollout_steps):
        pred_frame = pred_rollout[:, step * rollout_step : (step + 1) * rollout_step]
        true_frame = targets[:, step * rollout_step : (step + 1) * rollout_step]
        manual_sum += float(relative_l2_sum(pred_frame, true_frame))
    expected = manual_sum / (batch * n_rollout_steps)

    assert result == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# append_grid_nwc / append_grid_nhwc
# ---------------------------------------------------------------------------


def test_append_grid_nwc_shape_and_endpoints():
    """append_grid_nwc appends one grid channel spanning [0, grid_range]."""
    x = jnp.zeros((2, 5, 3))
    out = append_grid_nwc(x, grid_range=2.0)
    assert out.shape == (2, 5, 4)
    grid_channel = out[..., -1]
    assert float(grid_channel[0, 0]) == pytest.approx(0.0)
    assert float(grid_channel[0, -1]) == pytest.approx(2.0)
    # Grid is identical across the batch axis.
    assert jnp.allclose(grid_channel[0], grid_channel[1])


def test_append_grid_nhwc_shape_and_endpoints():
    """append_grid_nhwc appends two grid channels spanning their own ranges."""
    x = jnp.zeros((1, 4, 6, 2))
    out = append_grid_nhwc(x, grid_range=(1.0, 3.0))
    assert out.shape == (1, 4, 6, 4)
    grid_x = out[..., -2]
    grid_y = out[..., -1]
    assert float(grid_x[0, 0, 0]) == pytest.approx(0.0)
    assert float(grid_x[0, -1, 0]) == pytest.approx(1.0)
    assert float(grid_y[0, 0, 0]) == pytest.approx(0.0)
    assert float(grid_y[0, 0, -1]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# write_metrics
# ---------------------------------------------------------------------------


def test_write_metrics_writes_expected_payload(tmp_path):
    """write_metrics dumps history plus total-time fields as JSON."""
    history = {"train_rel_l2": [1.0, 0.5], "test_rel_l2": [1.2]}
    metrics_path = tmp_path / "metrics.json"
    total_start_time = __import__("time").time() - 5.0

    write_metrics(history, str(metrics_path), total_start_time)

    payload = json.loads(metrics_path.read_text())
    assert payload["train_rel_l2"] == [1.0, 0.5]
    assert payload["test_rel_l2"] == [1.2]
    assert payload["total_training_time_seconds"] >= 5.0
    assert payload["total_training_time_minutes"] == pytest.approx(
        payload["total_training_time_seconds"] / 60.0
    )
