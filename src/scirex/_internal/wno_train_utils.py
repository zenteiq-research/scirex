# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

import json
import logging
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.training import train_state

logger = logging.getLogger(__name__)


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    input_shape: tuple[int, ...],
    tx: optax.GradientTransformation,
    input_dtype: jnp.dtype = jnp.float32,
) -> train_state.TrainState:
    """Initialise a Flax TrainState from a dummy forward pass.

    Parameters
    ----------
    rng:
        PRNG key used for parameter initialisation.
    model:
        Flax module to initialise.
    input_shape:
        Full shape of a single batch input, e.g. `(1, nx, in_channels)`.
    tx:
        Optax optimiser (e.g. `optax.adam(lr)`).
    input_dtype:
        dtype for the dummy initialisation input. Defaults to float32.
    """
    dummy = jnp.zeros(input_shape, dtype=input_dtype)
    params = model.init(rng, dummy)["params"]
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


_DTYPE_MAP = {
    "float32": jnp.float32,
    "fp32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "bf16": jnp.bfloat16,
    "float16": jnp.float16,
    "fp16": jnp.float16,
}


def resolve_dtype(name: str) -> jnp.dtype:
    normalized = name.lower()
    if normalized not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype string: {name}")
    return _DTYPE_MAP[normalized]


def make_schedule(
    learning_rate: float,
    epochs: int,
    scheduler_step_size: int,
    scheduler_gamma: float,
    steps_per_epoch: int,
):
    """Piecewise-constant LR schedule shared by the WNO training scripts."""
    boundaries_and_scales = {}
    for epoch in range(scheduler_step_size, epochs, scheduler_step_size):
        boundaries_and_scales[epoch * steps_per_epoch] = scheduler_gamma
    return optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales=boundaries_and_scales,
    )


def plot_loss_curves(history: dict, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    train_epochs = range(len(history["train_rel_l2"]))
    test_epochs = history.get("evaluated_epochs", range(len(history["test_rel_l2"])))
    ax.semilogy(train_epochs, history["train_rel_l2"], lw=2, label="Train Rel-L2")
    ax.semilogy(test_epochs, history["test_rel_l2"], lw=2, ls="--", label="Test Rel-L2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Loss curve -> {save_path}")


def relative_l2_sum(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """Sum, over the batch axis, of each sample's relative L2 error.

    Layout-agnostic: flattens every non-batch axis, so the same
    implementation covers NWC (1D), NHWC (2D), and higher-rank layouts.
    """
    diff = y_pred - y_true
    diff_flat = diff.reshape((diff.shape[0], -1))
    true_flat = y_true.reshape((y_true.shape[0], -1))
    diff_norm = jnp.sqrt(jnp.sum(diff_flat * diff_flat, axis=1))
    true_norm = jnp.sqrt(jnp.sum(true_flat * true_flat, axis=1))
    return jnp.sum(diff_norm / (true_norm + 1e-8))


def per_frame_relative_l2(
    pred_rollout: jnp.ndarray,
    targets: jnp.ndarray,
    n_rollout_steps: int,
    rollout_step: int,
) -> float:
    """Per-frame relative L2 for an autoregressive rollout, averaged like the
    training metric.

    Splits the concatenated rollout back into its `n_rollout_steps` frames
    and treats (step, batch) jointly as the sample axis, rather than computing
    one relative-L2 over the whole flattened sequence (which is not comparable
    to the training metric).
    """
    pred_frames = jnp.moveaxis(
        pred_rollout.reshape(pred_rollout.shape[:-1] + (n_rollout_steps, rollout_step)),
        -2,
        0,
    )
    true_frames = jnp.moveaxis(
        targets.reshape(targets.shape[:-1] + (n_rollout_steps, rollout_step)),
        -2,
        0,
    )
    pred_frames_flat = pred_frames.reshape((-1,) + pred_frames.shape[2:])
    true_frames_flat = true_frames.reshape((-1,) + true_frames.shape[2:])
    return float(relative_l2_sum(pred_frames_flat, true_frames_flat)) / (
        pred_rollout.shape[0] * n_rollout_steps
    )


def append_grid_nwc(x: jnp.ndarray, grid_range: float) -> jnp.ndarray:
    batch_size, size_x = x.shape[:2]
    grid = jnp.linspace(0.0, grid_range, size_x, dtype=x.dtype)
    grid = jnp.reshape(grid, (1, size_x, 1))
    grid = jnp.broadcast_to(grid, (batch_size, size_x, 1))
    return jnp.concatenate((x, grid), axis=-1)


def append_grid_nhwc(x: jnp.ndarray, grid_range: tuple[float, float]) -> jnp.ndarray:
    batch_size, size_x, size_y = x.shape[:3]
    grid_x = jnp.linspace(0.0, grid_range[0], size_x, dtype=x.dtype)
    grid_y = jnp.linspace(0.0, grid_range[1], size_y, dtype=x.dtype)
    grid_x = jnp.reshape(grid_x, (1, size_x, 1, 1))
    grid_y = jnp.reshape(grid_y, (1, 1, size_y, 1))
    grid_x = jnp.broadcast_to(grid_x, (batch_size, size_x, size_y, 1))
    grid_y = jnp.broadcast_to(grid_y, (batch_size, size_x, size_y, 1))
    return jnp.concatenate((x, grid_x, grid_y), axis=-1)


def write_metrics(history: dict, metrics_path: str, total_start_time: float) -> None:
    total_elapsed = time.time() - total_start_time
    metrics_payload = {
        **history,
        "total_training_time_seconds": total_elapsed,
        "total_training_time_minutes": total_elapsed / 60.0,
    }
    with open(metrics_path, "w", encoding="ascii") as fp:
        json.dump(metrics_payload, fp, indent=2)
