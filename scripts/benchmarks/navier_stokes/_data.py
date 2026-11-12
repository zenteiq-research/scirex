# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Data split, loss and rollout of the Navier-Stokes benchmark.

A port of the data section and the train/test loops of the Transolver
paper's ``exp_ns.py``:

* data: ``u`` of ``NavierStokes_V1e-5_N1200_T20.mat``, vorticity on a 64x64
  grid at 20 time steps, viscosity 1e-5;
* split: first 1000 samples for training, last 200 for testing;
* input: frames ``0 .. t_in - 1`` (t = 1 .. 10; the file's ``a``, t = 0, is
  unused); target: frames ``t_in .. t_in + t_out - 1`` (t = 11 .. 20),
  predicted one at a time; no normalisation;
* training: each step is fed the ground-truth history (teacher forcing) and
  the loss is the relative L2 per sample, summed over the batch and the
  ``t_out`` steps;
* metric: each step is fed the model's own previous predictions
  (autoregressive rollout), and the relative L2 is taken over all
  ``t_out`` predicted frames of a sample at once (``test_l2_full`` in
  ``exp_ns.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io


def load_split(data_path: Path, protocol: dict) -> dict:
    """Train/test split as ``(n, H * W, t)`` arrays.

    ``a_*`` hold the ``t_in`` input frames, ``u_*`` the ``t_out`` target
    frames. Also returns the ``grid`` shape.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found; set NS_DATA_DIR or --data-dir to the "
            "folder holding NavierStokes_V1e-5_N1200_T20.mat"
        )
    u = scipy.io.loadmat(str(data_path))["u"]  # (N, H, W, T)
    t_in, t_out = protocol["t_in"], protocol["t_out"]
    ntrain, ntest = protocol["ntrain"], protocol["ntest"]
    n, h, w, t = u.shape
    if ntrain + ntest > n or t_in + t_out > t:
        raise ValueError(
            f"{data_path.name} holds {n} samples of {t} frames, fewer than "
            f"{ntrain} + {ntest} samples of {t_in} + {t_out} frames"
        )

    def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = x.reshape(x.shape[0], h * w, t).astype(np.float32)
        return x[..., :t_in], x[..., t_in : t_in + t_out]

    a_train, u_train = split(u[:ntrain])
    a_test, u_test = split(u[-ntest:])
    return dict(
        a_train=a_train, u_train=u_train, a_test=a_test, u_test=u_test, grid=(h, w)
    )


def rel_l2_sum(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Relative L2 per sample, summed over the batch (``TestLoss.rel`` with
    ``size_average=False``)."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return jnp.sum(diff / jnp.linalg.norm(true.reshape(b, -1), axis=1))


StepFn = Callable[[jnp.ndarray], jnp.ndarray]


def teacher_forced(
    step: StepFn, fx: jnp.ndarray, yy: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Training loss and predictions with teacher forcing.

    ``step`` maps a ``(batch, N, t_in)`` history to the next frame
    ``(batch, N, 1)``. Each step's history is rebuilt from the ground truth
    ``yy``, so no gradient flows from one step into the next. Returns the
    loss (summed over the batch and the steps) and the ``(batch, N, t_out)``
    predictions.
    """
    loss = jnp.zeros(())
    preds = []
    for t in range(yy.shape[-1]):
        y = yy[..., t : t + 1]
        im = step(fx)
        loss = loss + rel_l2_sum(im, y)
        preds.append(im)
        fx = jnp.concatenate([fx[..., 1:], y], axis=-1)
    return loss, jnp.concatenate(preds, axis=-1)


def rollout(step: StepFn, fx: jnp.ndarray, t_out: int) -> jnp.ndarray:
    """Autoregressive rollout: each step is fed the model's own previous
    predictions. Returns ``(batch, N, t_out)``."""
    preds = []
    for _ in range(t_out):
        im = step(fx)
        preds.append(im)
        fx = jnp.concatenate([fx[..., 1:], jax.lax.stop_gradient(im)], axis=-1)
    return jnp.concatenate(preds, axis=-1)


def copy_last_frame(fx: np.ndarray, t_out: int) -> np.ndarray:
    """The trivial baseline: every predicted frame is the last input frame."""
    return np.repeat(fx[..., -1:], t_out, axis=-1)
