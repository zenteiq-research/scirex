# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data, normalisation and losses of the Darcy benchmark.

The data are Li et al.'s own released ``.mat`` files (native 421x421,
``coeff`` in {3, 12} and ``sol`` already in physical units), strided by
``subsample_rate`` exactly as their code does (``x[:, ::5, ::5]`` -> 85x85).
Nothing is re-solved.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np


def _load_mat_fields(
    path: Path, n: int, subsample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """``coeff`` and ``sol`` of the first ``n`` samples of one ``.mat`` file,
    strided by ``subsample_rate``: two ``(n, H, W)`` float32 arrays."""
    from scipy.io import loadmat

    if not path.is_file():
        raise FileNotFoundError(
            f"Darcy .mat file not found: {path}. Download Li et al.'s "
            "piececonst_r421_N1024_smooth{1,2}.mat (e.g. from the Google Drive "
            "link in TapasTripura/WNO's README, 'Dataset-2') into the data "
            "directory (--data-dir or DARCY_DATA_DIR)."
        )
    data = loadmat(path.as_posix(), variable_names=("coeff", "sol"))
    fields = []
    for key in ("coeff", "sol"):
        if key not in data:
            raise KeyError(f"{path} has no '{key}' variable.")
        field = np.asarray(data[key])
        # Fewer samples than asked for would otherwise be silent: JAX clamps
        # out-of-range gather indices instead of raising.
        if field.shape[0] < n:
            raise ValueError(
                f"Requested {n} instances but {path} only has {field.shape[0]}."
            )
        field = field[:n, ::subsample_rate, ::subsample_rate]
        fields.append(field.astype(np.float32))
    return fields[0], fields[1]


def load_split(data_dir: Path, protocol: dict) -> dict[str, np.ndarray]:
    """The train and test permeability ``a`` and pressure ``u`` fields, each
    ``(n, H, W)`` float32."""
    data_dir = Path(data_dir)
    train_a, train_u = _load_mat_fields(
        data_dir / protocol["train_file"],
        protocol["n_train"],
        protocol["subsample_rate"],
    )
    test_a, test_u = _load_mat_fields(
        data_dir / protocol["test_file"],
        protocol["n_test"],
        protocol["subsample_rate"],
    )
    return dict(train_a=train_a, train_u=train_u, test_a=test_a, test_u=test_u)


class PixelNormalizer:
    """Per-pixel mean/std over the samples (axis 0), ``eps=1e-5``: Li et al.'s
    ``utilities3.UnitGaussianNormalizer``, also used by the WNO reference
    code."""

    def __init__(self, x: np.ndarray, eps: float = 1e-5) -> None:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.eps = eps

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * (self.std + self.eps) + self.mean


class GlobalNormalizer:
    """One mean/std per channel, over the samples and every point, with
    ``eps=1e-8`` added to the std: the Transolver reference code's
    ``UnitTransformer``. Fitted on a channels-last array."""

    def __init__(self, x: np.ndarray, eps: float = 1e-8) -> None:
        reduce_axes = tuple(range(x.ndim - 1))
        self.mean = x.mean(axis=reduce_axes, keepdims=True)
        self.std = x.std(axis=reduce_axes, keepdims=True) + eps

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self.std + self.mean


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Relative L2 error of one sample."""
    return float(np.linalg.norm(pred - true) / np.maximum(np.linalg.norm(true), 1e-12))


def batched_rel_l2_loss(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Mean over the batch of the per-sample relative L2 error of ``(B, N)``
    arrays."""
    num = jnp.sqrt(jnp.sum((pred - true) ** 2, axis=-1))
    den = jnp.maximum(jnp.sqrt(jnp.sum(true**2, axis=-1)), 1e-12)
    return jnp.mean(num / den)


def zero_border(grid: jnp.ndarray) -> jnp.ndarray:
    """Zeroes the outer 1-pixel ring of a ``(B, H, W, C)`` grid, as
    ``exp_darcy.py``'s crop-then-zero-pad does."""
    grid = grid.at[:, 0, :, :].set(0.0)
    grid = grid.at[:, -1, :, :].set(0.0)
    grid = grid.at[:, :, 0, :].set(0.0)
    grid = grid.at[:, :, -1, :].set(0.0)
    return grid


def central_diff(grid: jnp.ndarray, dx: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Zero-padded central differences ``(d/dx, d/dy)`` of a ``(B, H, W, C)``
    grid, as ``exp_darcy.py``'s ``central_diff``."""
    padded = jnp.pad(grid, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    grad_x = (padded[:, 1:-1, 2:, :] - padded[:, 1:-1, :-2, :]) / (2.0 * dx)
    grad_y = (padded[:, 2:, 1:-1, :] - padded[:, :-2, 1:-1, :]) / (2.0 * dx)
    return grad_x, grad_y


def gradient_regularization_loss(
    pred_grid: jnp.ndarray, true_grid: jnp.ndarray, dx: float
) -> jnp.ndarray:
    """``L_grad`` of Transolver's Darcy loss ``rel. L2 + 0.1 * L_grad`` (Wu et al.,
    Table 8): the relative L2 errors of the x- and y-gradients of the
    border-zeroed prediction and the truth, summed."""
    pred_zeroed = zero_border(pred_grid)
    pred_grad_x, pred_grad_y = central_diff(pred_zeroed, dx)
    true_grad_x, true_grad_y = central_diff(true_grid, dx)
    b = pred_grid.shape[0]
    loss_x = batched_rel_l2_loss(pred_grad_x.reshape(b, -1), true_grad_x.reshape(b, -1))
    loss_y = batched_rel_l2_loss(pred_grad_y.reshape(b, -1), true_grad_y.reshape(b, -1))
    return loss_x + loss_y
