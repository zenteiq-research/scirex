# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Loss and accuracy metrics for the Car-Design-ShapeNetCar benchmark.

The paper's training loss is split by field: pressure MSE on surface points
only (``surf``-masked) plus velocity MSE on every point, combined as
``loss_velo + press_weight * loss_press``. This is scale-dependent (mixes two
fields of different units and magnitudes) and normalized-field only, so it is
tracked during training but not used for the paper-comparison numbers -- see
:func:`rel_l2_metrics` for those.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np


def make_loss(module, press_weight: float) -> Callable:
    """Returns ``loss_fn(params, x, y, surf, neighbors_in=None,
    neighbors_out=None) -> (total, (loss_press, loss_velo))``.

    ``x``: ``(1, N, space_dim)``, ``y``/``surf``: ``(N, out_dim)``/``(N,)`` --
    unbatched past the leading singleton, since every sample is trained on its
    own (see this package's ``__init__.py``).

    ``neighbors_in``/``neighbors_out`` default to ``None`` and are forwarded
    only when given, so Transolver's call is bit-for-bit unaffected by their
    existence -- they are GINO's per-sample geometry (see
    ``_gino_geometry.py``/``_models.GINOCarWrapper``); Transolver ignores
    them (its own ``__call__`` has no such keywords, and ``None`` here means
    they are simply never passed).
    """

    def loss_fn(params, x, y, surf, neighbors_in=None, neighbors_out=None):
        extra = {}
        if neighbors_in is not None:
            extra["neighbors_in"] = neighbors_in
            extra["neighbors_out"] = neighbors_out
        out = module.apply(params, x=x, fx=None, **extra)[0]
        surf_f = surf.astype(out.dtype)
        n_surf = jnp.maximum(1.0, surf_f.sum())
        loss_press = jnp.sum(surf_f * (out[:, -1] - y[:, -1]) ** 2) / n_surf
        loss_velo = jnp.mean((out[:, :-1] - y[:, :-1]) ** 2)
        total = loss_velo + press_weight * loss_press
        return total, (loss_press, loss_velo)

    return loss_fn


def rel_l2(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-8) -> float:
    """Relative L2 error ``||pred - truth||_2 / (||truth||_2 + eps)``, the
    Transolver paper's Table 3 ("Volume"/"Surf") reporting convention --
    unlike :func:`make_loss`'s split MSE, this is scale-normalized and meant
    to be computed on denormalized (physical-unit) fields."""
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + eps))


def rel_l2_metrics(
    out_denorm: np.ndarray, y_denorm: np.ndarray, surf: np.ndarray
) -> dict:
    """Table 3-style relative L2: ``volume`` (velocity, every point) and
    ``surface`` (pressure, surface points only), both on denormalized fields."""
    return dict(
        volume=rel_l2(out_denorm[:, :-1], y_denorm[:, :-1]),
        surface=rel_l2(out_denorm[surf, -1], y_denorm[surf, -1]),
    )
