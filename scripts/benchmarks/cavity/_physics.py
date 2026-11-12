# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's loss and its metric.

The training loss is the paper's: mean squared error on the normalised fields
plus a soft mass-conservation penalty. The reported metric is the paper's RMSE,
pooled over (p, Ux, Uy) in m/s.

Differentiation is ``scirex``' own
:class:`~scirex.losses.FiniteDiff`, so the divergence
here uses the same central-difference stencils (and the same matched-order
one-sided stencils at the walls) as the rest of the package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from scirex.data import UnitGaussianNormalizer
from scirex.losses import FiniteDiff
from scripts.benchmarks.cavity._data import DOMAIN_SIZE, DX, FIELDS

U_REF = 1.0  # m/s, the Re = 10,000 test case's lid speed

_DIFF = FiniteDiff()

# Grid axes are (H = y, W = x), so Ux differentiates along the last spatial
# axis and Uy along the one before it.
_X_AXIS = -1
_Y_AXIS = -2


def divergence(vel: jnp.ndarray) -> jnp.ndarray:
    """``vel``: ``(..., H, W, 2)`` physical (Ux, Uy) -> ``(..., H, W)``.

    The cavity walls are not periodic, so ``periodic=False`` keeps the
    boundary rows and columns rather than dropping them.
    """
    dudx = _DIFF.derivative(
        vel[..., 0], axis=_X_AXIS, order=1, spacing=DX, periodic=False
    )
    dvdy = _DIFF.derivative(
        vel[..., 1], axis=_Y_AXIS, order=1, spacing=DX, periodic=False
    )
    return dudx + dvdy


def mass_conservation(vel_phys: jnp.ndarray) -> jnp.ndarray:
    """The paper's soft constraint, made dimensionless.

    The paper penalises the mean absolute per-cell mass imbalance. Its
    tabulated values carry the units "kg s^-1 m^2", which are not
    dimensionally consistent with a mass flux, so the absolute scale of its
    term cannot be reproduced exactly. What is reproduced is its role: a soft
    divergence penalty, small relative to the data term, at the paper's chosen
    weight. Here it is ``mean(|div u|) * L / U_ref``.
    """
    return jnp.mean(jnp.abs(divergence(vel_phys))) * DOMAIN_SIZE / U_REF


def make_loss(
    model: Any, normalizer: UnitGaussianNormalizer, mc_weight: float
) -> Callable[..., tuple[jnp.ndarray, tuple[jnp.ndarray, Any]]]:
    """MSE on normalised fields plus ``mc_weight`` times the physics term.

    The returned function reports ``(total, (data_loss, mass_conservation))``
    so the two can be logged apart.
    """

    def loss_fn(params: Any, xb: jnp.ndarray, yb: jnp.ndarray) -> Any:
        pred = model.apply(params, xb)
        data_loss = jnp.mean((pred - yb) ** 2)
        if mc_weight == 0.0:
            return data_loss, (data_loss, 0.0)
        pred_phys = normalizer.inverse_transform(pred)
        mc = mass_conservation(pred_phys[..., 1:3])
        return data_loss + mc_weight * mc, (data_loss, mc)

    return loss_fn


def rmse_metrics(pred_phys: np.ndarray, truth_phys: np.ndarray) -> dict:
    """The paper's metric: RMSE in m/s pooled over (p, Ux, Uy), plus the
    velocity-only and per-channel RMSE and the mean absolute divergence."""
    err = pred_phys - truth_phys
    div = np.asarray(divergence(jnp.asarray(pred_phys[..., 1:3])))
    return dict(
        rmse=float(np.sqrt(np.mean(err**2))),
        rmse_vel=float(np.sqrt(np.mean(err[..., 1:3] ** 2))),
        per_channel={
            name: float(np.sqrt(np.mean(err[..., channel] ** 2)))
            for channel, name in enumerate(FIELDS)
        },
        mean_abs_div=float(np.mean(np.abs(div))),
    )
