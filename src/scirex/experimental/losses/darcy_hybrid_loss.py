# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Model-agnostic hybrid (physics + optional data) loss for the 2D Darcy-flow
equation, for models whose native output is a flat point cloud rather than a
grid array.

The physics term itself -- the finite-difference strong-form residual
`-div(a * grad(u)) - f` -- already exists and is fully implemented by
`DarcyEqnLoss`; this module adds
no new PDE discretization. What it adds is the one piece of glue
`DarcyEqnLoss` doesn't provide: turning a *point-cloud* model's raw
output into the `(batch, n_x, n_y, 1)` grid array the finite-difference
stencils need.

GINO never outputs a grid: `__call__`
always returns a flat `(batch, n_out, out_channels)` tensor, even when
`output_queries` happens to be the coordinates of a regular grid. So to get
a grid-shaped prediction out of GINO, the *caller* must query it at exactly
the flattened, row-major coordinates of the Darcy grid, and
`DarcyHybridLossGINO` folds the resulting flat output back into
`(batch, n_x, n_y, 1)` using that same ordering before handing it to
`DarcyEqnLoss`. Grid-native models (FNO, WNO) need no such reshape --
their own hybrid-loss class, should one be added later, would follow the
naming precedent set for linear elasticity
(`LinearElasticityHybridLoss`
for the grid case, `...T3` for the point-cloud case): the plain
`DarcyHybridLoss` name is left free for that grid-native variant, not used
here.

Like every other hybrid loss in this repo, the physics term is always
active and the (optional) data term defaults to weight zero -- pure
physics-only training, matching the pretraining stage of Wang et al.,
*Pretrain finite element method: A pretraining and warm-start framework for
PDEs via physics-informed neural operators*, J. Mech. Phys. Solids 214
(2026) 106682 (https://github.com/yizheng-wang/PFEM), even though Darcy's
own discretization (finite-difference strong form) is not PFEM's own FEM
energy scheme.
"""

from typing import Any

import jax.numpy as jnp

from scirex.losses.data_losses import H1Loss, LpLoss
from scirex.losses.equation_losses import DarcyEqnLoss


class DarcyHybridLossGINO:
    """Combines `DarcyEqnLoss`
    (always active, physics term) with an optional weighted supervised data
    loss, for GINO's flat point-cloud output queried at a fixed regular grid.

    Three conventions meet in this class, which is exactly why it exists as
    a separate adapter rather than being folded into `DarcyEqnLoss` itself:

    * GINO's own output layout: flat, `(batch, n_x * n_y, 1)`, produced by
      querying GINO at the Darcy grid's coordinates (flattened in row-major
      order) via `output_queries`.
    * `DarcyEqnLoss`'s layout: channels-last grid, `(batch, n_x, n_y, 1)`.
    * `LpLoss`/`H1Loss`'s layout: channel-first, `(batch, 1, n_x, n_y)`
      -- matching the Darcy dataset's own channel-first ground-truth
      convention, so the labeled target `y` needs no reshaping, only the
      reshaped prediction does.

    Parameters
    ----------
    physics_loss : DarcyEqnLoss
        the physics (PDE-residual) term, always included
    grid_shape : tuple[int, int]
        `(n_x, n_y)`, the Darcy grid GINO was queried at -- used to fold
        GINO's flat output back into a grid, and must match the row-major
        ordering used to build `output_queries`
    data_loss : LpLoss or H1Loss, optional
        supervised data loss against the labeled pressure field, by default
        None. Construct with `d=2` (matching the channel-first
        `(batch, 1, n_x, n_y)` layout this class feeds it).
    data_weight : float, optional
        weight applied to `data_loss`, by default 0.0 (pure physics-only
        training, matching PFEM's own pretraining stage). A nonzero weight
        requires `data_loss` to be given.
    """

    def __init__(
        self,
        physics_loss: DarcyEqnLoss,
        grid_shape: tuple[int, int],
        data_loss: LpLoss | H1Loss | None = None,
        data_weight: float = 0.0,
    ) -> None:
        if data_weight != 0.0 and data_loss is None:
            raise ValueError("data_weight != 0 requires a data_loss to be given.")
        if len(grid_shape) != 2:
            raise ValueError(
                "DarcyHybridLossGINO expects grid_shape to have exactly 2 "
                f"entries (n_x, n_y), got {grid_shape}"
            )
        self.physics_loss = physics_loss
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.data_loss = data_loss
        self.data_weight = float(data_weight)

    @property
    def name(self) -> str:
        """str: Human-readable identifier."""
        return "DarcyHybridLossGINO"

    def _to_grid(self, y_pred: jnp.ndarray) -> jnp.ndarray:
        """Fold GINO's flat output into a channels-last grid.

        Parameters
        ----------
        y_pred : jnp.ndarray
            GINO's raw output, shape `(batch, n_x * n_y, 1)`

        Returns
        -------
        jnp.ndarray
            channels-last grid, shape `(batch, n_x, n_y, 1)`
        """
        n_x, n_y = self.grid_shape
        if y_pred.ndim != 3 or y_pred.shape[1] != n_x * n_y or y_pred.shape[2] != 1:
            raise ValueError(
                "DarcyHybridLossGINO expects y_pred of shape "
                f"(batch, {n_x * n_y}, 1) (GINO's flat output queried at the "
                f"{n_x}x{n_y} Darcy grid), got {y_pred.shape}"
            )
        return jnp.reshape(y_pred, (y_pred.shape[0], n_x, n_y, 1))

    def __call__(
        self,
        y_pred: jnp.ndarray,
        x: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Matches the `loss(out, **sample)` calling convention used by
        `Trainer`.

        Parameters
        ----------
        y_pred : jnp.ndarray
            GINO's raw output, shape `(batch, n_x * n_y, 1)`
        x : jnp.ndarray
            permeability field, already channels-last grid shape
            `(batch, n_x, n_y, 1)` (the model's own scalar input, read from
            the batch dict -- never passes through GINO's point-cloud output,
            so it needs no reshape here)
        y : jnp.ndarray, optional
            labeled pressure field, channel-first shape
            `(batch, 1, n_x, n_y)`, matching the Darcy dataset's own
            convention. Required if `data_weight != 0` (raises otherwise --
            see the class docstring's note on `enforce_dirichlet`).

        Returns
        -------
        jnp.ndarray
            the combined scalar loss -- an unnormalized training objective
            (physics residual, optionally plus a weighted data term), not a
            normalized "error" metric comparable across different
            `data_weight`/loss configurations.
        """
        del kwargs
        if x is None:
            raise ValueError("DarcyHybridLossGINO requires the permeability field `x`.")

        u_grid = self._to_grid(y_pred)
        total = self.physics_loss(u_grid, x=x)
        if self.data_weight != 0.0:
            if y is None:
                raise ValueError(
                    "DarcyHybridLossGINO was constructed with data_weight != 0 "
                    "(a nonzero-weight data term configured), but `y` was not "
                    "passed to __call__ -- the data term would otherwise be "
                    "silently dropped instead of applied, which is especially "
                    "unsafe with DarcyEqnLoss's default enforce_dirichlet=True: "
                    "the model would then get no supervision at all for its "
                    "boundary predictions."
                )
            if self.data_loss is None:
                raise ValueError(
                    "DarcyHybridLossGINO has data_weight != 0 but no `data_loss` was given."
                )
            u_grid_channel_first = jnp.transpose(u_grid, (0, 3, 1, 2))
            total = total + self.data_weight * self.data_loss(u_grid_channel_first, y=y)
        return total
