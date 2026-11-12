<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.losses

Data losses, and the finite-element building blocks (`Q4Galerkin`) behind the Ritz-energy loss of the
Darcy physics-only benchmark. The residual losses of the Darcy equation are listed below for
completeness; no released benchmark uses them.

```python
from scirex.losses import LpLoss
```

| Group | Names |
| --- | --- |
| Data losses | `LpLoss`, `H1Loss` |
| Darcy residual | `DarcyEqnLoss`, `DiffusionTerm`, `SourceTerm` |
| Derivatives | `AxisSpec`, `FiniteDiff` |
| Weak form | `Q4Galerkin` |

The other physics losses are in [`scirex.experimental.losses`](experimental.md#experimental-losses).
For how to choose among them, read [Physics-informed losses](../loss_guide/overview.md).

## Data losses

::: scirex.losses.LpLoss

::: scirex.losses.H1Loss

## Darcy residual loss

Finite-difference residual of the 2D Darcy equation, built from these terms.

::: scirex.losses.DarcyEqnLoss

::: scirex.losses.DiffusionTerm

::: scirex.losses.SourceTerm

## Derivative operators

::: scirex.losses.AxisSpec

::: scirex.losses.FiniteDiff

## Galerkin (weak-form) loss

::: scirex.losses.Q4Galerkin
