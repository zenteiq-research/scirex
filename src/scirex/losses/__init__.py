# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Loss functions for neural-operator training.

The names below are resolved lazily (PEP 562): `import scirex.losses` imports
nothing, and each loss module is imported on first access of one of its names.

The stable losses are the data losses (`LpLoss`, `H1Loss`) and the physics losses
used by the released benchmarks: `DarcyEqnLoss`, `Q4Galerkin` and `FiniteDiff`
with the term classes `DarcyEqnLoss` is built from. Every other loss
(`PDEResidualLoss` and its remaining terms, `NavierStokesEqnLoss`, the T3 and T6
Galerkin losses, `FourierDiff`, the automatic-differentiation residuals) lives in
`scirex.experimental.losses`. The old names still resolve from this package, with
a `DeprecationWarning`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from scirex._internal.deprecation import warn_deprecated
from scirex._internal.lazy import resolve_lazy_attr
from scirex.experimental.losses import EXPERIMENTAL_EXPORTS as _DEPRECATED_EXPORTS

if TYPE_CHECKING:
    from .data_losses import H1Loss, LpLoss
    from .differentiation import FiniteDiff
    from .equation_losses import DarcyEqnLoss
    from .galerkin_q4 import Q4Galerkin
    from .pde_terms import AxisSpec, DiffusionTerm, SourceTerm

__all__ = [
    "AxisSpec",
    "DarcyEqnLoss",
    "DiffusionTerm",
    "FiniteDiff",
    "H1Loss",
    "LpLoss",
    "Q4Galerkin",
    "SourceTerm",
]

_LAZY_EXPORTS: dict[str, str] = {
    "AxisSpec": "scirex.losses.pde_terms",
    "DarcyEqnLoss": "scirex.losses.equation_losses",
    "DiffusionTerm": "scirex.losses.pde_terms",
    "FiniteDiff": "scirex.losses.differentiation",
    "H1Loss": "scirex.losses.data_losses",
    "LpLoss": "scirex.losses.data_losses",
    "Q4Galerkin": "scirex.losses.galerkin_q4",
    "SourceTerm": "scirex.losses.pde_terms",
}


def __getattr__(name: str) -> Any:
    # Formerly exported here, now experimental: still resolvable for the deprecation
    # period. Policy: doc/dev_guide/api_policy.md (at least two minor releases).
    if name in _DEPRECATED_EXPORTS:
        warn_deprecated(
            f"scirex.losses.{name}",
            since="1.0.0",
            removal="1.2.0",
            alternative=f"scirex.experimental.losses.{name} (experimental, no stability guarantee)",
            stacklevel=3,
        )
        return getattr(importlib.import_module(_DEPRECATED_EXPORTS[name]), name)
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
