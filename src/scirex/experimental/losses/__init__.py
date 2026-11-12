# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Experimental physics-informed losses: no stability guarantee.

The stable `scirex.losses` package keeps the data losses (`LpLoss`, `H1Loss`)
and the losses used by the released benchmarks: `DarcyEqnLoss`, `Q4Galerkin` and
`FiniteDiff` with the term classes `DarcyEqnLoss` is built from (`AxisSpec`,
`DiffusionTerm`, `SourceTerm`). Everything else lives here: the generic
`PDEResidualLoss` framework and its remaining term kinds, the Navier-Stokes
residual, the T3 and T6 Galerkin losses, Fourier differentiation, and the
automatic-differentiation residuals for irregular geometry.

Names resolve lazily (PEP 562), so `import scirex.experimental.losses` does not
import JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from scirex.experimental.losses.ad_differentiation import ADDiff
    from scirex.experimental.losses.ad_pde_terms import (
        AdvectionTermAD,
        DiffusionTermAD,
        ReactionTermAD,
        SourceTermAD,
    )
    from scirex.experimental.losses.darcy_hybrid_loss import DarcyHybridLossGINO
    from scirex.experimental.losses.galerkin_t6 import (
        BatchedVariableCoefficientGalerkinResidualLossT6,
        T6Galerkin,
    )
    from scirex.losses.differentiation import FourierDiff
    from scirex.losses.equation_losses import NavierStokesEqnLoss, PDEResidualLoss
    from scirex.losses.galerkin import (
        BatchedVariableCoefficientGalerkinResidualLossT3,
        T3Galerkin,
    )
    from scirex.losses.pde_terms import (
        AdvectionTerm,
        DivergenceTerm,
        GradientTerm,
        ReactionTerm,
        TransientTerm,
    )

#: name -> the module that defines it (some of those modules are still in
#: `scirex.losses` because the stable losses share them)
EXPERIMENTAL_EXPORTS: dict[str, str] = {
    "ADDiff": "scirex.experimental.losses.ad_differentiation",
    "AdvectionTerm": "scirex.losses.pde_terms",
    "AdvectionTermAD": "scirex.experimental.losses.ad_pde_terms",
    "BatchedVariableCoefficientGalerkinResidualLossT3": "scirex.losses.galerkin",
    "BatchedVariableCoefficientGalerkinResidualLossT6": (
        "scirex.experimental.losses.galerkin_t6"
    ),
    "DarcyHybridLossGINO": "scirex.experimental.losses.darcy_hybrid_loss",
    "DiffusionTermAD": "scirex.experimental.losses.ad_pde_terms",
    "DivergenceTerm": "scirex.losses.pde_terms",
    "FourierDiff": "scirex.losses.differentiation",
    "GradientTerm": "scirex.losses.pde_terms",
    "NavierStokesEqnLoss": "scirex.losses.equation_losses",
    "PDEResidualLoss": "scirex.losses.equation_losses",
    "ReactionTerm": "scirex.losses.pde_terms",
    "ReactionTermAD": "scirex.experimental.losses.ad_pde_terms",
    "SourceTermAD": "scirex.experimental.losses.ad_pde_terms",
    "T3Galerkin": "scirex.losses.galerkin",
    "T6Galerkin": "scirex.experimental.losses.galerkin_t6",
    "TransientTerm": "scirex.losses.pde_terms",
}

__all__ = [
    "ADDiff",
    "AdvectionTerm",
    "AdvectionTermAD",
    "BatchedVariableCoefficientGalerkinResidualLossT3",
    "BatchedVariableCoefficientGalerkinResidualLossT6",
    "DarcyHybridLossGINO",
    "DiffusionTermAD",
    "DivergenceTerm",
    "FourierDiff",
    "GradientTerm",
    "NavierStokesEqnLoss",
    "PDEResidualLoss",
    "ReactionTerm",
    "ReactionTermAD",
    "SourceTermAD",
    "T3Galerkin",
    "T6Galerkin",
    "TransientTerm",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, EXPERIMENTAL_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
