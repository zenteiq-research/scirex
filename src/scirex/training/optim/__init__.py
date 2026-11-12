# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Optimizers for neural-operator training.

Provides an Adam/AdamW variant that handles complex-valued parameters, as used by
the spectral weights of Fourier neural operators. The names are resolved lazily
(PEP 562), so `import scirex.training.optim` does not import JAX or Optax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .adamw import adamw_cx, scale_by_adam_cx

__all__ = ["adamw_cx", "scale_by_adam_cx"]

_LAZY_EXPORTS: dict[str, str] = {
    "adamw_cx": "scirex.training.optim.adamw",
    "scale_by_adam_cx": "scirex.training.optim.adamw",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
