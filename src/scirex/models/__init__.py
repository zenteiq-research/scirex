# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Neural-operator model architectures and the model factory.

The names below are resolved lazily (PEP 562): `import scirex.models` imports
nothing heavy, and each architecture is imported on first access. A missing
third-party dependency (for example `jaxwt` for `WNO`) therefore raises
an `ImportError` that names it, but only when the architecture is used.

`get_model_jax` and `available_models` know every built-in architecture
without it having been imported first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .base import BaseModel, available_models, get_model_jax
    from .fno import FNO
    from .fno_classic import FNOClassic2d
    from .gino import GINO
    from .transolver import Transolver
    from .wno import WNO

__all__ = [
    "FNO",
    "GINO",
    "WNO",
    "BaseModel",
    "FNOClassic2d",
    "Transolver",
    "available_models",
    "get_model_jax",
]

_LAZY_EXPORTS: dict[str, str] = {
    "BaseModel": "scirex.models.base",
    "FNO": "scirex.models.fno",
    "FNOClassic2d": "scirex.models.fno_classic",
    "GINO": "scirex.models.gino",
    "Transolver": "scirex.models.transolver",
    "WNO": "scirex.models.wno",
    "available_models": "scirex.models.base",
    "get_model_jax": "scirex.models.base",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
