# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Experimental training loop: no stability guarantee.

`Trainer` drives any object that has the `train_step` / `__call__` contract
(the Car-CFD benchmark wraps GINO, Transolver and GIWO that way). The other
benchmarks train with their own loops. The stable `scirex.training` package keeps
only the optimizers (`adamw_cx`, `scirex.training.optim`).

Names resolve lazily (PEP 562), so `import scirex.experimental.training` does not
import JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from scirex.experimental.training.trainer import SimpleDataLoader, Trainer
    from scirex.experimental.training.training_state import (
        load_training_state,
        save_training_state,
    )

#: name -> the module that defines it
EXPERIMENTAL_EXPORTS: dict[str, str] = {
    "SimpleDataLoader": "scirex.experimental.training.trainer",
    "Trainer": "scirex.experimental.training.trainer",
    "load_training_state": "scirex.experimental.training.training_state",
    "save_training_state": "scirex.experimental.training.training_state",
}

__all__ = [
    "SimpleDataLoader",
    "Trainer",
    "load_training_state",
    "save_training_state",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, EXPERIMENTAL_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
