# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Training subpackage: the complex-aware optimizers.

The names in `__all__` are resolved lazily (PEP 562): `import scirex.training`
imports nothing heavy (no JAX, Optax, Weights & Biases or Matplotlib), and each
name is imported on first access. Optimizers are also available from
`scirex.training.optim`. The training loop (`Trainer`) and the training-state
helpers are in `scirex.experimental.training`; the old names still resolve from
here, with a `DeprecationWarning`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from scirex._internal.deprecation import warn_deprecated
from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .optim.adamw import adamw_cx

__all__ = ["adamw_cx"]

_LAZY_EXPORTS: dict[str, str] = {
    "adamw_cx": "scirex.training.optim.adamw",
}

# Formerly exported here, now experimental: still resolvable for the deprecation
# period (name -> module). They are deliberately not in `__all__`.
_MOVED_TO_EXPERIMENTAL: dict[str, str] = {
    "SimpleDataLoader": "scirex.experimental.training.trainer",
    "Trainer": "scirex.experimental.training.trainer",
    "load_training_state": "scirex.experimental.training.training_state",
    "save_training_state": "scirex.experimental.training.training_state",
}

# Names that used to be re-exported here from `wno_train_utils`. They now live
# in the private `scirex._internal.wno_train_utils` and are resolved lazily,
# with a DeprecationWarning, so existing imports keep working for now. They are
# deliberately not in `__all__`.
_DEPRECATED_REEXPORTS = frozenset(
    {
        "append_grid_nhwc",
        "append_grid_nwc",
        "create_train_state",
        "make_schedule",
        "per_frame_relative_l2",
        "plot_loss_curves",
        "relative_l2_sum",
        "resolve_dtype",
        "write_metrics",
    }
)


def __getattr__(name: str) -> Any:
    if name in _MOVED_TO_EXPERIMENTAL:
        warn_deprecated(
            f"scirex.training.{name}",
            since="1.0.0",
            removal="1.2.0",
            alternative=(
                f"scirex.experimental.training.{name} "
                "(experimental, no stability guarantee)"
            ),
            stacklevel=3,
        )
        return getattr(importlib.import_module(_MOVED_TO_EXPERIMENTAL[name]), name)
    if name in _DEPRECATED_REEXPORTS:
        warn_deprecated(
            f"scirex.training.{name}",
            since="1.0.0",
            removal="1.2.0",
            alternative=(
                f"scirex._internal.wno_train_utils.{name} (it is not public API)"
            ),
            stacklevel=3,
        )
        module = importlib.import_module("scirex._internal.wno_train_utils")
        return getattr(module, name)
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
