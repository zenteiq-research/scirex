# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Hyperparameter tuning with Optuna, tracked in MLflow.

Requires the `hpo` extra: `pip install "scirex[hpo]"`.

`import scirex.experimental.tuning` always works. The names in `__all__` are resolved lazily
(PEP 562) and importing them without the `hpo` extra installed raises an
`ImportError` that says so.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from scirex._internal.extras import missing_extra_error

if TYPE_CHECKING:
    from .tuner import TrialContext, suggest_params, tune

__all__ = ["TrialContext", "suggest_params", "tune"]

# Top-level packages tuner.py imports that are provided by the `hpo` extra.
_HPO_PACKAGES = ("mlflow", "optuna")


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module("scirex.experimental.tuning.tuner")
    except ModuleNotFoundError as err:
        explained = missing_extra_error(
            err,
            packages=_HPO_PACKAGES,
            extra="hpo",
            feature=f"scirex.experimental.tuning.{name}",
        )
        if explained is None:
            raise
        raise explained from err
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
