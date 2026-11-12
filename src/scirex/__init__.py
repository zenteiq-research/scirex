# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
ScireX neural operators: a JAX/Flax library for learning solution
operators of partial differential equations.

Exposes the top-level model factory `get_model_jax` together with the FNO,
WNO, GINO and Transolver architectures. Everything except the version string is
imported lazily (PEP 562), so `import scirex` does not import JAX; the first
access of `scirex.FNO` or `scirex.models` does.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from scirex._internal.aliases import install as _install_aliases
from scirex._internal.lazy import resolve_lazy_attr

# Old import paths of modules that moved keep working (with a DeprecationWarning);
# the table is scirex._internal.aliases.MOVED.
_install_aliases()

if TYPE_CHECKING:
    __version__: str  # read from the package metadata on first access, see __getattr__
    from .models import FNO, GINO, WNO, Transolver, get_model_jax

__all__ = ["FNO", "GINO", "WNO", "Transolver", "get_model_jax"]

_LAZY_EXPORTS: dict[str, str] = {
    "FNO": "scirex.models",
    "GINO": "scirex.models",
    "Transolver": "scirex.models",
    "WNO": "scirex.models",
    "get_model_jax": "scirex.models",
}

# Subpackages reachable as attributes of `scirex` without an explicit import.
_SUBMODULES = frozenset(
    {
        "config",
        "data",
        "experimental",
        "layers",
        "losses",
        "models",
        "training",
        "utils",
    }
)


def __getattr__(name: str) -> Any:
    if name == "__version__":
        # defined once, in pyproject.toml; looked up on first use so that
        # `import scirex` does not scan the installed distributions
        from scirex._internal.version import package_version

        globals()["__version__"] = package_version()
        return globals()["__version__"]
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_SUBMODULES, "__version__"})
