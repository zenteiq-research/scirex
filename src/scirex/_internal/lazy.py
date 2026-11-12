# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Helpers for package-level lazy exports (PEP 562).

Public packages such as `scirex.models` export their names through a module
level `__getattr__` so that `import scirex.models` does not import JAX, Flax or
any optional dependency. The first access of a name imports the module that defines
it. If that import fails because a third-party dependency is missing, the failure
is reported as an `ImportError` that names the dependency and the feature
that needs it.

This module must stay free of heavy imports.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, MutableMapping
from typing import Any

# Import name -> name on PyPI, for the packages whose absence we explain.
_PIP_NAMES: dict[str, str] = {
    "einops": "einops",
    "jaxwt": "jaxwt",
    "open3d": "open3d",
    "scipy": "scipy",
    "tensorly": "tensorly",
}


def missing_dependency_error(
    err: ModuleNotFoundError, feature: str
) -> ImportError | None:
    """Build an explanatory `ImportError` for a missing third-party package.

    Args:
        err: The `ModuleNotFoundError` raised while importing `feature`.
        feature: Human-readable name of what was being imported, e.g. `"WNO"`.

    Returns:
        An `ImportError` naming the missing package, or `None` if the missing
        module is part of ScireX itself (a real bug that must not be disguised).
    """
    missing = (err.name or "").split(".")[0]
    if not missing or missing == "scirex":
        return None
    pip_name = _PIP_NAMES.get(missing, missing)
    return ImportError(
        f"{feature} requires the third-party package {missing!r}, "
        "which is not installed. "
        f"Install it with `pip install {pip_name}`."
    )


def import_feature(module_name: str, feature: str) -> Any:
    """Import `module_name`, explaining a missing third-party dependency.

    Args:
        module_name: Absolute dotted module name.
        feature: Human-readable name of what needs the module.

    Returns:
        The imported module.

    Raises:
        ImportError: If a third-party dependency of the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        explained = missing_dependency_error(err, feature)
        if explained is None:
            raise
        raise explained from err


def resolve_lazy_attr(
    package: str,
    name: str,
    table: Mapping[str, str],
    namespace: MutableMapping[str, Any],
) -> Any:
    """Resolve `name` of `package` through `table` and cache it in `namespace`.

    Meant to be called from a module level `__getattr__`.

    Args:
        package: `__name__` of the package whose attribute is requested.
        name: The requested attribute.
        table: Maps each lazily exported name to the absolute name of the module
            that provides it.
        namespace: The package's `globals()`; the resolved value is stored there
            so that later lookups bypass `__getattr__`.

    Returns:
        The attribute.

    Raises:
        AttributeError: If `name` is not in `table`.
        ImportError: If a third-party dependency of the defining module is missing.
    """
    try:
        module_name = table[name]
    except KeyError:
        raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
    value = getattr(import_feature(module_name, name), name)
    namespace[name] = value
    return value
