# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Install hints for third-party packages that live in optional extras.

`pyproject.toml` groups packages that only some features need (`vtk` for the
benchmark loaders, `mlflow` and `optuna` for tuning) into extras. The helpers
here turn the bare `ModuleNotFoundError` for such a package into an
`ImportError` that names the feature and the extra to install.

This module must stay free of heavy imports.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def missing_extra_error(
    err: ModuleNotFoundError, *, packages: tuple[str, ...], extra: str, feature: str
) -> ImportError | None:
    """Build an `ImportError` that names the extra to install.

    Args:
        err: The `ModuleNotFoundError` that was raised.
        packages: Top-level import names that the extra provides.
        extra: Name of the extra in `pyproject.toml`, e.g. `"hpo"`.
        feature: Human-readable name of what needs the package.

    Returns:
        An `ImportError` with the install command, or `None` if the missing
        module is not one of `packages` (so the original error should propagate).
    """
    missing = (err.name or "").split(".")[0]
    if missing not in packages:
        return None
    return ImportError(
        f"{feature} requires {missing!r}, which is not installed. "
        f'Install the {extra!r} extra with `pip install "scirex[{extra}]"`.'
    )


def requires_extra(*packages: str, extra: str) -> Callable[[F], F]:
    """Decorate a function so that a missing `packages` import names the extra.

    The check happens when the function runs, not when it is defined, so code paths
    that do not need the package (for example a cache hit) keep working without it.
    """

    def decorator(func: F) -> F:
        feature = f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ModuleNotFoundError as err:
                explained = missing_extra_error(
                    err, packages=packages, extra=extra, feature=feature
                )
                if explained is None:
                    raise
                raise explained from err

        return wrapper  # type: ignore[return-value]

    return decorator
