# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Deprecation helpers implementing the policy in `doc/dev_guide/api_policy.md`.

A deprecated public name keeps working for at least two minor releases and emits
a `DeprecationWarning` that says when it was deprecated, when it can be removed
and what to use instead.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecation_message(
    name: str, *, since: str, removal: str, alternative: str | None = None
) -> str:
    """The standard text of the `DeprecationWarning` for `name`."""
    message = f"{name} is deprecated since ScireX {since} and will be removed in {removal} or later."
    if alternative:
        message += f" Use {alternative} instead."
    return message


def warn_deprecated(
    name: str,
    *,
    since: str,
    removal: str,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit the standard `DeprecationWarning` for `name`.

    Args:
        name: Dotted name of the deprecated object, e.g. `"scirex.training.resolve_dtype"`.
        since: Release in which the deprecation started.
        removal: Earliest release in which the name may be removed.
        alternative: What users should use instead, if anything.
        stacklevel: Passed to `warnings.warn`; `2` points at the caller of
            the function that calls this helper.
    """
    message = deprecation_message(
        name, since=since, removal=removal, alternative=alternative
    )
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    *, since: str, removal: str, alternative: str | None = None
) -> Callable[[F], F]:
    """Decorate a function or class so that using it emits a `DeprecationWarning`."""

    def decorator(obj: F) -> F:
        name = f"{obj.__module__}.{obj.__qualname__}"
        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def init(self: Any, *args: Any, **kwargs: Any) -> None:
                warn_deprecated(
                    name,
                    since=since,
                    removal=removal,
                    alternative=alternative,
                    stacklevel=3,
                )
                original_init(self, *args, **kwargs)

            obj.__init__ = init  # type: ignore[assignment, misc]
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(
                name,
                since=since,
                removal=removal,
                alternative=alternative,
                stacklevel=3,
            )
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
