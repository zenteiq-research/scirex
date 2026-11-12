# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data processors and normalizers (lazy exports, see `scirex.data`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .data_processors import DefaultDataProcessor
    from .normalizers import UnitGaussianNormalizer

__all__ = ["DefaultDataProcessor", "UnitGaussianNormalizer"]

_LAZY_EXPORTS: dict[str, str] = {
    "DefaultDataProcessor": "scirex.data.transforms.data_processors",
    "UnitGaussianNormalizer": "scirex.data.transforms.normalizers",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
