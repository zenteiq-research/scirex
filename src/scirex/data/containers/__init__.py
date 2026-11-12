# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""In-memory dataset containers (lazy exports, see `scirex.data`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .dict_dataset import DictDataset
    from .tensor_dataset import TensorDataset

__all__ = ["DictDataset", "TensorDataset"]

_LAZY_EXPORTS: dict[str, str] = {
    "DictDataset": "scirex.data.containers.dict_dataset",
    "TensorDataset": "scirex.data.containers.tensor_dataset",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
