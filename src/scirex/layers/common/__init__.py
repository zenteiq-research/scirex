# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Building blocks shared by several models.

Channel MLP, grid embeddings and complex-valued activations.

The names below are resolved lazily (PEP 562), so `import scirex.layers.common`
imports nothing heavy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .channel_mlp import ChannelMLP
    from .complex import cgelu, cselu, ctanh
    from .embeddings import GridEmbedding2D, GridEmbeddingND

__all__ = [
    "ChannelMLP",
    "GridEmbedding2D",
    "GridEmbeddingND",
    "cgelu",
    "cselu",
    "ctanh",
]

_LAZY_EXPORTS: dict[str, str] = {
    "ChannelMLP": "scirex.layers.common.channel_mlp",
    "GridEmbedding2D": "scirex.layers.common.embeddings",
    "GridEmbeddingND": "scirex.layers.common.embeddings",
    "cgelu": "scirex.layers.common.complex",
    "cselu": "scirex.layers.common.complex",
    "ctanh": "scirex.layers.common.complex",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
