# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Neural-operator layers: spectral convolutions, GNO blocks, embeddings,
normalization, padding, skip connections and supporting utilities.

The public layers are exported both here and from the two subpackages that
define them, `scirex.layers.blocks` (composite blocks, neighbor search) and
`scirex.layers.common` (building blocks shared by several models). The
names are resolved lazily (PEP 562), so `import scirex.layers` imports nothing
heavy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr
from scirex.layers.blocks import _LAZY_EXPORTS as _BLOCKS_EXPORTS
from scirex.layers.common import _LAZY_EXPORTS as _COMMON_EXPORTS

if TYPE_CHECKING:
    from .blocks import (
        BaseSpectralConv,
        FNOBlocks,
        GNOBlock,
        SpectralConv,
        WaveletConv,
        WNOBlock,
        build_neighbor_graph,
        edge_index_to_neighbors,
        kdtree_neighbor_search,
        neighbors_to_segments,
        precompute_neighbors,
    )
    from .common import (
        ChannelMLP,
        GridEmbedding2D,
        GridEmbeddingND,
        cgelu,
        cselu,
        ctanh,
    )

__all__ = [
    "BaseSpectralConv",
    "ChannelMLP",
    "FNOBlocks",
    "GNOBlock",
    "GridEmbedding2D",
    "GridEmbeddingND",
    "SpectralConv",
    "WNOBlock",
    "WaveletConv",
    "build_neighbor_graph",
    "cgelu",
    "cselu",
    "ctanh",
    "edge_index_to_neighbors",
    "kdtree_neighbor_search",
    "neighbors_to_segments",
    "precompute_neighbors",
]

_LAZY_EXPORTS: dict[str, str] = {**_BLOCKS_EXPORTS, **_COMMON_EXPORTS}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
