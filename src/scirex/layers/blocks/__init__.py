# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Composite layers: spectral convolutions, FNO, GNO and WNO blocks, neighbor search.

The names below are resolved lazily (PEP 562), so `import scirex.layers.blocks`
imports nothing heavy and `WaveletConv` and `WNOBlock` only require
`jaxwt` when they are actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .base_spectral_conv import BaseSpectralConv
    from .fno_block import FNOBlocks
    from .gno_block import GNOBlock
    from .neighbor_search import (
        build_neighbor_graph,
        edge_index_to_neighbors,
        kdtree_neighbor_search,
        neighbors_to_segments,
        precompute_neighbors,
    )
    from .spectral_convolution import SpectralConv
    from .wavelet_conv import WaveletConv
    from .wno_block import WNOBlock

__all__ = [
    "BaseSpectralConv",
    "FNOBlocks",
    "GNOBlock",
    "SpectralConv",
    "WNOBlock",
    "WaveletConv",
    "build_neighbor_graph",
    "edge_index_to_neighbors",
    "kdtree_neighbor_search",
    "neighbors_to_segments",
    "precompute_neighbors",
]

_LAZY_EXPORTS: dict[str, str] = {
    "BaseSpectralConv": "scirex.layers.blocks.base_spectral_conv",
    "FNOBlocks": "scirex.layers.blocks.fno_block",
    "GNOBlock": "scirex.layers.blocks.gno_block",
    "SpectralConv": "scirex.layers.blocks.spectral_convolution",
    "WNOBlock": "scirex.layers.blocks.wno_block",
    "WaveletConv": "scirex.layers.blocks.wavelet_conv",
    "build_neighbor_graph": "scirex.layers.blocks.neighbor_search",
    "edge_index_to_neighbors": "scirex.layers.blocks.neighbor_search",
    "kdtree_neighbor_search": "scirex.layers.blocks.neighbor_search",
    "neighbors_to_segments": "scirex.layers.blocks.neighbor_search",
    "precompute_neighbors": "scirex.layers.blocks.neighbor_search",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
