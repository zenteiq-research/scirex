<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.layers

The building blocks the models are made of. Use them to assemble your own operator.

```python
from scirex.layers import SpectralConv, ChannelMLP
```

| Group | Names |
| --- | --- |
| Convolution | `BaseSpectralConv`, `SpectralConv`, `WaveletConv` |
| Blocks | `FNOBlocks`, `WNOBlock`, `GNOBlock` |
| Neighbour search | `build_neighbor_graph`, `precompute_neighbors`, `kdtree_neighbor_search`, `edge_index_to_neighbors`, `neighbors_to_segments` |
| Embeddings and MLP | `GridEmbedding2D`, `GridEmbeddingND`, `ChannelMLP` |
| Activations | `cgelu`, `cselu`, `ctanh` (for complex-valued data) |

## Spectral and wavelet convolution

::: scirex.layers.BaseSpectralConv

::: scirex.layers.SpectralConv

::: scirex.layers.WaveletConv

## Operator blocks

::: scirex.layers.FNOBlocks

::: scirex.layers.WNOBlock

::: scirex.layers.GNOBlock

## Neighbor search and graphs

Helpers that build the radius graphs consumed by `GNOBlock`.

::: scirex.layers.build_neighbor_graph

::: scirex.layers.precompute_neighbors

::: scirex.layers.kdtree_neighbor_search

::: scirex.layers.edge_index_to_neighbors

::: scirex.layers.neighbors_to_segments

## Embeddings and MLP

::: scirex.layers.GridEmbedding2D

::: scirex.layers.GridEmbeddingND

::: scirex.layers.ChannelMLP

## Complex-valued activations

::: scirex.layers.cgelu

::: scirex.layers.cselu

::: scirex.layers.ctanh
