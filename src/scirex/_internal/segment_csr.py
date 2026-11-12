# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Segment-CSR reduction of edge features to node features for graph operations."""

from typing import Literal

import jax
import jax.numpy as jnp


def segment_csr(
    src: jnp.ndarray,
    segment_ids: jnp.ndarray,
    counts: jnp.ndarray,
    reduction: Literal["mean", "sum"] = "mean",
    use_scatter: bool = False,
) -> jnp.ndarray:
    """Reduce edge features to node features using pre-padded segment IDs.

    All inputs have static shapes across batches, making this function fully
    compatible with `jax.jit`.

    Parameters
    ----------
    src : jnp.ndarray
        Edge features, shape `[max_edges, C]` or `[B, max_edges, C]`.
        Padded edges must map to segment `n_out` (they are discarded).
    segment_ids : jnp.ndarray
        Integer array `[max_edges]` mapping each edge to its output node.
        Padded edges must have value `n_out` so their contribution is dropped.
    counts : jnp.ndarray
        Number of real edges per output node, shape `[n_out]`.
        `n_out` is derived from `counts.shape[0]` - a static compile-time
        constant, which avoids passing a Python int that JAX would trace.
    reduction : 'mean' or 'sum'
    use_scatter : bool
        Ignored - kept so existing callers with `use_scatter=...` still work.
    """
    if reduction not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    # n_out is static (shape attribute), safe to use inside jax.jit
    n_out = counts.shape[0]

    if src.ndim == 3:
        # batched input, shape (B, max_edges, C)
        out = jax.vmap(
            lambda s: jax.ops.segment_sum(s, segment_ids, num_segments=n_out + 1)
        )(src)
        out = out[:, :n_out, :]  # drop dummy segment
        if reduction == "mean":
            denom = jnp.maximum(counts, 1).astype(src.dtype)  # [n_out]
            out = out / denom[None, :, None]
    else:
        # unbatched input, shape (max_edges, C)
        out = jax.ops.segment_sum(src, segment_ids, num_segments=n_out + 1)
        out = out[:n_out]  # drop dummy segment
        if reduction == "mean":
            denom = jnp.maximum(counts, 1).astype(src.dtype)  # [n_out]
            out = out / denom[:, None]

    return out
