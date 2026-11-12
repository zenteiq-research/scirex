# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the GNO block."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.layers.blocks.gno_block import GNOBlock
from scirex.layers.blocks.gno_weighting_functions import dispatch_weighting_fn

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)

# Fixed variables matching the reference test structure
IN_CHANNELS = 3
OUT_CHANNELS = 3
MLP_HIDDEN_LAYERS = [16, 16, 16]
N_IN = 30
N_OUT = 20


def _make_neighbors(
    y: jnp.ndarray, x: jnp.ndarray, radius: float
) -> dict[str, jnp.ndarray]:
    """Build a neighbours dict (segment_ids + counts) from two point clouds."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.array(y))
    neighbor_lists = tree.query_ball_point(np.array(x), r=radius)
    all_indices, row_splits, current = [], [0], 0
    for nbrs in neighbor_lists:
        all_indices.extend(nbrs)
        current += len(nbrs)
        row_splits.append(current)
    idx = (
        jnp.array(all_indices, dtype=jnp.int32)
        if all_indices
        else jnp.zeros(0, jnp.int32)
    )
    row_splits_arr = jnp.array(row_splits, dtype=jnp.int32)
    counts = (row_splits_arr[1:] - row_splits_arr[:-1]).astype(jnp.int32)
    seg_ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)
    return {"neighbors_index": idx, "segment_ids": seg_ids, "counts": counts}


# ---------------------------------------------------------------------------
# Parametrized shape / forward pass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("coord_dim", [2])
@pytest.mark.parametrize("pos_embed_type", ["nerf", "transformer", None])
@pytest.mark.parametrize(
    "transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
def test_gno_block_output_shape(
    backend: str,
    batch_size: int,
    coord_dim: int,
    pos_embed_type: str | None,
    transform_type: str,
) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        y = jax.random.normal(k1, (N_IN, coord_dim))
        x = jax.random.normal(k2, (N_OUT, coord_dim))

        gno = GNOBlock(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            coord_dim=coord_dim,
            radius=0.5,
            transform_type=transform_type,
            pos_embedding_type=pos_embed_type,
            channel_mlp_layers=MLP_HIDDEN_LAYERS,
            use_open3d_neighbor_search=False,
        )

        f_y = None
        if transform_type != "linear":
            f_y = jax.random.normal(k3, (batch_size, N_IN, IN_CHANNELS))

        neighbors = _make_neighbors(y, x, radius=0.5)
        params = gno.init(key, y=y, x=x, f_y=f_y, neighbors=neighbors)
        out = cast(
            jnp.ndarray, gno.apply(params, y=y, x=x, f_y=f_y, neighbors=neighbors)
        )

        if transform_type != "linear":
            assert out.shape == (
                batch_size,
                N_OUT,
                OUT_CHANNELS,
            ), f"Expected ({batch_size}, {N_OUT}, {OUT_CHANNELS}), got {out.shape}"
        else:
            assert out.shape == (
                N_OUT,
                OUT_CHANNELS,
            ), f"Expected ({N_OUT}, {OUT_CHANNELS}), got {out.shape}"

        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gno_block_no_pos_embedding(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        y = jax.random.normal(key, (N_IN, 2))
        x = jax.random.normal(key, (N_OUT, 2))
        gno = GNOBlock(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            coord_dim=2,
            radius=0.5,
            transform_type="linear",
            pos_embedding_type=None,
            channel_mlp_layers=MLP_HIDDEN_LAYERS,
            use_open3d_neighbor_search=False,
        )
        neighbors = _make_neighbors(y, x, radius=0.5)
        params = gno.init(key, y=y, x=x, neighbors=neighbors)
        out = cast(jnp.ndarray, gno.apply(params, y=y, x=x, neighbors=neighbors))
        assert out.shape == (N_OUT, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gno_block_precomputed_neighbors_skips_search(backend: str) -> None:
    """Passing neighbors bypasses NeighborSearch; output shape must still be
    correct."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        y = jax.random.normal(key, (N_IN, 2))
        x = jax.random.normal(key, (N_OUT, 2))
        gno = GNOBlock(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            coord_dim=2,
            radius=0.5,
            transform_type="linear",
            pos_embedding_type=None,
            channel_mlp_layers=MLP_HIDDEN_LAYERS,
            use_open3d_neighbor_search=False,
        )
        neighbors = _make_neighbors(y, x, radius=0.5)
        params = gno.init(key, y=y, x=x, neighbors=neighbors)
        # Call twice with same precomputed neighbors - result must be identical
        out1 = cast(jnp.ndarray, gno.apply(params, y=y, x=x, neighbors=neighbors))
        out2 = cast(jnp.ndarray, gno.apply(params, y=y, x=x, neighbors=neighbors))
        assert jnp.allclose(out1, out2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gno_block_internal_neighbor_search(backend: str) -> None:
    """GNOBlock should work end-to-end when neighbors=None (runs NeighborSearch
    internally)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        y = jax.random.normal(key, (N_IN, 2))
        x = jax.random.normal(key, (N_OUT, 2))
        gno = GNOBlock(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            coord_dim=2,
            radius=0.5,
            transform_type="linear",
            pos_embedding_type=None,
            channel_mlp_layers=MLP_HIDDEN_LAYERS,
            use_open3d_neighbor_search=False,
        )
        # Init with precomputed neighbors, then call with neighbors=None
        neighbors = _make_neighbors(y, x, radius=0.5)
        params = gno.init(key, y=y, x=x, neighbors=neighbors)
        out = cast(jnp.ndarray, gno.apply(params, y=y, x=x))
        assert out.shape == (N_OUT, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gno_block_with_weighting_fn(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        y = jax.random.normal(key, (N_IN, 2))
        x = jax.random.normal(key, (N_OUT, 2))
        weighting_fn = dispatch_weighting_fn("half_cos", sq_radius=0.5, scale=1.0)
        gno = GNOBlock(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            coord_dim=2,
            radius=0.5,
            transform_type="linear",
            pos_embedding_type=None,
            channel_mlp_layers=MLP_HIDDEN_LAYERS,
            use_open3d_neighbor_search=False,
            weighting_fn=weighting_fn,
        )
        # weighting_fn requires weights in neighbors dict
        from scirex.layers.blocks.neighbor_search import kdtree_neighbor_search

        raw = kdtree_neighbor_search(
            np.array(y), np.array(x), radius=0.5, return_norm=True
        )
        row_splits = jnp.array(raw["neighbors_row_splits"], dtype=jnp.int32)
        counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
        seg_ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)
        neighbors = {
            "neighbors_index": jnp.array(raw["neighbors_index"], dtype=jnp.int32),
            "segment_ids": seg_ids,
            "counts": counts,
            "weights": jnp.array(raw["weights"], dtype=jnp.float32),
        }
        params = gno.init(key, y=y, x=x, neighbors=neighbors)
        out = cast(jnp.ndarray, gno.apply(params, y=y, x=x, neighbors=neighbors))
        assert out.shape == (N_OUT, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))
