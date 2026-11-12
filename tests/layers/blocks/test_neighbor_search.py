# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Tests fallback neighbor search on a small 2D grid
that was calculated manually.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.layers.blocks.neighbor_search import (
    NeighborSearch,
    kdtree_neighbor_search,
    native_neighbor_search,
    native_neighbor_search_chunked,
    native_neighbor_search_chunked_optimized,
    native_neighbor_search_numpy,
    neighbors_to_segments,
    precompute_neighbors,
)
from tests import tolerances

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)

# ---------------------------------------------------------------------------
# Manually-calculated CSR neighbor lists for a 5x5 grid on [0,1]x[0,1],
# radius = 0.3.
# ---------------------------------------------------------------------------
indices = [
    0,
    1,
    5,
    0,
    1,
    2,
    6,
    1,
    2,
    3,
    7,
    2,
    3,
    4,
    8,
    3,
    4,
    9,
    0,
    5,
    6,
    10,
    1,
    5,
    6,
    7,
    11,
    2,
    6,
    7,
    8,
    12,
    3,
    7,
    8,
    9,
    13,
    4,
    8,
    9,
    14,
    5,
    10,
    11,
    15,
    6,
    10,
    11,
    12,
    16,
    7,
    11,
    12,
    13,
    17,
    8,
    12,
    13,
    14,
    18,
    9,
    13,
    14,
    19,
    10,
    15,
    16,
    20,
    11,
    15,
    16,
    17,
    21,
    12,
    16,
    17,
    18,
    22,
    13,
    17,
    18,
    19,
    23,
    14,
    18,
    19,
    24,
    15,
    20,
    21,
    16,
    20,
    21,
    22,
    17,
    21,
    22,
    23,
    18,
    22,
    23,
    24,
    19,
    23,
    24,
]

splits = [
    0,
    3,
    7,
    11,
    15,
    18,
    22,
    27,
    32,
    37,
    41,
    45,
    50,
    55,
    60,
    64,
    68,
    73,
    78,
    83,
    87,
    90,
    94,
    98,
    102,
    105,
]


# ---------------------------------------------------------------------------
# 1:1 mirror of the reference test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_fallback_nb_search(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))

        return_dict = native_neighbor_search_chunked(
            data=coords, queries=coords, radius=0.3, return_norm=True
        )

        assert return_dict["neighbors_index"].tolist() == indices
        assert return_dict["neighbors_row_splits"].tolist() == splits

        # Compute squared norms independently via repeat/gather and compare weights
        num_reps = (
            return_dict["neighbors_row_splits"][1:]
            - return_dict["neighbors_row_splits"][:-1]
        )
        rep_queries = jnp.repeat(coords, num_reps, axis=0)
        rep_data = coords[return_dict["neighbors_index"]]
        rep_dist = rep_queries - rep_data
        squared_norm = (rep_dist**2).sum(axis=-1)

        assert jnp.allclose(
            return_dict["weights"], squared_norm, atol=tolerances.atol(1e-5)
        )


# ---------------------------------------------------------------------------
# Extra coverage: return_norm absent when not requested
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_chunked_no_norm_key_absent(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search_chunked(
            data=coords, queries=coords, radius=0.3, return_norm=False
        )
        assert "weights" not in result


# ---------------------------------------------------------------------------
# Extra coverage: empty neighbourhood (radius smaller than eps)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_chunked_empty_neighborhood(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search_chunked(coords, coords, radius=1e-9)
        row_splits = result["neighbors_row_splits"]
        counts = row_splits[1:] - row_splits[:-1]
        assert int(jnp.sum(counts)) == 0


# ---------------------------------------------------------------------------
# Extra coverage: kdtree_neighbor_search - counts must match reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_kdtree_counts_match_reference(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = mesh_grid.reshape(-1, 2).astype(np.float32)
        result = kdtree_neighbor_search(coords, coords, radius=0.3)
        row_splits = np.array(result["neighbors_row_splits"])
        counts = row_splits[1:] - row_splits[:-1]
        expected_counts = np.diff(splits)
        np.testing.assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize("backend", BACKENDS)
def test_kdtree_return_norm(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = mesh_grid.reshape(-1, 2).astype(np.float32)
        result = kdtree_neighbor_search(coords, coords, radius=0.3, return_norm=True)
        assert "weights" in result
        assert len(result["weights"]) == len(indices)
        assert np.all(np.isfinite(result["weights"]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_kdtree_no_norm_key_absent(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = mesh_grid.reshape(-1, 2).astype(np.float32)
        result = kdtree_neighbor_search(coords, coords, radius=0.3, return_norm=False)
        assert "weights" not in result


# ---------------------------------------------------------------------------
# Extra coverage: NeighborSearch Flax module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_neighbor_search_module(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        ns = NeighborSearch(use_open3d=False, return_norm=False)
        key = jax.random.PRNGKey(0)
        params = ns.init(key, data=coords, queries=coords, radius=0.3)
        result = cast(
            dict[str, jnp.ndarray],
            ns.apply(params, data=coords, queries=coords, radius=0.3),
        )
        assert result["neighbors_index"].tolist() == indices
        assert result["neighbors_row_splits"].tolist() == splits


@pytest.mark.parametrize("backend", BACKENDS)
def test_neighbor_search_module_return_norm(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        ns = NeighborSearch(use_open3d=False, return_norm=True)
        key = jax.random.PRNGKey(0)
        params = ns.init(key, data=coords, queries=coords, radius=0.3)
        result = cast(
            dict[str, jnp.ndarray],
            ns.apply(params, data=coords, queries=coords, radius=0.3),
        )
        assert "weights" in result
        assert result["weights"].shape == (len(indices),)


@pytest.mark.parametrize("backend", BACKENDS)
def test_neighbor_search_module_asymmetric(backend: str) -> None:
    """data and queries are different point clouds; validate split structure."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        data = jax.random.uniform(key, (50, 2))
        queries = jax.random.uniform(key, (30, 2))
        ns = NeighborSearch(use_open3d=False, return_norm=False)
        params = ns.init(key, data=data, queries=queries, radius=0.3)
        result = cast(
            dict[str, jnp.ndarray],
            ns.apply(params, data=data, queries=queries, radius=0.3),
        )
        row_splits = result["neighbors_row_splits"]
        assert len(row_splits) == 31  # n_queries + 1
        assert int(row_splits[0]) == 0
        assert int(row_splits[-1]) == len(result["neighbors_index"])


# ---------------------------------------------------------------------------
# native_neighbor_search (non-chunked, lines 115-199)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_neighbor_search_counts_match(backend: str) -> None:
    """native_neighbor_search neighbor counts match the reference splits."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search(data=coords, queries=coords, radius=0.3)
        row_splits = result["neighbors_row_splits"].tolist()
        assert row_splits == splits


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_neighbor_search_return_norm(backend: str) -> None:
    """native_neighbor_search with return_norm=True includes weights key."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search(
            data=coords, queries=coords, radius=0.3, return_norm=True
        )
        assert "weights" in result
        assert len(result["weights"]) == len(indices)


# ---------------------------------------------------------------------------
# native_neighbor_search_chunked_optimized (lines 435-492)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_chunked_optimized_counts_match(backend: str) -> None:
    """native_neighbor_search_chunked_optimized neighbor counts match reference."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search_chunked_optimized(
            data=coords, queries=coords, radius=0.3
        )
        row_splits = result["neighbors_row_splits"].tolist()
        assert row_splits == splits


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_chunked_optimized_return_norm(backend: str) -> None:
    """native_neighbor_search_chunked_optimized with return_norm=True includes
    weights."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search_chunked_optimized(
            data=coords, queries=coords, radius=0.3, return_norm=True
        )
        assert "weights" in result
        assert len(result["weights"]) == len(indices)


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_chunked_optimized_empty_neighbors(backend: str) -> None:
    """native_neighbor_search_chunked_optimized with tiny radius returns empty index."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = jnp.array(mesh_grid.reshape(-1, 2))
        result = native_neighbor_search_chunked_optimized(
            data=coords, queries=coords, radius=1e-9, return_norm=True
        )
        assert len(result["neighbors_index"]) == 0
        assert "weights" in result
        assert len(result["weights"]) == 0


# ---------------------------------------------------------------------------
# native_neighbor_search_numpy (lines 503-508)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_neighbor_search_numpy_counts_match(backend: str) -> None:
    """native_neighbor_search_numpy (wrapper) produces correct neighbor counts."""
    with jax.default_device(jax.devices(backend)[0]):
        mesh_grid = np.stack(
            np.meshgrid(*[np.linspace(0, 1, 5) for _ in range(2)], indexing="ij"),
            axis=-1,
        )
        coords = mesh_grid.reshape(-1, 2).astype(np.float32)
        result = native_neighbor_search_numpy(data=coords, queries=coords, radius=0.3)
        row_splits = result["neighbors_row_splits"].tolist()
        assert row_splits == splits


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_neighbor_search_numpy_accepts_numpy_arrays(backend: str) -> None:
    """native_neighbor_search_numpy converts numpy arrays to JAX internally."""
    with jax.default_device(jax.devices(backend)[0]):
        rng = np.random.default_rng(0)
        data = rng.random((20, 2)).astype(np.float32)
        queries = rng.random((10, 2)).astype(np.float32)
        result = native_neighbor_search_numpy(data=data, queries=queries, radius=0.4)
        assert "neighbors_index" in result
        assert "neighbors_row_splits" in result
        assert len(result["neighbors_row_splits"]) == 11  # n_queries + 1


# ---------------------------------------------------------------------------
# neighbors_to_segments / precompute_neighbors
# ---------------------------------------------------------------------------
# GNOBlock converts CRS to segments internally when it runs the search itself.
# These two expose the same conversion so a caller whose geometry never changes
# can run the search once and pass the result in on every step.


def test_neighbors_to_segments_matches_the_row_splits() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    raw = kdtree_neighbor_search(data=points, queries=points, radius=1.5)
    got = neighbors_to_segments(raw)

    np.testing.assert_array_equal(got["counts"], [2, 3, 2])
    np.testing.assert_array_equal(got["segment_ids"], [0, 0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(
        got["neighbors_index"], np.asarray(raw["neighbors_index"])
    )
    assert got["segment_ids"].shape[0] == got["neighbors_index"].shape[0]


def test_neighbors_to_segments_matches_what_gno_block_computes() -> None:
    """The conversion must agree with GNOBlock's own inline version, or a
    precomputed neighbour list would not be interchangeable with a live one."""
    rng = np.random.default_rng(0)
    points = rng.random((40, 2)).astype(np.float32)
    raw = kdtree_neighbor_search(data=points, queries=points, radius=0.3)
    got = neighbors_to_segments(raw)

    row_splits = np.asarray(raw["neighbors_row_splits"])
    counts = (row_splits[1:] - row_splits[:-1]).astype(np.int32)
    np.testing.assert_array_equal(got["counts"], counts)
    np.testing.assert_array_equal(
        got["segment_ids"], np.repeat(np.arange(counts.shape[0]), counts)
    )


def test_neighbors_to_segments_passes_weights_through() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    raw = kdtree_neighbor_search(
        data=points, queries=points, radius=1.5, return_norm=True
    )
    assert "weights" in neighbors_to_segments(raw)


def test_precompute_neighbors_refuses_to_strand_a_query() -> None:
    """A query with no neighbour contributes nothing to the kernel integral.
    That is nearly always too small a radius, so it is reported, not ignored."""
    data = np.array([[0.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="no neighbor within radius"):
        precompute_neighbors(data, queries, radius=0.5)


def test_precompute_neighbors_equals_search_then_convert() -> None:
    rng = np.random.default_rng(1)
    data = rng.random((30, 2)).astype(np.float32)
    queries = rng.random((11, 2)).astype(np.float32)
    got = precompute_neighbors(data, queries, radius=0.5)
    want = neighbors_to_segments(
        kdtree_neighbor_search(data=data, queries=queries, radius=0.5)
    )
    for key in ("neighbors_index", "segment_ids", "counts"):
        np.testing.assert_array_equal(got[key], want[key])
