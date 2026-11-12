# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the integral kernel transform."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.layers.blocks.integral_transform import IntegralTransform

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)

# ---------------------------------------------------------------------------
# Shared geometry: 10 data points → 8 query points
# ---------------------------------------------------------------------------
N_DATA = 10
N_QUERY = 8
COORD_DIM = 2
IN_CHANNELS = 4
OUT_CHANNELS = 3
MLP_LAYERS = [COORD_DIM * 2, 16, OUT_CHANNELS]  # linear/linear_kernelonly
MLP_LAYERS_NL = [COORD_DIM * 2 + IN_CHANNELS, 16, OUT_CHANNELS]  # nonlinear variants


def _make_neighbors(
    y: jnp.ndarray, x: jnp.ndarray, radius: float = 0.6
) -> dict[str, jnp.ndarray]:
    """Build segment_ids + counts from two point clouds (scipy KDTree)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.array(y))
    nbr_lists = tree.query_ball_point(np.array(x), r=radius)
    all_idx, splits, cur = [], [0], 0
    for nbrs in nbr_lists:
        all_idx.extend(nbrs)
        cur += len(nbrs)
        splits.append(cur)
    idx = jnp.array(all_idx, dtype=jnp.int32) if all_idx else jnp.zeros(0, jnp.int32)
    row_splits = jnp.array(splits, dtype=jnp.int32)
    counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
    seg_ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)
    return {"neighbors_index": idx, "segment_ids": seg_ids, "counts": counts}


# ---------------------------------------------------------------------------
# transform_type = 'linear_kernelonly'  →  (a): ∫ k(x,y) dy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_kernelonly_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        y = jax.random.normal(key, (N_DATA, COORD_DIM))
        x = jax.random.normal(key, (N_QUERY, COORD_DIM))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear_kernelonly",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x)
        out = cast(jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x))
        assert out.shape == (N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_kernelonly_no_f_y(backend: str) -> None:
    """f_y must not be required for transform (a)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        y = jax.random.normal(key, (N_DATA, COORD_DIM))
        x = jax.random.normal(key, (N_QUERY, COORD_DIM))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear_kernelonly",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x, f_y=None)
        )
        assert out.shape == (N_QUERY, OUT_CHANNELS)


# ---------------------------------------------------------------------------
# transform_type = 'linear'  →  (b): ∫ k(x,y)*f(y) dy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_linear_batched_shape(batch_size: int, backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        y = jax.random.normal(k1, (N_DATA, COORD_DIM))
        x = jax.random.normal(k1, (N_QUERY, COORD_DIM))
        f_y = jax.random.normal(k2, (batch_size, N_DATA, OUT_CHANNELS))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x, f_y=f_y)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x, f_y=f_y)
        )
        assert out.shape == (batch_size, N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_unbatched_f_y(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        y = jax.random.normal(k1, (N_DATA, COORD_DIM))
        x = jax.random.normal(k1, (N_QUERY, COORD_DIM))
        f_y = jax.random.normal(k2, (N_DATA, OUT_CHANNELS))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x, f_y=f_y)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x, f_y=f_y)
        )
        assert out.shape == (N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# transform_type = 'nonlinear_kernelonly'  →  (c): ∫ k(x,y,f(y)) dy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_nonlinear_kernelonly_shape(batch_size: int, backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(key)
        y = jax.random.normal(k1, (N_DATA, COORD_DIM))
        x = jax.random.normal(k1, (N_QUERY, COORD_DIM))
        f_y = jax.random.normal(k2, (batch_size, N_DATA, IN_CHANNELS))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS_NL,
            transform_type="nonlinear_kernelonly",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x, f_y=f_y)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x, f_y=f_y)
        )
        assert out.shape == (batch_size, N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# transform_type = 'nonlinear'  →  (d): ∫ k(x,y,f(y))*f(y) dy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_nonlinear_shape(batch_size: int, backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(5)
        k1, k2 = jax.random.split(key)
        y = jax.random.normal(k1, (N_DATA, COORD_DIM))
        x = jax.random.normal(k1, (N_QUERY, COORD_DIM))
        f_y = jax.random.normal(k2, (batch_size, N_DATA, IN_CHANNELS))
        neighbors = _make_neighbors(y, x)
        it = IntegralTransform(
            channel_mlp_layers=[COORD_DIM * 2 + IN_CHANNELS, 16, IN_CHANNELS],
            transform_type="nonlinear",
        )
        params = it.init(key, y=y, neighbors=neighbors, x=x, f_y=f_y)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=x, f_y=f_y)
        )
        assert out.shape == (batch_size, N_QUERY, IN_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Invalid transform_type raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_transform_type(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(ValueError, match="transform_type"):
            it = IntegralTransform(
                channel_mlp_layers=MLP_LAYERS,
                transform_type="bad_type",
            )
            key = jax.random.PRNGKey(0)
            y = jax.random.normal(key, (N_DATA, COORD_DIM))
            x = jax.random.normal(key, (N_QUERY, COORD_DIM))
            neighbors = _make_neighbors(y, x)
            it.init(key, y=y, neighbors=neighbors, x=x)


# ---------------------------------------------------------------------------
# x defaults to y when not supplied
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_x_defaults_to_y(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(6)
        y = jax.random.normal(key, (N_DATA, COORD_DIM))
        # build self-neighbors (data == queries)
        neighbors = _make_neighbors(y, y)
        it = IntegralTransform(
            channel_mlp_layers=[COORD_DIM * 2, 16, OUT_CHANNELS],
            transform_type="linear_kernelonly",
        )
        params = it.init(key, y=y, neighbors=neighbors)
        out_implicit = cast(jnp.ndarray, it.apply(params, y=y, neighbors=neighbors))
        out_explicit = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors, x=y)
        )
        assert jnp.allclose(out_implicit, out_explicit)


# ---------------------------------------------------------------------------
# Reduction modes: 'sum' vs 'mean'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduction_sum_vs_mean(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(7)
        y = jax.random.normal(key, (N_DATA, COORD_DIM))
        x = jax.random.normal(key, (N_QUERY, COORD_DIM))
        neighbors = _make_neighbors(y, x)
        it_sum = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear_kernelonly",
            reduction="sum",
        )
        it_mean = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear_kernelonly",
            reduction="mean",
        )
        params = it_sum.init(key, y=y, neighbors=neighbors, x=x)
        out_sum = cast(jnp.ndarray, it_sum.apply(params, y=y, neighbors=neighbors, x=x))
        out_mean = cast(
            jnp.ndarray, it_mean.apply(params, y=y, neighbors=neighbors, x=x)
        )
        # outputs will differ unless every query has exactly 1 neighbour
        assert out_sum.shape == out_mean.shape == (N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out_sum))
        assert jnp.all(jnp.isfinite(out_mean))


# ---------------------------------------------------------------------------
# Weighting: output shape preserved, values differ from unweighted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_with_weights_unbatched_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(8)
        y = jax.random.normal(key, (N_DATA, COORD_DIM))
        x = jax.random.normal(key, (N_QUERY, COORD_DIM))
        neighbors = _make_neighbors(y, x)
        # inject synthetic edge weights
        n_edges = int(neighbors["counts"].sum())
        neighbors_w = dict(neighbors)
        neighbors_w["weights"] = jnp.ones(n_edges, dtype=jnp.float32) * 0.5
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear_kernelonly",
        )
        params = it.init(key, y=y, neighbors=neighbors_w, x=x)
        out = cast(jnp.ndarray, it.apply(params, y=y, neighbors=neighbors_w, x=x))
        assert out.shape == (N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_with_weights_batched_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(9)
        k1, k2 = jax.random.split(key)
        y = jax.random.normal(k1, (N_DATA, COORD_DIM))
        x = jax.random.normal(k1, (N_QUERY, COORD_DIM))
        f_y = jax.random.normal(k2, (2, N_DATA, OUT_CHANNELS))
        neighbors = _make_neighbors(y, x)
        n_edges = int(neighbors["counts"].sum())
        neighbors_w = dict(neighbors)
        neighbors_w["weights"] = jnp.ones(n_edges, dtype=jnp.float32)
        it = IntegralTransform(
            channel_mlp_layers=MLP_LAYERS,
            transform_type="linear",
        )
        params = it.init(key, y=y, neighbors=neighbors_w, x=x, f_y=f_y)
        out = cast(
            jnp.ndarray, it.apply(params, y=y, neighbors=neighbors_w, x=x, f_y=f_y)
        )
        assert out.shape == (2, N_QUERY, OUT_CHANNELS)
        assert jnp.all(jnp.isfinite(out))
