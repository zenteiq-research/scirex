# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for ChannelMLP and LinearChannelMLP (JAX/Flax).

Covers output shape, channel dimension changes, spatial-dimension invariance,
dropout behaviour, and high-dimensional flattening, across CPU and GPU.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.channel_mlp import ChannelMLP, LinearChannelMLP

from ..conftest import available_backends
from ..conftest import init_apply as _init_apply

BACKENDS = available_backends()


# ---------------------------------------------------------------------------
# Test 1 - output shape matches (batch, out_channels, *spatial) for 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_ChannelMLP_output_shape(n_dim: int, backend: str) -> None:
    """ChannelMLP produces the expected (batch, out_channels, *spatial) shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        in_ch, out_ch = 4, 8
        size = [10] * n_dim
        x = jax.random.normal(key, (2, in_ch, *size), dtype=jnp.float32)
        mlp = ChannelMLP(in_channels=in_ch, out_channels=out_ch, n_dim=n_dim)
        out = _init_apply(mlp, x, key)
        assert out.shape == (2, out_ch, *size)


# ---------------------------------------------------------------------------
# Test 2 - same in/out channels (default out_channels=None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ChannelMLP_same_channels(backend: str) -> None:
    """ChannelMLP with out_channels=None preserves the channel dimension."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        in_ch = 6
        x = jax.random.normal(key, (2, in_ch, 16, 16), dtype=jnp.float32)
        mlp = ChannelMLP(in_channels=in_ch)
        out = _init_apply(mlp, x, key)
        assert out.shape == (2, in_ch, 16, 16)


# ---------------------------------------------------------------------------
# Test 3 - spatial size is arbitrary (no constraint from kernel)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("size", [7, 13, 32])
def test_ChannelMLP_spatial_invariance(size: int, backend: str) -> None:
    """ChannelMLP output spatial size matches input spatial size."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (1, 4, size, size), dtype=jnp.float32)
        mlp = ChannelMLP(in_channels=4, out_channels=4)
        out = _init_apply(mlp, x, key)
        assert out.shape[2:] == (size, size)


# ---------------------------------------------------------------------------
# Test 4 - high-dimensional input (4D spatial) flattens and unflattens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ChannelMLP_high_dim(backend: str) -> None:
    """ChannelMLP correctly handles 4D spatial tensors via internal flattening."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (2, 4, 8, 8, 8, 8), dtype=jnp.float32)
        mlp = ChannelMLP(in_channels=4, out_channels=8)
        out = _init_apply(mlp, x, key)
        assert out.shape == (2, 8, 8, 8, 8, 8)


# ---------------------------------------------------------------------------
# Test 5 - dropout=0 and dropout>0 both produce correct shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_ChannelMLP_dropout(dropout: float, backend: str) -> None:
    """ChannelMLP output shape is unchanged regardless of dropout rate."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (2, 4, 16), dtype=jnp.float32)
        mlp = ChannelMLP(in_channels=4, out_channels=4, dropout=dropout)
        out = _init_apply(mlp, x, key)
        assert out.shape == (2, 4, 16)


# ---------------------------------------------------------------------------
# Test 6 - LinearChannelMLP output shape for various layer configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layers", [[4, 8, 4], [4, 4], [4, 16, 8, 4]])
def test_LinearChannelMLP_output_shape(layers: list[int], backend: str) -> None:
    """LinearChannelMLP maps input (batch, in) → (batch, out) per layer spec."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (8, layers[0]), dtype=jnp.float32)
        mlp = LinearChannelMLP(layers=layers)
        out = _init_apply(mlp, x, key)
        assert out.shape == (8, layers[-1])


# ---------------------------------------------------------------------------
# Test 7 - LinearChannelMLP dropout does not affect shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_LinearChannelMLP_dropout(backend: str) -> None:
    """LinearChannelMLP output shape is unchanged with dropout > 0."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(6)
        x = jax.random.normal(key, (4, 8), dtype=jnp.float32)
        mlp = LinearChannelMLP(layers=[8, 16, 4], dropout=0.2)
        out = _init_apply(mlp, x, key)
        assert out.shape == (4, 4)
