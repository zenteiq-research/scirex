# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for skip connection modules (JAX/Flax):
  Identity, SoftGating, Flattened1dConv, and the skip_connection factory.

Covers output shapes, channel transformations, and the factory dispatch
for 'identity', 'linear', and 'soft-gating' types across CPU and GPU.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.skip_connections import (
    Flattened1dConv,
    Identity,
    SoftGating,
    skip_connection,
)

from ..conftest import available_backends
from ..conftest import init_apply as _init_apply

BACKENDS = available_backends()


# ---------------------------------------------------------------------------
# Test 1 - Identity returns input unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_Identity_shape(n_dim: int, backend: str) -> None:
    """Identity module returns tensor with unchanged shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        shape = (2, 4, *(8,) * n_dim)
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        mod = Identity()
        out = _init_apply(mod, x, key)
        assert out.shape == x.shape
        assert jnp.allclose(out, x)


# ---------------------------------------------------------------------------
# Test 2 - SoftGating preserves shape and applies channel weighting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_SoftGating_shape(n_dim: int, backend: str) -> None:
    """SoftGating output shape matches input shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        in_ch = 6
        shape = (2, in_ch, *(10,) * n_dim)
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        mod = SoftGating(in_features=in_ch, n_dim=n_dim)
        out = _init_apply(mod, x, key)
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_SoftGating_with_bias(backend: str) -> None:
    """SoftGating with bias=True produces correct output shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (2, 4, 8, 8), dtype=jnp.float32)
        mod = SoftGating(in_features=4, n_dim=2, bias=True)
        out = _init_apply(mod, x, key)
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_SoftGating_mismatched_features_raises(backend: str) -> None:
    """SoftGating raises when in_features != out_features."""
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(ValueError):
            mod = SoftGating(in_features=4, out_features=8, n_dim=2)
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (2, 4, 8, 8), dtype=jnp.float32)
            _init_apply(mod, x, key)


# ---------------------------------------------------------------------------
# Test 3 - Flattened1dConv changes channel dimension across 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_Flattened1dConv_shape(n_dim: int, backend: str) -> None:
    """Flattened1dConv maps (batch, in_ch, *spatial) → (batch, out_ch, *spatial)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        in_ch, out_ch = 4, 8
        size = (10,) * n_dim
        x = jax.random.normal(key, (2, in_ch, *size), dtype=jnp.float32)
        mod = Flattened1dConv(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        out = _init_apply(mod, x, key)
        assert out.shape == (2, out_ch, *size)


# ---------------------------------------------------------------------------
# Test 4 - skip_connection factory dispatches to correct module types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("skip_type", ["identity", "linear", "soft-gating"])
def test_skip_connection_factory_shape(skip_type: str, backend: str) -> None:
    """skip_connection factory returns a module with the correct output shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        in_ch = 4
        x = jax.random.normal(key, (2, in_ch, 16, 16), dtype=jnp.float32)
        mod = skip_connection(
            in_features=in_ch, out_features=in_ch, n_dim=2, skip_type=skip_type
        )
        out = _init_apply(mod, x, key)
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_skip_connection_linear_channel_change(backend: str) -> None:
    """skip_connection('linear') correctly maps in_ch → out_ch."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(5)
        x = jax.random.normal(key, (2, 4, 10, 10), dtype=jnp.float32)
        mod = skip_connection(
            in_features=4, out_features=8, n_dim=2, skip_type="linear"
        )
        out = _init_apply(mod, x, key)
        assert out.shape == (2, 8, 10, 10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_skip_connection_invalid_type_raises(backend: str) -> None:
    """skip_connection raises ValueError for an unknown skip_type."""
    with pytest.raises(ValueError):
        skip_connection(in_features=4, out_features=4, skip_type="unknown")
