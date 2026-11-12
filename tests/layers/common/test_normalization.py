# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for AdaIN, InstanceNorm, and BatchNorm (JAX/Flax).

No dedicated PyTorch reference test exists for these layers; coverage
mirrors the norm-variant checks exercised by the neuraloperator fno_block
test, applied directly to each normalization class across 1D/2D/3D spatial
inputs on CPU and GPU.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.normalization import AdaIN, BatchNorm, InstanceNorm
from tests import tolerances


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# InstanceNorm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_InstanceNorm_shape(n_dim: int, backend: str) -> None:
    """InstanceNorm preserves input shape across 1D/2D/3D spatial inputs."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 4, *(10,) * n_dim), dtype=jnp.float32)
        mod = InstanceNorm()
        variables = mod.init(key, x)
        out = cast(jnp.ndarray, mod.apply(variables, x))
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_InstanceNorm_zero_mean(backend: str) -> None:
    """InstanceNorm output has near-zero mean per sample/channel over spatial dims."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (2, 4, 16, 16), dtype=jnp.float32)
        mod = InstanceNorm()
        variables = mod.init(key, x)
        out = cast(jnp.ndarray, mod.apply(variables, x))
        mean = jnp.mean(out, axis=(2, 3))
        assert jnp.allclose(mean, jnp.zeros_like(mean), atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_BatchNorm_shape(n_dim: int, backend: str) -> None:
    """BatchNorm preserves input shape across 1D/2D/3D spatial inputs."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (2, 4, *(10,) * n_dim), dtype=jnp.float32)
        mod = BatchNorm(n_dim=n_dim, num_features=4)
        variables = mod.init(key, x)
        out, _ = mod.apply(variables, x, mutable=["batch_stats"])
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_BatchNorm_high_dim_shape(backend: str) -> None:
    """BatchNorm with n_dim>=4 flattens spatially and preserves full tensor shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (2, 4, 8, 8, 8, 8), dtype=jnp.float32)
        mod = BatchNorm(n_dim=4, num_features=4)
        variables = mod.init(key, x)
        out, _ = mod.apply(variables, x, mutable=["batch_stats"])
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# AdaIN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_AdaIN_shape(n_dim: int, backend: str) -> None:
    """AdaIN preserves input shape across 1D/2D/3D spatial inputs."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        embed_dim = 8
        in_channels = 4
        x = jax.random.normal(key, (2, in_channels, *(10,) * n_dim), dtype=jnp.float32)
        embedding = jax.random.normal(key, (embed_dim,), dtype=jnp.float32)
        mod = AdaIN(embed_dim=embed_dim, in_channels=in_channels)
        variables = mod.init(key, x, embedding)
        out = cast(jnp.ndarray, mod.apply(variables, x, embedding))
        assert out.shape == x.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_AdaIN_custom_mlp(backend: str) -> None:
    """AdaIN accepts an externally provided MLP and produces correct output shape."""
    with jax.default_device(jax.devices(backend)[0]):
        import flax.linen as nn

        key = jax.random.PRNGKey(5)
        embed_dim = 6
        in_channels = 3
        custom_mlp = nn.Sequential([nn.Dense(2 * in_channels)])
        x = jax.random.normal(key, (2, in_channels, 16, 16), dtype=jnp.float32)
        embedding = jax.random.normal(key, (embed_dim,), dtype=jnp.float32)
        mod = AdaIN(embed_dim=embed_dim, in_channels=in_channels, mlp=custom_mlp)
        variables = mod.init(key, x, embedding)
        out = cast(jnp.ndarray, mod.apply(variables, x, embedding))
        assert out.shape == x.shape
