# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for GridEmbedding2D, GridEmbeddingND, and SinusoidalEmbedding (JAX/Flax).

Mirrors the structure of the neuraloperator PyTorch reference tests
(test_grid_embeddings.py and test_sin_embeddings.py), extended with
CPU and GPU backend parametrization.
"""

import math
import random
from typing import cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.embeddings import (
    GridEmbedding2D,
    GridEmbeddingND,
    RotaryEmbedding2D,
    SinusoidalEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
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

batch_size = 4
num_freqs = 3
in_channels = 3
n_in = 2
max_pos = 10000


# ---------------------------------------------------------------------------
# Grid embedding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_GridEmbedding2D(backend: str) -> None:
    """GridEmbedding2D encodes correct (x,y) coordinates at a random grid point."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        grid_boundaries = [[0, 1], [0, 1]]
        pos_embed = GridEmbedding2D(in_channels=1, grid_boundaries=grid_boundaries)

        input_res = (20, 20)
        x = jax.random.normal(key, (1, 1, *input_res), dtype=jnp.float32)

        variables = pos_embed.init(key, x)
        x_out = cast(jnp.ndarray, pos_embed.apply(variables, x))

        index = [random.randint(0, res - 1) for res in input_res]
        true_coords = x_out[0, 1:, index[0], index[1]].squeeze()
        expected_coords = jnp.array([i / j for i, j in zip(index, input_res)])
        assert jnp.allclose(true_coords, expected_coords, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_GridEmbeddingND(dim: int, backend: str) -> None:
    """GridEmbeddingND encodes correct coordinates at a random grid point for 1D-4D."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        grid_boundaries = [[0, 1]] * dim
        pos_embed = GridEmbeddingND(
            in_channels=1, dim=dim, grid_boundaries=tuple(map(tuple, grid_boundaries))
        )

        input_res = [20] * dim
        x = jax.random.normal(key, (1, 1, *input_res), dtype=jnp.float32)

        variables = pos_embed.init(key, x)
        x_out = cast(jnp.ndarray, pos_embed.apply(variables, x))

        index = [random.randint(0, res - 1) for res in input_res]
        pos_channels = x_out[0, 1:, ...]
        indices = (slice(None), *index)
        true_coords = pos_channels[indices]
        expected_coords = jnp.array([i / j for i, j in zip(index, input_res)])
        assert jnp.allclose(true_coords, expected_coords, atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# Sinusoidal / NeRF embedding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_NeRFEmbedding(backend: str) -> None:
    """SinusoidalEmbedding (nerf) matches analytically computed sin/cos values."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        nerf_embed = SinusoidalEmbedding(
            in_channels=in_channels, num_frequencies=num_freqs, embedding_type="nerf"
        )
        # unbatched: (n_in, in_channels)
        unbatched_inputs = jnp.arange(in_channels, dtype=jnp.float32) * jnp.array(
            [[1.0], [0.5]]
        )

        variables = nerf_embed.init(key, unbatched_inputs)
        embeds = cast(jnp.ndarray, nerf_embed.apply(variables, unbatched_inputs))

        true_outputs = jnp.zeros((n_in, in_channels * num_freqs * 2), dtype=jnp.float32)
        for channel in range(in_channels):
            for wavenumber in range(num_freqs):
                for i in range(2):
                    idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                    freqs = 2**wavenumber * math.pi * unbatched_inputs[:, channel]
                    val = jnp.sin(freqs) if i == 0 else jnp.cos(freqs)
                    true_outputs = true_outputs.at[:, idx].set(val)

        assert jnp.allclose(embeds, true_outputs, atol=tolerances.atol(1e-5))

        # batched: (batch, n_in, in_channels)
        batched_inputs = jnp.stack([unbatched_inputs] * batch_size)
        embeds = cast(jnp.ndarray, nerf_embed.apply(variables, batched_inputs))

        true_outputs_b = jnp.zeros(
            (batch_size, n_in, in_channels * num_freqs * 2), dtype=jnp.float32
        )
        for channel in range(in_channels):
            for wavenumber in range(num_freqs):
                for i in range(2):
                    idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                    freqs = 2**wavenumber * math.pi * batched_inputs[:, :, channel]
                    val = jnp.sin(freqs) if i == 0 else jnp.cos(freqs)
                    true_outputs_b = true_outputs_b.at[:, :, idx].set(val)

        assert jnp.allclose(embeds, true_outputs_b, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_TransformerEmbedding(backend: str) -> None:
    """SinusoidalEmbedding (transformer) matches analytically computed sin/cos
    values."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        sin_embed = SinusoidalEmbedding(
            in_channels=in_channels,
            num_frequencies=num_freqs,
            embedding_type="transformer",
            max_positions=max_pos,
        )
        unbatched_inputs = jnp.arange(in_channels, dtype=jnp.float32) * jnp.array(
            [[1.0], [0.5]]
        )

        variables = sin_embed.init(key, unbatched_inputs)
        embeds = cast(jnp.ndarray, sin_embed.apply(variables, unbatched_inputs))

        true_outputs = jnp.zeros((n_in, in_channels * num_freqs * 2), dtype=jnp.float32)
        for channel in range(in_channels):
            for wavenumber in range(num_freqs):
                for i in range(2):
                    idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                    freqs = (
                        (1 / max_pos) ** (wavenumber / num_freqs * 2)
                    ) * unbatched_inputs[:, channel]
                    val = jnp.sin(freqs) if i == 0 else jnp.cos(freqs)
                    true_outputs = true_outputs.at[:, idx].set(val)

        assert jnp.allclose(embeds, true_outputs, atol=tolerances.atol(1e-5))

        # batched
        batched_inputs = jnp.stack([unbatched_inputs] * batch_size)
        embeds = cast(jnp.ndarray, sin_embed.apply(variables, batched_inputs))

        true_outputs_b = jnp.zeros(
            (batch_size, n_in, in_channels * num_freqs * 2), dtype=jnp.float32
        )
        for channel in range(in_channels):
            for wavenumber in range(num_freqs):
                for i in range(2):
                    idx = channel * (num_freqs * 2) + wavenumber * 2 + i
                    freqs = (
                        (1 / max_pos) ** (wavenumber / num_freqs * 2)
                    ) * batched_inputs[:, :, channel]
                    val = jnp.sin(freqs) if i == 0 else jnp.cos(freqs)
                    true_outputs_b = true_outputs_b.at[:, :, idx].set(val)

        assert jnp.allclose(embeds, true_outputs_b, atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# out_channels property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_GridEmbedding2D_out_channels(backend: str) -> None:
    """GridEmbedding2D.out_channels returns in_channels + 2."""
    with jax.default_device(jax.devices(backend)[0]):
        embed = GridEmbedding2D(in_channels=5)
        assert embed.out_channels == 7


@pytest.mark.parametrize("backend", BACKENDS)
def test_GridEmbeddingND_out_channels(backend: str) -> None:
    """GridEmbeddingND.out_channels returns in_channels + dim."""
    with jax.default_device(jax.devices(backend)[0]):
        embed = GridEmbeddingND(in_channels=3, dim=4, grid_boundaries=((0, 1),) * 4)
        assert embed.out_channels == 7


# ---------------------------------------------------------------------------
# Unbatched input paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_GridEmbedding2D_unbatched(backend: str) -> None:
    """GridEmbedding2D with batched=False squeezes the batch dim on output."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(10)
        embed = GridEmbedding2D(in_channels=1)
        # (channels, h, w) - no batch dim
        x = jax.random.normal(key, (1, 8, 8), dtype=jnp.float32)
        variables = embed.init(key, x[jnp.newaxis])
        out = cast(jnp.ndarray, embed.apply(variables, x, batched=False))
        assert out.shape == (3, 8, 8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_GridEmbeddingND_unbatched(backend: str) -> None:
    """GridEmbeddingND with batched=False adds a temporary batch dim."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(11)
        embed = GridEmbeddingND(in_channels=1, dim=2)
        x = jax.random.normal(key, (1, 8, 8), dtype=jnp.float32)
        variables = embed.init(key, x[jnp.newaxis])
        out = cast(jnp.ndarray, embed.apply(variables, x, batched=False))
        # GridEmbeddingND batched=False adds a temporary batch dim;
        # the channel dim grows by `dim` (1→3), spatial stays (8,8)
        assert out.shape[-3] == 3  # in_channels(1) + dim(2)
        assert out.shape[-2:] == (8, 8)


# ---------------------------------------------------------------------------
# RotaryEmbedding2D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_RotaryEmbedding2D_call_shape(backend: str) -> None:
    """RotaryEmbedding2D.__call__ returns shape (batch, n_points, dim)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(20)
        dim = 8
        embed = RotaryEmbedding2D(dim=dim)
        coords = jax.random.uniform(key, (2, 16))
        variables = embed.init(key, coords)
        out = cast(jnp.ndarray, embed.apply(variables, coords))
        assert out.shape == (2, 16, dim)


@pytest.mark.parametrize("backend", BACKENDS)
def test_RotaryEmbedding2D_apply_1d(backend: str) -> None:
    """apply_1d_rotary_pos_emb delegates to apply_rotary_pos_emb."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(21)
        t = jax.random.normal(key, (2, 16, 8), dtype=jnp.float32)
        freqs = jax.random.normal(key, (2, 16, 8), dtype=jnp.float32)
        out1 = RotaryEmbedding2D.apply_1d_rotary_pos_emb(t, freqs)
        out2 = apply_rotary_pos_emb(t, freqs)
        assert jnp.allclose(out1, out2, atol=tolerances.atol(1e-6))


@pytest.mark.parametrize("backend", BACKENDS)
def test_RotaryEmbedding2D_apply_2d(backend: str) -> None:
    """apply_2d_rotary_pos_emb splits last dim and applies rotary emb to each half."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(22)
        t = jax.random.normal(key, (2, 16, 8), dtype=jnp.float32)
        freqs_x = jax.random.normal(key, (2, 16, 4), dtype=jnp.float32)
        freqs_y = jax.random.normal(key, (2, 16, 4), dtype=jnp.float32)
        out = RotaryEmbedding2D.apply_2d_rotary_pos_emb(t, freqs_x, freqs_y)
        assert out.shape == t.shape


# ---------------------------------------------------------------------------
# rotate_half and apply_rotary_pos_emb utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_rotate_half(backend: str) -> None:
    """rotate_half produces (-x2, x1) from a split of the last dim."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=jnp.float32)
        out = rotate_half(x)
        expected = jnp.array([[[-3.0, -4.0, 1.0, 2.0]]], dtype=jnp.float32)
        assert jnp.allclose(out, expected, atol=tolerances.atol(1e-6))


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_rotary_pos_emb_zero_freqs(backend: str) -> None:
    """apply_rotary_pos_emb with zero freqs is identity (cos=1, sin=0)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(30)
        t = jax.random.normal(key, (2, 8, 4), dtype=jnp.float32)
        freqs = jnp.zeros((2, 8, 4), dtype=jnp.float32)
        out = apply_rotary_pos_emb(t, freqs)
        assert jnp.allclose(out, t, atol=tolerances.atol(1e-6))
