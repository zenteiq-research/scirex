# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for resample (JAX).

Mirrors the structure of the neuraloperator PyTorch reference test,
extended with CPU and GPU backend parametrization.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.resample import iterative_resample, resample


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


@pytest.mark.parametrize("backend", BACKENDS)
def test_resample(backend: str) -> None:
    """resample produces correct output shapes for 2D (bicubic) and 3D+ (spectral)
    paths."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        # 2D spatial - bicubic interpolation path
        a = jax.random.normal(key, (10, 20, 40, 50), dtype=jnp.float32)
        res_scale: list[float] = [2, 3]
        axis = [-2, -1]
        b = resample(a, res_scale, axis)
        assert b.shape[-1] == 3 * a.shape[-1] and b.shape[-2] == 2 * a.shape[-2]

        # 3D spatial - spectral (FFT) interpolation path
        a = jax.random.normal(key, (10, 20, 40, 50, 60), dtype=jnp.float32)
        res_scale = [0.5, 3, 4]
        axis = [-3, -2, -1]
        b = resample(a, res_scale, axis)
        assert (
            b.shape[-1] == 4 * a.shape[-1]
            and b.shape[-2] == 3 * a.shape[-2]
            and b.shape[-3] == int(0.5 * a.shape[-3])
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_resample_scalar_axis_none(backend: str) -> None:
    """Scalar res_scale with axis=None defaults to all spatial dims (lines 42-43)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        # shape (batch, channels, h, w) - axis=None should resample both h and w
        a = jax.random.normal(key, (2, 4, 8, 10), dtype=jnp.float32)
        b = resample(a, 2.0, axis=None)
        assert b.shape == (2, 4, 16, 20)


@pytest.mark.parametrize("backend", BACKENDS)
def test_resample_scalar_res_scale_int_axis(backend: str) -> None:
    """Scalar res_scale with int axis wraps into a list (lines 45-46).

    The single-axis rfft path in the source assumes a 3-D tensor; use 3-D input.
    """
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        a = jax.random.normal(key, (4, 8, 10), dtype=jnp.float32)
        b = resample(a, 2.0, axis=-1)
        assert b.shape[-1] == 20


@pytest.mark.parametrize("backend", BACKENDS)
def test_iterative_resample_list_axis_scalar_scale(backend: str) -> None:
    """iterative_resample with list axis and scalar scale exercises the multi-axis
    branch."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        a = jax.random.normal(key, (4, 8, 10), dtype=jnp.float32)
        b = iterative_resample(a, 2.0, [-2, -1])
        assert b.shape == (4, 8, 40)


@pytest.mark.parametrize("backend", BACKENDS)
def test_iterative_resample_list_axis_list_scale(backend: str) -> None:
    """iterative_resample with matching list axis and list scale exercises the same
    multi-axis branch."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        a = jax.random.normal(key, (4, 8, 10), dtype=jnp.float32)
        b = iterative_resample(a, [2.0, 3.0], [-2, -1])
        assert b.shape == (4, 8, 60)


@pytest.mark.parametrize("backend", BACKENDS)
def test_iterative_resample_scalar_axis(backend: str) -> None:
    """iterative_resample with scalar axis uses the rfft single-axis path
    (lines 135-148)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(5)
        a = jax.random.normal(key, (2, 4, 8, 10), dtype=jnp.float32)
        b = iterative_resample(a, 2.0, -1)
        assert b.shape[-1] == 20


@pytest.mark.parametrize("backend", BACKENDS)
def test_iterative_resample_raises_mismatched_list(backend: str) -> None:
    """iterative_resample raises when axis is not a list but scale is a list."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(6)
        a = jax.random.normal(key, (2, 4, 8, 10), dtype=jnp.float32)
        with pytest.raises(Exception):
            iterative_resample(a, [2.0, 3.0], -1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_iterative_resample_raises_length_mismatch(backend: str) -> None:
    """iterative_resample raises when axis list and scale list have different
    lengths."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(7)
        a = jax.random.normal(key, (2, 4, 8, 10), dtype=jnp.float32)
        with pytest.raises(Exception):
            iterative_resample(a, [2.0, 3.0, 4.0], [-2, -1])
