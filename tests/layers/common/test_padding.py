# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for DomainPadding (JAX/Flax).

Mirrors the structure of the neuraloperator PyTorch reference test,
extended with CPU and GPU backend parametrization.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.padding import DomainPadding


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
def test_DomainPadding(backend: str) -> None:
    """DomainPadding pads to the expected size and round-trips back to original
    shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, (2, 3, 10, 10), dtype=jnp.float32)

        padder = DomainPadding(domain_padding=0.2)
        variables = padder.init(key, data)
        padded = cast(
            jnp.ndarray,
            cast(jnp.ndarray, padder.apply(variables, data, method=padder.pad)),
        )

        # 0.2 * 10 = 2 padding per side → 10 + 2*2 = 14
        target_shape = list(padded.shape)
        target_shape[-1] = target_shape[-2] = 14
        assert list(padded.shape) == target_shape

        unpadded = cast(
            jnp.ndarray,
            cast(jnp.ndarray, padder.apply(variables, padded, method=padder.unpad)),
        )
        assert unpadded.shape == data.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_DomainPadding_none_scaling_factor(backend: str) -> None:
    """DomainPadding with resolution_scaling_factor=None falls back to 1 in setup."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        data = jax.random.normal(key, (2, 3, 10, 10), dtype=jnp.float32)
        padder = DomainPadding(domain_padding=0.2, resolution_scaling_factor=None)
        variables = padder.init(key, data)
        padded = cast(
            jnp.ndarray,
            cast(jnp.ndarray, padder.apply(variables, data, method=padder.pad)),
        )
        assert list(padded.shape[-2:]) == [14, 14]


@pytest.mark.parametrize("backend", BACKENDS)
def test_DomainPadding_verbose(backend: str) -> None:
    """DomainPadding.pad with verbose=True prints without error."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        data = jax.random.normal(key, (2, 3, 10, 10), dtype=jnp.float32)
        padder = DomainPadding(domain_padding=0.2)
        variables = padder.init(key, data)
        padded = cast(
            jnp.ndarray,
            cast(
                jnp.ndarray,
                padder.apply(variables, data, method=padder.pad, verbose=True),
            ),
        )
        assert list(padded.shape[-2:]) == [14, 14]


@pytest.mark.parametrize("backend", BACKENDS)
def test_DomainPadding_zero_padding_dim(backend: str) -> None:
    """DomainPadding with zero padding on one dimension hits the p==0 branch."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(3)
        data = jax.random.normal(key, (2, 3, 10, 10), dtype=jnp.float32)
        # First dim padded, second dim has zero padding
        padder = DomainPadding(domain_padding=[0.2, 0.0])
        variables = padder.init(key, data)
        padded = cast(
            jnp.ndarray,
            cast(jnp.ndarray, padder.apply(variables, data, method=padder.pad)),
        )
        assert padded.shape[-2] == 14
        assert padded.shape[-1] == 10  # unchanged


@pytest.mark.parametrize("backend", BACKENDS)
def test_DomainPadding_unpad_zero_padding(backend: str) -> None:
    """DomainPadding.unpad with zero-padding fraction hits the slice(None) branch."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        data = jax.random.normal(key, (2, 3, 14, 10), dtype=jnp.float32)
        padder = DomainPadding(domain_padding=[0.2, 0.0])
        variables = padder.init(key, data)
        unpadded = cast(
            jnp.ndarray,
            cast(jnp.ndarray, padder.apply(variables, data, method=padder.unpad)),
        )
        assert unpadded.shape[-1] == 10  # zero-padded dim unchanged


@pytest.mark.parametrize("size,frac", [(16, 0.1), (11, 0.15), (7, 0.2), (13, 0.2)])
def test_DomainPadding_unpad_exact_inverse(size: int, frac: float) -> None:
    """unpad must exactly recover the original size regardless of rounding.

    Regression test: round(padded / (1 + 2*frac)) is only an approximate
    inverse of round(frac*orig) and previously returned an off-by-one size
    for some (size, frac) pairs, e.g. size=16, frac=0.1 -> 17 instead of 16.
    """
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (2, 3, size, size), dtype=jnp.float32)
    padder = DomainPadding(domain_padding=frac)
    variables = padder.init(key, data)
    padded = cast(jnp.ndarray, padder.apply(variables, data, method=padder.pad))
    unpadded = cast(jnp.ndarray, padder.apply(variables, padded, method=padder.unpad))
    assert unpadded.shape == data.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_DomainPadding_cache_hit(backend: str) -> None:
    """Calling pad twice with same resolution within one apply hits the cache."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(5)
        data = jax.random.normal(key, (2, 3, 10, 10), dtype=jnp.float32)
        padder = DomainPadding(domain_padding=0.2)
        variables = padder.init(key, data)

        def pad_twice(
            module: DomainPadding, x: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            p1 = module.pad(x)
            p2 = module.pad(x)  # cache hit - same resolution
            return p1, p2

        p1, p2 = cast(
            "tuple[jnp.ndarray, jnp.ndarray]",
            cast(jnp.ndarray, padder.apply(variables, data, method=pad_twice)),
        )
        assert p1.shape == p2.shape
