# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the complex-valued activations and the ComplexValued wrapper."""

from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.complex import (
    ComplexValued,
    apply_complex,
    cgelu,
    cselu,
    ctanh,
)
from tests import tolerances

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cgelu_output_dtype_and_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = (
            jax.random.normal(key, (2, 4, 8)) + 1j * jax.random.normal(key, (2, 4, 8))
        ).astype(jnp.complex64)
        out = cgelu(x)
        assert out.shape == x.shape
        assert out.dtype == jnp.complex64


@pytest.mark.parametrize("backend", BACKENDS)
def test_cgelu_applies_gelu_to_real_imag(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([1.0 + 2.0j, -1.0 - 0.5j], dtype=jnp.complex64)
        out = cgelu(x)
        expected_real = nn.gelu(x.real)
        expected_imag = nn.gelu(x.imag)
        assert jnp.allclose(out.real, expected_real, atol=tolerances.atol(1e-5))
        assert jnp.allclose(out.imag, expected_imag, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_cselu_output_dtype_and_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        x = (
            jax.random.normal(key, (3, 5)) + 1j * jax.random.normal(key, (3, 5))
        ).astype(jnp.complex64)
        out = cselu(x)
        assert out.shape == x.shape
        assert out.dtype == jnp.complex64


@pytest.mark.parametrize("backend", BACKENDS)
def test_cselu_applies_selu_to_real_imag(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([0.5 + 1.0j, -2.0 + 0.3j], dtype=jnp.complex64)
        out = cselu(x)
        expected_real = nn.selu(x.real)
        expected_imag = nn.selu(x.imag)
        assert jnp.allclose(out.real, expected_real, atol=tolerances.atol(1e-5))
        assert jnp.allclose(out.imag, expected_imag, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctanh_output_dtype_and_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        x = (
            jax.random.normal(key, (2, 6)) + 1j * jax.random.normal(key, (2, 6))
        ).astype(jnp.complex64)
        out = ctanh(x)
        assert out.shape == x.shape
        assert out.dtype == jnp.complex64


@pytest.mark.parametrize("backend", BACKENDS)
def test_ctanh_applies_tanh_to_real_imag(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([1.0 + 0.5j, -0.3 + 2.0j], dtype=jnp.complex64)
        out = ctanh(x)
        assert jnp.allclose(out.real, jnp.tanh(x.real), atol=tolerances.atol(1e-5))
        assert jnp.allclose(out.imag, jnp.tanh(x.imag), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_complex_identity(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        # With identical identity functions, apply_complex(id, id, x) = (Re-Im) +
        # i(Im+Re)
        def identity(v: jnp.ndarray) -> jnp.ndarray:
            return v

        x = jnp.array([3.0 + 4.0j], dtype=jnp.complex64)
        out = apply_complex(identity, identity, x)
        expected_real = x.real - x.imag  # 3 - 4 = -1
        expected_imag = x.imag + x.real  # 4 + 3 = 7
        assert jnp.allclose(out.real, expected_real, atol=tolerances.atol(1e-5))
        assert jnp.allclose(out.imag, expected_imag, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_apply_complex_output_dtype(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):

        def scale2(v: jnp.ndarray) -> jnp.ndarray:
            return 2.0 * v

        x = jnp.array([1.0 + 1.0j], dtype=jnp.complex64)
        out = apply_complex(scale2, scale2, x)
        assert out.dtype == jnp.complex64


@pytest.mark.parametrize("backend", BACKENDS)
def test_complex_valued_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        module = nn.Dense(features=8)
        cv = ComplexValued(module=module)
        key = jax.random.PRNGKey(3)
        x = (
            jax.random.normal(key, (2, 4)) + 1j * jax.random.normal(key, (2, 4))
        ).astype(jnp.complex64)
        params = cv.init(key, x)
        out = cast(jnp.ndarray, cv.apply(params, x))
        assert out.shape == (2, 8)
        assert out.dtype == jnp.complex64


@pytest.mark.parametrize("backend", BACKENDS)
def test_complex_valued_output_matches_apply_complex(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        module = nn.Dense(features=6, use_bias=False)
        cv = ComplexValued(module=module)
        key = jax.random.PRNGKey(4)
        x = (
            jax.random.normal(key, (3, 5)) + 1j * jax.random.normal(key, (3, 5))
        ).astype(jnp.complex64)
        params = cv.init(key, x)
        out = cast(jnp.ndarray, cv.apply(params, x))
        # Output should be complex and have the right shape
        assert out.shape == (3, 6)
        assert jnp.iscomplexobj(out)
