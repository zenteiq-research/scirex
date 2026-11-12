# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the complex-half einsum utilities."""

import jax
import jax.numpy as jnp
import pytest

from scirex._internal.einsum_utils import (
    _view_as_complex,
    _view_as_real,
    einsum_complexhalf,
    einsum_complexhalf_two_input,
)
from tests import tolerances

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)


@pytest.mark.parametrize("backend", BACKENDS)
def test_view_as_real_shape_and_dtype(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=jnp.complex64)
        out = _view_as_real(x)
        assert out.shape == (2, 2)
        assert out.dtype == jnp.float32


@pytest.mark.parametrize("backend", BACKENDS)
def test_view_as_real_values(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=jnp.complex64)
        out = _view_as_real(x)
        assert jnp.allclose(out[0, :, 0], jnp.array([1.0, 3.0]))
        assert jnp.allclose(out[0, :, 1], jnp.array([2.0, 4.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_view_as_complex_roundtrip(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=jnp.complex64)
        reconstructed = _view_as_complex(_view_as_real(x))
        assert jnp.allclose(reconstructed.real, x.real, atol=tolerances.atol(1e-6))
        assert jnp.allclose(reconstructed.imag, x.imag, atol=tolerances.atol(1e-6))


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_two_input_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        a = (
            jax.random.normal(key, (4, 6)) + 1j * jax.random.normal(key, (4, 6))
        ).astype(jnp.complex64)
        b = (
            jax.random.normal(key, (6, 8)) + 1j * jax.random.normal(key, (6, 8))
        ).astype(jnp.complex64)
        out = einsum_complexhalf_two_input("ij,jk->ik", a, b)
        assert out.shape == (4, 8)
        assert jnp.iscomplexobj(out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_two_input_correctness(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        # Use small integer values so float16 precision is sufficient
        a = jnp.array([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=jnp.complex64)  # (1, 2)
        b = jnp.array([[2.0 + 0.0j], [0.0 + 2.0j]], dtype=jnp.complex64)  # (2, 1)
        # Expected: (1+0j)(2+0j) + (0+1j)(0+2j) = 2 + (-2) = 0, imag = 0 + 2 = 2
        # (1)(2) + (j)(2j) = 2 + 2j^2 = 2 - 2 = 0, imag part: 0 + 2 = 2
        out = einsum_complexhalf_two_input("ij,jk->ik", a, b)
        reference = jnp.einsum("ij,jk->ik", a, b)
        # float16 half precision: allow generous tolerance
        assert jnp.allclose(out.real, reference.real, atol=0.1)
        assert jnp.allclose(out.imag, reference.imag, atol=0.1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_two_input_assertion(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        a = jnp.ones((2, 3), dtype=jnp.complex64)
        b = jnp.ones((3, 4), dtype=jnp.complex64)
        jnp.ones((4, 5), dtype=jnp.complex64)
        with pytest.raises(AssertionError):
            einsum_complexhalf_two_input("ij,jk,kl->il", a, b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_dispatches_two_input(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(1)
        a = (
            jax.random.normal(key, (3, 4)) + 1j * jax.random.normal(key, (3, 4))
        ).astype(jnp.complex64)
        b = (
            jax.random.normal(key, (4, 5)) + 1j * jax.random.normal(key, (4, 5))
        ).astype(jnp.complex64)
        out_dispatch = einsum_complexhalf("ij,jk->ik", a, b)
        out_direct = einsum_complexhalf_two_input("ij,jk->ik", a, b)
        assert jnp.allclose(out_dispatch.real, out_direct.real, atol=1e-4)
        assert jnp.allclose(out_dispatch.imag, out_direct.imag, atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_output_is_complex(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(2)
        a = (
            jax.random.normal(key, (2, 3)) + 1j * jax.random.normal(key, (2, 3))
        ).astype(jnp.complex64)
        b = (
            jax.random.normal(key, (3, 4)) + 1j * jax.random.normal(key, (3, 4))
        ).astype(jnp.complex64)
        out = einsum_complexhalf("ij,jk->ik", a, b)
        assert jnp.iscomplexobj(out)
        assert out.shape == (2, 4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_three_input_shape(backend: str) -> None:
    """einsum_complexhalf with three inputs exercises the multi-input contraction
    path."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(4)
        a = (
            jax.random.normal(key, (2, 3)) + 1j * jax.random.normal(key, (2, 3))
        ).astype(jnp.complex64)
        b = (
            jax.random.normal(key, (3, 4)) + 1j * jax.random.normal(key, (3, 4))
        ).astype(jnp.complex64)
        c = (
            jax.random.normal(key, (4, 5)) + 1j * jax.random.normal(key, (4, 5))
        ).astype(jnp.complex64)
        out = einsum_complexhalf("ij,jk,kl->il", a, b, c)
        assert out.shape == (2, 5)
        assert jnp.iscomplexobj(out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_einsum_complexhalf_three_input_correctness(backend: str) -> None:
    """Three-input complexhalf result is close to the reference full-precision
    einsum."""
    with jax.default_device(jax.devices(backend)[0]):
        # Small integer-valued tensors so float16 rounding stays negligible.
        a = jnp.array([[1.0 + 0.0j, 0.0 + 1.0j]], dtype=jnp.complex64)  # (1, 2)
        b = jnp.array(
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=jnp.complex64
        )  # (2, 2)
        c = jnp.array([[2.0 + 0.0j], [0.0 + 1.0j]], dtype=jnp.complex64)  # (2, 1)
        reference = jnp.einsum("ij,jk,kl->il", a, b, c)
        out = einsum_complexhalf("ij,jk,kl->il", a, b, c)
        assert jnp.allclose(out.real, reference.real, atol=0.1)
        assert jnp.allclose(out.imag, reference.imag, atol=0.1)
