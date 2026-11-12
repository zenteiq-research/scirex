# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for BaseSpectralConv (JAX/Flax).

BaseSpectralConv is an abstract base class providing a default identity
transform used by skip connections.  Tests verify that the transform method
returns an unmodified copy of its input across common spatial shapes and
that concrete subclasses inherit the identity correctly.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.base_spectral_conv import BaseSpectralConv


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
# Minimal concrete subclass - BaseSpectralConv cannot be instantiated directly
# because Flax requires at least a __call__ method to be trainable.
# ---------------------------------------------------------------------------


class _ConcreteSpectralConv(BaseSpectralConv):
    """Trivial subclass that passes input through unchanged."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


# ---------------------------------------------------------------------------
# Test 1 - transform is an identity map across 1D/2D/3D spatial inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_BaseSpectralConv_transform_identity(dim: int, backend: str) -> None:
    """BaseSpectralConv.transform returns the input unchanged.

    Verifies value equality (allclose) and shape preservation for 1D, 2D,
    and 3D spatial inputs on both CPU and GPU.
    """
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        shape = (2, 4, *(8,) * dim)
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        module = _ConcreteSpectralConv()
        variables = module.init(key, x)
        module.apply(variables, x)

        # __call__ is identity, so transform output must equal input
        transformed = cast(
            jnp.ndarray, module.apply(variables, x, method=module.transform)
        )

        assert transformed.shape == x.shape
        assert jnp.allclose(transformed, x)


# ---------------------------------------------------------------------------
# Test 2 - transform works on complex inputs without modification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_BaseSpectralConv_transform_complex(backend: str) -> None:
    """BaseSpectralConv.transform preserves complex-valued inputs exactly.

    Ensures the identity transform does not drop imaginary parts or change dtype.
    """
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(7)
        xr = jax.random.normal(key, (2, 3, 8, 8), dtype=jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(8), (2, 3, 8, 8), dtype=jnp.float32)
        x = (xr + 1j * xi).astype(jnp.complex64)

        module = _ConcreteSpectralConv()
        variables = module.init(key, x)
        transformed = cast(
            jnp.ndarray, module.apply(variables, x, method=module.transform)
        )

        assert transformed.shape == x.shape
        assert jnp.issubdtype(transformed.dtype, jnp.complexfloating)
        assert jnp.allclose(transformed, x)
