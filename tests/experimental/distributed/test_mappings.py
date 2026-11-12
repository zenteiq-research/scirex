# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the model-parallel region mappings (JAX).

neuraloperator's ``mpu`` package has no test suite, so coverage follows the
Phase 1 ``layers/tests`` structure. The four mapping functions are custom_vjp
wrappers around the ``_reduce`` / ``_split`` / ``_gather`` primitives; with a
single process those primitives short-circuit to the identity, so the mappings
are identity in both the forward and backward pass. The mappings call
``get_model_parallel_group()`` (which asserts the comm layer is initialized), so
the tests patch it to return ``None`` - that selects the single-process
``process_count() == 1`` fast path without standing up a distributed runtime.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import scirex.experimental.distributed.mappings as mappings


@pytest.fixture(autouse=True)
def _single_process_group(monkeypatch):
    """Make get_model_parallel_group() select the single-process fast path."""
    monkeypatch.setattr(mappings, "get_model_parallel_group", lambda: None)


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# forward pass - identity in single-process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_copy_and_reduce_forward_identity(backend) -> None:
    """copy_to / reduce_from regions are identity forward with one process."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(8.0).reshape(2, 4)
        assert np.allclose(mappings.copy_to_model_parallel_region(x), x)
        assert np.allclose(mappings.reduce_from_model_parallel_region(x), x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scatter_and_gather_forward_identity(backend) -> None:
    """scatter_to / gather_from regions are identity forward with one process."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(8.0).reshape(2, 4)
        assert np.allclose(mappings.scatter_to_model_parallel_region(x, 0), x)
        assert np.allclose(mappings.gather_from_model_parallel_region(x, 1), x)


# ---------------------------------------------------------------------------
# backward pass - gradients flow through (identity in single-process)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: jnp.sum(mappings.copy_to_model_parallel_region(x)),
        lambda x: jnp.sum(mappings.reduce_from_model_parallel_region(x)),
        lambda x: jnp.sum(mappings.scatter_to_model_parallel_region(x, 0)),
        lambda x: jnp.sum(mappings.gather_from_model_parallel_region(x, 0)),
    ],
)
def test_mapping_gradients_are_identity(fn) -> None:
    """Each region's custom_vjp yields an all-ones gradient of sum(region(x))."""
    x = jnp.arange(8.0).reshape(2, 4)
    grad = jax.grad(fn)(x)
    assert grad.shape == x.shape
    assert np.allclose(grad, jnp.ones_like(x))
