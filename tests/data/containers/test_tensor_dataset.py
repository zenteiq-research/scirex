# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for TensorDataset and GeneralTensorDataset (JAX).

No dedicated PyTorch reference test exists for these datasets in neuraloperator,
so coverage follows the structure of the Phase 1 layer tests (CPU/GPU backend
parametrization) and exercises the dataset contract: length equals the batch
dimension, ``__getitem__`` returns the expected fields, optional per-field
transforms are applied, and size-mismatch inputs are rejected.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest

from scirex.data.containers.tensor_dataset import (
    GeneralTensorDataset,
    TensorDataset,
)


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
# TensorDataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_dataset_len(backend: str) -> None:
    """__len__ equals the leading (batch) dimension of x."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.zeros((6, 3, 4))
        y = jnp.zeros((6, 2))
        ds = TensorDataset(x, y)
        assert len(ds) == 6


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_dataset_getitem_shapes_and_keys(backend: str) -> None:
    """__getitem__ returns {'x', 'y'} sliced along the batch dim."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(6 * 3 * 4, dtype=jnp.float32).reshape(6, 3, 4)
        y = jnp.arange(6 * 2, dtype=jnp.float32).reshape(6, 2)
        ds = TensorDataset(x, y)
        item = ds[2]
        assert set(item.keys()) == {"x", "y"}
        assert item["x"].shape == (3, 4)
        assert item["y"].shape == (2,)
        assert jnp.allclose(item["x"], x[2])
        assert jnp.allclose(item["y"], y[2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_dataset_applies_transforms(backend: str) -> None:
    """transform_x / transform_y are applied to the corresponding fields."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.ones((4, 2))
        y = jnp.ones((4, 2))
        ds = TensorDataset(
            x, y, transform_x=lambda a: a * 2, transform_y=lambda a: a + 5
        )
        item = ds[0]
        assert jnp.allclose(item["x"], jnp.full((2,), 2.0))
        assert jnp.allclose(item["y"], jnp.full((2,), 6.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_dataset_size_mismatch_raises(backend: str) -> None:
    """Mismatched leading dims between x and y raise AssertionError."""
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(AssertionError):
            TensorDataset(jnp.zeros((4, 2)), jnp.zeros((3, 2)))


# ---------------------------------------------------------------------------
# GeneralTensorDataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_single_set_returns_bare_item(backend: str) -> None:
    """With a single set, __getitem__ returns the sliced array (not a list)."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(5 * 2, dtype=jnp.float32).reshape(5, 2)
        ds = GeneralTensorDataset([x], [None])
        item = ds[3]
        # single-set dataset: __getitem__ returns a bare Array, not a list
        assert jnp.allclose(cast(jax.Array, item), x[3])
        assert len(ds) == 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_multi_set_returns_list(backend: str) -> None:
    """With multiple sets, __getitem__ returns a list of sliced arrays."""
    with jax.default_device(jax.devices(backend)[0]):
        a = jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2)
        b = jnp.arange(4 * 3, dtype=jnp.float32).reshape(4, 3)
        ds = GeneralTensorDataset([a, b], [None, None])
        item = ds[1]
        assert isinstance(item, list)
        assert len(item) == 2
        assert jnp.allclose(item[0], a[1])
        assert jnp.allclose(item[1], b[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_applies_per_set_transforms(backend: str) -> None:
    """Per-set transforms are applied only where provided."""
    with jax.default_device(jax.devices(backend)[0]):
        a = jnp.ones((3, 2))
        b = jnp.ones((3, 2))
        ds = GeneralTensorDataset([a, b], [lambda t: t * 3, None])
        item = ds[0]
        assert jnp.allclose(item[0], jnp.full((2,), 3.0))
        assert jnp.allclose(item[1], jnp.full((2,), 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_single_set_transform_applied(backend: str) -> None:
    """Single-set transform is applied to the bare item."""
    with jax.default_device(jax.devices(backend)[0]):
        a = jnp.ones((3, 2))
        ds = GeneralTensorDataset([a], [lambda t: t - 1])
        assert jnp.allclose(cast(jax.Array, ds[0]), jnp.zeros((2,)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_set_transform_count_mismatch_raises(backend: str) -> None:
    """len(sets) must equal len(transforms)."""
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(AssertionError):
            GeneralTensorDataset([jnp.zeros((3, 2)), jnp.zeros((3, 2))], [None])


@pytest.mark.parametrize("backend", BACKENDS)
def test_general_set_size_mismatch_raises(backend: str) -> None:
    """All sets must share the same leading dimension."""
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(AssertionError):
            GeneralTensorDataset([jnp.zeros((4, 2)), jnp.zeros((3, 2))], [None, None])
