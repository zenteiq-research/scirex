# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for DictDataset (JAX).

No dedicated PyTorch reference test exists for DictDataset in neuraloperator,
so coverage follows the structure of the Phase 1 layer tests (CPU/GPU backend
parametrization) and exercises the basic dataset contract: ``__len__`` /
``__getitem__``, constant broadcasting, and non-mutation of the underlying list.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.data.containers.dict_dataset import DictDataset


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _sample_list(n: int = 3) -> list[dict]:
    return [
        {"x": jnp.full((2, 2), float(i)), "y": jnp.array(float(i))} for i in range(n)
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_len_matches_data_list(backend: str) -> None:
    """__len__ returns the number of entries in data_list."""
    with jax.default_device(jax.devices(backend)[0]):
        ds = DictDataset(_sample_list(4))
        assert len(ds) == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_getitem_returns_dict_with_expected_keys(backend: str) -> None:
    """__getitem__ returns the stored dict for the requested index."""
    with jax.default_device(jax.devices(backend)[0]):
        ds = DictDataset(_sample_list(3))
        item = ds[1]
        assert set(item.keys()) == {"x", "y"}
        assert jnp.allclose(item["x"], jnp.full((2, 2), 1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_constant_is_merged_into_each_item(backend: str) -> None:
    """Constant key/value pairs are added to every returned item."""
    with jax.default_device(jax.devices(backend)[0]):
        const = {"meta": jnp.array(7.0)}
        ds = DictDataset(_sample_list(2), constant=const)
        for i in range(len(ds)):
            item = ds[i]
            assert "meta" in item
            assert jnp.allclose(item["meta"], jnp.array(7.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_getitem_does_not_mutate_underlying_data(backend: str) -> None:
    """__getitem__ returns a copy; mutating it must not affect the dataset."""
    with jax.default_device(jax.devices(backend)[0]):
        data = _sample_list(2)
        ds = DictDataset(data, constant={"meta": jnp.array(1.0)})
        item = ds[1]
        item["x"] = jnp.zeros((2, 2))
        item["meta"] = jnp.array(-1.0)
        # Original entry (value 1.0) and constant are untouched by the mutation.
        assert jnp.allclose(ds[1]["x"], jnp.full((2, 2), 1.0))
        assert "meta" not in data[1]
        assert jnp.allclose(ds[1]["meta"], jnp.array(1.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_constant_keeps_only_original_keys(backend: str) -> None:
    """With constant=None, items contain only their original keys."""
    with jax.default_device(jax.devices(backend)[0]):
        ds = DictDataset(_sample_list(2))
        assert set(ds[0].keys()) == {"x", "y"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_iteration_via_indexing(backend: str) -> None:
    """Dataset supports sequential index access over its full range."""
    with jax.default_device(jax.devices(backend)[0]):
        ds = DictDataset(_sample_list(5))
        collected = [ds[i]["y"] for i in range(len(ds))]
        assert jnp.allclose(jnp.stack(collected), jnp.arange(5.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_index_out_of_range_raises(backend: str) -> None:
    """Indexing past the end raises IndexError (list semantics)."""
    with jax.default_device(jax.devices(backend)[0]):
        ds = DictDataset(_sample_list(2))
        with pytest.raises(IndexError):
            _ = ds[5]
