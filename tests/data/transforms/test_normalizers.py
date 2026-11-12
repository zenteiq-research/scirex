# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for UnitGaussianNormalizer (JAX/Flax).

Mirrors ``neuralop/data/transforms/tests/test_unit_gaussian_normalizer.py``:
the three reference tests (created-from-stats, fit-from-data, incremental
update) are reproduced against the functional JAX implementation. The
``dim`` argument of the PyTorch normalizer corresponds to ``axis`` here, and
``fit`` / ``partial_fit`` return new instances rather than mutating in place.
The spatial extent is reduced from the reference (40x50x60 -> 8x10x12) to keep
the JAX/XLA test fast while preserving the 5D reduction structure.
"""

import jax
import jax.numpy as jnp
import pytest
from flaky import flaky

from scirex.data.transforms.normalizers import (
    DictUnitGaussianNormalizer,
    Normalizer,
    UnitGaussianNormalizer,
)
from tests import tolerances

EPS = 1e-6
REDUCE_AXIS = (0, 2, 3, 4)


def _sample(key=0):
    return jax.random.uniform(jax.random.PRNGKey(key), (16, 3, 8, 10, 12)) * 2.5


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_created_from_stats() -> None:
    """Initialised from ground-truth mean/std, transform is invertible and
    yields ~zero mean / ~unit std."""
    x = _sample(0)
    mean = jnp.mean(x, axis=REDUCE_AXIS, keepdims=True)
    std = jnp.std(x, axis=REDUCE_AXIS, keepdims=True)

    normalizer = UnitGaussianNormalizer(mean=mean, std=std, eps=EPS)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert jnp.allclose(x_unnormalized, x, atol=1e-4)
    assert jnp.mean(x_normalized) <= EPS
    assert (jnp.std(x_normalized) - 1) <= EPS


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_from_data() -> None:
    """Fitting the whole batch at once recovers mean/std and is invertible."""
    x = _sample(1)
    mean = jnp.mean(x, axis=REDUCE_AXIS, keepdims=True)
    std = jnp.std(x, axis=REDUCE_AXIS, keepdims=True)

    normalizer = UnitGaussianNormalizer.create(axis=REDUCE_AXIS, eps=EPS).fit(x)

    assert jnp.allclose(normalizer.mean, mean, atol=tolerances.atol(1e-5))
    assert jnp.allclose(normalizer.std, std, rtol=EPS, atol=EPS)

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert jnp.allclose(x_unnormalized, x, atol=1e-4)
    assert jnp.mean(x_normalized) <= EPS
    assert (jnp.std(x_normalized) - 1) <= EPS


@flaky(max_runs=4, min_passes=3)
def test_UnitGaussianNormalizer_incremental_update() -> None:
    """Incremental (partial_fit) statistics match a single-pass fit and are
    invertible."""
    x = _sample(2)
    mean = jnp.mean(x, axis=REDUCE_AXIS, keepdims=True)

    normalizer = UnitGaussianNormalizer.create(axis=REDUCE_AXIS, eps=EPS).partial_fit(
        x, batch_size=2
    )

    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    assert jnp.allclose(x_unnormalized, x, atol=1e-4)
    assert jnp.mean(x_normalized) <= EPS
    assert (jnp.std(x_normalized) - 1) <= EPS
    # incremental mean matches the analytic mean (std uses an unbiased
    # correction, so it is only checked through the ~unit-std assertion above)
    assert jnp.allclose(normalizer.mean, mean, atol=1e-4)


def test_partial_fit_skips_empty_batch() -> None:
    """partial_fit on an empty leading dim is a no-op (returns unfitted state)."""
    normalizer = UnitGaussianNormalizer.create(axis=REDUCE_AXIS, eps=EPS)
    empty = jnp.zeros((0, 3, 8, 10, 12))
    result = normalizer.partial_fit(empty)
    assert result.mean is None
    assert result.n_elements == 0


def test_basic_normalizer_round_trip() -> None:
    """The plain Normalizer struct transform/inverse_transform round-trips."""
    x = _sample(3)
    mean = jnp.mean(x, axis=REDUCE_AXIS, keepdims=True)
    std = jnp.std(x, axis=REDUCE_AXIS, keepdims=True)
    norm = Normalizer(mean=mean, std=std, eps=EPS)
    assert jnp.allclose(norm.inverse_transform(norm.transform(x)), x, atol=1e-4)


# ---------------------------------------------------------------------------
# UnitGaussianNormalizer.from_dataset and the masked reduction path
# ---------------------------------------------------------------------------


def test_UnitGaussianNormalizer_from_dataset() -> None:
    """from_dataset fits one normalizer per requested key over a dict-dataset."""
    dataset = [
        {"a": jnp.ones((3,)), "b": jnp.zeros((3,))},
        {"a": 3.0 * jnp.ones((3,)), "b": jnp.zeros((3,))},
    ]
    instances = UnitGaussianNormalizer.from_dataset(dataset, axis=(0,), keys=["a"])
    # only the requested key is fitted
    assert set(instances.keys()) == {"a"}
    # the mean over the two samples {1, 3} is 2
    assert jnp.allclose(instances["a"].mean, 2.0, atol=tolerances.atol(1e-5))


def test_UnitGaussianNormalizer_from_dataset_infers_keys() -> None:
    """When keys is None, every field in the first sample gets a normalizer."""
    dataset = [{"x": jnp.ones((4,)), "y": 2.0 * jnp.ones((4,))}]
    instances = UnitGaussianNormalizer.from_dataset(dataset, axis=(0,))
    assert set(instances.keys()) == {"x", "y"}


def test_UnitGaussianNormalizer_create_with_int_axis() -> None:
    """A scalar int axis is normalized to a one-tuple."""
    normalizer = UnitGaussianNormalizer.create(axis=0)
    assert normalizer.axis == (0,)


def test_UnitGaussianNormalizer_call_delegates_to_transform() -> None:
    """__call__ is an alias for transform()."""
    x = _sample(6)
    normalizer = UnitGaussianNormalizer.create(axis=REDUCE_AXIS).fit(x)
    assert jnp.allclose(normalizer(x), normalizer.transform(x))


def test_UnitGaussianNormalizer_masked_fit_runs() -> None:
    """The masked reduction path produces finite, correctly-shaped statistics."""
    data = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4)
    mask = jnp.array([1, 1, 0, 0])
    normalizer = UnitGaussianNormalizer.create(axis=(0, 1), mask=mask).fit(data)
    assert normalizer.mean.shape == (1, 4)
    assert jnp.all(jnp.isfinite(normalizer.mean))
    assert jnp.all(jnp.isfinite(normalizer.std))


def test_UnitGaussianNormalizer_masked_incremental_runs() -> None:
    """The masked incremental path runs across multiple partial_fit batches."""
    data = jnp.arange(4 * 4, dtype=jnp.float32).reshape(4, 4)
    mask = jnp.array([1, 0, 1, 0])
    normalizer = UnitGaussianNormalizer.create(axis=(0, 1), mask=mask).partial_fit(
        data, batch_size=2
    )
    assert normalizer.n_elements > 0
    assert jnp.all(jnp.isfinite(normalizer.mean))


# ---------------------------------------------------------------------------
# DictUnitGaussianNormalizer
# ---------------------------------------------------------------------------


def _dict_normalizer():
    field = UnitGaussianNormalizer(
        mean=jnp.array([1.0, 1.0, 1.0]), std=jnp.array([2.0, 2.0, 2.0])
    )
    return DictUnitGaussianNormalizer.create(
        normalizer_dict={"a": field},
        input_mappings={"a": slice(0, 3)},
        return_mappings={"a": slice(0, 3)},
    )


def test_DictUnitGaussianNormalizer_round_trip() -> None:
    """transform normalizes the mapped slice; inverse_transform recovers it."""
    norm = _dict_normalizer()
    x = jnp.arange(6.0)
    transformed = norm.transform(x)
    # the unmapped tail (indices 3..5) is left untouched
    assert jnp.allclose(transformed[3:], x[3:])
    # __call__ delegates to transform
    assert jnp.allclose(norm(x), transformed)
    # the mapped slice round-trips through inverse_transform
    assert jnp.allclose(
        norm.inverse_transform(transformed)[:3], x[:3], atol=tolerances.atol(1e-5)
    )


def test_DictUnitGaussianNormalizer_key_mismatch_asserts() -> None:
    """create asserts when normalizer keys do not match the input/return mappings."""
    field = UnitGaussianNormalizer(mean=jnp.zeros((3,)), std=jnp.ones((3,)))
    with pytest.raises(AssertionError):
        DictUnitGaussianNormalizer.create(
            normalizer_dict={"a": field},
            input_mappings={"b": slice(0, 3)},
            return_mappings={"a": slice(0, 3)},
        )
    with pytest.raises(AssertionError):
        DictUnitGaussianNormalizer.create(
            normalizer_dict={"a": field},
            input_mappings={"a": slice(0, 3)},
            return_mappings={"b": slice(0, 3)},
        )


def test_DictUnitGaussianNormalizer_from_dataset() -> None:
    """from_dataset builds and fits the composite normalizer over a dict-dataset."""
    dataset = [
        {"a": jnp.ones((3,)), "b": jnp.zeros((3,))},
        {"a": 3.0 * jnp.ones((3,)), "b": jnp.zeros((3,))},
    ]
    norm = DictUnitGaussianNormalizer.from_dataset(
        dataset,
        input_mappings={"a": slice(0, 3)},
        return_mappings={"a": slice(0, 3)},
        axis=(0,),
    )
    assert isinstance(norm, DictUnitGaussianNormalizer)
    assert set(norm.normalizer_dict.keys()) == {"a"}
    assert jnp.allclose(norm.normalizer_dict["a"].mean, 2.0, atol=tolerances.atol(1e-5))
