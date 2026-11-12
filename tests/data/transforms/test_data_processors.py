# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the data processors (JAX/Flax).

Mirrors ``neuralop/data/transforms/tests/test_data_processor.py``:

* ``test_DefaultDataProcessor_pipeline`` -> preprocess/postprocess round-trip
* ``test_DefaultDataProcessor_train_eval`` -> wrap / train / eval / device state
* ``test_incremental_resolution`` -> IncrementalDataProcessor resolution decrease
* ``MGPatchingDataProcessor`` now lives in ``scirex.experimental.patching``; its
  (currently skipped) test moved to ``tests/experimental/test_patching.py``.

The PyTorch device selection (``torch.backends.cuda.is_built()``) is replaced
by JAX device discovery.
"""

import jax
import jax.numpy as jnp

from scirex.data.transforms.data_processors import (
    DataProcessor,
    DefaultDataProcessor,
    IncrementalDataProcessor,
)
from scirex.data.transforms.normalizers import UnitGaussianNormalizer


class _ConcreteProcessor(DataProcessor):
    """Minimal concrete DataProcessor to exercise the abstract base class."""

    def to(self, device):
        return self

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x

    def __call__(self, x, training: bool = True):
        return x


class _ToNormalizer:
    """A normalizer stub whose .to(device) returns self (for the to() test)."""

    def to(self, device):
        self.device = device
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def _device():
    try:
        if jax.devices("gpu"):
            return "gpu"
    except RuntimeError:
        pass
    return "cpu"


class _DummyModel:
    """Minimal stand-in for neuralop's DummyModel (only stored, not called here)."""

    def __init__(self, features: int = 10) -> None:
        self.features = features


class _CallableModel:
    """A callable stand-in model so the processors' __call__ pipeline can run."""

    def __call__(self, x):
        return x * 2.0


# ---------------------------------------------------------------------------
# DefaultDataProcessor
# ---------------------------------------------------------------------------


def test_DefaultDataProcessor_pipeline() -> None:
    """preprocess then postprocess recovers y through a (near-identity) normalizer."""
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 64, 64))
    y = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 64, 64))

    normalizer = UnitGaussianNormalizer(
        mean=jnp.zeros((1, 2, 1, 1)), std=jnp.ones((1, 2, 1, 1)), eps=1e-5
    )
    pipeline = DefaultDataProcessor(in_normalizer=normalizer, out_normalizer=normalizer)

    data = {"x": x, "y": y}  # y is kept as a separate immutable array reference
    xform_data = pipeline.preprocess(data)

    out = jax.random.normal(jax.random.PRNGKey(2), (1, 2, 64, 64))
    _, inv_xform_data = pipeline.postprocess(out, xform_data)

    assert jnp.allclose(inv_xform_data["y"], y, atol=1e-4)


def test_DefaultDataProcessor_train_eval() -> None:
    """wrap attaches a model; train()/eval() toggle the processor's mode; to() sets device."""
    device = _device()
    model = _DummyModel(features=10)

    normalizer = UnitGaussianNormalizer(
        mean=jnp.zeros((1, 2, 1, 1)), std=jnp.ones((1, 2, 1, 1)), eps=1e-5
    )
    pipeline = DefaultDataProcessor(in_normalizer=normalizer, out_normalizer=normalizer)
    wrapped_model = pipeline.wrap(model).to(device)

    assert wrapped_model.device == device
    assert wrapped_model.model is model

    wrapped_model.train()
    assert wrapped_model.training

    wrapped_model.eval()
    assert not wrapped_model.training


# ---------------------------------------------------------------------------
# IncrementalDataProcessor
# ---------------------------------------------------------------------------


def test_incremental_resolution() -> None:
    """IncrementalDataProcessor subsamples the spatial dims at dataset_indices."""
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16))
    y = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 16, 16))
    indice_list = [2, 3]

    data_transform = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=None,
        device=_device(),
        subsampling_rates=[2],
        dataset_resolution=16,
        dataset_indices=indice_list,
        epoch_gap=10,
        verbose=True,
    )

    x_new, y_new = data_transform.regularize_input_res(x, y)

    for i in indice_list:
        assert x_new.shape[i] < x.shape[i]
        assert y_new.shape[i] < y.shape[i]


def test_default_data_processor_call_runs_full_pipeline() -> None:
    """__call__ wires preprocess -> model -> postprocess and returns (out, dict)."""
    pipeline = DefaultDataProcessor().wrap(_CallableModel())
    data = {"x": jnp.ones((2, 3)), "y": jnp.ones((2, 3))}
    out, returned = pipeline(data, training=False)
    # the callable model doubles its input
    assert jnp.allclose(out, 2.0)
    assert "y" in returned


def test_incremental_index_to_sub_out_of_range() -> None:
    """Indices beyond the subsampling table clamp to the last entry."""
    proc = IncrementalDataProcessor(
        subsampling_rates=[2, 1], dataset_resolution=16, dataset_indices=[2, 3]
    )
    assert proc.index_to_sub_from_table(99) == 1


def test_incremental_epoch_wise_res_increase() -> None:
    """At an epoch on the gap boundary, the resolution index advances once."""
    proc = IncrementalDataProcessor(
        subsampling_rates=[2, 1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=2,
        verbose=True,
    )
    assert proc.current_index == 0
    proc.epoch_wise_res_increase(2)
    assert proc.current_index == 1
    assert proc.current_sub == 1
    assert proc.current_res == 16
    # the same epoch does not advance the index a second time
    proc.epoch_wise_res_increase(2)
    assert proc.current_index == 1


def test_incremental_step_regularizes() -> None:
    """step() advances the epoch schedule and subsamples x, y together."""
    proc = IncrementalDataProcessor(
        subsampling_rates=[2, 1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=2,
    )
    x = jnp.ones((1, 2, 16, 16))
    y = jnp.ones((1, 2, 16, 16))
    x_new, y_new = proc.step(epoch=0, x=x, y=y)
    # current_sub is 2 at index 0 -> spatial dims are halved
    assert x_new.shape[2] == 8
    assert y_new.shape[3] == 8


def test_incremental_preprocess_applies_input_normalizer() -> None:
    """preprocess normalizes x with the input normalizer and runs the step schedule.

    Note: out_normalizer is left None here because IncrementalDataProcessor.preprocess
    guards the y-normalization with ``self.train`` (the bound method, an nn.Module
    attribute that is not present imperatively); exercising that branch would require
    a production-code fix, which is out of scope for this test-only change.
    """
    in_normalizer = UnitGaussianNormalizer(
        mean=jnp.zeros((1, 2, 1, 1)), std=jnp.ones((1, 2, 1, 1)), eps=1e-5
    )
    proc = IncrementalDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=None,
        subsampling_rates=[1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=10,
    )
    proc.training = True
    data = {"x": jnp.ones((1, 2, 16, 16)), "y": jnp.ones((1, 2, 16, 16))}
    processed = proc.preprocess(data)
    assert {"x", "y"} <= set(processed.keys())
    assert processed["x"].shape == (1, 2, 16, 16)


def test_incremental_postprocess_denormalizes_in_eval() -> None:
    """postprocess inverse-transforms output and y with out_normalizer in eval."""
    out_normalizer = UnitGaussianNormalizer(
        mean=jnp.zeros((1, 2, 1, 1)), std=jnp.ones((1, 2, 1, 1)), eps=1e-5
    )
    proc = IncrementalDataProcessor(
        in_normalizer=None,
        out_normalizer=out_normalizer,
        subsampling_rates=[1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=10,
    )
    out = jnp.ones((1, 2, 16, 16))
    data = {"x": jnp.ones((1, 2, 16, 16)), "y": jnp.ones((1, 2, 16, 16))}
    result, returned = proc.postprocess(out, data, training=False)
    assert result.shape == out.shape
    assert "y" in returned


def test_incremental_to_sets_device() -> None:
    """to() updates the device (with no normalizers to relocate)."""
    proc = IncrementalDataProcessor(
        subsampling_rates=[1], dataset_resolution=16, dataset_indices=[2, 3]
    )
    assert proc.to("cpu").device == "cpu"


def test_abstract_data_processor_base() -> None:
    """A concrete DataProcessor subclass instantiates; base wrap/abstract bodies run."""
    proc = _ConcreteProcessor()
    model = _CallableModel()
    # default wrap stores the model and returns self
    assert proc.wrap(model) is proc
    assert proc.model is model
    # the abstract method bodies (pass) are reachable via the base class
    x = jnp.ones((2, 3))
    assert DataProcessor.to(proc, "cpu") is None
    assert DataProcessor.preprocess(proc, x) is None
    assert DataProcessor.postprocess(proc, x, {}) is None
    assert DataProcessor.__call__(proc, x) is None


def test_default_postprocess_uses_self_training_when_none() -> None:
    """postprocess(training=None) falls back to self.training set by eval()."""
    normalizer = UnitGaussianNormalizer(
        mean=jnp.zeros((1, 2, 1, 1)), std=jnp.ones((1, 2, 1, 1)), eps=1e-5
    )
    pipeline = DefaultDataProcessor(out_normalizer=normalizer)
    pipeline.eval()  # self.training = False
    out = jnp.ones((1, 2, 4, 4))
    result, _ = pipeline.postprocess(out, {"y": out})  # training defaults to None
    assert result.shape == out.shape


def test_incremental_to_relocates_normalizers() -> None:
    """IncrementalDataProcessor.to relocates both normalizers and sets device."""
    proc = IncrementalDataProcessor(
        in_normalizer=_ToNormalizer(),
        out_normalizer=_ToNormalizer(),
        subsampling_rates=[1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
    )
    moved = proc.to("cpu")
    assert moved.device == "cpu"
    assert moved.in_normalizer.device == "cpu"
    assert moved.out_normalizer.device == "cpu"


def test_incremental_call_runs_full_pipeline() -> None:
    """__call__ wires preprocess -> model -> postprocess (incremental processor)."""
    proc = IncrementalDataProcessor(
        subsampling_rates=[1],
        dataset_resolution=16,
        dataset_indices=[2, 3],
        epoch_gap=10,
    )
    proc.model = _CallableModel()
    data = {"x": jnp.ones((1, 2, 16, 16)), "y": jnp.ones((1, 2, 16, 16))}
    out, returned = proc(data, training=False)
    assert out.shape == (1, 2, 16, 16)
    assert "y" in returned
