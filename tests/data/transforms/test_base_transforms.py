# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the base transform classes (JAX/Flax).

No dedicated PyTorch reference test exists for these base classes in
neuraloperator, so coverage follows the structure of the Phase 1 layer tests
(CPU/GPU backend parametrization). It exercises CompositeTransform ordering /
round-trip behaviour and DictTransform per-field slicing.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.data.transforms.base_transforms import (
    CompositeTransform,
    DictTransform,
    Transform,
)
from tests import tolerances


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


class _AffineTransform(Transform):
    """Concrete Transform: x -> scale * x + shift (invertible)."""

    scale: float
    shift: float

    def transform(self, x):
        return self.scale * x + self.shift

    def inverse_transform(self, x):
        return (x - self.shift) / self.scale


class _Scaler:
    """Minimal per-field transform used by DictTransform tests."""

    def __init__(self, factor):
        self.factor = factor

    def transform(self, x):
        return x * self.factor

    def inverse_transform(self, x):
        return x / self.factor


# ---------------------------------------------------------------------------
# CompositeTransform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_composite_applies_in_order(backend) -> None:
    """transform applies sub-transforms left-to-right."""
    with jax.default_device(jax.devices(backend)[0]):
        # first +1, then *2  ->  (x + 1) * 2
        comp = CompositeTransform(
            transforms=[
                _AffineTransform(scale=1.0, shift=1.0),
                _AffineTransform(scale=2.0, shift=0.0),
            ]
        )
        x = jnp.arange(4.0)
        out = comp.transform(x)
        assert jnp.allclose(out, (x + 1.0) * 2.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_composite_inverse_is_round_trip(backend) -> None:
    """inverse_transform reverses transform (applied in reverse order)."""
    with jax.default_device(jax.devices(backend)[0]):
        comp = CompositeTransform(
            transforms=[
                _AffineTransform(scale=3.0, shift=-2.0),
                _AffineTransform(scale=0.5, shift=4.0),
            ]
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (5,))
        round_trip = comp.inverse_transform(comp.transform(x))
        assert jnp.allclose(round_trip, x, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_composite_empty_is_identity(backend) -> None:
    """An empty composite leaves the data unchanged."""
    with jax.default_device(jax.devices(backend)[0]):
        comp = CompositeTransform(transforms=[])
        x = jnp.arange(6.0)
        assert jnp.allclose(comp.transform(x), x)


# ---------------------------------------------------------------------------
# DictTransform
# ---------------------------------------------------------------------------


def _dict_transform():
    """Two channel-blocks scaled by different factors."""
    transforms = {"a": _Scaler(2.0), "b": _Scaler(10.0)}
    input_mappings = {
        "a": (slice(None), slice(0, 2)),
        "b": (slice(None), slice(2, 4)),
    }
    return DictTransform(transforms=transforms, input_mappings=input_mappings)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dict_transform_applies_per_field(backend) -> None:
    """Each field's slice is transformed with its own transform."""
    with jax.default_device(jax.devices(backend)[0]):
        dt = _dict_transform()
        x = jnp.ones((3, 4))
        out = dt.transform(x)
        assert jnp.allclose(out[:, 0:2], 2.0)
        assert jnp.allclose(out[:, 2:4], 10.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dict_transform_round_trip(backend) -> None:
    """inverse_transform recovers the original tensor."""
    with jax.default_device(jax.devices(backend)[0]):
        dt = _dict_transform()
        x = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
        round_trip = dt.inverse_transform(dt.transform(x))
        assert jnp.allclose(round_trip, x, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_dict_transform_key_mismatch_raises(backend) -> None:
    """Mismatched transforms / input_mappings keys are rejected at setup."""
    with jax.default_device(jax.devices(backend)[0]):
        dt = DictTransform(
            transforms={"a": _Scaler(2.0)},
            input_mappings={"b": (slice(None),)},
        )
        x = jnp.ones((2, 2))
        with pytest.raises(AssertionError):
            # setup() runs when the module is applied
            dt.init(jax.random.PRNGKey(0), x)


# ---------------------------------------------------------------------------
# Transform base class: abstract bodies, __call__, to
# ---------------------------------------------------------------------------


def test_transform_abstract_method_bodies() -> None:
    """The abstract Transform.transform / inverse_transform / to bodies are no-ops."""
    t = _AffineTransform(scale=2.0, shift=1.0)
    x = jnp.ones((2, 3))
    # invoking the base-class definitions directly executes their `pass` bodies
    assert Transform.transform(t, x) is None
    assert Transform.inverse_transform(t, x) is None
    # to() is a concrete identity method on the base class, not abstract
    assert Transform.to(t, "cpu") is t


def test_transform_call_delegates_to_transform() -> None:
    """Transform.__call__ (nn.compact) applies transform()."""
    t = _AffineTransform(scale=2.0, shift=1.0)
    x = jnp.arange(4.0)
    out = t.apply({}, x)
    assert jnp.allclose(out, 2.0 * x + 1.0)


def test_composite_transform_call_and_to() -> None:
    """CompositeTransform.__call__ applies transform; to() returns self."""
    comp = CompositeTransform(transforms=[_AffineTransform(scale=2.0, shift=1.0)])
    x = jnp.arange(4.0)
    assert jnp.allclose(comp.apply({}, x), 2.0 * x + 1.0)
    assert comp.to("cpu") is comp


# ---------------------------------------------------------------------------
# DictTransform: return_mappings branch, __call__, to
# ---------------------------------------------------------------------------


def _dict_transform_with_return_mappings():
    """A DictTransform whose return_mappings are identity (shape-preserving) slices."""
    return DictTransform(
        transforms={"a": _Scaler(2.0)},
        input_mappings={"a": (slice(None), slice(0, 2))},
        return_mappings={"a": (slice(None), slice(None))},
    )


def test_dict_transform_with_return_mappings_call() -> None:
    """__call__/setup run with return_mappings and the return-slice is applied."""
    dt = _dict_transform_with_return_mappings()
    x = jnp.ones((3, 2))
    # init() runs setup() (with the return_mappings assert) and __call__ -> transform
    out = dt.apply(dt.init(jax.random.PRNGKey(0), x), x)
    assert jnp.allclose(out, 2.0)


def test_dict_transform_with_return_mappings_inverse_and_to() -> None:
    """inverse_transform applies the return-slice branch; to() returns self."""
    dt = _dict_transform_with_return_mappings()
    x = jnp.ones((3, 2)) * 4.0
    decoded = dt.inverse_transform(x)
    assert jnp.allclose(decoded, 2.0)
    assert dt.to("cpu") is dt
