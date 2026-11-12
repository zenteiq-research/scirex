# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the AdamW optimizer (JAX).

Mirrors ``neuralop/training/tests/test_optim.py`` (the AdamW parts):

* ``test_correct_complex_adam_momentum`` - after one step on a complex parameter
  the second moment ``exp_avg_sq`` equals ``2 * x**2`` (with betas=(0.5, 0.5)),
  verifying the PyTorch-matching ``|g|**2`` second-moment rule for complex
  params.
* ``test_galore_adamw_rank`` - a GaLore parameter's ``exp_avg_sq`` is stored at
  low rank (``prod(galore_rank)`` elements).

The JAX optimizer follows a PyTorch-like API where each parameter exposes a
``.grad`` attribute and ``step()`` returns ``(loss, updates)`` with per-parameter
state keyed by ``(group_idx, param_idx)``; the tests use a small ``_Param``
wrapper to supply ``.grad``. Constructor validation and the state-dict round-trip
have no dedicated upstream test but exercise the rest of the public surface.
"""

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorly.tucker_tensor import validate_tucker_rank

from scirex._internal.serialization import load_msgpack, save_msgpack
from scirex.training.optim.adamw import AdamW, adamw_cx, scale_by_adam_cx
from scirex.training.optim.tensor_galore_projector import TensorGaLoreProjector
from tests import tolerances


class _Param:
    """Minimal PyTorch-like parameter holder exposing ``.grad``."""

    def __init__(self, value: jnp.ndarray, grad: jnp.ndarray | None = None) -> None:
        self.value = value
        self.grad = grad


# ---------------------------------------------------------------------------
# test_correct_complex_adam_momentum
# ---------------------------------------------------------------------------


def test_correct_complex_adam_momentum() -> None:
    """exp_avg_sq for a complex param is |g|**2 * (1 - beta2) = 2 * x**2."""
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
    param = _Param(((0.0 + 1.0j) * x).astype(jnp.complex64))
    # grad of |param|**2 is 0 + 2j*x for param = 1j*x (matches the reference)
    param.grad = (2.0j * x).astype(jnp.complex64)

    optimizer = AdamW(params=[param], betas=(0.5, 0.5))
    optimizer.step()

    # momentum = (1 - beta2) * |g|**2 = 0.5 * |2j x|**2 = 2 x**2
    momentum = optimizer.state[(0, 0)]["exp_avg_sq"]
    assert np.allclose(np.array(momentum), np.array(2 * x**2), atol=1e-4)


# ---------------------------------------------------------------------------
# test_galore_adamw_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("galore_param_pct", [0.25, 0.5, 1.0])
def test_galore_adamw_rank(galore_param_pct: float) -> None:
    """A GaLore param keeps its second moment at low rank (prod(rank) elems)."""
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8, 8))
    param = _Param(((0.0 + 1.0j) * x).astype(jnp.complex64), (x).astype(jnp.complex64))
    galore_param = _Param(
        ((1.0 + 0j) * x).astype(jnp.complex64), (x).astype(jnp.complex64)
    )
    galore_rank = validate_tucker_rank(galore_param.value.shape, galore_param_pct)

    optimizer = AdamW(
        params=[param],
        galore_params=[galore_param],
        galore_rank=galore_rank,
        betas=(0.5, 0.5),
    )
    optimizer.step()

    # galore param lives in the second group (group_idx=1, param_idx=0)
    momentum = optimizer.state[(1, 0)]["exp_avg_sq"]
    assert momentum.size == math.prod(galore_rank)


# ---------------------------------------------------------------------------
# Step behaviour for real-valued params
# ---------------------------------------------------------------------------


def test_step_produces_update_for_real_param() -> None:
    """A real-valued param with a gradient gets a finite update and state."""
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    param = _Param(x, jnp.ones_like(x))

    optimizer = AdamW(params=[param], lr=1e-2)
    _, updates = optimizer.step()

    assert (0, 0) in updates
    assert updates[(0, 0)].shape == x.shape
    assert bool(jnp.all(jnp.isfinite(updates[(0, 0)])))
    assert optimizer.state[(0, 0)]["step"] == 1


def test_step_skips_param_without_grad() -> None:
    """A param whose grad is None is skipped (no update, no state)."""
    param = _Param(jnp.ones((2, 2)), grad=None)
    optimizer = AdamW(params=[param])
    _, updates = optimizer.step()
    assert updates == {}
    assert (0, 0) not in optimizer.state


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"betas": (-0.1, 0.999)},
        {"betas": (0.9, 1.0)},
        {"eps": -1e-8},
    ],
)
def test_invalid_hyperparameters_raise(kwargs: dict) -> None:
    """Out-of-range lr / betas / eps are rejected at construction."""
    with pytest.raises(ValueError):
        AdamW(params=[], **kwargs)


# ---------------------------------------------------------------------------
# State-dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip() -> None:
    """state_dict -> load_state_dict restores optimizer state."""
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 4))
    optimizer = AdamW(params=[_Param(x, jnp.ones_like(x))], lr=1e-2)
    optimizer.step()
    saved = optimizer.state_dict()

    restored = AdamW(params=[_Param(x, jnp.ones_like(x))], lr=1e-2)
    restored.load_state_dict(saved)
    assert restored.state[(0, 0)]["step"] == 1
    assert np.allclose(
        np.array(restored.state[(0, 0)]["exp_avg"]),
        np.array(optimizer.state[(0, 0)]["exp_avg"]),
    )


# ---------------------------------------------------------------------------
# scale_by_adam_cx: lines 43-86
# ---------------------------------------------------------------------------


def test_scale_by_adam_cx_init_real_params() -> None:
    """scale_by_adam_cx init_fn zeros out mu and nu for real params (lines 43-51)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = scale_by_adam_cx(b1=0.9, b2=0.999, eps=1e-6)
        params = {"w": jnp.ones((3, 3), dtype=jnp.float32)}
        state = tx.init(params)
        assert state.count == 0
        assert np.allclose(np.array(state.mu["w"]), 0.0)
        assert np.allclose(np.array(state.nu["w"]), 0.0)
        assert state.nu["w"].dtype == jnp.float32


def test_scale_by_adam_cx_init_complex_params() -> None:
    """scale_by_adam_cx init_fn sets nu to float32 for complex params (lines 43-51)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = scale_by_adam_cx()
        params = {"w": jnp.ones((2, 2), dtype=jnp.complex64)}
        state = tx.init(params)
        # nu should be real (float32), not complex
        assert state.nu["w"].dtype == jnp.float32
        assert np.allclose(np.array(state.nu["w"]), 0.0)


def test_scale_by_adam_cx_update_real_params() -> None:
    """update_fn for real params runs standard Adam update (lines 53-84)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = scale_by_adam_cx(b1=0.9, b2=0.999, eps=1e-6)
        params = {"w": jnp.ones((2,), dtype=jnp.float32)}
        state = tx.init(params)
        grads = {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}
        updates, new_state = tx.update(grads, state, params)
        # updates should be finite and non-zero
        assert np.all(np.isfinite(np.array(updates["w"])))
        assert new_state.count == 1
        # nu should equal (1 - 0.999) * g^2 for step 1
        expected_nu = (1.0 - 0.999) * jnp.array([1.0, 4.0])
        assert np.allclose(
            np.array(new_state.nu["w"]),
            np.array(expected_nu),
            atol=tolerances.atol(1e-6),
        )


def test_scale_by_adam_cx_update_complex_params() -> None:
    """update_fn for complex params uses |g|^2 for nu, conjugates gradient
    (lines 53-84)."""
    with jax.default_device(jax.devices("cpu")[0]):
        b2 = 0.5
        tx = scale_by_adam_cx(b1=0.5, b2=b2, eps=1e-6)
        params = {"w": jnp.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=jnp.complex64)}
        state = tx.init(params)
        # gradient: g = 2j (pure imaginary)
        grads = {"w": jnp.array([0.0 + 2.0j, 0.0 + 2.0j], dtype=jnp.complex64)}
        updates, new_state = tx.update(grads, state, params)
        # nu = (1 - b2) * |g|^2 = 0.5 * 4 = 2 for each element
        assert new_state.nu["w"].dtype == jnp.float32
        assert np.allclose(np.array(new_state.nu["w"]), 2.0, atol=tolerances.atol(1e-5))
        # mu should store conj(g) = -2j
        assert np.allclose(
            np.array(new_state.mu["w"]),
            np.array([0.0 - 1.0j, 0.0 - 1.0j]),
            atol=tolerances.atol(1e-5),
        )
        # updates should be finite
        assert np.all(np.isfinite(np.abs(np.array(updates["w"]))))


def test_scale_by_adam_cx_multiple_steps() -> None:
    """Repeated update calls accumulate state correctly (lines 53-84)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = scale_by_adam_cx(b1=0.9, b2=0.999, eps=1e-6)
        params = {"w": jnp.ones((3,), dtype=jnp.float32)}
        state = tx.init(params)
        grads = {"w": jnp.ones((3,), dtype=jnp.float32)}
        for _ in range(3):
            updates, state = tx.update(grads, state, params)
        assert state.count == 3
        assert np.all(np.isfinite(np.array(updates["w"])))


# ---------------------------------------------------------------------------
# adamw_cx: line 98
# ---------------------------------------------------------------------------


def test_adamw_cx_is_gradient_transformation() -> None:
    """adamw_cx returns a valid optax GradientTransformation (line 98)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = adamw_cx(learning_rate=1e-3, weight_decay=0.01)
        params = {"w": jnp.ones((4,), dtype=jnp.float32)}
        state = tx.init(params)
        grads = {"w": jnp.ones((4,), dtype=jnp.float32)}
        updates, _new_state = tx.update(grads, state, params)
        # updates should be finite and negative (descending with lr > 0)
        assert np.all(np.isfinite(np.array(updates["w"])))
        assert np.all(np.array(updates["w"]) < 0)


def test_adamw_cx_with_complex_params() -> None:
    """adamw_cx works with complex-valued parameters (line 98)."""
    with jax.default_device(jax.devices("cpu")[0]):
        tx = adamw_cx(learning_rate=1e-3, weight_decay=0.0)
        params = {"w": jnp.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=jnp.complex64)}
        state = tx.init(params)
        grads = {"w": jnp.array([0.5 + 0.5j, 1.0 + 1.0j], dtype=jnp.complex64)}
        updates, _ = tx.update(grads, state, params)
        assert np.all(np.isfinite(np.abs(np.array(updates["w"]))))


# ---------------------------------------------------------------------------
# AdamW.step: closure (line 244) and weight_decay update (line 326)
# ---------------------------------------------------------------------------


def test_step_with_closure() -> None:
    """step(closure) calls closure and returns the loss (line 244)."""
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.ones((2, 2))
        param = _Param(x, jnp.ones_like(x))
        optimizer = AdamW(params=[param], lr=1e-2)

        called = []

        def closure() -> jnp.ndarray:
            called.append(True)
            return jnp.array(42.0)

        loss, _updates = optimizer.step(closure)
        assert len(called) == 1
        assert float(loss) == pytest.approx(42.0)


def test_step_with_weight_decay_updates_param() -> None:
    """step applies weight decay in the parameter update (line 326)."""
    with jax.default_device(jax.devices("cpu")[0]):
        # adamw.py's weight-decay line does
        # `param_update - group["lr"] * group["weight_decay"] * p`, where `p`
        # is each element of group["params"] directly (not `p.value`), so the
        # param itself must support multiplication by a float. _Param does
        # not; _ArrayParam wraps a jnp array and implements __mul__/__rmul__
        # to replicate that arithmetic.
        class _ArrayParam:
            """Param supporting arithmetic (weight decay multiplies the param
            directly)."""

            def __init__(
                self, value: jnp.ndarray, grad: jnp.ndarray | None = None
            ) -> None:
                # Expose as jnp array attributes
                self._val = value
                self.grad = grad

            def __mul__(self, other: float) -> jnp.ndarray:
                return self._val * other

            def __rmul__(self, other: float) -> jnp.ndarray:
                return self._val * other

        xp = jnp.ones((2, 2))
        # Run without weight decay
        opt_no_wd = AdamW(
            params=[_ArrayParam(xp, jnp.ones_like(xp))], lr=1e-2, weight_decay=0.0
        )
        _, updates_no_wd = opt_no_wd.step()

        # Run with weight decay
        opt_wd = AdamW(
            params=[_ArrayParam(xp, jnp.ones_like(xp))], lr=1e-2, weight_decay=0.1
        )
        _, updates_wd = opt_wd.step()

        # Weight decay should make the update larger in magnitude (more negative)
        update_no_wd = np.array(updates_no_wd[(0, 0)])
        update_wd = np.array(updates_wd[(0, 0)])
        assert np.all(
            update_wd < update_no_wd
        ), "weight_decay should produce a more negative update"


def test_step_without_correct_bias() -> None:
    """step with correct_bias=False skips bias correction (line 326 adjacent)."""
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.ones((3,))
        param = _Param(x, jnp.ones_like(x))
        optimizer = AdamW(params=[param], lr=1e-2, correct_bias=False)
        _, updates = optimizer.step()
        assert (0, 0) in updates
        assert np.all(np.isfinite(np.array(updates[(0, 0)])))


# ---------------------------------------------------------------------------
# GaLore step with monkeypatched projector and state-dict round-trip
# ---------------------------------------------------------------------------


class _GaLoreParam:
    """Param with numpy val and optional JAX grad (for GaLore monkeypatch tests)."""

    def __init__(self, val: np.ndarray, grad: np.ndarray | None) -> None:
        self.val = np.array(val)
        self.grad = jnp.asarray(grad) if grad is not None else None

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.array(self.val)


def test_adamw_basic_step_and_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """GaLore AdamW step and state-dict round-trip with a monkeypatched projector."""
    import scirex.training.optim.adamw as adamw_mod

    class _FakeProj:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def project(self, x: jnp.ndarray, step: int) -> jnp.ndarray:
            return x * 0.5

        def project_back(self, x: jnp.ndarray) -> jnp.ndarray:
            return x * 1.0

    monkeypatch.setattr(adamw_mod, "TensorGaLoreProjector", _FakeProj)

    p = _GaLoreParam(val=np.ones((2,)), grad=np.array([0.1, -0.2]))
    opt = AdamW(
        params=[p], lr=0.01, galore_params=[p], galore_rank=1, galore_update_proj_gap=1
    )
    p2 = _GaLoreParam(val=np.ones((2,)), grad=None)
    opt.add_param_group(
        {
            "params": [p2],
            "lr": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0.0,
            "correct_bias": True,
            "galore": False,
        }
    )
    _, updates = opt.step()
    assert isinstance(updates, dict)
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    assert "state" in sd


def test_state_dict_msgpack_round_trip_with_galore(tmp_path: Path) -> None:
    """opt.state_dict() survives an actual msgpack round trip (VA finding F-03).

    Exercises the hard case end to end with the real (not monkeypatched)
    projector: tuple-keyed state, an embedded TensorGaLoreProjector, and
    params dropped from param_groups must all come back correctly through
    a real file, not just an in-memory dict transform.
    """
    # Real (non-monkeypatched) Tucker decomposition needs >= 2 dims.
    val = np.ones((4, 4))
    grad = np.full((4, 4), 0.1)
    p = _GaLoreParam(val=val, grad=grad)
    opt = AdamW(
        params=[p], lr=0.01, galore_params=[p], galore_rank=1, galore_update_proj_gap=1
    )
    opt.step()

    galore_key = next(k for k, v in opt.state.items() if "projector" in v)
    assert isinstance(opt.state[galore_key]["projector"], TensorGaLoreProjector)

    sd = opt.state_dict()
    # No tuple keys and no raw params should have reached the serialized form.
    assert all(isinstance(key, list) for key, _ in sd["state"])
    assert all("params" not in group for group in sd["param_groups"])

    path = tmp_path / "optimizer.msgpack"
    save_msgpack(sd, path)
    loaded = load_msgpack(path)

    restored = AdamW(
        params=[p], lr=0.01, galore_params=[p], galore_rank=1, galore_update_proj_gap=1
    )
    restored.load_state_dict(loaded)

    assert restored.state[galore_key]["step"] == opt.state[galore_key]["step"]
    restored_projector = restored.state[galore_key]["projector"]
    assert isinstance(restored_projector, TensorGaLoreProjector)
    for a, b in zip(
        restored_projector.proj_tensor,
        opt.state[galore_key]["projector"].proj_tensor,
        strict=True,
    ):
        assert np.allclose(np.array(a), np.array(b))
    # params are kept from the live optimizer, never read back from disk.
    assert restored.param_groups[0]["params"][0] is p
