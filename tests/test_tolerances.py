# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The tolerance helper: float32 is scaled per backend, float64 is tightened."""

import jax.numpy as jnp
import pytest

from tests import tolerances


@pytest.fixture
def fake_backend(monkeypatch):
    def use(name: str) -> None:
        monkeypatch.setattr(tolerances.jax, "default_backend", lambda: name)

    return use


@pytest.mark.parametrize(
    "name, factor", [("cpu", 1.0), ("gpu", 10.0), ("tpu", 1000.0), ("rocm", 10.0)]
)
def test_float32_is_scaled_by_backend(fake_backend, name, factor):
    fake_backend(name)
    assert tolerances.atol(1e-5, jnp.float32) == pytest.approx(1e-5 * factor)
    assert tolerances.rtol(1e-6, jnp.float32) == pytest.approx(1e-6 * factor)


def test_cpu_float32_keeps_the_stated_value(fake_backend):
    fake_backend("cpu")
    assert tolerances.atol(1e-5) == 1e-5
    assert tolerances.both(rtol=1e-6, atol=1e-5) == {"rtol": 1e-6, "atol": 1e-5}


@pytest.mark.parametrize("name", ["cpu", "gpu", "tpu"])
def test_float64_is_tightened_and_not_scaled(fake_backend, name):
    fake_backend(name)
    assert tolerances.atol(1e-5, jnp.float64) == tolerances.FLOAT64_ATOL
    assert tolerances.rtol(1e-5, jnp.float64) == tolerances.FLOAT64_RTOL


def test_float64_never_loosens_a_tighter_request(fake_backend):
    fake_backend("cpu")
    assert tolerances.atol(1e-14, jnp.float64) == 1e-14


@pytest.mark.needs_x64
def test_default_dtype_decides_when_none_is_given(fake_backend):
    fake_backend("gpu")
    assert tolerances.atol(1e-5) == tolerances.FLOAT64_ATOL


def test_current_backend_is_known():
    assert tolerances.backend() in tolerances.FLOAT32_SCALE
