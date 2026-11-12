# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the GNO radial weighting functions."""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.gno_weighting_functions import (
    WEIGHTING_FN_REGISTRY,
    bump_cutoff,
    dispatch_weighting_fn,
    half_cos_cutoff,
    octic_cutoff,
    quadr_cutoff,
    quartic_cutoff,
)
from tests import tolerances

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)


# ---------------------------------------------------------------------------
# bump_cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_bump_cutoff_at_zero(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = bump_cutoff(jnp.array(0.0), radius=1.0, scale=1.0)
        # At x=0: out = exp(-1/(1 - 0 + eps)) * e ≈ e * exp(-1) = 1.0
        assert jnp.isfinite(val)
        assert float(val) > 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_bump_cutoff_scale(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array(0.3)
        v1 = bump_cutoff(x, radius=1.0, scale=1.0)
        v2 = bump_cutoff(x, radius=1.0, scale=2.0)
        assert jnp.allclose(v2, 2.0 * v1, atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_bump_cutoff_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.linspace(0.0, 1.0, 10)
        out = bump_cutoff(x, radius=1.0, scale=1.0)
        assert out.shape == (10,)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# half_cos_cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_half_cos_cutoff_at_zero(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = half_cos_cutoff(jnp.array(0.0), radius=1.0, scale=1.0)
        # At x=0: 0.5*cos(0) + 0.5 = 1.0
        assert jnp.allclose(val, jnp.array(1.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_half_cos_cutoff_at_radius(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = half_cos_cutoff(jnp.array(1.0), radius=1.0, scale=1.0)
        # At x=radius: 0.5*cos(pi) + 0.5 = 0.0
        assert jnp.allclose(val, jnp.array(0.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_half_cos_cutoff_scale(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.array(0.5)
        v1 = half_cos_cutoff(x, radius=1.0, scale=1.0)
        v2 = half_cos_cutoff(x, radius=1.0, scale=3.0)
        assert jnp.allclose(v2, 3.0 * v1, atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# quadr_cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_quadr_cutoff_at_zero(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = quadr_cutoff(jnp.array(0.0), radius=1.0, scale=1.0)
        # At x=0, d=0 < 0.5: 1 - 2*0 = 1.0
        assert jnp.allclose(val, jnp.array(1.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_quadr_cutoff_at_radius(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = quadr_cutoff(jnp.array(1.0), radius=1.0, scale=1.0)
        # At x=radius, d=1 >= 0.5: 2*(1-1)^2 = 0.0
        assert jnp.allclose(val, jnp.array(0.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_quadr_cutoff_continuity_at_half(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        # At d=0.5, both branches give the same value
        x = jnp.array(0.5)
        1 - 2 * 0.5**2  # 0.5
        2 * (1 - 0.5) ** 2  # 0.5
        val = quadr_cutoff(x, radius=1.0, scale=1.0)
        assert jnp.allclose(val, jnp.array(0.5), atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# quartic_cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_quartic_cutoff_at_zero(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = quartic_cutoff(jnp.array(0.0), radius=1.0, scale=1.0)
        # a*0 + c*0 + scale = scale
        assert jnp.allclose(val, jnp.array(1.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_quartic_cutoff_at_radius(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        r = 2.0
        val = quartic_cutoff(jnp.array(r), radius=r, scale=1.0)
        # a*r^4 + c*r^2 + 1 where a=1/r^4, c=-2/r^2
        # = 1 - 2 + 1 = 0
        assert jnp.allclose(val, jnp.array(0.0), atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# octic_cutoff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_octic_cutoff_at_zero(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = octic_cutoff(jnp.array(0.0), radius=1.0, scale=1.0)
        # At d=0: scale*(-0 + 0 - 0 + 1) = scale
        assert jnp.allclose(val, jnp.array(1.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_octic_cutoff_at_radius(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        val = octic_cutoff(jnp.array(1.0), radius=1.0, scale=1.0)
        # At d=1: -3 + 8 - 6 + 1 = 0
        assert jnp.allclose(val, jnp.array(0.0), atol=tolerances.atol(1e-5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_octic_cutoff_shape(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.linspace(0.0, 1.0, 20)
        out = octic_cutoff(x, radius=1.0, scale=2.0)
        assert out.shape == (20,)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# dispatch_weighting_fn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(WEIGHTING_FN_REGISTRY.keys()))
def test_dispatch_weighting_fn_all_names(name: str, backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        fn = dispatch_weighting_fn(name, sq_radius=1.0, scale=1.0)
        x = jnp.linspace(0.0, 1.0, 5)
        out = fn(x)
        assert out.shape == (5,)
        assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("backend", BACKENDS)
def test_dispatch_weighting_fn_unknown_raises(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        with pytest.raises(NotImplementedError):
            dispatch_weighting_fn("nonexistent_fn", sq_radius=1.0, scale=1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dispatch_weighting_fn_partial_kwargs(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        fn = dispatch_weighting_fn("half_cos", sq_radius=2.0, scale=3.0)
        val = fn(jnp.array(0.0))
        # half_cos at 0 with scale=3: 3*(0.5*cos(0) + 0.5) = 3.0
        assert jnp.allclose(val, jnp.array(3.0), atol=tolerances.atol(1e-5))
