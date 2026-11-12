# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the grid-based differentiation primitives FiniteDiff and
FourierDiff (JAX).

Both primitives are axis-agnostic (any array rank, differentiate along a
chosen ``axis``), so what's actually exercised here is 1D FiniteDiff --
periodic and non-periodic, orders 1 and 2, central/forward/backward schemes,
plus one batched/multi-channel case confirming the axis-generality -- and
FourierDiff exactness at a fixed resolution, with one explicit 2D
(arbitrary-axis) case. There is currently no dedicated 2D/3D FiniteDiff test,
no 3D FourierDiff test, and no laplacian/gradient/divergence/curl-combination
test in this file; combinations like a Laplacian are instead exercised where
they're actually built, in ``test_pde_terms.py`` (``DiffusionTerm``).
``test_nonuniform_fd``-equivalent coverage (meshless/scattered-point finite
differences) has no analogue here either -- deferred to a later phase
(irregular-geometry point-cloud models); see the documented skip in
``test_data_losses.py``.

Following the Phase 1 ``layers/tests`` structure, the numeric tests are
parametrized over the available CPU/GPU backends.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.losses.differentiation import FiniteDiff, FourierDiff


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _observed_order(err_coarse: float, err_fine: float, refinement: int = 2) -> float:
    """Observed order of accuracy from two grid-refinement error samples."""
    return float(jnp.log(err_coarse / err_fine) / jnp.log(refinement))


# ---------------------------------------------------------------------------
# test_finite_diff -- periodic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order", [1, 2])
def test_finite_diff_periodic_converges_at_second_order(
    backend: str, order: int
) -> None:
    """FiniteDiff (periodic) converges at 2nd order for sin(2*pi*x)."""
    with jax.default_device(jax.devices(backend)[0]):
        fd = FiniteDiff()

        def error(n: int) -> float:
            h = 1.0 / n
            x = jnp.arange(n) * h
            u = jnp.sin(2 * jnp.pi * x)
            d = fd.derivative(u, axis=0, order=order, spacing=h, periodic=True)
            if order == 1:
                exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
            else:
                exact = -((2 * jnp.pi) ** 2) * jnp.sin(2 * jnp.pi * x)
            return float(jnp.max(jnp.abs(d - exact)))

        err_coarse = error(32)
        err_fine = error(64)
        assert _observed_order(err_coarse, err_fine) == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_diff_periodic_batched_and_arbitrary_axis(backend: str) -> None:
    """FiniteDiff generalizes across a non-last axis of a batched tensor."""
    with jax.default_device(jax.devices(backend)[0]):
        fd = FiniteDiff()
        n = 64
        h = 1.0 / n
        x = jnp.arange(n) * h
        u_1d = jnp.sin(2 * jnp.pi * x)

        # (batch, n, channels): differentiate along axis=1
        u = jnp.tile(u_1d[None, :, None], (3, 1, 5))
        d = fd.derivative(u, axis=1, order=1, spacing=h, periodic=True)

        exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
        for b in range(3):
            for c in range(5):
                assert jnp.max(jnp.abs(d[b, :, c] - exact)) < 0.05
        # shape is preserved
        assert d.shape == u.shape


# ---------------------------------------------------------------------------
# test_finite_diff -- non-periodic
# ---------------------------------------------------------------------------


@pytest.mark.needs_x64
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order", [1, 2])
def test_finite_diff_non_periodic_converges_at_second_order(
    backend: str, order: int
) -> None:
    """FiniteDiff (non-periodic) converges at 2nd order, including boundary points."""
    with jax.default_device(jax.devices(backend)[0]):
        fd = FiniteDiff()

        def error(n: int) -> float:
            h = 1.0 / (n - 1)
            x = jnp.arange(n) * h
            # sin(pi*x): zero at both endpoints, matching a Dirichlet field
            u = jnp.sin(jnp.pi * x)
            d = fd.derivative(u, axis=0, order=order, spacing=h, periodic=False)
            if order == 1:
                exact = jnp.pi * jnp.cos(jnp.pi * x)
            else:
                exact = -(jnp.pi**2) * jnp.sin(jnp.pi * x)
            return float(jnp.max(jnp.abs(d - exact)))

        err_coarse = error(65)
        err_fine = error(129)
        assert _observed_order(err_coarse, err_fine) == pytest.approx(2.0, abs=0.3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("periodic", [True, False])
def test_finite_diff_directional_converges_at_first_order(
    backend: str, periodic: bool
) -> None:
    """forward/backward schemes converge at 1st order, away from the boundary
    fallback point (which uses the other one-sided difference instead)."""
    with jax.default_device(jax.devices(backend)[0]):
        fd = FiniteDiff()

        def error(n: int, scheme: str) -> float:
            if periodic:
                h = 1.0 / n
                x = jnp.arange(n) * h
            else:
                h = 1.0 / (n - 1)
                x = jnp.arange(n) * h
            u = jnp.sin(2 * jnp.pi * x)
            d = fd.derivative(
                u, axis=0, order=1, spacing=h, periodic=periodic, scheme=scheme
            )
            exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
            err = jnp.abs(d - exact)
            if not periodic:
                # exclude both boundary points: whichever one falls back to
                # the *other* one-sided difference isn't testing this scheme
                err = err[1:-1]
            return float(jnp.max(err))

        for scheme in ("forward", "backward"):
            err_coarse = error(64, scheme)
            err_fine = error(128, scheme)
            assert err_fine < err_coarse
            assert _observed_order(err_coarse, err_fine) == pytest.approx(1.0, abs=0.3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_finite_diff_directional_periodic_matches_hand_computation(
    backend: str,
) -> None:
    """forward/backward on a periodic axis match a direct jnp.roll computation."""
    with jax.default_device(jax.devices(backend)[0]):
        fd = FiniteDiff()
        u = jnp.array([1.0, 3.0, 6.0, 2.0])
        h = 0.5
        fwd = fd.derivative(
            u, axis=0, order=1, spacing=h, periodic=True, scheme="forward"
        )
        bwd = fd.derivative(
            u, axis=0, order=1, spacing=h, periodic=True, scheme="backward"
        )
        assert jnp.allclose(fwd, (jnp.roll(u, -1) - u) / h)
        assert jnp.allclose(bwd, (u - jnp.roll(u, 1)) / h)


def test_finite_diff_directional_non_periodic_falls_back_at_boundary() -> None:
    """At the domain edge, the missing-neighbor direction falls back to the
    other one-sided difference instead of reading out of bounds."""
    fd = FiniteDiff()
    u = jnp.array([1.0, 3.0, 6.0, 2.0, 5.0])
    h = 0.5

    fwd = fd.derivative(u, axis=0, order=1, spacing=h, periodic=False, scheme="forward")
    bwd = fd.derivative(
        u, axis=0, order=1, spacing=h, periodic=False, scheme="backward"
    )

    # forward at the last point has no i+1 -> falls back to backward there
    assert fwd[-1] == pytest.approx((u[-1] - u[-2]) / h)
    # backward at the first point has no i-1 -> falls back to forward there
    assert bwd[0] == pytest.approx((u[1] - u[0]) / h)
    # everywhere else, forward/backward are the plain one-sided differences
    assert jnp.allclose(fwd[:-1], (u[1:] - u[:-1]) / h)
    assert jnp.allclose(bwd[1:], (u[1:] - u[:-1]) / h)


def test_finite_diff_rejects_bad_scheme() -> None:
    """scheme must be one of central/forward/backward."""
    fd = FiniteDiff()
    u = jnp.arange(8.0)
    with pytest.raises(ValueError):
        fd.derivative(u, axis=0, order=1, spacing=1.0, scheme="downwind")


def test_finite_diff_rejects_directional_scheme_for_order_2() -> None:
    """forward/backward are only defined for order=1 (diffusion doesn't upwind)."""
    fd = FiniteDiff()
    u = jnp.arange(8.0)
    with pytest.raises(ValueError):
        fd.derivative(u, axis=0, order=2, spacing=1.0, scheme="forward")


def test_finite_diff_non_periodic_rejects_too_few_points() -> None:
    """FiniteDiff (non-periodic) needs at least 4 points along `axis`."""
    fd = FiniteDiff()
    u = jnp.arange(3.0)
    with pytest.raises(ValueError):
        fd.derivative(u, axis=0, order=1, spacing=1.0, periodic=False)


def test_finite_diff_rejects_unsupported_order() -> None:
    """FiniteDiff only supports order in (1, 2)."""
    fd = FiniteDiff()
    u = jnp.arange(8.0)
    with pytest.raises(ValueError):
        fd.derivative(u, axis=0, order=3, spacing=1.0, periodic=True)


# ---------------------------------------------------------------------------
# test_fourier_diff
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("order", [1, 2])
def test_fourier_diff_exact_at_fixed_coarse_resolution(
    backend: str, order: int
) -> None:
    """FourierDiff is accurate to ~machine precision at a fixed, coarse n.

    Unlike FiniteDiff, spectral differentiation of a smooth periodic
    function does not need grid refinement to become accurate -- it should
    already be near-exact at a modest resolution.
    """
    with jax.default_device(jax.devices(backend)[0]):
        fdiff = FourierDiff()
        n = 32
        h = 1.0 / n
        x = jnp.arange(n) * h
        u = jnp.sin(2 * jnp.pi * x)

        d = fdiff.derivative(u, axis=0, order=order, spacing=h)
        if order == 1:
            exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
        else:
            exact = -((2 * jnp.pi) ** 2) * jnp.sin(2 * jnp.pi * x)

        assert float(jnp.max(jnp.abs(d - exact))) < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_fourier_diff_2d_arbitrary_axis(backend: str) -> None:
    """FourierDiff generalizes to a chosen axis of a 2D periodic field."""
    with jax.default_device(jax.devices(backend)[0]):
        fdiff = FourierDiff()
        n = 32
        h = 1.0 / n
        x = jnp.arange(n) * h
        xx, yy = jnp.meshgrid(x, x, indexing="ij")
        u = jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)

        du_dx = fdiff.derivative(u, axis=0, order=1, spacing=h)
        exact_dx = 2 * jnp.pi * jnp.cos(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)
        assert float(jnp.max(jnp.abs(du_dx - exact_dx))) < 1e-3

        du_dy = fdiff.derivative(u, axis=1, order=1, spacing=h)
        exact_dy = -2 * jnp.pi * jnp.sin(2 * jnp.pi * xx) * jnp.sin(2 * jnp.pi * yy)
        assert float(jnp.max(jnp.abs(du_dy - exact_dy))) < 1e-3

        # shape is preserved
        assert du_dx.shape == u.shape
        assert du_dy.shape == u.shape


def test_fourier_diff_rejects_negative_order() -> None:
    """FourierDiff requires order >= 0."""
    fdiff = FourierDiff()
    u = jnp.arange(8.0)
    with pytest.raises(ValueError):
        fdiff.derivative(u, axis=0, order=-1, spacing=1.0)


def test_fourier_diff_rejects_integer_dtype() -> None:
    """An integer-dtype input must be rejected, not silently truncated:
    the result is cast back to u's dtype, so an int dtype would otherwise
    truncate the derivative to integers with no warning."""
    fdiff = FourierDiff()
    u = jnp.arange(8, dtype=jnp.int32)
    with pytest.raises(ValueError):
        fdiff.derivative(u, axis=0)


def test_fourier_diff_gradient_matches_per_axis_derivative() -> None:
    """gradient() over multiple axes agrees with individual derivative() calls."""
    fdiff = FourierDiff()
    n = 16
    h = 1.0 / n
    x = jnp.arange(n) * h
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    u = jnp.sin(2 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy)

    du_dx, du_dy = fdiff.gradient(u, axes=(0, 1), spacing=h)
    assert jnp.allclose(du_dx, fdiff.derivative(u, axis=0, order=1, spacing=h))
    assert jnp.allclose(du_dy, fdiff.derivative(u, axis=1, order=1, spacing=h))
