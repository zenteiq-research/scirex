# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the reusable PDE-term primitives (JAX): AxisSpec,
DiffusionTerm, AdvectionTerm, TransientTerm, ReactionTerm, SourceTerm, and
the generic PDEResidualLoss composite that assembles a PDE residual from a
signed sum of them.

* ``test_axisspec_*`` -- spacing convention (periodic vs. non-periodic) and
  dispatch to FiniteDiff vs. FourierDiff.
* ``test_diffusionterm_*`` / ``test_advectionterm_*`` / etc. -- each term is
  tested standalone via a plain context dict, independent of any specific
  PDE loss (mirrors how FiniteDiff/FourierDiff were tested standalone).
* ``test_advectionterm_upwind_*`` -- upwind vs. central scheme selection,
  including that upwind picks the stencil matching the local velocity sign.
* ``test_pderesidualloss_*`` -- the signed-sum composite, and that
  DarcyEqnLoss's own manufactured-solution/Dirichlet behavior is unaffected
  by being built from terms (regression coverage lives in
  test_equation_losses.py; these tests exercise the composite directly).
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.losses.differentiation import FiniteDiff, FourierDiff
from scirex.losses.equation_losses import PDEResidualLoss
from scirex.losses.pde_terms import (
    AdvectionTerm,
    AxisSpec,
    DiffusionTerm,
    DivergenceTerm,
    GradientTerm,
    ReactionTerm,
    SourceTerm,
    TransientTerm,
)
from tests import tolerances


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
# AxisSpec
# ---------------------------------------------------------------------------


def test_axisspec_periodic_spacing() -> None:
    """Periodic axis: h = measure / n (no duplicated endpoint)."""
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=2.0, periodic=True)
    u = jnp.zeros((1, 8, 1))
    assert axis.spacing(u) == pytest.approx(2.0 / 8)


def test_axisspec_non_periodic_spacing() -> None:
    """Non-periodic axis: h = measure / (n - 1) (both endpoints sampled)."""
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=2.0, periodic=False)
    u = jnp.zeros((1, 9, 1))
    assert axis.spacing(u) == pytest.approx(2.0 / 8)


def test_axisspec_rejects_non_positive_measure() -> None:
    """measure <= 0 would otherwise make spacing() return 0 (or negative),
    which FiniteDiff/FourierDiff then silently turn into inf/nan derivatives
    instead of raising."""
    with pytest.raises(ValueError):
        AxisSpec(axis=1, differentiator=FiniteDiff(), measure=0.0)
    with pytest.raises(ValueError):
        AxisSpec(axis=1, differentiator=FiniteDiff(), measure=-1.0)


def test_axisspec_dispatches_to_fourierdiff() -> None:
    """A FourierDiff-backed axis is spectrally exact, unlike FiniteDiff."""
    n = 32
    h = 1.0 / n
    x = jnp.arange(n) * h
    u = jnp.sin(2 * jnp.pi * x)

    axis = AxisSpec(axis=0, differentiator=FourierDiff(), measure=1.0, periodic=True)
    d = axis.derivative(u, order=1)
    exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
    assert float(jnp.max(jnp.abs(d - exact))) < 1e-3


def test_axisspec_fourierdiff_rejects_non_central_scheme() -> None:
    """FourierDiff has no directional bias -- only 'central' is valid."""
    axis = AxisSpec(axis=0, differentiator=FourierDiff())
    u = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 16))
    with pytest.raises(ValueError):
        axis.derivative(u, order=1, scheme="forward")


def test_axisspec_fourierdiff_rejects_non_periodic() -> None:
    """FourierDiff only supports the periodic spacing convention -- pairing
    it with periodic=False must raise instead of silently using the wrong
    spacing formula."""
    axis = AxisSpec(axis=0, differentiator=FourierDiff(), periodic=False)
    u = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 16))
    with pytest.raises(ValueError):
        axis.derivative(u, order=1)


# ---------------------------------------------------------------------------
# DiffusionTerm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_diffusionterm_matches_manual_flux_form(backend: str) -> None:
    """DiffusionTerm's div(a * grad(u)) matches computing it by hand."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 16
        h = 1.0 / (n - 1)
        key_u, key_a = jax.random.split(jax.random.PRNGKey(0))
        u = jax.random.normal(key_u, (2, n, n))
        a = 1.0 + jax.random.uniform(key_a, (2, n, n))

        fd = FiniteDiff()
        axes = [
            AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False),
            AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False),
        ]
        term = DiffusionTerm(axes=axes, field="u", coeff="a")
        result = term({"u": u, "a": a})

        du_dx = fd.derivative(u, axis=1, order=1, spacing=h, periodic=False)
        du_dy = fd.derivative(u, axis=2, order=1, spacing=h, periodic=False)
        expected = fd.derivative(
            a * du_dx, axis=1, order=1, spacing=h, periodic=False
        ) + fd.derivative(a * du_dy, axis=2, order=1, spacing=h, periodic=False)

        assert jnp.allclose(result, expected)


def test_diffusionterm_constant_coeff() -> None:
    """A constant (non-string) coeff is used directly, not looked up in ctx."""
    n = 8
    h = 1.0 / (n - 1)
    u = jnp.sin(jnp.pi * jnp.arange(n) * h)[None, :]
    fd = FiniteDiff()
    axes = [AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)]
    term = DiffusionTerm(axes=axes, field="u", coeff=2.0)
    result = term({"u": u})

    du_dx = fd.derivative(u, axis=1, order=1, spacing=h, periodic=False)
    expected = fd.derivative(2.0 * du_dx, axis=1, order=1, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_diffusionterm_rejects_repeated_axis() -> None:
    """A repeated axis index would silently double-count that axis's
    divergence contribution instead of raising."""
    fd = FiniteDiff()
    ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    with pytest.raises(AssertionError):
        DiffusionTerm(axes=[ax, ax], field="u", coeff=1.0)


def test_diffusionterm_rejects_collapsed_coeff_shape() -> None:
    """A coeff field reduced/collapsed along an axis (e.g. via an accidental
    jnp.mean) would otherwise silently broadcast against the field's own
    shape instead of raising -- verified with a genuinely non-constant
    coeff so the result would be silently wrong, not just unchecked."""
    n = 8
    u = jnp.sin(jnp.linspace(0.0, 3.14, n))[None, :]
    coeff_collapsed = jnp.mean(
        jnp.linspace(1.0, 2.0, n)[None, :], axis=1, keepdims=True
    )  # shape (1, 1), not (1, 8)
    fd = FiniteDiff()
    ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    term = DiffusionTerm(axes=[ax], field="u", coeff="a")
    with pytest.raises(ValueError):
        term({"u": u, "a": coeff_collapsed})


def test_diffusionterm_order_defaults_to_one() -> None:
    """order isn't passed -- must still behave as the standard 2nd-order
    diffusion operator, matching every other existing DiffusionTerm test."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert DiffusionTerm(axes=[ax]).order == 1


def test_diffusionterm_order_two_matches_manual_computation() -> None:
    """order is forwarded to both the flux and divergence derivative calls
    -- not silently pinned to 1 regardless of what's requested."""
    n = 12
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(9), (1, n))
    coeff = 2.0
    fd = FiniteDiff()
    ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    term = DiffusionTerm(axes=[ax], field="u", coeff=coeff, order=2)
    result = term({"u": u})

    d2u_dx2 = fd.derivative(u, axis=1, order=2, spacing=h, periodic=False)
    expected = fd.derivative(
        coeff * d2u_dx2, axis=1, order=2, spacing=h, periodic=False
    )
    assert jnp.allclose(result, expected)


def test_diffusionterm_scheme_defaults_to_central() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert DiffusionTerm(axes=[ax]).scheme == "central"


def test_diffusionterm_forward_scheme_matches_manual_computation() -> None:
    """scheme is forwarded to both the flux and divergence derivative calls
    -- not silently pinned to central regardless of what's requested."""
    n = 10
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(12), (1, n))
    coeff = 1.5
    fd = FiniteDiff()
    ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    term = DiffusionTerm(axes=[ax], field="u", coeff=coeff, scheme="forward")
    result = term({"u": u})

    du_dx = fd.derivative(
        u, axis=1, order=1, spacing=h, periodic=False, scheme="forward"
    )
    expected = fd.derivative(
        coeff * du_dx, axis=1, order=1, spacing=h, periodic=False, scheme="forward"
    )
    assert jnp.allclose(result, expected)
    # a directional scheme must actually change the result vs central
    central = DiffusionTerm(axes=[ax], field="u", coeff=coeff)({"u": u})
    assert not jnp.allclose(result, central)


def test_diffusionterm_rejects_bad_scheme() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        DiffusionTerm(axes=[ax], scheme="upwind")


def test_diffusionterm_rejects_directional_scheme_with_order_two() -> None:
    """order=2 only combines with scheme="central" -- must be caught at
    construction time, not only on the first __call__ inside FiniteDiff."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        DiffusionTerm(axes=[ax], order=2, scheme="forward")


# ---------------------------------------------------------------------------
# AdvectionTerm
# ---------------------------------------------------------------------------


def test_advectionterm_self_advection_central() -> None:
    """velocity=None means self-advection: u * du/dx."""
    n = 16
    h = 1.0 / n
    u = jnp.sin(2 * jnp.pi * jnp.arange(n) * h)[None, :]
    axes = [AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=True)]
    term = AdvectionTerm(axes=axes, field="u", velocity=None, scheme="central")
    result = term({"u": u})

    du_dx = FiniteDiff().derivative(u, axis=1, order=1, spacing=h, periodic=True)
    assert jnp.allclose(result, u * du_dx)


def test_advectionterm_external_velocity_field() -> None:
    """velocity as a ctx key advects `field`, not itself."""
    n = 16
    h = 1.0 / n
    x = jnp.arange(n) * h
    u = jnp.sin(2 * jnp.pi * x)[None, :]
    c = jnp.full((1, n), 3.0)
    axes = [AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=True)]
    term = AdvectionTerm(axes=axes, field="u", velocity="c", scheme="central")
    result = term({"u": u, "c": c})

    du_dx = FiniteDiff().derivative(u, axis=1, order=1, spacing=h, periodic=True)
    assert jnp.allclose(result, c * du_dx)


def test_advectionterm_upwind_picks_stencil_by_velocity_sign() -> None:
    """upwind uses backward where velocity>=0, forward where velocity<0."""
    n = 8
    h = 1.0 / n
    u = jnp.arange(n, dtype=jnp.float32)[None, :]
    axes = [AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=True)]

    # constant positive velocity -> pure backward difference everywhere
    v_pos = jnp.ones((1, n))
    result_pos = AdvectionTerm(axes=axes, field="u", velocity="v", scheme="upwind")(
        {"u": u, "v": v_pos}
    )
    backward = FiniteDiff().derivative(
        u, axis=1, order=1, spacing=h, periodic=True, scheme="backward"
    )
    assert jnp.allclose(result_pos, v_pos * backward)

    # constant negative velocity -> pure forward difference everywhere
    v_neg = -jnp.ones((1, n))
    result_neg = AdvectionTerm(axes=axes, field="u", velocity="v", scheme="upwind")(
        {"u": u, "v": v_neg}
    )
    forward = FiniteDiff().derivative(
        u, axis=1, order=1, spacing=h, periodic=True, scheme="forward"
    )
    assert jnp.allclose(result_neg, v_neg * forward)


def test_advectionterm_2d_per_axis_velocity_components() -> None:
    """A sequence of ctx keys supplies one velocity component per axis."""
    n = 8
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(1), (1, n, n))
    v_x = jnp.full((1, n, n), 2.0)
    v_y = jnp.full((1, n, n), -1.0)
    fd = FiniteDiff()
    axes = [
        AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False),
        AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False),
    ]
    term = AdvectionTerm(axes=axes, field="u", velocity=("vx", "vy"), scheme="central")
    result = term({"u": u, "vx": v_x, "vy": v_y})

    du_dx = fd.derivative(u, axis=1, order=1, spacing=h, periodic=False)
    du_dy = fd.derivative(u, axis=2, order=1, spacing=h, periodic=False)
    assert jnp.allclose(result, v_x * du_dx + v_y * du_dy)


def test_advectionterm_rejects_bad_scheme() -> None:
    """scheme must be one of central/upwind."""
    axes = [AxisSpec(axis=1, differentiator=FiniteDiff())]
    with pytest.raises(ValueError):
        AdvectionTerm(axes=axes, scheme="downwind")


def test_advectionterm_rejects_upwind_with_order_not_one() -> None:
    """scheme="upwind" only combines with order=1 -- must be caught at
    construction time, not only on the first __call__ inside FiniteDiff."""
    axes = [AxisSpec(axis=1, differentiator=FiniteDiff())]
    with pytest.raises(ValueError):
        AdvectionTerm(axes=axes, scheme="upwind", order=2)


def test_advectionterm_rejects_repeated_axis() -> None:
    """A repeated axis index would silently double-count that axis's
    contribution instead of raising."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(AssertionError):
        AdvectionTerm(axes=[ax, ax])


def test_advectionterm_rejects_collapsed_velocity_shape() -> None:
    """A velocity field reduced/collapsed along an axis would otherwise
    silently broadcast against the field's own shape instead of raising."""
    n = 8
    u = jnp.sin(jnp.linspace(0.0, 3.14, n))[None, :]
    v_collapsed = jnp.mean(
        jnp.linspace(0.5, 1.5, n)[None, :], axis=1, keepdims=True
    )  # shape (1, 1), not (1, 8)
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = AdvectionTerm(axes=[ax], field="u", velocity="v")
    with pytest.raises(ValueError):
        term({"u": u, "v": v_collapsed})


def test_advectionterm_order_defaults_to_one() -> None:
    """order isn't passed -- must still behave as first-order advection,
    matching every other existing AdvectionTerm test."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert AdvectionTerm(axes=[ax]).order == 1


def test_advectionterm_order_two_matches_manual_second_derivative() -> None:
    """order is forwarded straight through to the axis's derivative call --
    not silently pinned to 1 regardless of what's requested."""
    n = 9
    h = 1.0 / (n - 1)
    x = jnp.linspace(0.0, 1.0, n)
    u = (x**3)[None, :]  # d^2u/dx^2 = 6x
    v = jnp.full((1, n), 2.0)
    fd = FiniteDiff()
    ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    term = AdvectionTerm(axes=[ax], field="u", velocity="v", scheme="central", order=2)
    result = term({"u": u, "v": v})

    d2u_dx2 = fd.derivative(u, axis=1, order=2, spacing=h, periodic=False)
    assert jnp.allclose(result, v * d2u_dx2)


# ---------------------------------------------------------------------------
# GradientTerm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradientterm_matches_manual_derivative(backend: str) -> None:
    """GradientTerm's d(field)/d(axis) matches calling FiniteDiff by hand."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 12
        h = 1.0 / (n - 1)
        p = jax.random.normal(jax.random.PRNGKey(2), (2, n, n))
        fd = FiniteDiff()
        ax = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
        term = GradientTerm(axis=ax, field="p")
        result = term({"p": p})

        expected = fd.derivative(p, axis=1, order=1, spacing=h, periodic=False)
        assert jnp.allclose(result, expected)


def test_gradientterm_honors_axis_index() -> None:
    """A gradient along axis=2 must differ from axis=1 for a field that
    actually varies differently along each axis -- catches an
    axis-index bug that would silently differentiate the wrong axis."""
    n = 10
    h = 1.0 / (n - 1)
    x = jnp.arange(n) * h
    y = jnp.arange(n) * h
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    p = (X**2 + 3.0 * Y)[None, :, :]  # dp/dx = 2x, dp/dy = 3 (different)

    fd = FiniteDiff()
    ax_x = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    ax_y = AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False)

    grad_x = GradientTerm(axis=ax_x, field="p")({"p": p})
    grad_y = GradientTerm(axis=ax_y, field="p")({"p": p})

    assert not jnp.allclose(grad_x, grad_y)
    assert jnp.allclose(grad_x[:, 2:-2, 2:-2], (2 * X)[None, 2:-2, 2:-2], atol=1e-2)
    assert jnp.allclose(grad_y[:, 2:-2, 2:-2], 3.0, atol=tolerances.atol(1e-6))


def test_gradientterm_default_field_is_u() -> None:
    """The default `field` argument reads ctx["u"], matching every other
    term's default."""
    n = 8
    u = jnp.arange(n, dtype=jnp.float32)[None, :]
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=True)
    result = GradientTerm(axis=ax)({"u": u})
    expected = FiniteDiff().derivative(
        u, axis=1, order=1, spacing=1.0 / n, periodic=True
    )
    assert jnp.allclose(result, expected)


def test_gradientterm_supports_fourierdiff_axis() -> None:
    """GradientTerm must dispatch through AxisSpec generically -- it should
    work with a spectral (FourierDiff) axis just as well as FiniteDiff,
    since it never calls a differentiator directly itself."""
    n = 32
    h = 1.0 / n
    x = jnp.arange(n) * h
    p = jnp.sin(2 * jnp.pi * x)[None, :]
    ax = AxisSpec(axis=1, differentiator=FourierDiff(), measure=1.0, periodic=True)
    result = GradientTerm(axis=ax, field="p")({"p": p})
    exact = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)[None, :]
    assert float(jnp.max(jnp.abs(result - exact))) < 1e-3


def test_gradientterm_matches_transientterm_for_same_axis_and_field() -> None:
    """GradientTerm and TransientTerm are the same underlying operation
    (single-axis, single-field, first derivative) -- they must agree
    exactly for the same axis/field, only their PDE-role name differs.
    Locks in that equivalence so a future change to one doesn't silently
    drift from the other."""
    n = 10
    t = jnp.linspace(0.0, 1.0, n)
    u = (t**3)[None, :]
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert jnp.allclose(
        GradientTerm(axis=ax, field="u")({"u": u}),
        TransientTerm(axis=ax, field="u")({"u": u}),
    )


def test_gradientterm_order_two_matches_manual_second_derivative() -> None:
    """order isn't hardcoded to 1 -- e.g. a Laplacian-component term reuses
    this same building block via order=2."""
    n = 9
    h = 1.0 / (n - 1)
    x = jnp.arange(n) * h
    p = (x**3)[None, :]  # d^2p/dx^2 = 6x
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = GradientTerm(axis=ax, field="p", order=2)
    result = term({"p": p})
    expected = FiniteDiff().derivative(p, axis=1, order=2, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_gradientterm_scheme_defaults_to_central() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert GradientTerm(axis=ax).scheme == "central"


def test_gradientterm_forward_scheme_matches_manual_computation() -> None:
    """scheme isn't hardcoded to central -- forwarded straight through to
    axis.derivative."""
    n = 8
    h = 1.0 / (n - 1)
    p = jax.random.normal(jax.random.PRNGKey(13), (1, n))
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    result = GradientTerm(axis=ax, field="p", scheme="forward")({"p": p})
    expected = FiniteDiff().derivative(
        p, axis=1, order=1, spacing=h, periodic=False, scheme="forward"
    )
    assert jnp.allclose(result, expected)


def test_gradientterm_rejects_bad_scheme() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        GradientTerm(axis=ax, scheme="upwind")


def test_gradientterm_rejects_directional_scheme_with_order_two() -> None:
    """order=2 only combines with scheme="central" -- must be caught at
    construction time, not only on the first __call__ inside FiniteDiff."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        GradientTerm(axis=ax, order=2, scheme="forward")


# ---------------------------------------------------------------------------
# DivergenceTerm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_divergenceterm_matches_manual_sum(backend: str) -> None:
    """div(velocity) = u_x + v_y matches computing it by hand."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 12
        h = 1.0 / (n - 1)
        key_u, key_v = jax.random.split(jax.random.PRNGKey(3))
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        fd = FiniteDiff()
        axes = [
            AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False),
            AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False),
        ]
        term = DivergenceTerm(fields=["u", "v"], axes=axes)
        result = term({"u": u, "v": v})

        du_dx = fd.derivative(u, axis=1, order=1, spacing=h, periodic=False)
        dv_dy = fd.derivative(v, axis=2, order=1, spacing=h, periodic=False)
        assert jnp.allclose(result, du_dx + dv_dy)


def test_divergenceterm_three_components() -> None:
    """Generalizes beyond 2D: a 3-component field (u, v, w) over 3 axes
    sums all three contributions, not just the first two."""
    n = 8
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(4), (1, n, n, n))
    v = jax.random.normal(jax.random.PRNGKey(5), (1, n, n, n))
    w = jax.random.normal(jax.random.PRNGKey(6), (1, n, n, n))
    fd = FiniteDiff()
    axes = [
        AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False),
        AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False),
        AxisSpec(axis=3, differentiator=fd, measure=1.0, periodic=False),
    ]
    term = DivergenceTerm(fields=["u", "v", "w"], axes=axes)
    result = term({"u": u, "v": v, "w": w})

    expected = (
        fd.derivative(u, axis=1, order=1, spacing=h, periodic=False)
        + fd.derivative(v, axis=2, order=1, spacing=h, periodic=False)
        + fd.derivative(w, axis=3, order=1, spacing=h, periodic=False)
    )
    assert jnp.allclose(result, expected)


def test_divergenceterm_single_component_reduces_to_one_derivative() -> None:
    """The degenerate 1-field/1-axis case must still work (an empty-loop or
    off-by-one bug in the sum would show up here first)."""
    n = 10
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(7), (1, n))
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    result = DivergenceTerm(fields=["u"], axes=[ax])({"u": u})
    expected = FiniteDiff().derivative(u, axis=1, order=1, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_divergenceterm_field_axis_pairing_is_positional() -> None:
    """fields[i] must pair with axes[i] -- swapping which field is paired
    with which axis must change the result, catching a bug where the
    fields/axes zip is silently mismatched."""
    n = 10
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 1.0, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u = (X**2)[None, :, :]  # d/dx = 2x, d/dy = 0
    v = (Y**3)[None, :, :]  # d/dx = 0, d/dy = 3y^2

    fd = FiniteDiff()
    ax_x = AxisSpec(axis=1, differentiator=fd, measure=1.0, periodic=False)
    ax_y = AxisSpec(axis=2, differentiator=fd, measure=1.0, periodic=False)

    correct = DivergenceTerm(fields=["u", "v"], axes=[ax_x, ax_y])({"u": u, "v": v})
    swapped = DivergenceTerm(fields=["v", "u"], axes=[ax_x, ax_y])({"u": u, "v": v})
    assert not jnp.allclose(correct, swapped)


def test_divergenceterm_rejects_empty_axes() -> None:
    """An empty fields/axes pair would silently give a divergence of 0.0
    everywhere instead of raising."""
    with pytest.raises(AssertionError):
        DivergenceTerm(fields=[], axes=[])


def test_divergenceterm_rejects_mismatched_fields_axes_length() -> None:
    """A different number of fields and axes has no well-defined pairing --
    must raise instead of silently truncating via zip."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(AssertionError):
        DivergenceTerm(fields=["u", "v"], axes=[ax])


def test_divergenceterm_rejects_repeated_axis() -> None:
    """A repeated axis index would silently double-count that axis's
    contribution instead of raising."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(AssertionError):
        DivergenceTerm(fields=["u", "v"], axes=[ax, ax])


def test_divergenceterm_rejects_collapsed_field_shape() -> None:
    """A component field reduced/collapsed along an axis would otherwise
    silently broadcast against the reference (first) field's shape instead
    of raising."""
    n = 8
    u = jnp.sin(jnp.linspace(0.0, 3.14, n))[None, :, None] * jnp.ones((1, n, n))
    v_collapsed = jnp.mean(
        jnp.linspace(0.5, 1.5, n)[None, None, :] * jnp.ones((1, n, n)),
        axis=2,
        keepdims=True,
    )  # shape (1, n, 1), not (1, n, n)
    ax_x = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    ax_y = AxisSpec(axis=2, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = DivergenceTerm(fields=["u", "v"], axes=[ax_x, ax_y])
    with pytest.raises(ValueError):
        term({"u": u, "v": v_collapsed})


def test_divergenceterm_order_two_matches_manual_computation() -> None:
    """order isn't hardcoded to 1 -- forwarded to each component's
    derivative call."""
    n = 9
    h = 1.0 / (n - 1)
    x = jnp.arange(n) * h
    u = (x**3)[None, :]  # d^2u/dx^2 = 6x
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = DivergenceTerm(fields=["u"], axes=[ax], order=2)
    result = term({"u": u})
    expected = FiniteDiff().derivative(u, axis=1, order=2, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_divergenceterm_scheme_defaults_to_central() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    assert DivergenceTerm(fields=["u"], axes=[ax]).scheme == "central"


def test_divergenceterm_forward_scheme_matches_manual_computation() -> None:
    """scheme isn't hardcoded to central -- forwarded to each component's
    derivative call."""
    n = 8
    h = 1.0 / (n - 1)
    u = jax.random.normal(jax.random.PRNGKey(14), (1, n))
    ax = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = DivergenceTerm(fields=["u"], axes=[ax], scheme="forward")
    result = term({"u": u})
    expected = FiniteDiff().derivative(
        u, axis=1, order=1, spacing=h, periodic=False, scheme="forward"
    )
    assert jnp.allclose(result, expected)


def test_divergenceterm_rejects_bad_scheme() -> None:
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        DivergenceTerm(fields=["u"], axes=[ax], scheme="upwind")


def test_divergenceterm_rejects_directional_scheme_with_order_two() -> None:
    """order=2 only combines with scheme="central" -- must be caught at
    construction time, not only on the first __call__ inside FiniteDiff."""
    ax = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        DivergenceTerm(fields=["u"], axes=[ax], order=2, scheme="forward")


# ---------------------------------------------------------------------------
# TransientTerm / ReactionTerm / SourceTerm
# ---------------------------------------------------------------------------


def test_transientterm_matches_manual_derivative() -> None:
    n = 9
    h = 1.0 / (n - 1)
    t = jnp.arange(n) * h
    u = (t**2)[None, :]  # du/dt = 2t
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = TransientTerm(axis=axis, field="u")
    result = term({"u": u})
    expected = FiniteDiff().derivative(u, axis=1, order=1, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_transientterm_order_two_matches_manual_second_derivative() -> None:
    """order isn't hardcoded to 1 -- differentiating twice in time (e.g. a
    wave equation's u_tt) must be reachable without a new class."""
    n = 9
    h = 1.0 / (n - 1)
    t = jnp.arange(n) * h
    u = (t**3)[None, :]  # d^2u/dt^2 = 6t
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    term = TransientTerm(axis=axis, field="u", order=2)
    result = term({"u": u})
    expected = FiniteDiff().derivative(u, axis=1, order=2, spacing=h, periodic=False)
    assert jnp.allclose(result, expected)


def test_transientterm_scheme_defaults_to_central() -> None:
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    assert TransientTerm(axis=axis).scheme == "central"


def test_transientterm_forward_scheme_matches_manual_computation() -> None:
    """scheme isn't hardcoded to central -- forwarded straight through to
    axis.derivative."""
    n = 9
    h = 1.0 / (n - 1)
    t = jax.random.normal(jax.random.PRNGKey(15), (1, n))
    axis = AxisSpec(axis=1, differentiator=FiniteDiff(), measure=1.0, periodic=False)
    result = TransientTerm(axis=axis, field="u", scheme="forward")({"u": t})
    expected = FiniteDiff().derivative(
        t, axis=1, order=1, spacing=h, periodic=False, scheme="forward"
    )
    assert jnp.allclose(result, expected)


def test_transientterm_rejects_bad_scheme() -> None:
    axis = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        TransientTerm(axis=axis, scheme="upwind")


def test_transientterm_rejects_directional_scheme_with_order_two() -> None:
    """order=2 only combines with scheme="central" -- must be caught at
    construction time, not only on the first __call__ inside FiniteDiff."""
    axis = AxisSpec(axis=1, differentiator=FiniteDiff())
    with pytest.raises(ValueError):
        TransientTerm(axis=axis, order=2, scheme="forward")


def test_reactionterm_constant_and_field_coeff() -> None:
    u = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(ReactionTerm(field="u", coeff=2.0)({"u": u}), 2.0 * u)

    r = jnp.array([1.0, 0.0, -1.0])
    assert jnp.allclose(ReactionTerm(field="u", coeff="r")({"u": u, "r": r}), r * u)


def test_reactionterm_rejects_collapsed_coeff_shape() -> None:
    """A coeff field reduced/collapsed to a different shape than the field
    would otherwise silently broadcast instead of raising."""
    u = jnp.array([1.0, 2.0, 3.0])
    r_collapsed = jnp.array([1.0, 0.0])  # shape (2,), not (3,)
    term = ReactionTerm(field="u", coeff="r")
    with pytest.raises(ValueError):
        term({"u": u, "r": r_collapsed})


def test_sourceterm_constant_and_field() -> None:
    assert SourceTerm(1.0)({}) == 1.0
    f = jnp.array([1.0, 2.0])
    assert jnp.allclose(SourceTerm("f")({"f": f}), f)
    assert SourceTerm()({}) == 0.0  # default is 0.0


# ---------------------------------------------------------------------------
# PDEResidualLoss
# ---------------------------------------------------------------------------


def test_pderesidualloss_sums_signed_terms() -> None:
    """residual = sum(sign * term(ctx)) for every active term."""
    u = jnp.array([2.0, 4.0])
    loss = PDEResidualLoss(
        terms={
            "a": (1.0, lambda ctx: ctx["u"]),
            "b": (-2.0, lambda ctx: ctx["u"] + 1.0),
        }
    )
    result = loss.residual({"u": u})
    # 1*u - 2*(u+1) = -u - 2
    assert jnp.allclose(result, -u - 2.0)


def test_pderesidualloss_abs_reduction_and_take_root() -> None:
    u = jnp.array([3.0, 4.0])
    loss_mean = PDEResidualLoss(
        terms={"id": (1.0, lambda ctx: ctx["u"])}, reduction="mean"
    )
    loss_sum = PDEResidualLoss(
        terms={"id": (1.0, lambda ctx: ctx["u"])}, reduction="sum"
    )

    # squared residuals are 9, 16 -> mean=12.5, sum=25
    assert float(loss_mean.abs({"u": u}, take_root=False)) == pytest.approx(12.5)
    assert float(loss_sum.abs({"u": u}, take_root=False)) == pytest.approx(25.0)
    assert float(loss_mean.abs({"u": u}, take_root=True)) == pytest.approx(12.5**0.5)


def test_pderesidualloss_rejects_bad_reduction() -> None:
    with pytest.raises(AssertionError):
        PDEResidualLoss(terms={"id": (1.0, lambda ctx: ctx["u"])}, reduction="median")


def test_pderesidualloss_rejects_empty_terms() -> None:
    """An empty terms dict would silently give a residual of 0.0 everywhere
    (a loss that always reports the PDE as satisfied) -- must raise instead."""
    with pytest.raises(ValueError):
        PDEResidualLoss(terms={})


def test_pderesidualloss_rejects_mismatched_array_term_shapes() -> None:
    """Two array-valued terms with different shapes must raise -- silently
    broadcasting them together (e.g. (2, 8) against (8,)) almost always
    means one term's output has the wrong shape."""
    loss = PDEResidualLoss(
        terms={
            "a": (1.0, lambda ctx: jnp.ones((2, 8))),
            "b": (1.0, lambda ctx: jnp.ones((8,))),
        }
    )
    with pytest.raises(ValueError):
        loss.residual({})


def test_pderesidualloss_rejects_term_collapsed_to_0d_array() -> None:
    """A term whose array output has been fully reduced/collapsed down to a
    0-d array must still raise against a correctly-shaped term -- a 0-d
    jnp array is not the same thing as a genuine Python float, and treating
    it as an always-allowed scalar broadcast was the actual loophole here:
    it broadcasts against any shape with no error, same as any other
    mismatched array shape."""
    loss = PDEResidualLoss(
        terms={
            "array": (1.0, lambda ctx: jnp.ones((8,))),
            "collapsed": (1.0, lambda ctx: jnp.mean(jnp.arange(8.0))),  # 0-d
        }
    )
    with pytest.raises(ValueError):
        loss.residual({})


def test_pderesidualloss_allows_scalar_term_to_broadcast_against_array() -> None:
    """A scalar (constant) term is allowed to broadcast against an
    array-valued term -- e.g. DarcyEqnLoss's constant SourceTerm against
    DiffusionTerm's array output. Only array-vs-array mismatches must raise."""
    u = jnp.array([2.0, 4.0])
    loss = PDEResidualLoss(
        terms={
            "array": (1.0, lambda ctx: ctx["u"]),
            "scalar": (-1.0, lambda ctx: 1.0),
        }
    )
    result = loss.residual({"u": u})
    assert jnp.allclose(result, u - 1.0)


def test_pderesidualloss_name() -> None:
    term = {"id": (1.0, lambda ctx: ctx["u"])}
    assert PDEResidualLoss(terms=term, name="Foo").name == "Foo"
    assert PDEResidualLoss(terms=term).name == "PDEResidualLoss"
