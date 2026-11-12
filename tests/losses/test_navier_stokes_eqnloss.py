# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the PDE-residual (physics-informed) loss NavierStokesEqnLoss
(JAX) -- steady, incompressible 2D Navier-Stokes in primitive
(velocity-pressure) form, the lid-driven-cavity benchmark.

* ``test_navier_stokes_eqnloss_residual_converges_with_manufactured_solution``
  -- a manufactured, divergence-free ``(u, v)`` pair (Taylor-Green-like) plus
  a smooth ``p``, giving closed-form momentum/continuity residuals for the
  *continuous* PDE at a given Reynolds number, isolated from Dirichlet/pin
  enforcement (``enforce_dirichlet=False``, ``pin_pressure=False``) so it
  purely tests discretization accuracy of the four terms
  (``AdvectionTerm``, ``GradientTerm``, ``DiffusionTerm``, ``DivergenceTerm``)
  wired together.
* ``test_navier_stokes_eqnloss_dirichlet_*`` / ``*_pin_*`` -- the hard
  boundary-condition and pressure-pinning behavior discussed alongside the
  implementation: the residual is invariant to the predicted field's
  boundary/pinned values, and (checked via ``jax.grad``) no gradient flows
  back to the model's own prediction at those points.
* ``test_navier_stokes_eqnloss_weights_*`` -- the three coupled equations
  are combined by a per-equation weighted sum, not silently averaged.
* Shape/argument validation and the ``__call__`` / ``Trainer`` calling
  convention (``loss(out, **sample)``, ignoring ``x`` and ``y``).
* ``test_navier_stokes_eqnloss_residual_unsteady_*`` / ``*_abs_unsteady_*``
  -- the transient (``du/dt``/``dv/dt``) extension added on top of the
  steady equations. These reuse (rather than re-derive) the steady
  manufactured-solution guarantee above: the transient term's own
  correctness is checked against a linear-in-time perturbation, for which
  a first-order finite difference is *exact* (zero truncation error, not
  merely convergent), and against the ``y_pred_next == y_pred`` case, which
  must reduce the new methods to exactly the already-verified steady
  ``residual``/``abs``.

Following the ``losses/tests`` structure, the numeric tests are parametrized
over the available CPU/GPU backends.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.losses.equation_losses import NavierStokesEqnLoss


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _manufactured_taylor_green(
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A divergence-free ``(u, v)`` pair plus a smooth ``p`` on the unit
    square, ``n`` points per axis (both endpoints sampled).

    ``u = sin(pi x) cos(pi y)``, ``v = -cos(pi x) sin(pi y)``: their
    divergence is exactly zero everywhere, not just after cancellation
    across the domain, which is why the manufactured continuity residual
    below is exactly zero. ``p = sin(pi x) sin(pi y)`` is otherwise
    arbitrary (smooth, non-constant, non-separable-trivially).

    Returns
    -------
    X, Y : jnp.ndarray, shape (n, n)
        the grid coordinates (``indexing="ij"``, matching AxisSpec's
        axis=1 -> x, axis=2 -> y convention)
    u, v, p : jnp.ndarray, shape (n, n)
    """
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 1.0, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
    v = -jnp.cos(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    p = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    return X, Y, u, v, p


def _analytic_residuals(
    X: jnp.ndarray, Y: jnp.ndarray, reynolds: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Closed-form momentum-x/momentum-y/continuity residuals of the
    *continuous* PDE for the manufactured fields above, at the given `Re`
    -- hand-derived from u, v, p and cross-checked numerically against a
    high-resolution finite-difference evaluation (max interior error
    ~4e-5 at n=400) before being hardcoded here.
    """
    pi = jnp.pi
    r_x = (
        pi * jnp.sin(pi * X) * jnp.cos(pi * X)
        + pi * jnp.cos(pi * X) * jnp.sin(pi * Y)
        + (2 * pi**2 / reynolds) * jnp.sin(pi * X) * jnp.cos(pi * Y)
    )
    r_y = (
        pi * jnp.sin(pi * Y) * jnp.cos(pi * Y)
        + pi * jnp.sin(pi * X) * jnp.cos(pi * Y)
        - (2 * pi**2 / reynolds) * jnp.cos(pi * X) * jnp.sin(pi * Y)
    )
    r_c = jnp.zeros_like(X)
    return r_x, r_y, r_c


@pytest.mark.needs_x64
@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_residual_converges_with_manufactured_solution(
    backend: str,
) -> None:
    """The discretized residual -> the known closed-form continuous
    residual as the grid is refined, for all three coupled equations at
    once. Interior-only (a boundary halo is excluded), the same caveat
    DarcyEqnLoss's own convergence test documents: chaining two
    finite-difference operators (as DiffusionTerm's flux form and the
    advection/gradient terms all do) degrades accuracy near a one-sided
    boundary stencil.
    """
    with jax.default_device(jax.devices(backend)[0]):
        reynolds = 37.0
        loss = NavierStokesEqnLoss(
            reynolds=reynolds, enforce_dirichlet=False, pin_pressure=False
        )

        def errors(n: int) -> dict[str, float]:
            X, Y, u, v, p = _manufactured_taylor_green(n)
            y_pred = jnp.stack([u, v, p], axis=-1)[None]
            computed = loss.residual(y_pred)
            r_x, r_y, r_c = _analytic_residuals(X, Y, reynolds)
            analytic = {"momentum_x": r_x, "momentum_y": r_y, "continuity": r_c}
            return {
                name: float(
                    jnp.max(jnp.abs((computed[name][0] - analytic[name])[3:-3, 3:-3]))
                )
                for name in ("momentum_x", "momentum_y", "continuity")
            }

        err_coarse = errors(33)
        err_fine = errors(65)

        for name in ("momentum_x", "momentum_y"):
            assert err_fine[name] < err_coarse[name]
            assert err_fine[name] < 0.01
            order = float(jnp.log(err_coarse[name] / err_fine[name]) / jnp.log(2.0))
            assert order == pytest.approx(2.0, abs=0.5)

        # continuity's analytic residual is exactly 0 everywhere (u, v is a
        # divergence-free pair by construction), so its discretization
        # error is already at floating-point noise level at the coarse
        # grid -- a strict decrease/order check would be flaky, so just
        # check both resolutions stay at that noise floor.
        assert err_coarse["continuity"] < 1e-6
        assert err_fine["continuity"] < 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_dirichlet_residual_invariant_to_boundary(
    backend: str,
) -> None:
    """With enforce_dirichlet=True, the residual ignores the predicted
    boundary values of u and v."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0, enforce_dirichlet=True)
        n = 10
        key_u, key_v, key_p, key_pert = jax.random.split(jax.random.PRNGKey(0), 4)
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        r1 = loss.residual(y_pred)

        perturbation = jax.random.normal(key_pert, y_pred.shape)
        y_perturbed = y_pred
        for edge in (
            (slice(None), 0, slice(None), slice(0, 2)),
            (slice(None), -1, slice(None), slice(0, 2)),
            (slice(None), slice(None), 0, slice(0, 2)),
            (slice(None), slice(None), -1, slice(0, 2)),
        ):
            y_perturbed = y_perturbed.at[edge].add(perturbation[edge])

        r2 = loss.residual(y_perturbed)
        for name in ("momentum_x", "momentum_y", "continuity"):
            assert jnp.allclose(r1[name], r2[name])


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_dirichlet_cuts_off_boundary_gradient(
    backend: str,
) -> None:
    """enforce_dirichlet=True gives exactly zero gradient at every boundary
    point of u and v; without it, the boundary generally does receive
    gradient."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        key_u, key_v, key_p = jax.random.split(jax.random.PRNGKey(1), 3)
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        loss_hard = NavierStokesEqnLoss(
            reynolds=100.0, enforce_dirichlet=True, pin_pressure=False
        )
        loss_soft = NavierStokesEqnLoss(
            reynolds=100.0, enforce_dirichlet=False, pin_pressure=False
        )

        grad_hard = jax.grad(lambda yp: loss_hard.abs(yp))(y_pred)
        grad_soft = jax.grad(lambda yp: loss_soft.abs(yp))(y_pred)

        # hard constraint: exactly zero gradient at every boundary point,
        # for both velocity components (channels 0 and 1), not pressure
        for ch in (0, 1):
            assert jnp.all(grad_hard[:, 0, :, ch] == 0.0)
            assert jnp.all(grad_hard[:, -1, :, ch] == 0.0)
            assert jnp.all(grad_hard[:, :, 0, ch] == 0.0)
            assert jnp.all(grad_hard[:, :, -1, ch] == 0.0)

        assert not jnp.all(grad_soft[:, 0, :, 0] == 0.0)


def test_navier_stokes_eqnloss_apply_dirichlet_lid_and_walls() -> None:
    """The lid (y_max edge) gets u=lid_velocity, v=0; the other three walls
    get u=v=0. At the four corners (where an x-edge and the lid's y-edge
    overlap), the y-edge (lid) value wins, matching DarcyEqnLoss's
    x-edges-then-y-edges convention."""
    loss = NavierStokesEqnLoss(reynolds=100.0, lid_velocity=2.5)
    n = 6
    u = jax.random.normal(jax.random.PRNGKey(9), (1, n, n))
    v = jax.random.normal(jax.random.PRNGKey(10), (1, n, n))

    out_u = loss._apply_dirichlet(u, loss._u_boundary)
    out_v = loss._apply_dirichlet(v, loss._v_boundary)

    # interior untouched
    assert jnp.allclose(out_u[:, 1:-1, 1:-1], u[:, 1:-1, 1:-1])
    assert jnp.allclose(out_v[:, 1:-1, 1:-1], v[:, 1:-1, 1:-1])

    # non-corner wall points (x_min, x_max, y_min) are no-slip
    assert jnp.all(out_u[:, 0, 1:-1] == 0.0)
    assert jnp.all(out_u[:, -1, 1:-1] == 0.0)
    assert jnp.all(out_u[:, 1:-1, 0] == 0.0)
    assert jnp.all(out_v[:, 0, 1:-1] == 0.0)
    assert jnp.all(out_v[:, -1, 1:-1] == 0.0)
    assert jnp.all(out_v[:, 1:-1, 0] == 0.0)

    # lid (y_max), excluding corners: u = lid_velocity, v = 0
    assert jnp.all(out_u[:, 1:-1, -1] == 2.5)
    assert jnp.all(out_v[:, 1:-1, -1] == 0.0)

    # corners: the lid's y-edge value wins over the wall's x-edge value
    assert out_u[0, 0, -1] == 2.5
    assert out_u[0, -1, -1] == 2.5
    assert out_v[0, 0, -1] == 0.0
    assert out_v[0, -1, -1] == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_pin_pressure_residual_invariant_to_p(
    backend: str,
) -> None:
    """With pin_pressure=True, the residual ignores the predicted value of
    p at the pinned point (0, 0) -- since only p's gradient enters the
    residual, perturbing every OTHER point of p still changes the residual
    (it isn't a global no-op), only the single pinned point is masked out."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0, pin_pressure=True)
        n = 10
        key_u, key_v, key_p, key_pert = jax.random.split(jax.random.PRNGKey(2), 4)
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        r1 = loss.residual(y_pred)

        perturbation = jax.random.normal(key_pert, (2, n, n))
        p_perturbed = p.at[:, 0, 0].add(perturbation[:, 0, 0])
        y_perturbed = jnp.stack([u, v, p_perturbed], axis=-1)

        r2 = loss.residual(y_perturbed)
        for name in ("momentum_x", "momentum_y", "continuity"):
            assert jnp.allclose(r1[name], r2[name])


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_pin_pressure_cuts_off_gradient_at_pinned_point(
    backend: str,
) -> None:
    """pin_pressure=True gives exactly zero gradient w.r.t. p at (0, 0);
    without it, that point generally does receive gradient."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        key_u, key_v, key_p = jax.random.split(jax.random.PRNGKey(3), 3)
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        loss_pinned = NavierStokesEqnLoss(
            reynolds=100.0, enforce_dirichlet=False, pin_pressure=True
        )
        loss_free = NavierStokesEqnLoss(
            reynolds=100.0, enforce_dirichlet=False, pin_pressure=False
        )

        grad_pinned = jax.grad(lambda yp: loss_pinned.abs(yp))(y_pred)
        grad_free = jax.grad(lambda yp: loss_free.abs(yp))(y_pred)

        assert jnp.all(grad_pinned[:, 0, 0, 2] == 0.0)
        assert not jnp.all(grad_free[:, 0, 0, 2] == 0.0)


def test_navier_stokes_eqnloss_pressure_pin_value_is_configurable() -> None:
    """pressure_pin_value overrides the default 0.0 pin target."""
    loss = NavierStokesEqnLoss(reynolds=100.0, pressure_pin_value=3.0)
    n = 6
    y_pred = jnp.zeros((1, n, n, 3))
    ctx = loss._context(y_pred)
    p = jnp.asarray(ctx["p"])
    assert float(p[0, 0, 0]) == pytest.approx(3.0)


def test_navier_stokes_eqnloss_weights_default_to_one_and_scale_contributions() -> None:
    """The three equations are summed with per-equation weights (default
    all 1.0); a custom weights dict actually scales each equation's
    contribution, verified against the internal per-equation
    PDEResidualLoss.abs() values directly."""
    n = 10
    key_u, key_v, key_p = jax.random.split(jax.random.PRNGKey(4), 3)
    u = jax.random.normal(key_u, (1, n, n))
    v = jax.random.normal(key_v, (1, n, n))
    p = jax.random.normal(key_p, (1, n, n))
    y_pred = jnp.stack([u, v, p], axis=-1)

    default_loss = NavierStokesEqnLoss(reynolds=100.0, enforce_dirichlet=False)
    ctx = default_loss._context(y_pred)
    r_x = default_loss._momentum_x.abs(ctx)
    r_y = default_loss._momentum_y.abs(ctx)
    r_c = default_loss._continuity.abs(ctx)
    assert float(default_loss.abs(y_pred)) == pytest.approx(
        float(r_x + r_y + r_c), rel=1e-5
    )

    weighted_loss = NavierStokesEqnLoss(
        reynolds=100.0,
        enforce_dirichlet=False,
        weights={"momentum_x": 2.0, "momentum_y": 0.0, "continuity": 0.0},
    )
    assert float(weighted_loss.abs(y_pred)) == pytest.approx(float(2.0 * r_x), rel=1e-5)


def test_navier_stokes_eqnloss_weights_rejects_incomplete_or_unknown_dict() -> None:
    """A weights dict missing an equation, or naming one that doesn't
    exist, fails loudly at construction instead of silently defaulting the
    missing equation's weight to something unintended."""
    with pytest.raises(ValueError):
        NavierStokesEqnLoss(reynolds=100.0, weights={"momentum_x": 1.0})

    with pytest.raises(ValueError):
        NavierStokesEqnLoss(
            reynolds=100.0,
            weights={
                "momentum_x": 1.0,
                "momentum_y": 1.0,
                "continuity": 1.0,
                "extra": 1.0,
            },
        )


def test_navier_stokes_eqnloss_rejects_non_positive_reynolds() -> None:
    """reynolds has no standard default and must be > 0."""
    with pytest.raises(ValueError):
        NavierStokesEqnLoss(reynolds=0.0)
    with pytest.raises(ValueError):
        NavierStokesEqnLoss(reynolds=-10.0)


def test_navier_stokes_eqnloss_rejects_wrong_length_measure() -> None:
    """measure must have exactly one entry per spatial dim (2)."""
    with pytest.raises(ValueError):
        NavierStokesEqnLoss(reynolds=100.0, measure=[2.0])
    with pytest.raises(ValueError):
        NavierStokesEqnLoss(reynolds=100.0, measure=[2.0, 1.0, 3.0])


def test_navier_stokes_eqnloss_accepts_sequence_measure() -> None:
    """measure can be a per-axis Sequence[float] (non-square domain); a
    different domain length must actually change the residual, not just
    fail to crash."""
    n = 8
    u = jax.random.normal(jax.random.PRNGKey(5), (1, n, n))
    v = jax.random.normal(jax.random.PRNGKey(6), (1, n, n))
    p = jax.random.normal(jax.random.PRNGKey(7), (1, n, n))
    y_pred = jnp.stack([u, v, p], axis=-1)

    loss_rect = NavierStokesEqnLoss(
        reynolds=100.0, measure=[2.0, 1.0], enforce_dirichlet=False
    )
    loss_square = NavierStokesEqnLoss(
        reynolds=100.0, measure=1.0, enforce_dirichlet=False
    )
    assert loss_rect.measure == [2.0, 1.0]
    r_rect = loss_rect.residual(y_pred)
    r_square = loss_square.residual(y_pred)
    assert not jnp.allclose(r_rect["momentum_x"], r_square["momentum_x"])


def test_navier_stokes_eqnloss_rejects_wrong_channel_count() -> None:
    """NavierStokesEqnLoss requires a 3-channel (u, v, p) prediction."""
    loss = NavierStokesEqnLoss(reynolds=100.0)
    y_pred = jnp.zeros((1, 8, 8, 1))
    with pytest.raises(ValueError):
        loss.residual(y_pred)


def test_navier_stokes_eqnloss_rejects_too_few_grid_points() -> None:
    """NavierStokesEqnLoss needs at least 4 grid points per spatial axis."""
    loss = NavierStokesEqnLoss(reynolds=100.0)
    y_pred = jnp.zeros((1, 3, 8, 3))
    with pytest.raises(ValueError):
        loss.residual(y_pred)


def test_navier_stokes_eqnloss_rejects_bad_reduction() -> None:
    """An unsupported reduction is rejected at construction."""
    with pytest.raises(AssertionError):
        NavierStokesEqnLoss(reynolds=100.0, reduction="median")


def test_navier_stokes_eqnloss_residual_returns_three_correctly_shaped_fields() -> None:
    """residual() returns a dict with exactly the three equation names,
    each shaped like the input grid (batch, n_x, n_y)."""
    n = 8
    y_pred = jnp.zeros((2, n, n, 3))
    loss = NavierStokesEqnLoss(reynolds=100.0)
    res = loss.residual(y_pred)
    assert set(res) == {"momentum_x", "momentum_y", "continuity"}
    for field in res.values():
        assert field.shape == (2, n, n)


def test_navier_stokes_eqnloss_call_ignores_x_and_y() -> None:
    """__call__ matches Trainer's loss(out, **sample) convention -- neither
    x nor y is required or used, since reynolds is a fixed config constant,
    not a per-sample field."""
    n = 8
    loss = NavierStokesEqnLoss(reynolds=100.0, enforce_dirichlet=False)
    y_pred = jax.random.normal(jax.random.PRNGKey(8), (1, n, n, 3))
    dummy = jax.random.normal(jax.random.PRNGKey(9), (1, n, n, 3))

    val_bare = loss(y_pred)
    val_with_kwargs = loss(y_pred, x=dummy, y=dummy)
    assert float(val_bare) == pytest.approx(float(val_with_kwargs))
    assert float(val_bare) == pytest.approx(float(loss.abs(y_pred)))


def test_navier_stokes_eqnloss_name_property() -> None:
    """`.name` identifies the loss, matching the LpLoss/H1Loss/DarcyEqnLoss
    convention."""
    assert NavierStokesEqnLoss(reynolds=100.0).name == "NavierStokesEqnLoss"


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_residual_unsteady_matches_steady_when_no_time_variation(
    backend: str,
) -> None:
    """With y_pred_next == y_pred, du/dt = dv/dt = 0 exactly regardless of
    dt, so residual_unsteady must reduce to exactly the steady residual()."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0)
        n = 10
        key_u, key_v, key_p = jax.random.split(jax.random.PRNGKey(20), 3)
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        steady = loss.residual(y_pred)
        unsteady = loss.residual_unsteady(y_pred, y_pred, dt=0.37)
        for name in ("momentum_x", "momentum_y", "continuity"):
            assert jnp.allclose(steady[name], unsteady[name])


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_abs_unsteady_matches_abs_when_no_time_variation(
    backend: str,
) -> None:
    """Same no-time-variation equivalence as residual_unsteady, through the
    scalar abs_unsteady()/abs() pair (covers the weighting/reduction path,
    not just the pointwise residual)."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0)
        n = 10
        y_pred = jax.random.normal(jax.random.PRNGKey(21), (2, n, n, 3))

        assert float(loss.abs_unsteady(y_pred, y_pred, dt=1.5)) == pytest.approx(
            float(loss.abs(y_pred)), rel=1e-5
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_unsteady_recovers_linear_time_derivative(
    backend: str,
) -> None:
    """u(t) = u0 + a*t has du/dt = a exactly; a first-order finite
    difference of a linear function has zero truncation error, so the
    transient term added to momentum_x/momentum_y must equal `a` exactly
    (away from the Dirichlet-enforced boundary, where both u0 and u0+a*t
    are overwritten to the same fixed value and the perturbation is
    masked out identically on both sides)."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0, enforce_dirichlet=True)
        n = 10
        key_u, key_v, key_p, key_au, key_av = jax.random.split(
            jax.random.PRNGKey(22), 5
        )
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        a_u = jax.random.normal(key_au, (2, n, n))
        a_v = jax.random.normal(key_av, (2, n, n))
        dt = 0.5
        y_next = y_pred.at[..., 0].add(a_u * dt).at[..., 1].add(a_v * dt)

        res_steady = loss.residual(y_pred)
        res_unsteady = loss.residual_unsteady(y_pred, y_next, dt=dt)

        interior = (slice(None), slice(1, -1), slice(1, -1))
        diff_x = (
            res_unsteady["momentum_x"][interior] - res_steady["momentum_x"][interior]
        )
        diff_y = (
            res_unsteady["momentum_y"][interior] - res_steady["momentum_y"][interior]
        )
        assert jnp.allclose(diff_x, a_u[interior], atol=1e-4)
        assert jnp.allclose(diff_y, a_v[interior], atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_residual_unsteady_reversing_time_order_flips_sign(
    backend: str,
) -> None:
    """Swapping which prediction is "current" vs "next" must flip the sign
    of the added transient term (a correctness property a sign-flip bug in
    the finite difference would violate)."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0, enforce_dirichlet=True)
        n = 10
        key_u, key_v, key_p, key_au, key_av = jax.random.split(
            jax.random.PRNGKey(23), 5
        )
        u = jax.random.normal(key_u, (2, n, n))
        v = jax.random.normal(key_v, (2, n, n))
        p = jax.random.normal(key_p, (2, n, n))
        y_pred = jnp.stack([u, v, p], axis=-1)

        a_u = jax.random.normal(key_au, (2, n, n))
        a_v = jax.random.normal(key_av, (2, n, n))
        dt = 0.5
        y_next = y_pred.at[..., 0].add(a_u * dt).at[..., 1].add(a_v * dt)

        forward = loss.residual_unsteady(y_pred, y_next, dt=dt)
        backward = loss.residual_unsteady(y_next, y_pred, dt=dt)

        interior = (slice(None), slice(1, -1), slice(1, -1))
        added_forward = (
            forward["momentum_x"][interior]
            - loss.residual(y_pred)["momentum_x"][interior]
        )
        added_backward = (
            backward["momentum_x"][interior]
            - loss.residual(y_next)["momentum_x"][interior]
        )
        assert jnp.allclose(added_forward, -added_backward, atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_residual_unsteady_continuity_unaffected(
    backend: str,
) -> None:
    """Continuity (mass conservation) has no time-derivative term -- it
    must come back byte-identical to the steady residual regardless of
    y_pred_next or dt."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = NavierStokesEqnLoss(reynolds=100.0)
        n = 10
        y_pred = jax.random.normal(jax.random.PRNGKey(24), (2, n, n, 3))
        y_next = jax.random.normal(jax.random.PRNGKey(25), (2, n, n, 3))

        steady = loss.residual(y_pred)
        unsteady = loss.residual_unsteady(y_pred, y_next, dt=0.1)
        assert jnp.array_equal(steady["continuity"], unsteady["continuity"])


def test_navier_stokes_eqnloss_residual_unsteady_returns_shaped_fields() -> None:
    """residual_unsteady() returns the same dict shape/keys as residual()."""
    n = 8
    y_pred = jnp.zeros((2, n, n, 3))
    y_next = jnp.zeros((2, n, n, 3))
    loss = NavierStokesEqnLoss(reynolds=100.0)
    res = loss.residual_unsteady(y_pred, y_next, dt=1.0)
    assert set(res) == {"momentum_x", "momentum_y", "continuity"}
    for field in res.values():
        assert field.shape == (2, n, n)


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_boundary_margin_defaults_to_full_field(
    backend: str,
) -> None:
    """boundary_margin=0 (the default) must reproduce the pre-existing
    full-field reduction exactly -- this option is opt-in, and every caller
    written before it existed must be unaffected."""
    with jax.default_device(jax.devices(backend)[0]):
        y_pred = jax.random.normal(jax.random.PRNGKey(41), (2, 12, 12, 3))
        default = NavierStokesEqnLoss(reynolds=100.0)
        explicit = NavierStokesEqnLoss(reynolds=100.0, boundary_margin=0)
        assert float(explicit.abs(y_pred)) == pytest.approx(
            float(default.abs(y_pred)), rel=1e-6
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_boundary_margin_matches_manual_crop(
    backend: str,
) -> None:
    """The cropped norm must equal reducing the *same* pointwise residual over
    the interior by hand -- i.e. the margin only changes which points are
    reduced, never how the residual itself is computed (the Dirichlet/pinning
    constraints still use the full field, so the interior values are
    unchanged)."""
    with jax.default_device(jax.devices(backend)[0]):
        m, n = 3, 16
        y_pred = jax.random.normal(jax.random.PRNGKey(42), (2, n, n, 3))
        loss = NavierStokesEqnLoss(reynolds=100.0, boundary_margin=m)
        res = loss.residual(y_pred)  # full-field pointwise residual
        expected = sum(
            float(jnp.sqrt(jnp.mean(v[:, m:-m, m:-m] ** 2))) for v in res.values()
        )
        assert float(loss.abs(y_pred)) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_navier_stokes_eqnloss_boundary_margin_applies_to_unsteady_too(
    backend: str,
) -> None:
    """abs_unsteady() shares abs()'s reduction, so the margin must crop there
    as well -- the transient term is exactly where the wrong-units bug lived,
    and it is the path the hybrid trainers actually optimize."""
    with jax.default_device(jax.devices(backend)[0]):
        m, n = 2, 14
        key = jax.random.PRNGKey(43)
        y_pred = jax.random.normal(key, (2, n, n, 3))
        y_next = y_pred + 0.05 * jax.random.normal(jax.random.PRNGKey(44), (2, n, n, 3))
        loss = NavierStokesEqnLoss(reynolds=100.0, boundary_margin=m)
        res = loss.residual_unsteady(y_pred, y_next, dt=0.3)
        expected = sum(
            float(jnp.sqrt(jnp.mean(v[:, m:-m, m:-m] ** 2))) for v in res.values()
        )
        assert float(loss.abs_unsteady(y_pred, y_next, dt=0.3)) == pytest.approx(
            expected, rel=1e-5
        )


def test_navier_stokes_eqnloss_boundary_margin_rejects_bad_values() -> None:
    """A negative/non-int margin is a configuration error, and one wide enough
    to leave no interior would silently reduce over an empty array (nan)."""
    with pytest.raises(ValueError, match="non-negative int"):
        NavierStokesEqnLoss(reynolds=100.0, boundary_margin=-1)
    with pytest.raises(ValueError, match="non-negative int"):
        NavierStokesEqnLoss(reynolds=100.0, boundary_margin=2.5)

    loss = NavierStokesEqnLoss(reynolds=100.0, boundary_margin=5)
    with pytest.raises(ValueError, match="no interior points"):
        loss.abs(jnp.zeros((1, 10, 10, 3)))
