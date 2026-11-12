# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the PDE-residual (physics-informed) loss DarcyEqnLoss (JAX).

* ``test_darcy_eqnloss_residual_converges_with_manufactured_solution`` --
  a manufactured (u, a) pair chosen so ``-div(a * grad(u)) = 1`` holds
  *exactly*, with a genuinely non-constant coefficient ``a(x) = 1 + x`` (so
  the flux-form discretization -- differencing ``a * grad(u)``, never ``a``
  itself -- is actually exercised), isolated from Dirichlet enforcement
  (``enforce_dirichlet=False``) so it purely tests discretization accuracy.
* ``test_darcy_eqnloss_dirichlet_*`` -- the hard-Dirichlet-constraint
  behavior discussed alongside the implementation: the residual is
  invariant to the predicted field's boundary values, and (checked via
  ``jax.grad``) no gradient flows back to the model's boundary predictions
  through this loss when ``enforce_dirichlet=True``.
* Shape/argument validation and the ``__call__`` / ``Trainer`` calling
  convention (``loss(out, **sample)``, ignoring ``y``).

Following the ``losses/tests`` structure, the numeric tests are parametrized
over the available CPU/GPU backends.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.losses.equation_losses import DarcyEqnLoss


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _manufactured_uxa(n: int) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    """u(x,y) and a(x,y) (constant in y) with -div(a * grad(u)) = 1 exactly.

    Derived by choosing flux F(x) = a(x) u'(x) = C - x directly (so
    F'(x) = -1), then integrating u'(x) = F(x)/a(x) with a(x) = 1 + x, and
    fixing constants so u(0) = u(1) = 0. Since neither u nor a depends on
    y, the y-divergence term is exactly zero at every grid point -- the
    entire residual comes from the (non-constant-coefficient) x-direction
    flux-divergence, which is exactly what the flux-form discretization
    needs to get right.

    Returns
    -------
    u : jnp.ndarray, shape (1, n, n, 1)
    a : jnp.ndarray, shape (1, n, n, 1)
    h : float
        grid spacing (unit square, n points including both endpoints)
    """
    h = 1.0 / (n - 1)
    x = jnp.arange(n) * h
    c = 1.0 / jnp.log(2.0) - 1.0
    u_x = -x + (1.0 + c) * jnp.log(1.0 + x)
    a_x = 1.0 + x

    u = jnp.tile(u_x[:, None], (1, n))[None, :, :, None]
    a = jnp.tile(a_x[:, None], (1, n))[None, :, :, None]
    return u, a, h


@pytest.mark.needs_x64
@pytest.mark.parametrize("backend", BACKENDS)
def test_darcy_eqnloss_residual_converges_with_manufactured_solution(
    backend: str,
) -> None:
    """The flux-form residual -> 0 as the grid is refined.

    The two domain-boundary points (x=0, x=1) converge at only 1st order:
    the divergence step there differences the *already-approximated*
    (O(h^2)-accurate) flux values at the neighboring points, and dividing
    by h amplifies that input error to O(h) -- a generic property of
    differencing an already-differenced quantity at a one-sided boundary,
    not specific to this stencil choice. This is why DarcyEqnLoss hard-
    enforces the (exactly known) Dirichlet value at the boundary by default
    rather than relying on finite-difference accuracy to recover it there.
    The interior (unaffected by that compounding) converges at the full
    2nd order the stencils are designed for.
    """
    with jax.default_device(jax.devices(backend)[0]):
        loss = DarcyEqnLoss(forcing=1.0, enforce_dirichlet=False)

        def errors(n: int) -> tuple[float, float]:
            u, a, _ = _manufactured_uxa(n)
            r = loss.residual(u, a)[0, :, 0]
            interior = float(jnp.max(jnp.abs(r[2:-2])))
            boundary = float(jnp.max(jnp.abs(jnp.array([r[0], r[-1]]))))
            return interior, boundary

        interior_coarse, boundary_coarse = errors(65)
        interior_fine, boundary_fine = errors(129)

        assert interior_fine < interior_coarse
        assert interior_fine < 0.001
        interior_order = float(jnp.log(interior_coarse / interior_fine) / jnp.log(2.0))
        assert interior_order == pytest.approx(2.0, abs=0.3)

        # boundary still converges, just at the (expected, lower) 1st order
        assert boundary_fine < boundary_coarse
        boundary_order = float(jnp.log(boundary_coarse / boundary_fine) / jnp.log(2.0))
        assert boundary_order == pytest.approx(1.0, abs=0.3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_darcy_eqnloss_dirichlet_residual_invariant_to_boundary(
    backend: str,
) -> None:
    """With enforce_dirichlet=True, the residual ignores the predicted boundary."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = DarcyEqnLoss(enforce_dirichlet=True)
        key_u, key_a, key_pert = jax.random.split(jax.random.PRNGKey(0), 3)
        n = 8
        u = jax.random.normal(key_u, (2, n, n, 1))
        a = 1.0 + jax.random.uniform(key_a, (2, n, n, 1))

        r1 = loss.residual(u, a)

        # perturb only the boundary rows/columns of u
        perturbation = jax.random.normal(key_pert, u.shape)
        u_perturbed = u
        u_perturbed = u_perturbed.at[:, 0, :, :].add(perturbation[:, 0, :, :])
        u_perturbed = u_perturbed.at[:, -1, :, :].add(perturbation[:, -1, :, :])
        u_perturbed = u_perturbed.at[:, :, 0, :].add(perturbation[:, :, 0, :])
        u_perturbed = u_perturbed.at[:, :, -1, :].add(perturbation[:, :, -1, :])

        r2 = loss.residual(u_perturbed, a)
        assert jnp.allclose(r1, r2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_darcy_eqnloss_dirichlet_cuts_off_boundary_gradient(backend: str) -> None:
    """enforce_dirichlet=True gives exactly zero gradient at boundary points."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        key_u, key_a = jax.random.split(jax.random.PRNGKey(1))
        u = jax.random.normal(key_u, (2, n, n, 1))
        a = 1.0 + jax.random.uniform(key_a, (2, n, n, 1))

        loss_hard = DarcyEqnLoss(enforce_dirichlet=True)
        loss_soft = DarcyEqnLoss(enforce_dirichlet=False)

        grad_hard = jax.grad(lambda up: loss_hard.abs(up, a))(u)
        grad_soft = jax.grad(lambda up: loss_soft.abs(up, a))(u)

        # hard constraint: exactly zero gradient at every boundary point
        assert jnp.all(grad_hard[:, 0, :, :] == 0.0)
        assert jnp.all(grad_hard[:, -1, :, :] == 0.0)
        assert jnp.all(grad_hard[:, :, 0, :] == 0.0)
        assert jnp.all(grad_hard[:, :, -1, :] == 0.0)

        # without the constraint, the boundary generally does receive gradient
        assert not jnp.all(grad_soft[:, 0, :, :] == 0.0)


def test_darcy_eqnloss_boundary_value_defaults_and_resolves_scalar() -> None:
    """boundary_value defaults to 0.0 on every edge, and a single scalar
    passed in resolves to that same constant applied to all four edges."""
    default = DarcyEqnLoss()
    assert default.boundary_value == {
        "x_min": 0.0,
        "x_max": 0.0,
        "y_min": 0.0,
        "y_max": 0.0,
    }

    scalar = DarcyEqnLoss(boundary_value=5.0)
    assert scalar.boundary_value == {
        "x_min": 5.0,
        "x_max": 5.0,
        "y_min": 5.0,
        "y_max": 5.0,
    }


def test_darcy_eqnloss_boundary_value_accepts_per_edge_dict() -> None:
    """A dict with exactly the four edge keys lets each edge differ."""
    per_edge = {"x_min": 1.0, "x_max": 2.0, "y_min": 3.0, "y_max": 4.0}
    loss = DarcyEqnLoss(boundary_value=per_edge)
    assert loss.boundary_value == per_edge


def test_darcy_eqnloss_boundary_value_rejects_incomplete_or_unknown_dict() -> None:
    """A dict missing an edge, or naming an edge that doesn't exist, fails
    loudly at construction instead of silently defaulting the missing edge
    to 0."""
    with pytest.raises(ValueError):
        DarcyEqnLoss(boundary_value={"x_min": 1.0, "x_max": 2.0, "y_min": 3.0})

    with pytest.raises(ValueError):
        DarcyEqnLoss(
            boundary_value={"x_min": 1.0, "x_max": 2.0, "y_min": 3.0, "top": 4.0}
        )


def test_darcy_eqnloss_apply_dirichlet_per_edge_with_corner_precedence() -> None:
    """Each edge gets its own configured value; at the four corners (where
    an x-edge and a y-edge overlap), the y-edge value wins, since y-edges
    are applied after x-edges."""
    loss = DarcyEqnLoss(
        boundary_value={"x_min": 1.0, "x_max": 2.0, "y_min": 3.0, "y_max": 4.0}
    )
    n = 6
    u = jax.random.normal(jax.random.PRNGKey(9), (1, n, n))
    out = loss._apply_dirichlet(u)

    # interior untouched
    assert jnp.allclose(out[:, 1:-1, 1:-1], u[:, 1:-1, 1:-1])

    # non-corner edge points take their own edge's value
    assert jnp.all(out[:, 0, 1:-1] == 1.0)
    assert jnp.all(out[:, -1, 1:-1] == 2.0)
    assert jnp.all(out[:, 1:-1, 0] == 3.0)
    assert jnp.all(out[:, 1:-1, -1] == 4.0)

    # corners: y-edge value wins over x-edge value
    assert out[0, 0, 0] == 3.0
    assert out[0, -1, 0] == 3.0
    assert out[0, 0, -1] == 4.0
    assert out[0, -1, -1] == 4.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_darcy_eqnloss_nonzero_boundary_value_residual_still_invariant_to_boundary(
    backend: str,
) -> None:
    """With a nonzero (and per-edge distinct) boundary_value, the residual
    is still computed from the enforced values, not the model's raw
    boundary prediction -- same invariance guarantee as the default case,
    end-to-end through residual()."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = DarcyEqnLoss(
            enforce_dirichlet=True,
            boundary_value={"x_min": 1.0, "x_max": 2.0, "y_min": 3.0, "y_max": 4.0},
        )
        key_u, key_a, key_pert = jax.random.split(jax.random.PRNGKey(2), 3)
        n = 8
        u = jax.random.normal(key_u, (2, n, n, 1))
        a = 1.0 + jax.random.uniform(key_a, (2, n, n, 1))

        r1 = loss.residual(u, a)

        perturbation = jax.random.normal(key_pert, u.shape)
        u_perturbed = u
        u_perturbed = u_perturbed.at[:, 0, :, :].add(perturbation[:, 0, :, :])
        u_perturbed = u_perturbed.at[:, -1, :, :].add(perturbation[:, -1, :, :])
        u_perturbed = u_perturbed.at[:, :, 0, :].add(perturbation[:, :, 0, :])
        u_perturbed = u_perturbed.at[:, :, -1, :].add(perturbation[:, :, -1, :])

        r2 = loss.residual(u_perturbed, a)
        assert jnp.allclose(r1, r2)


def test_darcy_eqnloss_rejects_wrong_length_measure() -> None:
    """measure must have exactly one entry per spatial dim (2 for Darcy);
    a mismatched length must be rejected explicitly, not fail later with an
    unrelated IndexError or silently ignore extra entries."""
    with pytest.raises(ValueError):
        DarcyEqnLoss(measure=[2.0])
    with pytest.raises(ValueError):
        DarcyEqnLoss(measure=[2.0, 1.0, 3.0])


def test_darcy_eqnloss_accepts_0d_array_measure_and_boundary_value() -> None:
    """A 0-d jax/numpy array (e.g. jnp.array(2.0)) behaves like a plain
    float scalar for both measure and boundary_value, not like a Sequence."""
    loss = DarcyEqnLoss(measure=jnp.array(2.0), boundary_value=jnp.array(5.0))
    assert loss.measure == [2.0, 2.0]
    assert loss.boundary_value == {
        "x_min": 5.0,
        "x_max": 5.0,
        "y_min": 5.0,
        "y_max": 5.0,
    }


def test_darcy_eqnloss_rejects_broadcastable_batch_mismatch() -> None:
    """y_pred and x must share the same batch size. A size-1 batch on one
    side would otherwise silently broadcast against the other instead of
    raising -- verified specifically for the broadcastable case (one side
    is 1), since a genuinely incompatible mismatch already raises via numpy
    broadcasting rules."""
    loss = DarcyEqnLoss()
    y_pred = jnp.zeros((2, 8, 8, 1))
    x = jnp.ones((1, 8, 8, 1))
    with pytest.raises(ValueError):
        loss.residual(y_pred, x)


def test_darcy_eqnloss_rejects_wrong_channel_count() -> None:
    """DarcyEqnLoss requires single-channel (..., 1) inputs."""
    loss = DarcyEqnLoss()
    u = jnp.zeros((1, 8, 8, 2))
    a = jnp.ones((1, 8, 8, 1))
    with pytest.raises(ValueError):
        loss.residual(u, a)


def test_darcy_eqnloss_rejects_too_few_grid_points() -> None:
    """DarcyEqnLoss needs at least 4 grid points per spatial axis."""
    loss = DarcyEqnLoss()
    u = jnp.zeros((1, 3, 8, 1))
    a = jnp.ones((1, 3, 8, 1))
    with pytest.raises(ValueError):
        loss.residual(u, a)


def test_darcy_eqnloss_rejects_bad_reduction() -> None:
    """An unsupported reduction is rejected at construction."""
    with pytest.raises(AssertionError):
        DarcyEqnLoss(reduction="median")


def test_darcy_eqnloss_accepts_sequence_measure() -> None:
    """measure can be a per-axis Sequence[float] (non-square domain), not
    just a single scalar applied to both spatial dims."""
    loss = DarcyEqnLoss(measure=[2.0, 1.0], enforce_dirichlet=False)
    assert loss.measure == [2.0, 1.0]

    n = 8
    u = jax.random.normal(jax.random.PRNGKey(5), (1, n, n, 1))
    a = 1.0 + jax.random.uniform(jax.random.PRNGKey(6), (1, n, n, 1))

    r_rect = loss.residual(u, a)
    r_square = DarcyEqnLoss(measure=1.0, enforce_dirichlet=False).residual(u, a)
    # a different domain length along x means a different grid spacing h_x,
    # so the two residuals must actually differ -- not just "not crash"
    assert not jnp.allclose(r_rect, r_square)


def test_darcy_eqnloss_diffusion_order_defaults_to_one() -> None:
    """diffusion_order isn't passed -- must still give the standard
    2nd-order-overall Darcy diffusion operator, matching every other
    existing DarcyEqnLoss test."""
    assert DarcyEqnLoss().diffusion_order == 1


def test_darcy_eqnloss_diffusion_order_is_forwarded_to_diffusion_term() -> None:
    """diffusion_order isn't hardcoded inside DiffusionTerm's construction --
    changing it must actually change the residual for the same input."""
    n = 8
    u = jax.random.normal(jax.random.PRNGKey(10), (1, n, n, 1))
    a = 1.0 + jax.random.uniform(jax.random.PRNGKey(11), (1, n, n, 1))

    r_order1 = DarcyEqnLoss(diffusion_order=1, enforce_dirichlet=False).residual(u, a)
    r_order2 = DarcyEqnLoss(diffusion_order=2, enforce_dirichlet=False).residual(u, a)
    assert not jnp.allclose(r_order1, r_order2)


def test_darcy_eqnloss_diffusion_scheme_defaults_to_central() -> None:
    """diffusion_scheme isn't passed -- must still give the standard
    central-difference Darcy diffusion operator, matching every other
    existing DarcyEqnLoss test."""
    assert DarcyEqnLoss().diffusion_scheme == "central"


def test_darcy_eqnloss_diffusion_scheme_is_forwarded_to_diffusion_term() -> None:
    """diffusion_scheme isn't hardcoded inside DiffusionTerm's construction
    -- changing it must actually change the residual for the same input."""
    n = 8
    u = jax.random.normal(jax.random.PRNGKey(16), (1, n, n, 1))
    a = 1.0 + jax.random.uniform(jax.random.PRNGKey(17), (1, n, n, 1))

    r_central = DarcyEqnLoss(enforce_dirichlet=False).residual(u, a)
    r_forward = DarcyEqnLoss(
        diffusion_scheme="forward", enforce_dirichlet=False
    ).residual(u, a)
    assert not jnp.allclose(r_central, r_forward)


def test_darcy_eqnloss_rejects_directional_scheme_with_order_two() -> None:
    """diffusion_order=2 only combines with diffusion_scheme="central" --
    DarcyEqnLoss must reject the combination as soon as it's constructed
    (via the underlying DiffusionTerm), not only on the first residual()."""
    with pytest.raises(ValueError):
        DarcyEqnLoss(diffusion_order=2, diffusion_scheme="forward")


def test_darcy_eqnloss_call_ignores_y_and_requires_x() -> None:
    """__call__ matches Trainer's loss(out, **sample) convention."""
    loss = DarcyEqnLoss(enforce_dirichlet=False)
    n = 8
    u = jax.random.normal(jax.random.PRNGKey(2), (1, n, n, 1))
    a = 1.0 + jax.random.uniform(jax.random.PRNGKey(3), (1, n, n, 1))
    y = jax.random.normal(jax.random.PRNGKey(4), (1, n, n, 1))

    val_with_y = loss(u, x=a, y=y)
    val_without_y = loss(u, x=a)
    assert float(val_with_y) == pytest.approx(float(val_without_y))
    assert float(val_with_y) == pytest.approx(float(loss.abs(u, a)))

    with pytest.raises(ValueError):
        loss(u, y=y)


def test_darcy_eval_metric_finalizes_global_rms_across_batches() -> None:
    """Eval aggregation should root once after summing squared residuals."""
    loss = DarcyEqnLoss(reduction="sum", enforce_dirichlet=False)
    n = 8
    u = jax.random.normal(jax.random.PRNGKey(7), (4, n, n, 1))
    a = 1.0 + jax.random.uniform(jax.random.PRNGKey(8), (4, n, n, 1))

    whole_sse = loss.eval_batch_loss(u, x=a)
    split_sse = loss.eval_batch_loss(u[:1], x=a[:1]) + loss.eval_batch_loss(
        u[1:], x=a[1:]
    )
    expected = jnp.sqrt(jnp.sum(loss.residual(u, a) ** 2) / u.shape[0])

    assert float(split_sse) == pytest.approx(float(whole_sse), rel=1e-5)
    mean_reduced_loss = DarcyEqnLoss(reduction="mean", enforce_dirichlet=False)
    assert float(mean_reduced_loss.eval_batch_loss(u, x=a)) == pytest.approx(
        float(whole_sse),
        rel=1e-5,
    )
    assert loss.finalize_eval_metric(split_sse, u.shape[0]) == pytest.approx(
        float(expected), rel=1e-5
    )


def test_darcy_eqnloss_name_property() -> None:
    """`.name` identifies the loss, matching the LpLoss/H1Loss convention."""
    assert DarcyEqnLoss().name == "DarcyEqnLoss"
