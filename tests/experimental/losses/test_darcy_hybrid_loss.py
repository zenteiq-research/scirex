# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for :class:`~scirex.experimental.losses.darcy_hybrid_loss.DarcyHybridLossGINO`.

``DarcyEqnLoss``'s own discretization accuracy (manufactured-solution
convergence, Dirichlet-invariance, etc.) is already covered by
``test_equation_losses.py`` and isn't re-tested here. What's new -- and
what these tests actually check -- is the one thing this wrapper adds: the
reshape from GINO's flat point-cloud output back into a grid.

* ``test_reshape_equivalent_to_calling_darcyeqnloss_directly`` -- the core
  correctness check: flattening a known grid field in row-major order and
  running it through the wrapper must give exactly the same physics loss as
  calling ``DarcyEqnLoss`` directly on the original grid-shaped field.
* Construction/shape/argument validation.
* The ``__call__`` / ``Trainer`` calling convention (``loss(out, **sample)``).
* ``data_weight``'s physics-only default vs. a nonzero weight actually
  changing the loss.

Following the ``losses/tests`` structure, the numeric tests are parametrized
over the available CPU/GPU backends.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.experimental.losses.darcy_hybrid_loss import DarcyHybridLossGINO
from scirex.losses.data_losses import LpLoss
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_reshape_equivalent_to_calling_darcyeqnloss_directly(backend: str) -> None:
    """Flattening a grid field row-major and running it through the wrapper
    must match calling DarcyEqnLoss directly on the unflattened grid --
    this is the one piece of new logic the wrapper adds, so it's the one
    thing that actually needs checking here."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        batch = 2
        key_u, key_a = jax.random.split(jax.random.PRNGKey(0))
        u_grid = jax.random.normal(key_u, (batch, n, n, 1))
        a_grid = 1.0 + jax.random.uniform(key_a, (batch, n, n, 1))

        physics_loss = DarcyEqnLoss(enforce_dirichlet=False)
        expected = physics_loss(u_grid, x=a_grid)

        wrapper = DarcyHybridLossGINO(physics_loss=physics_loss, grid_shape=(n, n))
        y_pred_flat = jnp.reshape(u_grid, (batch, n * n, 1))
        actual = wrapper(y_pred_flat, x=a_grid)

        assert float(actual) == pytest.approx(float(expected))


def test_data_weight_nonzero_requires_data_loss() -> None:
    """data_weight != 0 must be rejected at construction without a data_loss,
    same guard as LinearElasticityHybridLoss."""
    with pytest.raises(ValueError):
        DarcyHybridLossGINO(
            physics_loss=DarcyEqnLoss(), grid_shape=(8, 8), data_weight=0.5
        )


def test_grid_shape_must_have_two_entries() -> None:
    """grid_shape is (n_x, n_y) -- anything else is rejected explicitly."""
    with pytest.raises(ValueError):
        DarcyHybridLossGINO(physics_loss=DarcyEqnLoss(), grid_shape=(8,))
    with pytest.raises(ValueError):
        DarcyHybridLossGINO(physics_loss=DarcyEqnLoss(), grid_shape=(8, 8, 8))


def test_rejects_wrong_flat_shape() -> None:
    """y_pred must be the flat (batch, n_x*n_y, 1) shape GINO would produce
    when queried at the configured grid -- a mismatched point count or an
    already-grid-shaped array must fail loudly, not silently misreshape."""
    wrapper = DarcyHybridLossGINO(physics_loss=DarcyEqnLoss(), grid_shape=(8, 8))
    x = jnp.ones((1, 8, 8, 1))

    with pytest.raises(ValueError):
        wrapper(jnp.zeros((1, 63, 1)), x=x)  # wrong point count

    with pytest.raises(ValueError):
        wrapper(jnp.zeros((1, 8, 8, 1)), x=x)  # already grid-shaped, not flat

    with pytest.raises(ValueError):
        wrapper(jnp.zeros((1, 64, 2)), x=x)  # wrong channel count


def test_requires_permeability_field_x() -> None:
    """__call__ raises if x (the permeability field) isn't given."""
    wrapper = DarcyHybridLossGINO(physics_loss=DarcyEqnLoss(), grid_shape=(8, 8))
    y_pred = jnp.zeros((1, 64, 1))
    with pytest.raises(ValueError):
        wrapper(y_pred)


@pytest.mark.parametrize("backend", BACKENDS)
def test_call_matches_trainer_convention_and_ignores_y_by_default(
    backend: str,
) -> None:
    """__call__(out, **sample) works whether or not `y` is present in the
    batch dict, and with the default data_weight=0.0, `y` is ignored (pure
    physics-only loss) exactly like DarcyEqnLoss and LinearElasticityHybridLoss."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        batch = 1
        key_u, key_a, key_y = jax.random.split(jax.random.PRNGKey(1), 3)
        u_grid = jax.random.normal(key_u, (batch, n, n, 1))
        a_grid = 1.0 + jax.random.uniform(key_a, (batch, n, n, 1))
        y_pred_flat = jnp.reshape(u_grid, (batch, n * n, 1))
        y_label = jax.random.normal(key_y, (batch, 1, n, n))

        wrapper = DarcyHybridLossGINO(
            physics_loss=DarcyEqnLoss(enforce_dirichlet=False), grid_shape=(n, n)
        )

        val_with_y = wrapper(y_pred_flat, x=a_grid, y=y_label)
        val_without_y = wrapper(y_pred_flat, x=a_grid)
        assert float(val_with_y) == pytest.approx(float(val_without_y))


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonzero_data_weight_adds_weighted_data_term(backend: str) -> None:
    """With data_weight > 0, the total loss is exactly physics + weight *
    data_loss(reshaped prediction, label) -- checked against a manual
    computation of both terms, not just "the loss changed"."""
    with jax.default_device(jax.devices(backend)[0]):
        n = 8
        batch = 2
        key_u, key_a, key_y = jax.random.split(jax.random.PRNGKey(2), 3)
        u_grid = jax.random.normal(key_u, (batch, n, n, 1))
        a_grid = 1.0 + jax.random.uniform(key_a, (batch, n, n, 1))
        y_pred_flat = jnp.reshape(u_grid, (batch, n * n, 1))
        y_label = jax.random.normal(key_y, (batch, 1, n, n))

        physics_loss = DarcyEqnLoss(enforce_dirichlet=False)
        data_loss = LpLoss(d=2, p=2)
        weight = 0.3

        wrapper = DarcyHybridLossGINO(
            physics_loss=physics_loss,
            grid_shape=(n, n),
            data_loss=data_loss,
            data_weight=weight,
        )
        actual = wrapper(y_pred_flat, x=a_grid, y=y_label)

        expected_physics = physics_loss(u_grid, x=a_grid)
        u_grid_channel_first = jnp.transpose(u_grid, (0, 3, 1, 2))
        expected_data = data_loss(u_grid_channel_first, y=y_label)
        expected = expected_physics + weight * expected_data

        assert float(actual) == pytest.approx(float(expected), rel=1e-5)

        # and it must genuinely differ from the physics-only loss
        physics_only = DarcyHybridLossGINO(physics_loss=physics_loss, grid_shape=(n, n))
        assert float(actual) != pytest.approx(
            float(physics_only(y_pred_flat, x=a_grid))
        )


def test_nonzero_data_weight_requires_y_at_call_time() -> None:
    """With data_weight != 0, calling without `y` must raise -- not silently
    fall back to the physics-only loss, which would be especially unsafe
    with DarcyEqnLoss's default enforce_dirichlet=True (no boundary
    supervision at all in that case)."""
    wrapper = DarcyHybridLossGINO(
        physics_loss=DarcyEqnLoss(),
        grid_shape=(8, 8),
        data_loss=LpLoss(d=2, p=2),
        data_weight=0.5,
    )
    y_pred = jnp.zeros((1, 64, 1))
    x = jnp.ones((1, 8, 8, 1))
    with pytest.raises(ValueError):
        wrapper(y_pred, x=x)


def test_name_property() -> None:
    """`.name` identifies the loss, matching the repo-wide convention."""
    wrapper = DarcyHybridLossGINO(physics_loss=DarcyEqnLoss(), grid_shape=(8, 8))
    assert wrapper.name == "DarcyHybridLossGINO"
