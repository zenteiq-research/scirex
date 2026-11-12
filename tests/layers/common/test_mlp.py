# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the generic MLP layer (JAX/Flax) used by Transolver
(``src/scirex/layers/common/mlp.py``).

Covers output shape for arbitrary feature lists, the "activation on all but
last layer" rule (and the ``activation_last`` override), dropout behavior in
train vs. eval mode, and the residual-connection rule (only applied to
"interior" layers, i.e. not the first or last).
"""

from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.mlp import MLP
from tests import tolerances


def _init_and_apply(model: Any, x: Any, key: Any = None, **call_kwargs: Any) -> Any:
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, x, **call_kwargs)
    return model.apply(variables, x, **call_kwargs), variables


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("features", [[8], [16, 8], [32, 16, 8], [64, 64, 32, 4]])
def test_MLP_output_shape(features: Any) -> None:
    """Output's last dim always matches features[-1], batch/point dims preserved."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 10, 5), dtype=jnp.float32)
    model = MLP(features=features)
    out, _ = _init_and_apply(model, x, key)
    assert out.shape == (3, 10, features[-1])


def test_MLP_single_layer_is_plain_linear() -> None:
    """A single-feature MLP with activation_last=False is just one Dense (no
    activation)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 4, 6), dtype=jnp.float32)
    model = MLP(features=[3], activation_last=False)
    variables = model.init(key, x)
    out = cast(jax.Array, model.apply(variables, x))

    kernel = variables["params"]["Dense_0"]["kernel"]
    bias = variables["params"]["Dense_0"]["bias"]
    expected = x @ kernel + bias
    assert jnp.allclose(out, expected, atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# Activation placement: applied to all but the last layer, unless
# activation_last=True
# ---------------------------------------------------------------------------


def test_MLP_activation_last_false_leaves_final_layer_linear() -> None:
    """With activation_last=False, the final Dense output is not passed through
    activation."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    # Use relu so a linear final layer can produce negative values (a
    # post-relu output cannot), letting us detect whether relu was applied.
    model = MLP(features=[16, 8], activation=nn.relu, activation_last=False)
    out, _ = _init_and_apply(model, x, key)
    assert bool(
        (out < 0).any()
    ), "expected negative outputs when the final Dense is not activated"


def test_MLP_activation_last_true_activates_final_layer() -> None:
    """With activation_last=True and relu, every output is non-negative."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    model = MLP(features=[16, 8], activation=nn.relu, activation_last=True)
    out, _ = _init_and_apply(model, x, key)
    assert bool((out >= 0).all())


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def test_MLP_dropout_deterministic_is_reproducible() -> None:
    """deterministic=True gives identical output across repeated calls (no dropout
    applied)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    model = MLP(features=[8, 8], dropout_rate=0.5)
    variables = model.init(
        {"params": key, "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
    )
    out1 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(2)}
        ),
    )
    out2 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(3)}
        ),
    )
    assert jnp.allclose(out1, out2)


def test_MLP_dropout_train_mode_changes_output() -> None:
    """deterministic=False with different dropout rngs produces different outputs."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    model = MLP(features=[16, 16, 8], dropout_rate=0.5)
    variables = model.init(
        {"params": key, "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
    )
    out1 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(2)}
        ),
    )
    out2 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(3)}
        ),
    )
    assert not jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Residual connections: only applied to interior layers (0 < i < len - 1)
# ---------------------------------------------------------------------------


def test_MLP_residual_requires_matching_interior_dims() -> None:
    """residual=True with equal interior feature sizes runs without shape errors."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    # features = [16, 16, 16, 8]: interior indices 1, 2 both have size 16,
    # so x + x_in is shape-valid at each residual step.
    model = MLP(features=[16, 16, 16, 8], residual=True)
    out, _ = _init_and_apply(model, x, key)
    assert out.shape == (2, 5, 8)


def test_MLP_no_residual_at_first_or_last_layer() -> None:
    """residual=True on a 2-layer MLP is a no-op (no interior layer exists to skip)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 5, 4), dtype=jnp.float32)
    model_res = MLP(features=[8, 8], residual=True)
    model_no_res = MLP(features=[8, 8], residual=False)
    variables = model_res.init(key, x)
    out_res = cast(jax.Array, model_res.apply(variables, x))
    out_no_res = cast(jax.Array, model_no_res.apply(variables, x))
    assert jnp.allclose(out_res, out_no_res)
