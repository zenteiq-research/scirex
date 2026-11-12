# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for FNOClassic2d."""

import jax
import jax.numpy as jnp
import pytest

from scirex.models.fno_classic import FNOClassic2d


@pytest.mark.parametrize("in_channels,out_channels", [(1, 1), (2, 3)])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_FNOClassic2d_shape(in_channels: int, out_channels: int, n_layers: int) -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        model = FNOClassic2d(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes1=4,
            n_modes2=4,
            hidden_channels=8,
            n_layers=n_layers,
        )
        x = jax.random.normal(key, (2, in_channels, 16, 16), dtype=jnp.float32)
        params = model.init(key, x)
        out = model.apply(params, x)
        assert out.shape == (2, out_channels, 16, 16)
        assert jnp.isrealobj(out)


def test_FNOClassic2d_gradient_flows_to_every_parameter() -> None:
    """Every parameter (spectral weights, lifting, projection, skips,
    pointwise MLPs) must receive a nonzero gradient -- guards against a
    disconnected pathway in the block-assembly loop."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        model = FNOClassic2d(
            in_channels=1, out_channels=1, n_modes1=4, n_modes2=4, hidden_channels=8
        )
        x = jax.random.normal(key, (2, 1, 16, 16), dtype=jnp.float32)
        params = model.init(key, x)

        def loss_fn(p: dict) -> jnp.ndarray:
            return jnp.sum(model.apply(p, x) ** 2)

        grads = jax.grad(loss_fn)(params)
        flat_grads = jax.tree_util.tree_leaves(grads)
        flat_params = jax.tree_util.tree_leaves(params)
        assert len(flat_grads) == len(flat_params)
        for g in flat_grads:
            assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_FNOClassic2d_registered_in_model_registry() -> None:
    from scirex.models.base import BaseModel

    assert "fnoclassic2d" in BaseModel._models
