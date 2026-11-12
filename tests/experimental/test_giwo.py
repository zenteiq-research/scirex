# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for GIWO (JAX/Flax), the GNO-in / WNO / GNO-out composition.

No PyTorch reference exists for this one -- unlike GINO/Transolver, it isn't
a published architecture, just GINO's own GNO shell with its latent FNO
swapped for a WNO (see ``src/scirex/experimental/giwo.py``'s module docstring).
So this mirrors ``test_gino.py``'s *forward-shape* and *every-parameter-
gets-a-gradient* checks, the portable core of that file, rather than a
paper-matched sweep.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import scirex
from scirex.experimental.giwo import GIWO

IN_CHANNELS = 3
OUT_CHANNELS = 2
N_IN = 40
N_OUT = 40
LATENT_SIZE = 6


def _latent_queries(coord_dim: int = 3) -> jnp.ndarray:
    axes = [jnp.linspace(0, 1, LATENT_SIZE)] * coord_dim
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="xy"))
    grid = jnp.transpose(mesh, (*range(1, coord_dim + 1), 0))
    return grid[None]


def _build(
    batch_size: int = 1,
    transform: str = "nonlinear_kernelonly",
    in_channels: int = IN_CHANNELS,
) -> tuple[GIWO, dict[str, Any]]:
    model = GIWO(
        in_channels=in_channels,
        out_channels=OUT_CHANNELS,
        in_gno_radius=0.3,
        out_gno_radius=0.3,
        in_gno_transform_type=transform,
        in_gno_channel_mlp_hidden_layers=(16, 16),
        out_gno_channel_mlp_hidden_layers=(16, 16),
        gno_use_open3d=False,
        wno_size=(LATENT_SIZE,) * 3,
        wno_hidden_channels=8,
        wno_n_layers=2,
        wno_level=1,
    )
    inputs = dict(
        input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 3)),
        latent_queries=_latent_queries(),
        output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 3)),
        x=jax.random.normal(jax.random.PRNGKey(2), (batch_size, N_IN, in_channels)),
    )
    return model, inputs


def _init_apply(model: GIWO, inputs: dict[str, Any]) -> tuple[Any, Any]:
    params = model.init(jax.random.PRNGKey(1), **inputs)
    return model.apply(params, **inputs), params


def _count_unused_params(model: GIWO, params: Any, inputs: dict[str, Any]) -> int:
    def loss(p: Any) -> jnp.ndarray:
        out = cast(jnp.ndarray, model.apply(p, **inputs))
        return jnp.sum(jnp.abs(out))

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    return sum(1 for g in leaves if not bool((g != 0).any()))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_giwo_forward_and_unused_params(batch_size: int) -> None:
    """GIWO forward gives [batch, n_out, out_channels]; all params get a gradient."""
    model, inputs = _build(batch_size=batch_size)
    out, params = _init_apply(model, inputs)

    assert list(out.shape) == [batch_size, N_OUT, OUT_CHANNELS]
    assert bool(jnp.all(jnp.isfinite(out)))
    assert _count_unused_params(model, params, inputs) == 0

    if batch_size > 1:

        def loss_x(x: jnp.ndarray) -> jnp.ndarray:
            local = dict(inputs)
            local["x"] = x
            return jnp.sum(model.apply(params, **local)[0])

        grad_x = jax.grad(loss_x)(inputs["x"])
        assert bool((grad_x[1:] == 0).all())
        assert bool((grad_x[0] != 0).any())


@pytest.mark.parametrize("transform", ["linear", "nonlinear_kernelonly", "nonlinear"])
def test_giwo_transform_types(transform: str) -> None:
    """Each in-GNO transform type produces the correct output shape.

    GIWO has no separate lifting layer between the input GNO and the latent
    WNO (unlike GINO, whose ``lifting`` ChannelMLP can bridge a channel-count
    mismatch) -- see ``giwo.py``'s module docstring. So every mode except
    ``"nonlinear_kernelonly"`` needs the input GNO's in/out channel counts to
    match (``IntegralTransform`` multiplies the kernel output by ``f_y``
    elementwise in those modes); ``"nonlinear_kernelonly"`` does not, which is
    exactly why :class:`~scripts.benchmarks.shapenetcar._models.GIWOCarWrapper`
    always uses it.
    """
    in_channels = IN_CHANNELS if transform == "nonlinear_kernelonly" else 8
    model, inputs = _build(transform=transform, in_channels=in_channels)
    out, _ = _init_apply(model, inputs)
    assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


def test_giwo_params_independent_of_point_count() -> None:
    """Parameter count depends on the fixed latent grid, not on n_in/n_out."""
    model, inputs = _build()
    _, params_small = _init_apply(model, inputs)

    big_inputs = dict(inputs)
    big_inputs["input_geom"] = jax.random.normal(jax.random.PRNGKey(5), (1, 200, 3))
    big_inputs["output_queries"] = jax.random.normal(jax.random.PRNGKey(6), (1, 150, 3))
    big_inputs["x"] = jax.random.normal(jax.random.PRNGKey(7), (1, 200, IN_CHANNELS))
    _, params_big = _init_apply(model, big_inputs)

    count = lambda p: sum(x.size for x in jax.tree_util.tree_leaves(p))
    assert count(params_small) == count(params_big)


def test_get_model_jax_builds_giwo_without_importing_it_first() -> None:
    """``scirex.models`` no longer imports GIWO; ``get_model_jax`` loads it on demand."""
    code = textwrap.dedent(
        """
        import sys
        import scirex.models
        from scirex.models.base import BaseModel, get_model_jax

        assert "scirex.experimental.giwo" not in sys.modules
        assert "giwo" not in BaseModel._models
        assert not hasattr(scirex.models, "GIWO")

        class Config(dict):
            model = {
                "model_arch": "GIWO",
                "data_channels": 3,
                "out_channels": 2,
                "gno_use_open3d": False,
                "wno_size": (6, 6, 6),
                "wno_hidden_channels": 8,
                "wno_n_layers": 2,
                "wno_level": 1,
            }

        model = get_model_jax(Config())
        assert type(model).__module__ == "scirex.experimental.giwo", type(model)
        """
    )
    src_dir = Path(scirex.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(src_dir), "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
