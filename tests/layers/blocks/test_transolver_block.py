# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for TransolverBlock (JAX/Flax port of the reference
``Transolver_block``: ``fx = Attn(LN1(fx)) + fx; fx = mlp(LN2(fx)) + fx;
if last_layer: mlp2(LN3(fx))``).

Covers output shape across the three geometry dispatch branches (irregular,
structured_2d, structured_3d), the pre-norm + residual structure (three
LayerNorms present, none for the non-last-layer case beyond the two
attn/mlp norms), the ``last_layer`` head changing output dim via ``out_dim``,
and gradient flow through every parameter.
"""

from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.transolver_block import TransolverBlock


def _init_and_apply(
    model: nn.Module, x: jnp.ndarray, key: jax.Array | None = None, **call_kwargs: Any
) -> tuple[jnp.ndarray, Any]:
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, x, **call_kwargs)
    return cast(jnp.ndarray, model.apply(variables, x, **call_kwargs)), variables


def _count_layernorms(params: dict) -> int:
    count = 0

    def walk(d: dict) -> None:
        nonlocal count
        for k, v in d.items():
            if isinstance(v, dict):
                if "scale" in v and "bias" in v and len(v) == 2:
                    # heuristic: LayerNorm params are exactly {scale, bias}
                    count += 1
                walk(v)

    walk(params)
    return count


# ---------------------------------------------------------------------------
# Output shape across geometries
# ---------------------------------------------------------------------------


def test_transolver_block_irregular_output_shape() -> None:
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 20, 32
    x = jax.random.normal(key, (B, N, hidden), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=4, hidden_dim=hidden, geometry="irregular", slice_num=8
    )
    out, _ = _init_and_apply(block, x, key)
    assert out.shape == (B, N, hidden)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_transolver_block_structured_2d_output_shape() -> None:
    key = jax.random.PRNGKey(0)
    H, W, hidden, B = 4, 4, 16, 2
    x = jax.random.normal(key, (B, H * W, hidden), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2, hidden_dim=hidden, geometry="structured_2d", H=H, W=W, slice_num=4
    )
    out, _ = _init_and_apply(block, x, key)
    assert out.shape == (B, H * W, hidden)


def test_transolver_block_structured_3d_output_shape() -> None:
    key = jax.random.PRNGKey(0)
    H, W, D, hidden, B = 2, 2, 2, 16, 2
    x = jax.random.normal(key, (B, H * W * D, hidden), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2,
        hidden_dim=hidden,
        geometry="structured_3d",
        H=H,
        W=W,
        D=D,
        slice_num=4,
    )
    out, _ = _init_and_apply(block, x, key)
    assert out.shape == (B, H * W * D, hidden)


def test_transolver_block_unknown_geometry_raises() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 5, 8), dtype=jnp.float32)
    block = TransolverBlock(num_heads=2, hidden_dim=8, geometry="bogus", slice_num=4)
    with pytest.raises(ValueError, match="Unknown geometry"):
        block.init(key, x)


# ---------------------------------------------------------------------------
# Pre-norm + residual structure (the bug this file guards against: a prior
# version of TransolverBlock had no LayerNorm and no residual at all).
# ---------------------------------------------------------------------------


def test_transolver_block_has_two_layernorms_when_not_last_layer() -> None:
    """Non-last-layer block has exactly 2 LayerNorm instances (pre-attn, pre-mlp)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2, hidden_dim=16, geometry="irregular", slice_num=4, last_layer=False
    )
    variables = block.init(key, x)
    assert _count_layernorms(variables["params"]) == 2


def test_transolver_block_has_three_layernorms_when_last_layer() -> None:
    """last_layer=True adds a third LayerNorm (pre-final-Dense)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2,
        hidden_dim=16,
        geometry="irregular",
        slice_num=4,
        last_layer=True,
        out_dim=3,
    )
    variables = block.init(key, x)
    assert _count_layernorms(variables["params"]) == 3


def test_transolver_block_residual_makes_output_differ_from_raw_submodule_output() -> (
    None
):
    """The block's residuals mean output != a plain Attn->MLP pipeline without skips.

    We check this indirectly: scaling the input by a large constant should
    not blow up the output the way a norm-free, residual-free stack might,
    since LayerNorm normalizes the attention/MLP branch inputs regardless of
    input scale, and residuals reintroduce the (now-normalized-invariant)
    raw signal.
    """
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2, hidden_dim=16, geometry="irregular", slice_num=4
    )
    variables = block.init(key, x)
    out_small = cast(jax.Array, block.apply(variables, x))
    out_large = cast(jax.Array, block.apply(variables, x * 1000.0))
    # LayerNorm makes the attn/mlp branches scale-invariant; only the
    # residual carries the raw (scaled) signal through, so output should
    # scale roughly linearly with input, not explode combinatorially.
    assert bool(jnp.all(jnp.isfinite(out_large)))
    assert not jnp.allclose(out_small, out_large)


# ---------------------------------------------------------------------------
# last_layer head dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dim", [1, 4, 7])
def test_transolver_block_last_layer_out_dim(out_dim: int) -> None:
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 12, 16
    x = jax.random.normal(key, (B, N, hidden), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2,
        hidden_dim=hidden,
        geometry="irregular",
        slice_num=4,
        last_layer=True,
        out_dim=out_dim,
    )
    out, _ = _init_and_apply(block, x, key)
    assert out.shape == (B, N, out_dim)


# ---------------------------------------------------------------------------
# Dropout and gradient flow
# ---------------------------------------------------------------------------


def test_transolver_block_dropout_train_vs_eval() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2, hidden_dim=16, geometry="irregular", slice_num=4, dropout_rate=0.5
    )
    variables = block.init(
        {"params": key, "dropout": jax.random.PRNGKey(1)}, x, deterministic=True
    )

    out1 = cast(
        jax.Array,
        block.apply(
            variables, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(2)}
        ),
    )
    out2 = cast(
        jax.Array,
        block.apply(
            variables, x, deterministic=True, rngs={"dropout": jax.random.PRNGKey(3)}
        ),
    )
    assert jnp.allclose(out1, out2)

    out3 = cast(
        jax.Array,
        block.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(2)}
        ),
    )
    out4 = cast(
        jax.Array,
        block.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(3)}
        ),
    )
    assert not jnp.allclose(out3, out4)


def test_transolver_block_no_unused_params() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2,
        hidden_dim=16,
        geometry="irregular",
        slice_num=4,
        last_layer=True,
        out_dim=3,
    )
    variables = block.init(key, x)

    def loss(params: dict) -> jnp.ndarray:
        out = cast(jax.Array, block.apply({"params": params}, x))
        return jnp.sum(out**2)

    grads = jax.grad(loss)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool((g != 0).any()) for g in leaves)


@pytest.mark.parametrize("act", [nn.gelu, nn.relu, nn.silu])
def test_transolver_block_activation_options(act: Callable) -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    block = TransolverBlock(
        num_heads=2, hidden_dim=16, geometry="irregular", slice_num=4, act=act
    )
    out, _ = _init_and_apply(block, x, key)
    assert out.shape == (2, 10, 16)


# ---------------------------------------------------------------------------
# Parameter-tree key names (checkpoint compatibility)
# ---------------------------------------------------------------------------
# Flax derives the key of an unnamed submodule from its class name, so renaming
# a class can silently change the param tree and break existing checkpoints.
# These are the key paths produced by the pre-rename ``Transolver_Block_Jax``.

_BLOCK_COMMON_KEYS = {
    "Dense_0/bias",
    "Dense_0/kernel",
    "LayerNorm_0/bias",
    "LayerNorm_0/scale",
    "LayerNorm_1/bias",
    "LayerNorm_1/scale",
    "LayerNorm_2/bias",
    "LayerNorm_2/scale",
    "MLP_0/Dense_0/bias",
    "MLP_0/Dense_0/kernel",
    "MLP_0/Dense_1/bias",
    "MLP_0/Dense_1/kernel",
    "attn/temperature",
}
_STRUCTURED_ATTN_KEYS = {
    "attn/Conv_0/bias",
    "attn/Conv_0/kernel",
    "attn/Conv_1/bias",
    "attn/Conv_1/kernel",
    "attn/Dense_0/bias",
    "attn/Dense_0/kernel",
    "attn/Dense_1/kernel",
    "attn/Dense_2/kernel",
    "attn/Dense_3/kernel",
    "attn/Dense_4/bias",
    "attn/Dense_4/kernel",
}
_EXPECTED_BLOCK_KEYS = {
    "irregular": _BLOCK_COMMON_KEYS
    | {
        "attn/Dense_0/bias",
        "attn/Dense_0/kernel",
        "attn/Dense_1/bias",
        "attn/Dense_1/kernel",
        "attn/Dense_2/bias",
        "attn/Dense_2/kernel",
        "attn/Dense_3/kernel",
        "attn/Dense_4/kernel",
        "attn/Dense_5/kernel",
        "attn/Dense_6/bias",
        "attn/Dense_6/kernel",
    },
    "structured_2d": _BLOCK_COMMON_KEYS | _STRUCTURED_ATTN_KEYS,
    "structured_3d": _BLOCK_COMMON_KEYS | _STRUCTURED_ATTN_KEYS,
}


def _key_paths(tree: dict, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else k
        paths |= _key_paths(v, path) if isinstance(v, dict) else {path}
    return paths


@pytest.mark.parametrize("geometry", ["irregular", "structured_2d", "structured_3d"])
def test_transolver_block_param_tree_keys_unchanged(geometry: str) -> None:
    """Class renames must not change the param-tree keys of saved checkpoints."""
    kw: dict[str, Any] = {
        "irregular": {},
        "structured_2d": {"H": 4, "W": 4},
        "structured_3d": {"H": 2, "W": 2, "D": 2},
    }[geometry]
    n = 8 if geometry == "structured_3d" else 16
    block = TransolverBlock(
        num_heads=2,
        hidden_dim=16,
        geometry=geometry,
        slice_num=4,
        last_layer=True,
        **kw,
    )
    x = jnp.ones((1, n, 16), dtype=jnp.float32)
    shapes = jax.eval_shape(block.init, jax.random.PRNGKey(0), x)
    assert _key_paths(shapes["params"]) == _EXPECTED_BLOCK_KEYS[geometry]
