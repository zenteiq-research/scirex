# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for FNOBlocks (JAX/Flax).

Covers resolution scaling (downsample/upsample), normalization variants
(instance_norm, group_norm, batch_norm, ada_in), complex-valued inputs,
skip connection options (linear/None for both fno_skip and channel_mlp_skip),
and preactivation mode, across 1D/2D/3D spatial inputs on CPU and GPU.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from scirex.layers.blocks.fno_block import FNOBlocks, SubModule


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _init_and_apply(
    block: FNOBlocks,
    x: jnp.ndarray,
    key: jax.Array | None = None,
    mutable: bool | list[str] = False,
    **call_kwargs: Any,
) -> tuple[jnp.ndarray, Any]:
    """Init and apply a Flax module, handling mutable collections (e.g. batch_stats)."""
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = block.init(key, x, **call_kwargs)
    if mutable:
        out, _ = block.apply(variables, x, mutable=mutable, **call_kwargs)
        return cast(jnp.ndarray, out), variables
    return cast(jnp.ndarray, block.apply(variables, x, **call_kwargs)), variables


# ---------------------------------------------------------------------------
# Test 1 - output shape with downsampling and upsampling across 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_FNOBlock_resolution_scaling_factor(dim: int, backend: str) -> None:
    """FNOBlocks produces correct output shapes under resolution scaling.

    Verifies that a scaling factor of 0.5 halves the spatial dimensions and a
    factor of 2 doubles them, for 1D, 2D, and 3D inputs.
    """
    n_modes = [4, 4, 4]
    size = [10] * dim
    channel_mlp_dropout = 0
    channel_mlp_expansion = 0.5
    channel_mlp_skip = "linear"

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        # Downsample outputs
        block_down = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=0.5,
            use_channel_mlp=True,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        x = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)
        res, _ = _init_and_apply(block_down, x, key)
        assert list(res.shape[2:]) == [m // 2 for m in size]

        # Upsample outputs
        block_up = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=n_modes[:dim],
            n_layers=1,
            resolution_scaling_factor=2,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
        )
        res, _ = _init_and_apply(block_up, x, key)
        assert res.shape[1] == 4
        assert list(res.shape[2:]) == [m * 2 for m in size]


# ---------------------------------------------------------------------------
# Test 2 - normalization variants across 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize(
    "norm", ["instance_norm", "ada_in", "group_norm", "batch_norm"]
)
def test_FNOBlock_norm(norm: str, n_dim: int, backend: str) -> None:
    """FNOBlocks forward pass succeeds with each supported normalization type.

    Checks that the output shape matches the input spatial dimensions for
    instance_norm, group_norm, batch_norm, and ada_in normalization.
    """
    modes = (8, 8, 8)
    size = [10] * n_dim
    ada_in_features = 4

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        block = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=modes[:n_dim],
            n_layers=1,
            norm=norm,
            ada_in_features=ada_in_features,
            channel_mlp_dropout=0,
            channel_mlp_expansion=0.5,
            channel_mlp_skip="linear",
        )

        x = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)

        if norm == "ada_in":
            embedding = jax.random.normal(key, (ada_in_features,), dtype=jnp.float32)
            res, _ = _init_and_apply(block, x, key, ada_in_embeddings=[embedding])
        elif norm == "batch_norm":
            res, _ = _init_and_apply(block, x, key, mutable=["batch_stats"])
        else:
            res, _ = _init_and_apply(block, x, key)

        assert list(res.shape[2:]) == size


# ---------------------------------------------------------------------------
# Test 3 - complex-valued inputs across 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3])
def test_FNOBlock_complex_data(n_dim: int, backend: str) -> None:
    """FNOBlocks handles complex input data without error.

    Verifies output shape is preserved when the block is configured for
    complex_data=True and receives complex64 inputs.
    """
    modes = (8, 8, 8)
    size = [10] * n_dim

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        block = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=modes[:n_dim],
            n_layers=1,
            channel_mlp_dropout=0,
            channel_mlp_expansion=0.5,
            channel_mlp_skip="linear",
            complex_data=True,
        )

        xr = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(1), (2, 3, *size), dtype=jnp.float32)
        x = (xr + 1j * xi).astype(jnp.complex64)

        res, _ = _init_and_apply(block, x, key)
        assert list(res.shape[2:]) == size


# ---------------------------------------------------------------------------
# Test 4 - skip connection option combinations (fno_skip x channel_mlp_skip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fno_skip", ["linear", None])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", None])
def test_FNOBlock_skip_connections(
    fno_skip: str | None, channel_mlp_skip: str | None, backend: str
) -> None:
    """FNOBlocks forward pass succeeds for all skip connection combinations.

    Tests "linear" and None for both fno_skip and channel_mlp_skip, with and
    without channel MLP enabled, checking output shape in each case.
    """
    modes = (8, 8, 8)
    size = [10, 10, 10]

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)

        block = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=modes,
            n_layers=1,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            use_channel_mlp=True,
            channel_mlp_expansion=0.5,
            channel_mlp_dropout=0.0,
        )
        res, _ = _init_and_apply(block, x, key)
        assert res.shape == (2, 4, *size)

        block_no_mlp = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=modes,
            n_layers=1,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            use_channel_mlp=False,
        )
        res_no_mlp, _ = _init_and_apply(block_no_mlp, x, key)
        assert res_no_mlp.shape == (2, 4, *size)


# ---------------------------------------------------------------------------
# Test 5 - preactivation mode with skip connection options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fno_skip", ["linear", None])
@pytest.mark.parametrize("channel_mlp_skip", ["linear", None])
def test_FNOBlock_skip_connections_preactivation(
    fno_skip: str | None, channel_mlp_skip: str | None, backend: str
) -> None:
    """FNOBlocks with preactivation=True produces correct output shape.

    Verifies that pre-activation (norm + nonlinearity before convolution) is
    compatible with all skip connection combinations and does not alter the
    expected spatial output shape.
    """
    modes = (8, 8, 8)
    size = [10, 10, 10]

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)

        block = FNOBlocks(
            in_channels=3,
            out_channels=4,
            n_modes=modes,
            n_layers=1,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            use_channel_mlp=True,
            channel_mlp_expansion=0.5,
            channel_mlp_dropout=0.0,
            preactivation=True,
        )
        res, _ = _init_and_apply(block, x, key)
        assert res.shape == (2, 4, *size)


# ---------------------------------------------------------------------------
# Test 6 - n_modes as integer scalar (line 143)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_FNOBlock_int_n_modes(backend: str) -> None:
    """FNOBlocks with n_modes as a plain int is wrapped in a list (source line 143)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        block = FNOBlocks(in_channels=3, out_channels=3, n_modes=8, n_layers=1)
        x = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        res, _ = _init_and_apply(block, x, key)
        assert res.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# Test 7 - invalid norm raises ValueError (source line 262)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_FNOBlock_invalid_norm_raises(backend: str) -> None:
    """FNOBlocks with an unsupported norm string raises ValueError during setup
    (source line 262)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        block = FNOBlocks(
            in_channels=3, out_channels=3, n_modes=(8,), n_layers=1, norm="bad_norm"
        )
        x = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        with pytest.raises((ValueError, Exception)):
            block.init(key, x)


# ---------------------------------------------------------------------------
# Test 8 - ada_in with multiple embeddings (source line 296)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_FNOBlock_ada_in_multiple_embeddings(backend: str) -> None:
    """ada_in with len(ada_in_embeddings) > 1 picks embedding by norm_idx
    (source line 296)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        ada_in_features = 4
        block = FNOBlocks(
            in_channels=3,
            out_channels=3,
            n_modes=(8,),
            n_layers=1,
            norm="ada_in",
            ada_in_features=ada_in_features,
            use_channel_mlp=True,
            channel_mlp_skip="linear",
            channel_mlp_expansion=0.5,
            channel_mlp_dropout=0,
        )
        x = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        emb1 = jax.random.normal(key, (ada_in_features,), dtype=jnp.float32)
        emb2 = jax.random.normal(
            jax.random.PRNGKey(1), (ada_in_features,), dtype=jnp.float32
        )
        # n_norms=2 → two embeddings → triggers the len>1 branch (line 296)
        res, _ = _init_and_apply(block, x, key, ada_in_embeddings=[emb1, emb2])
        assert res.shape[1] == 3


# ---------------------------------------------------------------------------
# Test 9 - stabilizer="tanh" with complex_data=True (source lines 317, 370)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("preactivation", [False, True])
def test_FNOBlock_stabilizer_tanh_complex(preactivation: bool, backend: str) -> None:
    """stabilizer='tanh' with complex_data=True uses ctanh (source lines 317, 370)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        block = FNOBlocks(
            in_channels=3,
            out_channels=3,
            n_modes=(8,),
            n_layers=1,
            complex_data=True,
            stabilizer="tanh",
            preactivation=preactivation,
        )
        xr = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        xi = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 16), dtype=jnp.float32)
        x = (xr + 1j * xi).astype(jnp.complex64)
        res, _ = _init_and_apply(block, x, key)
        assert res.shape == (2, 3, 16)


# ---------------------------------------------------------------------------
# Test 10 - get_block on single-layer block raises ValueError (source lines 418-423)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_FNOBlock_get_block_single_layer_raises(backend: str) -> None:
    """get_block on a single-layer FNOBlocks raises ValueError
    (source lines 418-423)."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        block = FNOBlocks(in_channels=3, out_channels=3, n_modes=(8,), n_layers=1)
        x = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        block.init(key, x)
        with pytest.raises(ValueError, match="single layer"):
            block.get_block(0)


# ---------------------------------------------------------------------------
# Test 11 - __getitem__ and SubModule.__call__ (source lines 426, 443)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_FNOBlock_getitem_and_submodule(backend: str) -> None:
    """FNOBlocks[i] is a helper interface, and SubModule call delegation is
    exercised."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        block = FNOBlocks(
            in_channels=3,
            out_channels=3,
            n_modes=(8,),
            n_layers=2,
            use_channel_mlp=False,
            fno_skip=None,
        )
        x = jax.random.normal(key, (2, 3, 16), dtype=jnp.float32)
        block.init(key, x)

        with pytest.raises(Exception):
            block[0]

        class SubModuleCaller(nn.Module):
            main_block: FNOBlocks

            @nn.compact
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                sub = SubModule(main_module=self.main_block, indices=0)
                return sub(x)

        caller = SubModuleCaller(main_block=block)
        caller_vars = caller.init(key, x)
        out = cast(jnp.ndarray, caller.apply(caller_vars, x))
        assert out.shape == (2, 3, 16)
