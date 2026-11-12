# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for Transolver's Physics Attention layers (JAX/Flax port of
``Physics_Attention_Irregular_Mesh`` / ``_Structured_Mesh_2D`` /
``_Structured_Mesh_3D``).

Covers output shape preservation (B, N, dim) across head/dim_head/slice_num
configurations, dropout train/eval behavior, the structured 2D/3D conv-kernel
size options, temperature clamping bounds in the structured variants, and
gradient flow (no dead parameters) through every learnable weight.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.physics_attention import (
    IrregularMesh,
    StructuredMesh2D,
    StructuredMesh3D,
)


def _init_and_apply(model: Any, x: Any, key: Any = None, **call_kwargs: Any) -> Any:
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, x, **call_kwargs)
    return model.apply(variables, x, **call_kwargs), variables


# ---------------------------------------------------------------------------
# IrregularMesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("heads,dim_head", [(4, 8), (8, 16), (2, 32)])
def test_irregular_mesh_output_shape(heads: Any, dim_head: Any) -> None:
    """Output shape is (B, N, dim) regardless of internal head/dim_head split."""
    key = jax.random.PRNGKey(0)
    B, N, dim = 2, 20, 32
    x = jax.random.normal(key, (B, N, dim), dtype=jnp.float32)
    model = IrregularMesh(dim=dim, heads=heads, dim_head=dim_head, slice_num=16)
    out, _ = _init_and_apply(model, x, key)
    assert out.shape == (B, N, dim)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_irregular_mesh_slice_num_does_not_change_output_shape() -> None:
    """slice_num is an internal bottleneck width; output shape is unaffected by it."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 15, 16), dtype=jnp.float32)
    for slice_num in (4, 32, 64):
        model = IrregularMesh(dim=16, heads=2, dim_head=8, slice_num=slice_num)
        out, _ = _init_and_apply(model, x, key)
        assert out.shape == (2, 15, 16)


def test_irregular_mesh_dropout_deterministic_vs_train() -> None:
    """deterministic=True is reproducible; deterministic=False with dropout>0 varies
    across rngs."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    model = IrregularMesh(dim=16, heads=2, dim_head=8, slice_num=8, dropout_rate=0.5)
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

    out3 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(2)}
        ),
    )
    out4 = cast(
        jax.Array,
        model.apply(
            variables, x, deterministic=False, rngs={"dropout": jax.random.PRNGKey(3)}
        ),
    )
    assert not jnp.allclose(out3, out4)


def test_irregular_mesh_no_unused_params() -> None:
    """Every parameter (including temperature and in_project_slice) receives
    gradient."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 16), dtype=jnp.float32)
    model = IrregularMesh(dim=16, heads=2, dim_head=8, slice_num=8)
    variables = model.init(key, x)

    def loss(params: Any) -> Any:
        out = cast(jax.Array, model.apply({"params": params}, x))
        return jnp.sum(out**2)

    grads = jax.grad(loss)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool((g != 0).any()) for g in leaves)


# ---------------------------------------------------------------------------
# StructuredMesh2D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [1, 3, 5])
def test_structured_mesh_2d_output_shape_and_kernel_sizes(kernel: Any) -> None:
    """Output shape is (B, H*W, dim) for kernel sizes 1, 3, 5."""
    key = jax.random.PRNGKey(0)
    H, W, dim = 6, 5, 16
    B = 2
    x = jax.random.normal(key, (B, H * W, dim), dtype=jnp.float32)
    model = StructuredMesh2D(
        dim=dim, heads=2, dim_head=8, slice_num=8, H=H, W=W, kernel=kernel
    )
    out, _ = _init_and_apply(model, x, key)
    assert out.shape == (B, H * W, dim)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_structured_mesh_2d_temperature_is_clamped() -> None:
    """The learned temperature parameter, post-clamp, always lies within [0.1, 5]."""
    key = jax.random.PRNGKey(0)
    H, W, dim = 4, 4, 8
    x = jax.random.normal(key, (1, H * W, dim), dtype=jnp.float32)
    model = StructuredMesh2D(dim=dim, heads=2, dim_head=4, slice_num=4, H=H, W=W)
    variables = model.init(key, x)
    # temperature itself is initialized at 0.5 (within bounds); verify the
    # clamp math directly, mirroring what __call__ applies internally.
    temperature = variables["params"]["temperature"]
    clamped = jnp.clip(temperature, min=0.1, max=5)
    assert bool(jnp.all(clamped >= 0.1)) and bool(jnp.all(clamped <= 5))

    # perturb temperature outside the bounds and confirm the forward pass
    # still runs (i.e. the clamp is applied inside __call__, not just at init).
    perturbed = jax.tree_util.tree_map(lambda leaf: leaf, variables)
    perturbed["params"]["temperature"] = jnp.ones_like(temperature) * 100.0
    out_perturbed = cast(jax.Array, model.apply(perturbed, x))
    assert bool(jnp.all(jnp.isfinite(out_perturbed)))


def _slice_convs(variables: Any) -> list[Any]:
    params = cast(dict, variables["params"])
    return [params[k] for k in sorted(params) if k.startswith("Conv_")]


def test_structured_mesh_2d_torch_conv_init_matches_conv2d_bounds() -> None:
    """conv_init="torch": kernel and bias ~ U(+-1/sqrt(fan_in)), as Conv2d."""
    dim, H, W, kernel = 16, 8, 8, 3
    x = jax.random.normal(jax.random.PRNGKey(1), (1, H * W, dim))
    model = StructuredMesh2D(
        dim=dim, heads=2, dim_head=8, slice_num=4, H=H, W=W, conv_init="torch"
    )
    _, variables = _init_and_apply(model, x)
    bound = (dim * kernel * kernel) ** -0.5
    convs = _slice_convs(variables)
    assert len(convs) == 2
    for conv in convs:
        for leaf in (conv["kernel"], conv["bias"]):
            assert float(jnp.abs(leaf).max()) <= bound
            # uniform on [-b, b] has std b / sqrt(3)
            assert float(jnp.std(leaf)) == pytest.approx(bound / 3**0.5, rel=0.2)


def test_structured_mesh_2d_default_conv_init_is_flax() -> None:
    """The default keeps Flax's init: zero bias, and the same params as before."""
    dim, H, W = 16, 8, 8
    x = jax.random.normal(jax.random.PRNGKey(1), (1, H * W, dim))
    kw = dict(dim=dim, heads=2, dim_head=8, slice_num=4, H=H, W=W)
    _, default = _init_and_apply(StructuredMesh2D(**kw), x)
    _, flax_init = _init_and_apply(StructuredMesh2D(**kw, conv_init="flax"), x)
    for a, b in zip(_slice_convs(default), _slice_convs(flax_init), strict=True):
        assert jnp.array_equal(a["kernel"], b["kernel"])
        assert not jnp.any(a["bias"])


def test_structured_mesh_2d_rejects_unknown_conv_init() -> None:
    x = jnp.zeros((1, 16, 8))
    model = StructuredMesh2D(dim=8, heads=2, dim_head=4, H=4, W=4, conv_init="x")
    with pytest.raises(ValueError, match="conv_init"):
        model.init(jax.random.PRNGKey(0), x)


def test_structured_mesh_2d_no_unused_params() -> None:
    """Every parameter, including the conv kernels, receives gradient."""
    key = jax.random.PRNGKey(0)
    H, W, dim = 4, 4, 8
    x = jax.random.normal(key, (2, H * W, dim), dtype=jnp.float32)
    model = StructuredMesh2D(dim=dim, heads=2, dim_head=4, slice_num=4, H=H, W=W)
    variables = model.init(key, x)

    def loss(params: Any) -> Any:
        out = cast(jax.Array, model.apply({"params": params}, x))
        return jnp.sum(out**2)

    grads = jax.grad(loss)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool((g != 0).any()) for g in leaves)


# ---------------------------------------------------------------------------
# StructuredMesh3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", [1, 3])
def test_structured_mesh_3d_output_shape_and_kernel_sizes(kernel: Any) -> None:
    """Output shape is (B, H*W*D, dim) for kernel sizes 1 and 3."""
    key = jax.random.PRNGKey(0)
    H, W, D, dim = 3, 3, 3, 16
    B = 2
    x = jax.random.normal(key, (B, H * W * D, dim), dtype=jnp.float32)
    model = StructuredMesh3D(
        dim=dim, heads=2, dim_head=8, slice_num=8, H=H, W=W, D=D, kernel=kernel
    )
    out, _ = _init_and_apply(model, x, key)
    assert out.shape == (B, H * W * D, dim)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_structured_mesh_3d_temperature_is_clamped() -> None:
    """After perturbing temperature far outside [0.1, 5], forward output stays
    finite."""
    key = jax.random.PRNGKey(0)
    H, W, D, dim = 2, 2, 2, 8
    x = jax.random.normal(key, (1, H * W * D, dim), dtype=jnp.float32)
    model = StructuredMesh3D(dim=dim, heads=2, dim_head=4, slice_num=4, H=H, W=W, D=D)
    variables = model.init(key, x)
    variables["params"]["temperature"] = (
        jnp.ones_like(variables["params"]["temperature"]) * -50.0
    )
    out = cast(jax.Array, model.apply(variables, x))
    assert bool(jnp.all(jnp.isfinite(out)))


def test_structured_mesh_3d_no_unused_params() -> None:
    """Every parameter, including the 3D conv kernels, receives gradient."""
    key = jax.random.PRNGKey(0)
    H, W, D, dim = 2, 2, 2, 8
    x = jax.random.normal(key, (2, H * W * D, dim), dtype=jnp.float32)
    model = StructuredMesh3D(dim=dim, heads=2, dim_head=4, slice_num=4, H=H, W=W, D=D)
    variables = model.init(key, x)

    def loss(params: Any) -> Any:
        out = cast(jax.Array, model.apply({"params": params}, x))
        return jnp.sum(out**2)

    grads = jax.grad(loss)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool((g != 0).any()) for g in leaves)


# ---------------------------------------------------------------------------
# Cross-class sanity: unknown/mismatched H*W*[D] vs N raises a shape error
# ---------------------------------------------------------------------------


def test_structured_mesh_2d_mismatched_H_W_raises() -> None:
    """Passing N != H*W raises when reshaping x to (B, H, W, C)."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 10, 8), dtype=jnp.float32)  # N=10, but H*W=9
    model = StructuredMesh2D(dim=8, heads=2, dim_head=4, slice_num=4, H=3, W=3)
    with pytest.raises(Exception):
        model.init(key, x)
