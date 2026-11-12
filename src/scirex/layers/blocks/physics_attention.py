# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


def _conv_init_kwargs(conv_init: str, fan_in: int) -> dict:
    """`kernel_init`/`bias_init` for `nn.Conv` under `conv_init`."""
    if conv_init == "flax":
        return {}
    if conv_init == "torch":
        bound = fan_in**-0.5

        def bias_init(
            key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32
        ) -> jax.Array:
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        return dict(
            kernel_init=nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform"),
            bias_init=bias_init,
        )
    raise ValueError(f"conv_init must be 'flax' or 'torch', got {conv_init!r}")


class IrregularMesh(nn.Module):
    """Physics attention of the Transolver for points of an unstructured mesh, input shape `(batch, points, dim)`.

    The points are softly assigned to `slice_num` learned slices, attention is applied
    between the slices, and the result is sent back to the points. The cost grows with
    the number of slices, not with the square of the number of points.

    Parameters
    ----------
    dim : int
        Number of input and output channels.
    heads : int, optional
        Number of attention heads. Default: 8.
    dim_head : int, optional
        Channels per head. Default: 64.
    dropout_rate : float, optional
        Dropout rate. Default: 0.0.
    slice_num : int, optional
        Number of slices. Default: 64 (32 for the 3D grid).
    """

    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout_rate: float = 0.0
    slice_num: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head**-0.5
        B, N, C = x.shape

        ## Slice
        fx_mid = (
            nn.Dense(inner_dim, use_bias=True)(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N, C
        x_mid = (
            nn.Dense(inner_dim, use_bias=True)(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N , C

        temperature = self.param(
            "temperature",
            lambda key, shape: jnp.full(shape, 0.5, dtype=jnp.float32),
            (1, self.heads, 1, 1),
        )  # 1, H, 1, 1

        in_project_slice = nn.Dense(
            self.slice_num,
            use_bias=True,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
        )(x_mid)  # B, H, N, G

        # The irregular-mesh reference never clamps temperature before this
        # division, unlike StructuredMesh2D/3D below -- this mirrors
        # Physics_Attention_Irregular_Mesh upstream exactly; don't add a
        # clamp here to match the structured variants.
        slice_weights = nn.softmax(
            in_project_slice / temperature, axis=-1
        )  # B, H, N, G
        slice_norms = slice_weights.sum(axis=2)  # B, H, G
        slice_token = jnp.einsum(
            "bhnc, bhng -> bhgc", fx_mid, slice_weights
        )  # B, H, G, C
        # 1e-5 matches Physics_Attention_Irregular_Mesh's reference epsilon
        # (and the StructuredMesh2D/3D variants below), for exact parity.
        slice_token = slice_token / (slice_norms[..., None] + 1e-5)  # B, H, G, C

        ## Attention Among Slice Tokens
        q_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C
        k_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C
        v_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C

        dots = (
            jnp.matmul(q_slice_token, k_slice_token.transpose(0, 1, 3, 2)) * scale
        )  # B, H, G, G
        attn = nn.softmax(dots, axis=-1)  # B, H, G, G
        if self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)

        out_slice_token = jnp.matmul(attn, v_slice_token)  # B, H, G, D

        ## Deslice
        out_x = jnp.einsum(
            "bhgd, bhng -> bhnd", out_slice_token, slice_weights
        )  # B, H, N, C
        out_x = rearrange(out_x, "b h n c -> b n (h c)")  # B, N, H*C
        out_x = nn.Dense(self.dim, use_bias=True)(out_x)  # B, N, C
        if self.dropout_rate > 0:
            out_x = nn.Dropout(rate=self.dropout_rate)(
                out_x, deterministic=deterministic
            )

        return out_x


class StructuredMesh2D(nn.Module):
    """Physics attention of the Transolver for a regular 2D grid of shape `H x W`, input shape `(batch, H*W, dim)`.

    The points are softly assigned to `slice_num` learned slices, attention is applied
    between the slices, and the result is sent back to the points. The cost grows with
    the number of slices, not with the square of the number of points.

    Parameters
    ----------
    dim : int
        Number of input and output channels.
    heads : int, optional
        Number of attention heads. Default: 8.
    dim_head : int, optional
        Channels per head. Default: 64.
    dropout_rate : float, optional
        Dropout rate. Default: 0.0.
    slice_num : int, optional
        Number of slices. Default: 64 (32 for the 3D grid).
    """

    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout_rate: float = 0.0
    slice_num: int = 64
    # H=101, W=31 mirror Physics_Attention_Structured_Mesh_2D's own upstream
    # defaults (airfoil's mesh shape) - kept only for parity with the
    # reference constructor signature. TransolverBlock always passes H
    # and W explicitly for every benchmark, so these defaults are never
    # actually relied on in practice.
    H: int = 101
    W: int = 31
    kernel: int = 3
    # Initialisation of the two slice-projection convolutions: "flax" keeps
    # Flax's default (lecun_normal kernel, zero bias); "torch" uses
    # torch.nn.Conv2d's default (kaiming_uniform(a=sqrt(5)), i.e.
    # U(+-1/sqrt(fan_in)) for the kernel, and U(+-1/sqrt(fan_in)) for the bias),
    # for ports of PyTorch recipes.
    conv_init: str = "flax"

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head**-0.5
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C)  # B, H, W, C
        conv_kwargs = _conv_init_kwargs(self.conv_init, C * self.kernel**2)

        ## Slice
        fx_mid = (
            nn.Conv(
                features=inner_dim,
                kernel_size=[self.kernel, self.kernel],
                strides=1,
                padding=[(self.kernel // 2, self.kernel // 2)] * 2,
                use_bias=True,
                **conv_kwargs,
            )(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N, C

        x_mid = (
            nn.Conv(
                features=inner_dim,
                kernel_size=[self.kernel, self.kernel],
                strides=1,
                padding=[(self.kernel // 2, self.kernel // 2)] * 2,
                use_bias=True,
                **conv_kwargs,
            )(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N , C

        temperature = self.param(
            "temperature",
            lambda key, shape: jnp.full(shape, 0.5, dtype=jnp.float32),
            (1, self.heads, 1, 1),
        )  # 1, H, 1, 1

        temp_clamp = jnp.clip(temperature, min=0.1, max=5)

        in_project_slice = nn.Dense(
            self.slice_num,
            use_bias=True,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
        )(x_mid)  # B, H, N, G

        slice_weights = nn.softmax(in_project_slice / temp_clamp, axis=-1)  # B, H, N, G
        slice_norms = slice_weights.sum(axis=2)  # B, H, G
        slice_token = jnp.einsum(
            "bhnc, bhng -> bhgc", fx_mid, slice_weights
        )  # B, H, G, C
        slice_token = slice_token / (slice_norms[..., None] + 1e-5)  # B, H, G, C

        ## Attention Among Slice Tokens
        q_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C
        k_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C
        v_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, G, C

        dots = (
            jnp.matmul(q_slice_token, k_slice_token.transpose(0, 1, 3, 2)) * scale
        )  # B, H, G, G
        attn = nn.softmax(dots, axis=-1)  # B, H, G, G
        if self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)

        out_slice_token = jnp.matmul(attn, v_slice_token)  # B, H, G, C

        ## Deslice
        out_x = jnp.einsum(
            "bhgc, bhng -> bhnc", out_slice_token, slice_weights
        )  # B, H, N, C
        out_x = rearrange(out_x, "b h n c -> b n (h c)")  # B, N, H*C
        out_x = nn.Dense(self.dim, use_bias=True)(out_x)  # B, N, C
        if self.dropout_rate > 0:
            out_x = nn.Dropout(rate=self.dropout_rate)(
                out_x, deterministic=deterministic
            )

        return out_x


class StructuredMesh3D(nn.Module):
    """Physics attention of the Transolver for a regular 3D grid of shape `H x W x D`, input shape `(batch, H*W*D, dim)`.

    The points are softly assigned to `slice_num` learned slices, attention is applied
    between the slices, and the result is sent back to the points. The cost grows with
    the number of slices, not with the square of the number of points.

    Parameters
    ----------
    dim : int
        Number of input and output channels.
    heads : int, optional
        Number of attention heads. Default: 8.
    dim_head : int, optional
        Channels per head. Default: 64.
    dropout_rate : float, optional
        Dropout rate. Default: 0.0.
    slice_num : int, optional
        Number of slices. Default: 64 (32 for the 3D grid).
    """

    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout_rate: float = 0.0
    slice_num: int = 32
    D: int = 32
    H: int = 32
    W: int = 32
    kernel: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        inner_dim = self.dim_head * self.heads
        scale = self.dim_head**-0.5
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C)  # B, H, W, D, C

        ## Slice
        fx_mid = (
            nn.Conv(
                features=inner_dim,
                kernel_size=[self.kernel, self.kernel, self.kernel],
                strides=(1, 1, 1),
                padding=[(self.kernel // 2, self.kernel // 2)] * 3,
                use_bias=True,
            )(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N, C

        x_mid = (
            nn.Conv(
                features=inner_dim,
                kernel_size=[self.kernel, self.kernel, self.kernel],
                strides=(1, 1, 1),
                padding=[(self.kernel // 2, self.kernel // 2)] * 3,
                use_bias=True,
            )(x)
            .reshape(B, N, self.heads, self.dim_head)
            .transpose(0, 2, 1, 3)
        )  # B, H, N , C

        temperature = self.param(
            "temperature",
            lambda key, shape: jnp.full(shape, 0.5, dtype=jnp.float32),
            (1, self.heads, 1, 1),
        )  # 1, H, 1, 1

        temp_clamp = jnp.clip(temperature, min=0.1, max=5)

        in_project_slice = nn.Dense(
            self.slice_num,
            use_bias=True,
            kernel_init=nn.initializers.orthogonal(scale=1.0),
        )(x_mid)  # B, H, N, D

        slice_weights = nn.softmax(in_project_slice / temp_clamp, axis=-1)  # B, H, N, D
        slice_norms = slice_weights.sum(axis=2)  # B, H, D
        slice_token = jnp.einsum(
            "bhnc, bhnd -> bhdc", fx_mid, slice_weights
        )  # B, H, D, C
        slice_token = slice_token / (slice_norms[..., None] + 1e-5)  # B, H, D, C

        ## Attention Among Slice Tokens
        q_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, D, C
        k_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, D, C
        v_slice_token = nn.Dense(self.dim_head, use_bias=False)(
            slice_token
        )  # B, H, D, C

        dots = (
            jnp.matmul(q_slice_token, k_slice_token.transpose(0, 1, 3, 2)) * scale
        )  # B, H, D, D
        attn = nn.softmax(dots, axis=-1)  # B, H, D, D
        if self.dropout_rate > 0:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=deterministic)

        out_slice_token = jnp.matmul(attn, v_slice_token)  # B, H, D, C

        ## Deslice
        out_x = jnp.einsum(
            "bhdc, bhnd -> bhnc", out_slice_token, slice_weights
        )  # B, H, N, C
        out_x = rearrange(out_x, "b h n c -> b n (h c)")  # B, N, H*C
        out_x = nn.Dense(self.dim, use_bias=True)(out_x)  # B, N, C
        if self.dropout_rate > 0:
            out_x = nn.Dropout(rate=self.dropout_rate)(
                out_x, deterministic=deterministic
            )

        return out_x
