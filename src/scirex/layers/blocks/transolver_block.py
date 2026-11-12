# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from scirex.layers.blocks.physics_attention import (
    IrregularMesh,
    StructuredMesh2D,
    StructuredMesh3D,
)
from scirex.layers.common.mlp import MLP


class TransolverBlock(nn.Module):
    """One Transolver block: physics attention and an MLP, each with a residual connection.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    hidden_dim : int
        Number of channels.
    dropout_rate : float, optional
        Dropout rate. Default: 0.0.
    geometry : str, optional
        `"irregular"`, `"structured_2d"` or `"structured_3d"`. The structured options
        need `H` and `W` (and `D` in 3D). Default: `"irregular"`.
    mlp_ratio : int, optional
        Width of the MLP as a multiple of `hidden_dim`. Default: 4.
    last_layer : bool, optional
        Whether this is the last block, which projects to `out_dim` channels. Default: False.
    out_dim : int, optional
        Number of output channels of the last block. Default: 1.
    slice_num : int, optional
        Number of slices of the attention. Default: 32.
    """

    num_heads: int
    hidden_dim: int
    dropout_rate: float = 0.0
    geometry: str = "irregular"  # "irregular" | "structured_2d" | "structured_3d"
    act: Callable = nn.gelu
    mlp_ratio: int = 4
    last_layer: bool = False
    out_dim: int = 1
    slice_num: int = 32
    H: int | None = None
    W: int | None = None
    D: int | None = None
    kernel: int = 3
    # Flax's LayerNorm default. torch.nn.LayerNorm uses 1e-5; ports of
    # PyTorch recipes that need bit-level parity can pass that instead.
    layer_norm_epsilon: float = 1e-6
    # Init of the structured-2D slice convolutions, see StructuredMesh2D.
    conv_init: str = "flax"

    def setup(self) -> None:
        dim_head = self.hidden_dim // self.num_heads
        common: dict[str, Any] = dict(
            dim=self.hidden_dim,
            heads=self.num_heads,
            dim_head=dim_head,
            dropout_rate=self.dropout_rate,
            slice_num=self.slice_num,
        )
        if self.geometry == "irregular":
            self.attn: Any = IrregularMesh(**common)
        elif self.geometry == "structured_2d":
            assert (
                self.H is not None and self.W is not None
            ), "geometry='structured_2d' requires H and W to be set."
            self.attn = StructuredMesh2D(
                **common,
                H=self.H,
                W=self.W,
                kernel=self.kernel,
                conv_init=self.conv_init,
            )
        elif self.geometry == "structured_3d":
            assert (
                self.H is not None and self.W is not None and self.D is not None
            ), "geometry='structured_3d' requires H, W, and D to be set."
            self.attn = StructuredMesh3D(
                **common, H=self.H, W=self.W, D=self.D, kernel=self.kernel
            )
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = (
            self.attn(
                nn.LayerNorm(epsilon=self.layer_norm_epsilon)(x),
                deterministic=deterministic,
            )
            + x
        )
        x = (
            MLP(
                features=[self.hidden_dim * self.mlp_ratio, self.hidden_dim],
                activation=self.act,
                dropout_rate=0.0,
                activation_last=False,
                residual=False,
            )(
                nn.LayerNorm(epsilon=self.layer_norm_epsilon)(x),
                deterministic=deterministic,
            )
            + x
        )
        if self.last_layer:
            x = nn.Dense(self.out_dim)(nn.LayerNorm(epsilon=self.layer_norm_epsilon)(x))
        return x
