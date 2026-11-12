# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Geometry-Informed Wavelet Neural Operator (GIWO).

Not a published architecture -- a "Geo-WNO" built the same way GINO is built,
by swapping GINO's latent operator. `GINO`
reads an irregular point cloud onto a fixed latent grid through an input GNO,
runs an FNO over that grid, then reads back off the grid onto arbitrary
output points through a second GNO. This class keeps that outer GNO-in /
GNO-out shell verbatim (same `GNOBlock`
import, same call signature on each) and replaces only the middle piece: a
`WNO` instead of GINO's
`lifting + FNOBlocks`.

That swap is simpler than it sounds because `WNO` already bundles its
own lifting and projection (`fc0`/`fc1`/`fc2`) around its wavelet
blocks, where GINO's `lifting` (a `ChannelMLP`) and `fno_blocks`
are separate pieces assembled by `latent_embedding`. So
`GIWO.latent_embedding` is a single `self.wno(in_p)` call, channels
last throughout -- no need for GINO's own channels-first permute dance around
`self.lifting`/`self.fno_blocks`, since that dance exists only to match
`FNOBlocks`'s channels-first
convention, which `WNO` does not share.

`wno_size` is the one genuinely new knob this adds over GINO: WNO's wavelet
transform needs the latent grid's spatial shape fixed at construction (its
subband weights are shaped from it), the same way GINO's own `fno_n_modes`
is fixed at construction instead of inferred from `latent_queries` at call
time.
"""

from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from ..layers.blocks.gno_block import GNOBlock
from ..layers.blocks.gno_weighting_functions import dispatch_weighting_fn
from ..layers.blocks.wno_block import mish
from ..layers.common.channel_mlp import ChannelMLP
from ..models.base import BaseModel
from ..models.wno import WNO


class GIWO(BaseModel, name="giwo"):
    """GNO-in -> WNO (latent) -> GNO-out -> projection.

    Parameters
    ----------
    in_channels, out_channels : int
        Feature dimension of the input/output point-cloud functions.
    gno_coord_dim : int, optional
        Geometric dimension of input/output queries. Default: 3
    in_gno_radius, out_gno_radius : float, optional
        Neighbor-search radius for the input/output GNO. Same role as GINO's
        own fields of the same name -- a neighbor-search cutoff, not a model
        weight, so it does not move the parameter count.
    in_gno_transform_type : str, optional
        Unlike GINO (whose separate `lifting` ChannelMLP can bridge a
        channel-count mismatch between the input GNO's output and the
        latent operator's width), GIWO feeds the input GNO's output straight
        into the WNO with no lifting layer in between -- so every mode
        except `"nonlinear_kernelonly"` (the default) requires
        `in_channels == wno_hidden_channels`, since
        `IntegralTransform`
        multiplies the kernel output by `f_y` elementwise in those modes.
    wno_size : tuple[int, ...]
        Spatial shape of the fixed latent grid (e.g. `(10, 10, 10)`),
        passed straight through to `WNO`'s own `size`.
    wno_hidden_channels, wno_n_layers, wno_level, wno_wavelet : see `WNO`
        Sized the same empirical way GINO's `fno_hidden_channels` is: by
        direct parameter-count measurement against a budget, not a
        closed-form formula (see `scripts/benchmarks/shapenetcar/_config.py`).
    """

    in_channels: int
    out_channels: int
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    in_gno_radius: float = 0.14
    out_gno_radius: float = 0.8
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    in_gno_pos_embed_type: str | None = "nerf"
    out_gno_pos_embed_type: str | None = "nerf"
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000
    in_gno_channel_mlp_hidden_layers: Any = (80, 80, 80)
    out_gno_channel_mlp_hidden_layers: Any = (512, 256)
    gno_channel_mlp_non_linearity: Callable = nn.gelu
    gno_use_open3d: bool = True
    gno_use_torch_scatter: bool = False
    gno_weighting_function: Any = None
    gno_weight_function_scale: float = 1
    wno_size: Any = (10, 10, 10)
    wno_hidden_channels: int = 64
    wno_n_layers: int = 4
    wno_level: int = 2
    wno_wavelet: str = "db4"
    wno_mode: str = "symmetric"
    wno_use_grid: bool = True
    # Matches lid_driven_cavity's GridWNO: float32, not WNO's own bfloat16
    # default -- the subband weights stay float32 regardless (jaxwt's filter
    # banks are float64), so bfloat16 only adds casts, and float32 keeps this
    # comparable to GINO (`fno_block_precision="full"`) and Transolver,
    # neither of which run any part of their forward pass in reduced
    # precision.
    wno_compute_dtype: Any = jnp.float32
    wno_activation: Callable = mish
    projection_channel_ratio: int = 4

    def setup(self) -> None:
        gno_use_open3d = self.gno_use_open3d
        if self.gno_coord_dim != 3 and gno_use_open3d:
            gno_use_open3d = False

        self.gno_in = GNOBlock(
            in_channels=self.in_channels,
            out_channels=self.wno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=self.in_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            radius=self.in_gno_radius,
            reduction="mean",
            weighting_fn=None,
            channel_mlp_layers=list(self.in_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.in_gno_transform_type,
            use_torch_scatter_reduce=self.gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        self.wno = WNO(
            in_channels=self.wno_hidden_channels,
            out_channels=self.wno_hidden_channels,
            hidden_channels=self.wno_hidden_channels,
            n_layers=self.wno_n_layers,
            level=self.wno_level,
            size=tuple(self.wno_size),
            wavelet=self.wno_wavelet,
            mode=self.wno_mode,
            use_grid=self.wno_use_grid,
            compute_dtype=self.wno_compute_dtype,
            activation=self.wno_activation,
        )

        if self.gno_weighting_function is not None:
            weight_fn = dispatch_weighting_fn(
                self.gno_weighting_function,
                sq_radius=self.out_gno_radius**2,
                scale=self.gno_weight_function_scale,
            )
        else:
            weight_fn = None

        self.gno_out = GNOBlock(
            in_channels=self.wno_hidden_channels,
            out_channels=self.wno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction="sum",
            weighting_fn=weight_fn,
            pos_embedding_type=self.out_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            channel_mlp_layers=list(self.out_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.out_gno_transform_type,
            use_torch_scatter_reduce=self.gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        projection_channels = self.projection_channel_ratio * self.wno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=self.wno_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
        )

    def latent_embedding(self, in_p: Any) -> Any:
        """Run the latent WNO. `in_p`: (batch, n_1, ..., n_k, wno_hidden_channels)."""
        return self.wno(in_p)

    def encode(
        self,
        input_geom: Any,
        latent_queries: Any,
        x: Any = None,
        neighbors_in: dict | None = None,
    ) -> tuple[Any, Any]:
        """Input GNO -> WNO: input mesh -> latent-grid embedding.

        Same contract as `encode`
        (split out for the same reason: a caller can reuse one `encode`
        across several `decode` calls), minus the `latent_features`/
        `ada_in` options GINO exposes, which nothing in this repo uses for
        GIWO yet.
        """
        batch_size = 1 if x is None else x.shape[0]

        input_geom = jnp.squeeze(input_geom, axis=0)
        latent_queries = jnp.squeeze(latent_queries, axis=0)

        in_p = self.gno_in(
            y=input_geom,
            x=latent_queries.reshape((-1, latent_queries.shape[-1])),
            f_y=x,
            neighbors=neighbors_in,
        )

        grid_shape = latent_queries.shape[:-1]
        in_p = in_p.reshape((batch_size, *grid_shape, -1))

        latent_embed = self.latent_embedding(in_p)
        latent_embed = latent_embed.reshape(batch_size, -1, self.wno_hidden_channels)

        return latent_embed, latent_queries

    def decode(
        self,
        latent_embed: Any,
        latent_queries: Any,
        output_queries: Any,
        neighbors_out: dict | None = None,
    ) -> Any:
        """Output GNO + projection: latent-grid embedding -> output queries.

        Same contract as `decode`.
        """
        output_queries = jnp.squeeze(output_queries, axis=0)

        out = self.gno_out(
            y=latent_queries.reshape((-1, latent_queries.shape[-1])),
            x=output_queries,
            f_y=latent_embed,
            neighbors=neighbors_out,
        )
        out = jnp.transpose(out, (0, 2, 1))
        out = jnp.transpose(self.projection(out), (0, 2, 1))
        return out

    def __call__(
        self,
        input_geom: Any,
        latent_queries: Any,
        output_queries: Any,
        x: Any = None,
        neighbors_in: dict | None = None,
        neighbors_out: dict | None = None,
        **kwargs: Any,
    ) -> Any:
        """GNO-in -> WNO -> GNO-out + projection. Thin wrapper around
        `encode` then `decode`, same as GINO's own `__call__`."""
        del kwargs
        latent_embed, latent_queries = self.encode(
            input_geom=input_geom,
            latent_queries=latent_queries,
            x=x,
            neighbors_in=neighbors_in,
        )
        return self.decode(
            latent_embed=latent_embed,
            latent_queries=latent_queries,
            output_queries=output_queries,
            neighbors_out=neighbors_out,
        )
