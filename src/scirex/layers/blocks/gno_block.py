# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Graph Neural Operator (GNO) block combining neighbor search and an integral kernel
transform.
"""

from collections.abc import Callable
from typing import Any, Literal

import flax.linen as nn
import jax.numpy as jnp

from ..common.channel_mlp import LinearChannelMLP
from ..common.embeddings import SinusoidalEmbedding
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch


class GNOBlock(nn.Module):
    """Graph Neural Operator layer

    It is a resolution-invariant operator that maps a function defined
    over one coordinate mesh to another defined over another coordinate mesh using
    a pointwise kernel integral that takes contributions from neighbors of distance 1
    within a graph constructed via neighbor search with a specified radius.
    The exact implementation details of the GNO layer architecture are discussed
    in [1].

    Optionally, if provided, the input and output queries can have a positional
    embedding applied using the argument pos_embedding.

    The kernel integral computed in IntegralTransform computes one of:
    (a) \\int_{A(x)} k(x, y) dy
    (b) \\int_{A(x)} k(x, y) * f(y) dy
    (c) \\int_{A(x)} k(x, y, f(y)) dy
    (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy

    Parameters
    ----------
    in_channels : int
        Number of channels in input function. Only used if transform_type
        is (c) "nonlinear" or (d) "nonlinear_kernelonly"
    out_channels : int
        Number of channels in output function
    coord_dim : int
        Dimension of domain on which x and y are defined
    radius : float
        Radius in which to search for neighbors
    weighting_fn : Callable, optional
        Optional squared-norm weighting function to use for Mollified GNO layer
        by default None. See scirex.layers.gno_weighting_functions for more details.
    reduction : Literal['sum', 'mean']
        Whether to aggregate information from each neighborhood in the
        integral transform by summing ('sum') or averaging ('mean'), by default 'sum'

    Other Parameters
    ----------------
    transform_type : str, optional
        Which integral transform to compute. Options: 'linear_kernelonly', 'linear',
        'nonlinear_kernelonly', 'nonlinear'.
        The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b) [DEFAULT]
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter, by default 'linear'
    pos_embedding_type: literal {'transformer', 'nerf'} | None, optional
        Type of positional embedding to use during the kernel integral transform.
        Options: 'transformer', 'nerf', None. See
        `scirex.layers.embeddings.SinusoidalEmbedding` for more details,
        by default 'transformer'
    pos_embedding_channels : int, optional
        Per-channel dimension of optional positional embedding to use, by default 32
    pos_embedding_max_positions: int, optional
        `max_positions` parameter for SinusoidalEmbedding of type `'transformer'`. If
        `pos_embedding_type != 'transformer'`, this value is not used, by default 10000
    channel_mlp_layers : List[int], optional
        List of layer widths to dynamically construct
        LinearChannelMLP network to parameterize kernel k, by default None
    channel_mlp_non_linearity : callable, optional
        Activation function for ChannelMLPLinear above, by default nn.gelu
    channel_mlp : nn.Module, optional
        ChannelMLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        ChannelMLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the
        kernel integral. If you have more specific needs than the LinearChannelMLP,
        this argument allows you to pass your own Module to parameterize the kernel k.
        Default None, by default None
    use_torch_scatter_reduce : bool, optional
        Has no effect: the grouped reductions always use the JAX segment
        reduction of the package. Kept for compatibility, by default False.
    use_open3d_neighbor_search : bool, optional
        Whether to use open3d for fast 3D neighbor search, by default True.
        open3d works on 3D coordinates only: for any other coordinate
        dimension set this to False, otherwise setup raises an
        ``AssertionError``.

    References
    ----------
    [1] Zongyi Li, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya,
        Anima Anandkumar (2020). "Neural Operator: Graph Kernel Network for
        Partial Differential Equations." ArXiV, https://arxiv.org/pdf/2003.03485.
    """

    in_channels: int
    out_channels: int
    coord_dim: int
    radius: float
    transform_type: str = "linear"
    weighting_fn: Callable | None = None
    reduction: Literal["mean", "sum"] = "sum"
    pos_embedding_type: str | None = "transformer"
    pos_embedding_channels: int = 32
    pos_embedding_max_positions: int = 10000
    channel_mlp_layers: Any = None  # List[int], default [128, 256, 128]
    channel_mlp_non_linearity: Callable = nn.gelu
    channel_mlp: Any = None
    use_torch_scatter_reduce: bool = False
    use_open3d_neighbor_search: bool = True

    def setup(self) -> None:
        """Build the positional embedding, neighbor search, kernel ChannelMLP, and
        integral-transform submodules. Flax lifecycle hook.
        """
        # Apply sinusoidal positional embedding
        if self.pos_embedding_type in ["nerf", "transformer"]:
            self.pos_embedding: SinusoidalEmbedding | None = SinusoidalEmbedding(
                in_channels=self.coord_dim,
                num_frequencies=self.pos_embedding_channels,
                embedding_type=self.pos_embedding_type,
                max_positions=self.pos_embedding_max_positions,
            )
        else:
            self.pos_embedding = None

        # Create in-to-out nb search module
        if self.use_open3d_neighbor_search:
            assert self.coord_dim == 3, (
                f"Error: open3d is only designed for 3D data, "
                f"GNO instantiated for dim={self.coord_dim}"
            )
        self.neighbor_search = NeighborSearch(
            use_open3d=self.use_open3d_neighbor_search,
            return_norm=self.weighting_fn is not None,
        )

        # create proper kernel input channel dim
        if self.pos_embedding is None:
            # x and y dim will be coordinate dim if no pos embedding is applied
            kernel_in_dim = self.coord_dim * 2
            kernel_in_dim_str = "dim(y) + dim(x)"
        else:
            # x and y dim will be embedding dim if pos embedding is applied
            kernel_in_dim = self.pos_embedding.out_channels * 2
            kernel_in_dim_str = "dim(y_embed) + dim(x_embed)"

        if (
            self.transform_type == "nonlinear"
            or self.transform_type == "nonlinear_kernelonly"
        ):
            kernel_in_dim += self.in_channels
            kernel_in_dim_str += " + dim(f_y)"

        channel_mlp = self.channel_mlp
        channel_mlp_layers = (
            self.channel_mlp_layers
            if self.channel_mlp_layers is not None
            else [128, 256, 128]
        )

        if channel_mlp is not None:
            assert channel_mlp.in_channels == kernel_in_dim, (
                f"Error: expected ChannelMLP to take input with {kernel_in_dim} "
                f"channels (feature channels={kernel_in_dim_str}), "
                f"got {channel_mlp.in_channels}."
            )
            assert channel_mlp.out_channels == self.out_channels, (
                f"Error: expected ChannelMLP to have out_channels={self.out_channels} "
                f"but got {channel_mlp.in_channels}."
            )
        elif channel_mlp_layers is not None:
            channel_mlp_layers = list(channel_mlp_layers)
            if channel_mlp_layers[0] != kernel_in_dim:
                channel_mlp_layers = [kernel_in_dim, *channel_mlp_layers]
            if channel_mlp_layers[-1] != self.out_channels:
                channel_mlp_layers = [*channel_mlp_layers, self.out_channels]
            channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers,
                non_linearity=self.channel_mlp_non_linearity,
            )

        # Create integral transform module
        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            transform_type=self.transform_type,
            use_torch_scatter=self.use_torch_scatter_reduce,
            weighting_fn=self.weighting_fn,
            reduction=self.reduction,
        )

    def __call__(
        self,
        y: jnp.ndarray,
        x: jnp.ndarray,
        f_y: jnp.ndarray | None = None,
        neighbors: dict[str, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Compute a GNO neighbor search and kernel integral transform.

        Parameters
        ----------
        y : jnp.ndarray of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        x : jnp.ndarray of shape [m, d1], default None
            m points of dimension d1 over which the
            output function is defined. Must share domain
            with y
        f_y : jnp.ndarray of shape [batch, n, d2] or [n, d2], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        neighbors : dict, optional
            Precomputed neighbor search result from a prior call to
            self.neighbor_search(). If provided, the neighbor search
            is skipped entirely. Must have the same format as the
            output of NeighborSearch.forward().

        Returns
        -------
        out_features : jnp.ndarray of shape [batch, m, d3] or [m, d3]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        if neighbors is None:
            raw = self.neighbor_search(data=y, queries=x, radius=self.radius)
            row_splits = raw["neighbors_row_splits"]
            counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
            segment_ids = jnp.repeat(
                jnp.arange(counts.shape[0], dtype=jnp.int32), counts
            )
            neighbors_dict = {
                "neighbors_index": raw["neighbors_index"],
                "segment_ids": segment_ids,
                "counts": counts,
            }
            if "weights" in raw:
                neighbors_dict["weights"] = raw["weights"]
        else:
            neighbors_dict = neighbors

        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x

        out_features = self.integral_transform(
            y=y_embed, x=x_embed, neighbors=neighbors_dict, f_y=f_y
        )

        return out_features
