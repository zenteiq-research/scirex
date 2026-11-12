# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Integral kernel transform: the message-passing operation underlying the GNO."""

from collections.abc import Callable
from typing import Any, Literal

import flax.linen as nn
import jax.numpy as jnp

from ..._internal.segment_csr import segment_csr
from ..common.channel_mlp import LinearChannelMLP, gelu_exact


class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO).

    It computes one of the following:
        (a) \\int_{A(x)} k(x, y) dy
        (b) \\int_{A(x)} k(x, y) * f(y) dy
        (c) \\int_{A(x)} k(x, y, f(y)) dy
        (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)

    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : nn.Module, optional
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        MLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the
        kernel integral, by default None
    channel_mlp_layers : list, optional
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class, by default None
    channel_mlp_non_linearity : callable, optional
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None, by default `gelu_exact` (the exact
        erf GELU). Callers that build their own `channel_mlp` (e.g. GINO's
        GNOBlock) are unaffected by this default.
    transform_type : str, optional
        Which integral transform to compute. Options: 'linear_kernelonly', 'linear',
        'nonlinear_kernelonly', 'nonlinear'.
        The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter, by default 'linear'
    use_torch_scatter : bool, optional
        Has no effect: the grouped reductions always use the JAX segment
        reduction of the package. Kept for compatibility, by default False.
    """

    channel_mlp: Any = None
    channel_mlp_layers: Any = None
    channel_mlp_non_linearity: Callable = gelu_exact
    transform_type: str = "linear"
    weighting_fn: Any = None
    reduction: Literal["mean", "sum"] = "sum"
    use_torch_scatter: bool = False

    def setup(self) -> None:
        assert self.channel_mlp is not None or self.channel_mlp_layers is not None

        if self.transform_type not in (
            "linear_kernelonly",
            "linear",
            "nonlinear_kernelonly",
            "nonlinear",
        ):
            raise ValueError(
                f"Got transform_type={self.transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if self.channel_mlp is None:
            self._channel_mlp = LinearChannelMLP(
                layers=self.channel_mlp_layers,
                non_linearity=self.channel_mlp_non_linearity,
            )
        else:
            self._channel_mlp = self.channel_mlp

    def __call__(
        self,
        y: jnp.ndarray,
        neighbors: dict[str, jnp.ndarray],
        x: jnp.ndarray | None = None,
        f_y: jnp.ndarray | None = None,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute a kernel integral transform. Assumes x=y if not specified.

        Integral is taken w.r.t. the neighbors.

        If no weights are given, a Monte-Carlo approximation is made.

        For transforms of type 0 or 2, out channels must be
        the same as the channels of f.

        Parameters
        ----------
        y : jnp.ndarray of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        neighbors : dict
            The sets $A(x)$ given in CRS format. The
            dict must contain the keys "neighbors_index",
            "segment_ids", and "counts". For descriptions
            see NeighborSearch and pad_neighbor_data.
            If batch > 1, the neighbors must be constant
            across the entire batch.
        x : jnp.ndarray of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : jnp.ndarray of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : jnp.ndarray of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            $1/|A(x)|$ is used.

        Returns
        -------
        out_features : jnp.ndarray of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]]

        # batching only matters if f_y (latent embedding) values are provided
        batched = False
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                in_features = f_y[neighbors["neighbors_index"]]

        # Use padded segment_ids to gather x for each edge (fixed shape, JIT-safe).
        # x is always 2D [n_query, D]; clip avoids OOB on dummy padded entries
        # (their segment = n_out gets discarded by segment_csr anyway).
        safe_seg_ids = jnp.clip(neighbors["segment_ids"], 0, x.shape[0] - 1)
        self_features = x[safe_seg_ids]  # [max_edges, D_x]

        agg_features = jnp.concatenate([rep_features, self_features], axis=-1)

        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                agg_features = jnp.tile(
                    agg_features, [batch_size] + [1] * agg_features.ndim
                )
            agg_features = jnp.concatenate([agg_features, in_features], axis=-1)

        rep_features = self._channel_mlp(agg_features)

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            if rep_features.ndim == 2 and batched:
                rep_features = jnp.tile(
                    jnp.expand_dims(rep_features, 0),
                    [batch_size] + [1] * rep_features.ndim,
                )
            rep_features = rep_features * in_features

        nbr_weights = neighbors.get("weights")
        if nbr_weights is None:
            nbr_weights = weights
        if nbr_weights is None and self.weighting_fn is not None:
            raise KeyError(
                "if a weighting function is provided, your neighborhoods must "
                "contain weights."
            )
        reduction: Literal["mean", "sum"]
        if nbr_weights is not None:
            nbr_weights = jnp.expand_dims(nbr_weights, -1)
            if rep_features.ndim == 3:
                nbr_weights = jnp.expand_dims(nbr_weights, 0)
            if self.weighting_fn is not None:
                nbr_weights = self.weighting_fn(nbr_weights)
            rep_features = rep_features * nbr_weights
            reduction = "sum"
        else:
            reduction = self.reduction

        out_features = segment_csr(
            rep_features,
            neighbors["segment_ids"],
            neighbors["counts"],
            reduction=reduction,
        )
        return out_features
