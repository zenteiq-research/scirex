# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Dimension-agnostic normalization layers (AdaIN, instance norm, batch norm) for neural
operators.
"""

import logging
import warnings
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) layer for style transfer in neural
    operators.

    AdaIN performs instance normalization followed by adaptive scaling and shifting
    based on an external embedding vector. This allows for style transfer by
    modulating the output characteristics based on a conditioning signal.

    The layer first normalizes the input using instance normalization, then applies
    learned scaling (weight) and shifting (bias) parameters derived from an embedding
    vector through a multi-layer perceptron.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector used for style conditioning
    in_channels : int
        Number of input channels to normalize
    mlp : nn.Module, optional
        Multi-layer perceptron that maps embedding to (weight, bias) parameters.
        Should output 2*in_channels values. If None, a default MLP is created
        with architecture: Linear(embed_dim, 512) -> GELU -> Linear(512, 2*in_channels)
    eps : float, optional
        Small value added to the denominator for numerical stability in normalization.
        Default is 1e-5.
    """

    embed_dim: int
    in_channels: int
    mlp: nn.Module | None = None
    eps: float = 1e-5

    def setup(self) -> None:
        if self.mlp is None:
            self._mlp: nn.Module = nn.Sequential(
                [
                    nn.Dense(512),
                    nn.gelu,
                    nn.Dense(2 * self.in_channels),
                ]
            )
        else:
            self._mlp = self.mlp
        self._group_norm = nn.GroupNorm(num_groups=self.in_channels, epsilon=self.eps)

    def __call__(self, x: jnp.ndarray, embedding: jnp.ndarray) -> jnp.ndarray:
        """Apply adaptive instance normalization to the input tensor.

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor to normalize
        embedding : jnp.ndarray
            Embedding vector for style conditioning
        """
        embedding = embedding.reshape(
            self.embed_dim,
        )
        weight, bias = jnp.split(self._mlp(embedding), 2, axis=0)

        # GroupNorm expects features-last; transpose in/out for channel-first tensors
        x = jnp.moveaxis(x, 1, -1)
        x = self._group_norm(x)
        x = jnp.moveaxis(x, -1, 1)
        # apply per-channel affine transform (weight/bias shape: (in_channels,))
        # x shape: (..., in_channels) or (batch, in_channels, *spatial)
        # broadcast weight/bias over spatial dims
        shape = (1, self.in_channels) + (1,) * (x.ndim - 2)
        return x * weight.reshape(shape) + bias.reshape(shape)


class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators.

    InstanceNorm normalizes each sample in the batch independently, computing
    mean and variance across spatial dimensions for each sample and channel
    separately. This is useful when the statistical properties of each sample
    are distinct and should be treated separately.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters to pass to instance normalization.
        Common parameters include:

        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
    """

    kwargs: Any = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply instance normalization to the input tensor."""
        size = x.shape
        eps = (self.kwargs or {}).get("eps", 1e-5)
        # normalize over all spatial dims (everything except batch and channel)
        axes = tuple(range(2, x.ndim))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        assert x.shape == size
        return x


class BatchNorm(nn.Module):
    """Dimension-agnostic batch normalization layer for neural operators.

    BatchNorm normalizes data across the entire batch, computing a single mean
    and standard deviation for all samples combined. This is the most common
    form of normalization and is effective when batch statistics are a good
    approximation of the overall data distribution.

    For dimensions > 3, the layer automatically flattens spatial dimensions
    and uses BatchNorm, as high-dimensional batch norm requires flattening.

    Parameters
    ----------
    n_dim : int
        Spatial dimension of input data (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
        Determined by FNOBlocks.n_dim. If n_dim > 3, spatial dimensions are
        flattened before applying batch norm.
    num_features : int
        Number of channels in the input tensor to be normalized
    **kwargs : dict, optional
        Additional parameters to pass to the underlying batch normalization layer.
        Common parameters include:

        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - use_bias : bool, optional
            If True, apply learnable bias. Default is True.
        - use_scale : bool, optional
            If True, apply learnable scale. Default is True.
    """

    n_dim: int
    num_features: int
    kwargs: Any = None

    def setup(self) -> None:
        if self.n_dim > 3:
            warnings.warn(
                "JAX does not implement batch norm for dimensions higher "
                "than 3. We manually flatten the spatial dimension of 4+D tensors "
                "to apply batch norm.",
                stacklevel=2,
            )
        _kwargs = self.kwargs or {}
        momentum = _kwargs.get("momentum", 0.1)
        eps = _kwargs.get("eps", 1e-5)
        use_bias = _kwargs.get("affine", True)
        use_scale = _kwargs.get("affine", True)
        self.norm = nn.BatchNorm(
            momentum=momentum,
            epsilon=eps,
            use_bias=use_bias,
            use_scale=use_scale,
        )

    def __call__(
        self, x: jnp.ndarray, use_running_average: bool = False
    ) -> jnp.ndarray:
        """Apply batch normalization to the input tensor."""
        size = x.shape

        # in 4+D, we flatten and use batchnorm on flattened spatial dim
        if self.n_dim >= 4:
            x = x.reshape(size[0], size[1], -1)
        x = self.norm(x, use_running_average=use_running_average)

        # if flattening occurred, unflatten
        if self.n_dim >= 4:
            x = x.reshape(size)

        assert x.shape == size
        return x
