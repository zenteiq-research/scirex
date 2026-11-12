# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Channel-wise multi-layer perceptrons applied pointwise across spatial dimensions."""

import math
from collections.abc import Callable
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

# Exact (erf) GELU.  The flax/jax `nn.gelu` default uses the tanh
# approximation; `approximate=False` selects the exact error-function form,
# which matches PyTorch's default `nn.GELU`.
gelu_exact = partial(jax.nn.gelu, approximate=False)


def _pytorch_conv_init(in_features: int) -> Callable[..., jnp.ndarray]:
    """Weight/bias initializer matching PyTorch Conv1d defaults.

    Both kernel (Kaiming uniform, a=sqrt(5)) and bias use
    Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) when kernel_size=1, so a single
    function covers both.
    """
    bound = 1.0 / math.sqrt(in_features)

    def init(
        key: jax.Array, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


class ChannelMLP(nn.Module):
    """Multi-layer perceptron applied channel-wise across spatial dimensions.

    ChannelMLP applies a series of 1D convolutions and nonlinearities to the channel
    dimension of input tensors, making it invariant to spatial resolution.

    Uses Conv with kernel_size=1 (equivalent to PyTorch Conv1d with kernel_size=1),
    which performs linear transformations on the channel dimension while preserving
    spatial structure.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int, optional
        Number of output channels. If None, defaults to in_channels.
    hidden_channels : int, optional
        Number of hidden channels in intermediate layers. If None, defaults to
        in_channels.
    n_layers : int, optional
        Number of linear layers in the MLP, by default 2
    n_dim : int, optional
        Spatial dimension of input (unused but kept for compatibility), by default 2
    non_linearity : callable, optional
        Nonlinear activation function to apply between layers, by default nn.gelu
    dropout : float, optional
        Dropout probability applied after each layer (except the last).
        If 0, no dropout is applied, by default 0.0
    """

    in_channels: int
    out_channels: int | None = None
    hidden_channels: int | None = None
    n_layers: int = 2
    n_dim: int = 2
    non_linearity: Callable = nn.gelu
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass through the channel MLP.

        Parameters
        ----------
        x : jnp.ndarray
            Input array of shape (batch, in_channels, *spatial_dims)
        training : bool
            Whether in training mode (controls dropout), by default False

        Returns
        -------
        jnp.ndarray
            Output array of shape (batch, out_channels, *spatial_dims)
        """
        out_channels = (
            self.in_channels if self.out_channels is None else self.out_channels
        )
        hidden_channels = (
            self.in_channels if self.hidden_channels is None else self.hidden_channels
        )

        reshaped = False
        size = list(x.shape)

        # Handle high-dimensional inputs (4D+) by flattening spatial dimensions
        # (batch, channels, x1, x2, ...) -> (batch, channels, -1)
        if x.ndim > 3:
            x = x.reshape((*size[:2], -1))
            reshaped = True

        # In JAX/Flax, Conv1d(kernel_size=1) ≡ nn.Conv with kernel_size=(1,) on
        # (batch, spatial, channels). PyTorch Conv1d expects (batch, channels, length).
        # Flax nn.Conv expects (batch, spatial..., channels). So we transpose:
        # (batch, channels, length) -> (batch, length, channels) before conv,
        # then transpose back after.
        x = jnp.transpose(x, (0, 2, 1))  # (batch, length, channels)

        for i in range(self.n_layers):
            in_feats = self.in_channels if i == 0 else hidden_channels

            if i == 0 and i == (self.n_layers - 1):
                features = out_channels
            elif i == 0:
                features = hidden_channels
            elif i == (self.n_layers - 1):
                features = out_channels
            else:
                features = hidden_channels

            x = nn.Conv(
                features=features,
                kernel_size=(1,),
                kernel_init=_pytorch_conv_init(in_feats),
                bias_init=_pytorch_conv_init(in_feats),
                name=f"fc_{i}",
            )(x)

            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        x = jnp.transpose(x, (0, 2, 1))  # (batch, channels, length)

        # Restore original spatial dimensions if input was reshaped
        if reshaped:
            x = x.reshape((size[0], out_channels, *size[2:]))

        return x


class LinearChannelMLP(nn.Module):
    """
    Multi-layer perceptron using fully connected layers for channel processing.

    Parameters
    ----------
    layers : list of int
        List defining the architecture: [in_channels, hidden1, ..., out_channels]
        Must have at least 2 elements.
    non_linearity : callable, optional
        Nonlinear activation function to apply between layers, by default nn.gelu
    dropout : float, optional
        Dropout probability applied after each layer (except the last).
        If 0, no dropout is applied, by default 0.0
    """

    layers: list[int]
    non_linearity: Callable = nn.gelu
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Forward pass through the linear channel MLP.

        Parameters
        ----------
        x : jnp.ndarray
            Input array of shape (batch, in_channels) or (batch*spatial, in_channels)
        training : bool
            Whether in training mode (controls dropout), by default False

        Returns
        -------
        jnp.ndarray
            Output array of shape (batch, out_channels) or (batch*spatial, out_channels)
        """
        n_layers = len(self.layers) - 1
        assert (
            n_layers >= 1
        ), "Error: trying to instantiate a LinearChannelMLP with only one linear layer."

        for i in range(n_layers):
            # [PARITY-FIX] Match PyTorch nn.Linear defaults: kaiming_uniform(a=sqrt(5))
            # kernel + uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) bias. Flax Dense
            # defaults to lecun_normal + zeros.
            # To restore the Flax defaults, drop the custom initializers below.
            in_size = self.layers[i]
            x = nn.Dense(
                features=self.layers[i + 1],
                kernel_init=_pytorch_conv_init(in_size),
                bias_init=_pytorch_conv_init(in_size),
                name=f"fc_{i}",
            )(x)
            # [/PARITY-FIX]
            if i < n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        return x
