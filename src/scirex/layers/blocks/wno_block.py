# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Wavelet Neural Operator block: wavelet convolution plus a pointwise linear path."""

from collections.abc import Callable

import jax.numpy as jnp
from flax import linen as nn

from .wavelet_conv import WaveletConv


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation, `x * tanh(softplus(x))`."""
    return x * jnp.tanh(nn.softplus(x))


class WNOBlock(nn.Module):
    """Generic channels-last WNO block matching the upstream structure.

    Each block computes a wavelet operator plus a pointwise channel mixing term,
    then applies the non-linearity except for the last block.
    """

    hidden_channels: int
    level: int = 1
    size: int | tuple[int, ...] = 1024
    wavelet: str = "db4"
    mode: str = "symmetric"
    wavelet_precision: str = "fastest"
    compute_dtype: jnp.dtype = jnp.bfloat16
    activation: Callable | None = mish

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the wavelet convolution and the pointwise path, then the activation."""
        wavelet_out = WaveletConv(
            in_channels=x.shape[-1],
            out_channels=self.hidden_channels,
            level=self.level,
            size=self.size,
            wavelet=self.wavelet,
            mode=self.mode,
            precision=self.wavelet_precision,
        )(x)
        pointwise_out = nn.Dense(
            features=self.hidden_channels,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            name="pointwise",
        )(x)

        out = wavelet_out + pointwise_out
        if self.activation is not None:
            out = self.activation(out)
        return out
