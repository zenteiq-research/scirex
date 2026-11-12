# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Base class for spectral convolution layers."""

import flax.linen as nn
import jax.numpy as jnp


class BaseSpectralConv(nn.Module):
    """Base Class for Spectral Convolutions

    Use it when you want to build your own FNO-type Neural Operators
    """

    # Note: 'device' is not needed in JAX - device placement is handled
    # explicitly via jax.device_put() at the call site, not stored on the module.

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transforms an input x for a skip connection, by default just an identity map

        If your function transforms the input then you should also implement this
        transform method so the skip connection can also work.

        Typical usecases are:

        * Your upsample or downsample the input in the Spectral conv: the skip
          connection has to be similarly scaled. This allows you to deal with it
          however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to
          be applied to the skip connection too.
        """
        return x
