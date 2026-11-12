# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The 2D FNO architecture from Li et al. ("Fourier Neural Operator for
Parametric Partial Differential Equations", 2020/2021), ported to
JAX/Flax, with one deliberate deviation from that paper's literal
release (see below).

Differs from this repo's general-purpose `FNO`
in three concrete ways:

- Lifting and projection are single linear layers (plus one hidden
  linear layer in projection), not the small 2-layer ChannelMLPs the
  general `FNO` uses.
- Each block has one skip pathway (a single pointwise convolution), not
  the general `FNO`'s two-pathway skip design (a separate channel-MLP
  skip alongside the pointwise-conv skip).
- Most importantly, its spectral-convolution layer
  (`ClassicSpectralConv2d`)
  independently learns separate weight tensors for the low-frequency and
  wrapped high-frequency row bands of the spectrum, rather than deriving
  one band from the other by symmetry the way the general `FNO`'s
  `SpectralConv` does -- for the same nominal mode count, this gives
  the spectral layer roughly twice the learnable frequency-domain
  parameters.

One deliberate deviation from the literal 2020/2021 release: each block
also runs a small pointwise MLP (`_PointwiseMLP`) on the
spectral-conv output before the skip is added. Neither the original
paper's own released code nor PINO's official FNO baseline
(`github.com/neuraloperator/physics_informed`, `models/fourier2d.py`)
has this. It's a convention that also appears in other public FNO2d
implementations (e.g. VINO's FNO baseline,
`github.com/eshaghi-ms/VINO`), and it's kept here on purpose because
it measurably improves accuracy on this benchmark -- not because it is
part of the literal original design.

Kept as its own small, self-contained model rather than a configuration
of the general `FNO`, since none of that class's generality (Tucker
factorization, separable channels, complex-valued data, resolution
rescaling) applies here -- this is a narrow port of one specific
published design (plus the one noted MLP deviation), 2D and real-data
only.
"""

import flax.linen as nn
import jax.numpy as jnp

from ..layers.blocks.spectral_convolution_classic import ClassicSpectralConv2d
from ..layers.common.channel_mlp import _pytorch_conv_init
from ..layers.common.embeddings import GridEmbedding2D
from .base import BaseModel


def _channel_last(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.transpose(x, (0, 2, 3, 1))


def _channel_first(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.transpose(x, (0, 3, 1, 2))


class _PointwiseMLP(nn.Module):
    """Two 1x1 convolutions with a GELU in between, applied channel-wise.

    Not part of the original 2020/2021 FNO2d release or PINO's own FNO
    baseline -- see the module docstring for why it's kept anyway.
    """

    in_channels: int
    mid_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """`x`: `(batch, channels, H, W)` -> same shape, `out_channels`."""
        x = _channel_last(x)
        x = nn.Conv(
            self.mid_channels,
            kernel_size=(1, 1),
            kernel_init=_pytorch_conv_init(self.in_channels),
            bias_init=_pytorch_conv_init(self.in_channels),
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Conv(
            self.out_channels,
            kernel_size=(1, 1),
            kernel_init=_pytorch_conv_init(self.mid_channels),
            bias_init=_pytorch_conv_init(self.mid_channels),
        )(x)
        return _channel_first(x)


class FNOClassic2d(BaseModel, name="FNOClassic2d"):
    """The 2020/2021 FNO2d architecture, 2D and real-data only, with one
    deliberate deviation from the literal release -- see the module
    docstring.

    Parameters
    ----------
    in_channels : int
        Number of input channels *before* the built-in grid positional
        embedding (which always adds 2 more: x and y coordinates).
    out_channels : int
    n_modes1, n_modes2 : int
        Retained Fourier modes; see
        `ClassicSpectralConv2d`.
    hidden_channels : int
        Channel width of the spectral layers (the original design's
        `width`).
    n_layers : int, optional
        Number of spectral-conv blocks, by default 4 (the original
        design's fixed depth).
    projection_hidden_channels : int, optional
        Hidden width of the projection head's single hidden linear
        layer, by default 128 (the original design's fixed value).
    split_complex_params : bool, optional
        Passed through to
        `ClassicSpectralConv2d`.
        Set it when training this model with a moment-based optimizer
        such as Adam, whose state would otherwise be complex-valued; see
        that class for the full rationale. False by default, which keeps
        the original design's parameter layout.
    """

    in_channels: int
    out_channels: int
    n_modes1: int
    n_modes2: int
    hidden_channels: int
    n_layers: int = 4
    projection_hidden_channels: int = 128
    split_complex_params: bool = False

    def setup(self) -> None:
        """Create the embedding, lifting, spectral, pointwise and projection layers."""
        # PyTorch's default nn.Linear/nn.Conv2d init (Kaiming-uniform
        # weight, uniform-bounded bias -- never zero) rather than Flax's
        # own default (lecun_normal weight, zero bias): matches the
        # original design's own initialization exactly, not just its
        # architecture. FNO training has proven sensitive to init
        # scale/distribution throughout this benchmark family, so this
        # is not cosmetic.
        embedded_channels = self.in_channels + 2  # GridEmbedding2D adds x, y
        self.grid_embedding = GridEmbedding2D(in_channels=self.in_channels)
        self.lifting = nn.Dense(
            self.hidden_channels,
            kernel_init=_pytorch_conv_init(embedded_channels),
            bias_init=_pytorch_conv_init(embedded_channels),
        )
        self.convs = [
            ClassicSpectralConv2d(
                self.hidden_channels,
                self.hidden_channels,
                self.n_modes1,
                self.n_modes2,
                split_complex_params=self.split_complex_params,
            )
            for _ in range(self.n_layers)
        ]
        self.mlps = [
            _PointwiseMLP(
                self.hidden_channels, self.hidden_channels, self.hidden_channels
            )
            for _ in range(self.n_layers)
        ]
        self.skips = [
            nn.Conv(
                self.hidden_channels,
                kernel_size=(1, 1),
                kernel_init=_pytorch_conv_init(self.hidden_channels),
                bias_init=_pytorch_conv_init(self.hidden_channels),
            )
            for _ in range(self.n_layers)
        ]
        self.projection_hidden = nn.Dense(
            self.projection_hidden_channels,
            kernel_init=_pytorch_conv_init(self.hidden_channels),
            bias_init=_pytorch_conv_init(self.hidden_channels),
        )
        self.projection_out = nn.Dense(
            self.out_channels,
            kernel_init=_pytorch_conv_init(self.projection_hidden_channels),
            bias_init=_pytorch_conv_init(self.projection_hidden_channels),
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """`x`: `(batch, in_channels, H, W)` ->
        `(batch, out_channels, H, W)`.
        """
        x = self.grid_embedding(x)

        x = _channel_last(x)
        x = self.lifting(x)
        x = _channel_first(x)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = _channel_first(self.skips[i](_channel_last(x)))
            x = x1 + x2
            if i < self.n_layers - 1:
                x = nn.gelu(x, approximate=False)

        x = _channel_last(x)
        x = self.projection_hidden(x)
        x = nn.gelu(x, approximate=False)
        x = self.projection_out(x)
        return _channel_first(x)
