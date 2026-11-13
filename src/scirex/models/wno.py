# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Wavelet Neural Operator (WNO); requires the optional `jaxwt` package."""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

from ..layers.blocks.wno_block import WNOBlock, mish
from .base import BaseModel


def _normalize_padding(
    padding: int | float | list[int | float] | tuple[int | float, ...],
    spatial_shape: tuple[int, ...],
) -> list[int]:
    if isinstance(padding, (int, float)):
        padding_values = [padding] * len(spatial_shape)
    else:
        padding_values = list(padding)

    if len(padding_values) != len(spatial_shape):
        raise ValueError(
            f"Padding length mismatch: got {len(padding_values)} values for {len(spatial_shape)} spatial dims."
        )

    normalized = []
    for pad, size in zip(padding_values, spatial_shape):
        if isinstance(pad, float):
            normalized.append(int(round(pad * size)))
        else:
            normalized.append(int(pad))
    return normalized


def _default_grid_boundaries(ndim: int) -> tuple[tuple[float, float], ...]:
    return tuple((0.0, 1.0) for _ in range(ndim))


def _append_grid_nhwc(
    x: jnp.ndarray,
    grid_boundaries: Sequence[tuple[float, float]],
) -> jnp.ndarray:
    batch_size = x.shape[0]
    spatial_shape = x.shape[1:-1]
    grid_components: list[jnp.ndarray] = []

    for dim_size, (start, end) in zip(spatial_shape, grid_boundaries):
        axis_grid = jnp.linspace(start, end, dim_size, dtype=x.dtype)
        reshape_shape = [1] * x.ndim
        reshape_shape[len(grid_components) + 1] = dim_size
        reshape_shape[-1] = 1
        axis_grid = jnp.reshape(axis_grid, tuple(reshape_shape))
        broadcast_shape = (batch_size, *spatial_shape, 1)
        grid_components.append(jnp.broadcast_to(axis_grid, broadcast_shape))

    return jnp.concatenate((x, *grid_components), axis=-1)


class WNO(BaseModel, name="WNO"):
    """N-dimensional Wavelet Neural Operator.

    This keeps the original WNO2d recipe conceptually intact:
    lift once, apply repeated wavelet-plus-pointwise blocks, then project via
    hidden -> projection_hidden_channels -> out_channels.

    The implementation remains dimension-agnostic by operating on channels-last
    tensors of shape (batch, spatial_1, ..., spatial_n, channels).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input function. When `use_grid=True` the
        model appends `ndim` positional grid channels internally, so the
        actual first Dense layer receives `in_channels + ndim` features.
    out_channels : int
        Number of channels in the output function.
    hidden_channels : int
        Width of each WNO block (number of channels throughout the operator).
    n_layers : int
        Number of WNO blocks to stack.
    level : int
        Number of wavelet decomposition levels per WNOBlock.
    size : int or Tuple[int, ...]
        Spatial size(s) used by WaveletConv inside each block. Pass a single
        int for 1-D data or a tuple matching the spatial dimensions.

    Other Parameters
    ----------------
    wavelet : str, optional
        Wavelet name accepted by jaxwt (pywt-style, e.g. "db4"). Default: "db4"
    mode : str, optional
        Padding mode for the wavelet transform. Default: "symmetric"
    wavelet_precision : str, optional
        Precision mode for the wavelet convolution. Default: "fastest"
    use_grid : bool, optional
        When True, appends coordinate grid channels to the input before lifting.
        Set to False if the grid has already been appended externally. Default: True
    grid_boundaries : Tuple[Tuple[float, float], ...], optional
        Per-dimension (start, end) for the coordinate grid. Default: ((0,1),)*ndim
    padding : int, float, or List, optional
        Spatial padding added before WNO blocks and removed after. Default: 0.0
    padding_side : str, optional
        Where to apply padding: "end" or "symmetric". Default: "end"
    compute_dtype : jnp.dtype, optional
        Dtype used for intermediate computations. Default: jnp.bfloat16
    param_dtype : jnp.dtype, optional
        Dtype used for model parameters, except WaveletConv's subband weights,
        which are always float32 regardless of this setting (see
        `scirex.layers.blocks.wavelet_conv.WaveletConv` for why). Default: jnp.float32
    activation : Callable, optional
        Non-linearity applied after each WNO block except the last. Default: mish
    projection_activation : Callable, optional
        Non-linearity applied between the two projection Dense layers. Default: nn.gelu
    projection_hidden_channels : int, optional
        Hidden width of the two-layer projection head. Default: 128

    Examples
    --------
    >>> from scirex.models import WNO
    >>> model = WNO(in_channels=1, out_channels=1, hidden_channels=64,
    ...             n_layers=4, level=3, size=128)
    """

    in_channels: int
    out_channels: int
    hidden_channels: int
    n_layers: int
    level: int
    size: int | tuple[int, ...]
    wavelet: str = "db4"
    mode: str = "symmetric"
    wavelet_precision: str = "fastest"
    use_grid: bool = True
    grid_boundaries: tuple[tuple[float, float], ...] | None = None
    padding: int | float | list[int | float] = 0.0
    padding_side: str = "end"
    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32
    activation: Callable = mish
    projection_activation: Callable = nn.gelu
    projection_hidden_channels: int = 128

    def setup(self) -> None:
        """Create the lifting layer, the WNO blocks and the projection head."""
        # Lifting layer: maps input channels (+ optional grid channels) → hidden_channels.
        self.fc0 = nn.Dense(
            features=self.hidden_channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

        # Stack of WNO blocks; last block has no activation.
        self.wno_blocks = [
            WNOBlock(
                hidden_channels=self.hidden_channels,
                level=self.level,
                size=self.size,
                wavelet=self.wavelet,
                mode=self.mode,
                wavelet_precision=self.wavelet_precision,
                compute_dtype=self.compute_dtype,
                activation=self.activation if i < self.n_layers - 1 else None,
            )
            for i in range(self.n_layers)
        ]

        # Two-layer projection head.
        self.fc1 = nn.Dense(
            features=self.projection_hidden_channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )
        self.fc2 = nn.Dense(
            features=self.out_channels,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """WNO forward pass.

        1. Optional grid-coordinate appending
        2. Lifting Dense layer (fc0)
        3. Optional spatial padding
        4. `n_layers` WNO blocks
        5. Padding removal
        6. Projection head (fc1 → activation → fc2)

        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape `(batch, s_1, ..., s_n, in_channels)`.

        Returns
        -------
        jnp.ndarray
            Output tensor of shape `(batch, s_1, ..., s_n, out_channels)`.
        """
        ndim = x.ndim - 2
        if ndim < 1:
            raise ValueError("WNO expects at least one spatial dimension.")

        x = x.astype(self.compute_dtype)

        if self.use_grid:
            grid_boundaries = self.grid_boundaries or _default_grid_boundaries(ndim)
            if len(grid_boundaries) != ndim:
                raise ValueError(
                    f"grid_boundaries has {len(grid_boundaries)} entries, expected {ndim}."
                )
            x = _append_grid_nhwc(x, grid_boundaries)

        x = self.fc0(x)

        spatial_shape = tuple(x.shape[1:-1])
        pad_amounts = None
        if isinstance(self.padding, (list, tuple)):
            needs_pad = any(p > 0 for p in self.padding)
        else:
            needs_pad = self.padding > 0

        if needs_pad:
            pad_amounts = _normalize_padding(self.padding, spatial_shape)
            if self.padding_side == "end":
                pad_width = [(0, 0)] + [(0, pad) for pad in pad_amounts] + [(0, 0)]
            elif self.padding_side == "symmetric":
                pad_width = [(0, 0)] + [(pad, pad) for pad in pad_amounts] + [(0, 0)]
            else:
                raise ValueError(
                    f"Unsupported padding_side='{self.padding_side}'. Expected 'end' or 'symmetric'."
                )
            x = jnp.pad(x, pad_width, mode="constant")

        for block in self.wno_blocks:
            x = block(x)

        if pad_amounts is not None:
            # Symmetric padding added `pad` points before the data, so the
            # original region starts at `pad`; with "end" padding it starts at 0.
            slices = [slice(None)]
            for dim_size, pad in zip(spatial_shape, pad_amounts):
                start = pad if self.padding_side == "symmetric" else 0
                slices.append(slice(start, start + dim_size))
            slices.append(slice(None))
            x = x[tuple(slices)]

        x = self.fc1(x)
        x = self.projection_activation(x)
        x = self.fc2(x)
        return x
