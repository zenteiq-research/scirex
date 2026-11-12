# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Generic n-dimensional spectral resampling / interpolation utilities."""

import itertools
from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
from jax.image import scale_and_translate


def resample(
    x: jnp.ndarray,
    res_scale: float | Sequence[float],
    axis: int | Sequence[int] | None,
    output_shape: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """A module for generic n-dimentional interpolation.

    For 1D and 2D inputs, the `resample` function uses JAX's built-in spatial
    interpolators for efficiency, applying linear interpolation for 1D data and
    bicubic interpolation for 2D data directly in the spatial domain.

    For 3D or higher-dimensional inputs, the `resample` function switches to a
    spectral interpolation method based on the Fourier transform. The input is
    transformed into the frequency domain using a real n-dimensional FFT, which
    decomposes the signal into its frequency components. By resizing this frequency
    representation and then applying an inverse FFT, the function achieves smooth,
    alias-free interpolation that preserves the signal's overall structure.

    Parameters
    ----------
    x : jnp.ndarray
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If
            res_scale is scalar, then isotropic scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """
    if isinstance(res_scale, float | int):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale] * len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
            res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(
            cast("Sequence[int]", axis)
        ), "length of res_scale and axis are not same"

    # axis=None is only valid alongside a scalar res_scale, normalized to a
    # Sequence[int] above; a non-scalar res_scale requires the caller to have
    # already passed a matching Sequence[int] axis.
    axis = cast("Sequence[int]", axis)
    res_scale = cast("Sequence[float]", res_scale)

    old_size = x.shape[-len(axis) :]
    if output_shape is None:
        new_size = tuple([int(round(s * r)) for (s, r) in zip(old_size, res_scale)])
    else:
        new_size = output_shape

    if len(axis) == 1:
        # JAX equivalent of F.interpolate(..., mode="linear", align_corners=True)
        # jax.image.scale_and_translate works on (batch, spatial..., channels)
        # x is (batch, channels, spatial) → transpose to (batch, spatial, channels)
        x_t = jnp.transpose(x, (0, 2, 1))
        scale = jnp.array([new_size[0] / old_size[0]])
        translation = jnp.array(
            [(new_size[0] - 1) / 2 - scale[0] * (old_size[0] - 1) / 2]
        )
        out = scale_and_translate(
            x_t,
            shape=(x.shape[0], new_size[0], x.shape[1]),
            spatial_dims=(1,),
            scale=scale,
            translation=translation,
            method="linear",
        )
        return jnp.transpose(out, (0, 2, 1))

    if len(axis) == 2:
        # JAX equivalent of F.interpolate(..., mode="bicubic", align_corners=True)
        x_t = jnp.transpose(x, (0, 2, 3, 1))  # (batch, h, w, channels)
        scale = jnp.array([new_size[0] / old_size[0], new_size[1] / old_size[1]])
        translation = jnp.array(
            [
                (new_size[0] - 1) / 2 - scale[0] * (old_size[0] - 1) / 2,
                (new_size[1] - 1) / 2 - scale[1] * (old_size[1] - 1) / 2,
            ]
        )
        out = scale_and_translate(
            x_t,
            shape=(x.shape[0], new_size[0], new_size[1], x.shape[1]),
            spatial_dims=(1, 2),
            scale=scale,
            translation=translation,
            method="bicubic",
        )
        return jnp.transpose(out, (0, 3, 1, 2))

    X = jnp.fft.rfftn(x.astype(jnp.float32), norm="forward", axes=axis)

    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1  # Redundant last coefficient
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X.shape[-len(axis) :])]
    out_fft = jnp.zeros([x.shape[0], x.shape[1], *new_fft_size], dtype=jnp.complex64)

    mode_indexing = [((None, m // 2), (-m // 2, None)) for m in new_fft_size_c[:-1]] + [
        ((None, new_fft_size_c[-1]),)
    ]
    for boundaries in itertools.product(*mode_indexing):
        idx_tuple = tuple([slice(None), slice(None)] + [slice(*b) for b in boundaries])
        out_fft = out_fft.at[idx_tuple].set(X[idx_tuple])

    y = jnp.fft.irfftn(out_fft, s=new_size, norm="forward", axes=axis)

    return y


def iterative_resample(
    x: jnp.ndarray, res_scale: float | list[float], axis: int | list[int]
) -> jnp.ndarray:
    """Resample `x` along one or several axes by spectral interpolation, one axis at a time.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    res_scale : float or list[float]
        Resolution scale factor, one per axis if a list.
    axis : int or list[int]
        Axis or axes to resample.

    Returns
    -------
    jnp.ndarray
        The resampled array.

    Raises
    ------
    Exception
        If `axis` is not a list but `res_scale` is, or if their lengths differ.
    """
    if isinstance(axis, list) and isinstance(res_scale, float | int):
        res_scale = [res_scale] * len(axis)
    if not isinstance(axis, list) and isinstance(res_scale, list):
        raise Exception("Axis is not a list but Scale factors are")
    if (
        isinstance(axis, list)
        and isinstance(res_scale, list)
        and len(res_scale) != len(axis)
    ):
        raise Exception("Axis and Scale factor are in different sizes")

    if isinstance(axis, list):
        # guarded above: axis being a list guarantees res_scale is a list too,
        # either passed in that way or normalized on line 136.
        res_scale_list = cast("list[float]", res_scale)
        for i in range(len(res_scale_list)):
            rs = res_scale_list[i]
            a = axis[i]
            x = resample(x, rs, a)
        return x

    # guarded above: axis not being a list guarantees res_scale is a scalar float.
    res_scale = cast("float", res_scale)
    old_res = x.shape[axis]
    X = jnp.fft.rfft(x.astype(jnp.float32), axis=axis, norm="forward")
    newshape = list(x.shape)
    new_res = int(round(res_scale * newshape[axis]))
    newshape[axis] = new_res // 2 + 1

    Y = jnp.zeros(newshape, dtype=X.dtype)

    modes = min(new_res, old_res)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, modes // 2 + 1)
    Y = Y.at[tuple(sl)].set(X[tuple(sl)])
    y = jnp.fft.irfft(Y, n=new_res, axis=axis, norm="forward")
    return y
