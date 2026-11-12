# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Wavelet-domain convolution layer of the WNO; requires the `jaxwt` package."""

from collections.abc import Sequence
from functools import lru_cache
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import jaxwt
from flax import linen as nn

_SUPPORTED_MODES = {
    "symmetric": "symmetric",
    "reflect": "reflect",
    "zero": "zero",
    "constant": "zero",
}

# jnp.pad has no "zero" mode (only jaxwt does) - translate _normalize_mode's
# output to a mode jnp.pad actually accepts.
_JNP_PAD_MODES = {
    "symmetric": "symmetric",
    "reflect": "reflect",
    "zero": "constant",
}


def _normalize_size(size: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(size, int):
        return (size,) * ndim

    size_tuple = tuple(size)
    if len(size_tuple) != ndim:
        raise ValueError(f"Expected size with {ndim} entries, got {size_tuple}.")
    return size_tuple


def _normalize_mode(mode: str) -> str:
    normalized = _SUPPORTED_MODES.get(mode.lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported wavelet boundary mode '{mode}'. "
            "jaxwt supports 'symmetric', 'reflect', and 'zero'."
        )
    return normalized


def _effective_level_wno(
    input_size: tuple[int, ...], base_size: tuple[int, ...], level: int
) -> int:
    def _floor_log2_ratio(numerator: int, denominator: int) -> int:
        ratio = max(1, numerator // denominator)
        return ratio.bit_length() - 1

    ref_input = input_size[-1]
    ref_base = base_size[-1]
    factor = 0
    if ref_input > ref_base:
        factor = _floor_log2_ratio(ref_input, ref_base)
    elif ref_input < ref_base:
        factor = -_floor_log2_ratio(ref_base, ref_input)
    return max(0, level + factor)


def _subband_codes(ndim: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=ndim))


def _subband_name(code: tuple[int, ...], ndim: int) -> str:
    if ndim == 1:
        return "weights_approx" if code == (0,) else "weights_detail"
    if ndim == 2:
        labels = {0: "l", 1: "h"}
        return "w_" + "".join(labels[bit] for bit in code)
    return "w_" + "".join("l" if bit == 0 else "h" for bit in code)


# ---------------------------------------------------------------------------
# Layout helpers - used both at the NCHW boundary and by _mode_sizes/_wavedec_nd
# ---------------------------------------------------------------------------


def _to_jaxwt_layout(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    if ndim == 1:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 2:
        return jnp.transpose(x, (0, 3, 1, 2))
    if ndim == 3:
        return jnp.transpose(x, (0, 4, 1, 2, 3))
    raise ValueError(
        "WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs."
    )


def _from_jaxwt_array(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    if ndim == 1:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 2:
        return jnp.transpose(x, (0, 2, 3, 1))
    if ndim == 3:
        return jnp.transpose(x, (0, 2, 3, 4, 1))
    raise ValueError(
        "WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs."
    )


def _coeffs_from_jaxwt_layout(coeffs, ndim: int):
    """Convert jaxwt NCHW coefficients to NHWC - used only by _mode_sizes."""
    if ndim == 1:
        return [_from_jaxwt_array(coeff, ndim) for coeff in coeffs]
    if ndim == 2:
        converted: list[Any] = [_from_jaxwt_array(coeffs[0], ndim)]
        converted.extend(
            tuple(_from_jaxwt_array(part, ndim) for part in detail)
            for detail in coeffs[1:]
        )
        return converted
    converted = [_from_jaxwt_array(coeffs[0], ndim)]
    converted.extend(
        {key: _from_jaxwt_array(value, ndim) for key, value in detail.items()}
        for detail in coeffs[1:]
    )
    return converted


def _wavedec_nd(x: jnp.ndarray, wavelet: str, mode: str, level: int, ndim: int):
    """NHWC → wavedec → NHWC coefficients. Used only by _mode_sizes."""
    x_cf = _to_jaxwt_layout(x, ndim)
    if ndim == 1:
        coeffs = jaxwt.wavedec(x_cf, wavelet, mode=mode, level=level)
    elif ndim == 2:
        coeffs = jaxwt.wavedec2(x_cf, wavelet, mode=mode, level=level)
    elif ndim == 3:
        coeffs = jaxwt.wavedec3(x_cf, wavelet, mode=mode, level=level)
    else:
        # Unreachable: _to_jaxwt_layout above already raises for this ndim.
        raise ValueError(
            "WaveletConv with jaxwt currently supports only 1D, 2D, and 3D inputs."
        )  # pragma: no cover
    return _coeffs_from_jaxwt_layout(coeffs, ndim)


@lru_cache(maxsize=128)
def _mode_sizes(
    size: tuple[int, ...], level: int, wavelet: str, mode: str, ndim: int
) -> tuple[int, ...]:
    if level == 0:
        return size

    dummy = jnp.zeros((1,) + size + (1,), dtype=jnp.float32)
    coeffs = _wavedec_nd(dummy, wavelet, mode, level, ndim)
    return tuple(coeffs[0].shape[1:-1])


# ---------------------------------------------------------------------------
# NCHW-native helpers - used by the fast forward path in WaveletConv
# ---------------------------------------------------------------------------


def _apply_weight_nchw(coeff: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """Apply a subband weight in NCHW format using einsum.

    coeff:  (B, Cin, *spatial)
    weight: (Cin, Cout, *mode_sizes)
    out:    (B, Cout, *spatial)  - zero-padded where spatial > mode_sizes

    Coefficients are cast to the weight dtype (float32) so that float64
    filter banks in jaxwt never silently promote the computation to float64.
    """
    spatial = coeff.shape[2:]
    mode_spatial = weight.shape[2:]
    active = tuple(min(c, m) for c, m in zip(spatial, mode_spatial))

    ndim = len(spatial)
    sp = "".join(chr(ord("x") + k) for k in range(ndim))
    es = f"bi{sp},io{sp}->bo{sp}"

    w_sl = (slice(None), slice(None)) + tuple(slice(0, d) for d in active)
    wa = weight[w_sl]  # always float32 (see init_fn)
    c_sl = (slice(None), slice(None)) + tuple(slice(0, d) for d in active)

    if jnp.iscomplexobj(coeff):
        ca = coeff[c_sl]
        real_dtype = wa.dtype  # float32
        cr = jnp.real(ca).astype(real_dtype)
        ci = jnp.imag(ca).astype(real_dtype)
        # Weights are always real - complex(a+bi) * w = (a*w) + (b*w)i.
        # jax.lax.complex avoids the '1j' Python literal which promotes to
        # complex128 when JAX x64 mode is on.
        out = jax.lax.complex(
            jnp.einsum(es, cr, wa),
            jnp.einsum(es, ci, wa),
        )
    else:
        # Cast to weight dtype: jaxwt filter banks are float64, so wavedec
        # can return float64 even for float32 input. Force float32 here.
        ca = coeff[c_sl].astype(wa.dtype)
        out = jnp.einsum(es, ca, wa)

    if active == spatial:
        return out
    pw = ((0, 0), (0, 0)) + tuple((0, s - a) for a, s in zip(active, spatial))
    return jnp.pad(out, pw)


def _weighted_detail_nchw(detail, weights: dict, ndim: int):
    """Apply per-subband weights to jaxwt detail coefficients (NCHW)."""
    if ndim == 1:
        return _apply_weight_nchw(detail, weights[(1,)])
    if ndim == 2:
        lh, hl, hh = detail
        return (
            _apply_weight_nchw(lh, weights[(0, 1)]),
            _apply_weight_nchw(hl, weights[(1, 0)]),
            _apply_weight_nchw(hh, weights[(1, 1)]),
        )
    # 3D - jaxwt wavedec3 returns a dict at each level
    return {
        "aad": _apply_weight_nchw(detail["aad"], weights[(0, 0, 1)]),
        "ada": _apply_weight_nchw(detail["ada"], weights[(0, 1, 0)]),
        "add": _apply_weight_nchw(detail["add"], weights[(0, 1, 1)]),
        "daa": _apply_weight_nchw(detail["daa"], weights[(1, 0, 0)]),
        "dad": _apply_weight_nchw(detail["dad"], weights[(1, 0, 1)]),
        "dda": _apply_weight_nchw(detail["dda"], weights[(1, 1, 0)]),
        "ddd": _apply_weight_nchw(detail["ddd"], weights[(1, 1, 1)]),
    }


def _zero_detail_nchw(detail, out_channels: int, ndim: int, dtype=None):
    """Build zero-filled detail coefficients (NCHW) for unused decomp levels."""

    def _z(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros(
            (t.shape[0], out_channels) + t.shape[2:],
            dtype=t.dtype if dtype is None else dtype,
        )

    if ndim == 1:
        return _z(detail)
    if ndim == 2:
        return tuple(_z(part) for part in detail)
    return {k: _z(v) for k, v in detail.items()}


def _pad_to_level_multiple(
    x: jnp.ndarray, input_size: tuple[int, ...], effective_level: int, mode: str
) -> tuple[jnp.ndarray, tuple[int, ...]]:
    scale_factor = 2**effective_level
    target_size = tuple(
        ((dim - 1) // scale_factor + 1) * scale_factor for dim in input_size
    )
    pad_sizes = tuple(
        target - current for target, current in zip(target_size, input_size)
    )

    if any(pad_sizes):
        jnp_pad_mode = _JNP_PAD_MODES[mode]
        if jnp_pad_mode == "reflect":
            for dim, pad in zip(input_size, pad_sizes):
                if pad >= dim:
                    raise ValueError(
                        f"Cannot pad dimension of size {dim} by {pad} elements "
                        "with mode='reflect': reflect padding requires the pad "
                        "width to be smaller than the dimension size. Use "
                        "mode='symmetric' or 'zero', or increase the input "
                        "resolution."
                    )
        x = jnp.pad(
            x,
            ((0, 0),) + tuple((0, pad) for pad in pad_sizes) + ((0, 0),),
            mode=jnp_pad_mode,
        )

    return x, pad_sizes


def _crop_spatial(x: jnp.ndarray, target_shape: tuple[int, ...]) -> jnp.ndarray:
    slices = (
        (slice(None),) + tuple(slice(0, dim) for dim in target_shape) + (slice(None),)
    )
    return x[slices]


class WaveletConv(nn.Module):
    """
    N-dimensional Wavelet Convolution layer following the WNO architecture.

    The decomposition and reconstruction are delegated to `jaxwt`. The input
    is transposed to NCHW once on entry, all wavelet-domain operations run in
    NCHW (jaxwt's native layout), and the result is transposed back to NHWC
    once on exit. Subband weights are applied with jnp.einsum so XLA can
    lower the operation to a cuBLAS batched GEMM.
    """

    in_channels: int
    out_channels: int
    level: int = 1
    size: int | tuple[int, ...] = 1024
    wavelet: str = "db4"
    mode: str = "symmetric"
    precision: str = "fastest"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the wavelet convolution to `x` of shape `(batch, *spatial, ch)`."""
        ndim = x.ndim - 2
        if ndim < 1:
            raise ValueError("WaveletConv expects at least one spatial dimension.")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {x.shape[-1]}."
            )

        # Preserve input dtype so we can restore it after jaxwt (whose filter
        # banks are float64 and can silently upcast the reconstruction output).
        compute_dtype = x.dtype

        normalized_mode = _normalize_mode(self.mode)
        training_size = _normalize_size(self.size, ndim)
        input_size = tuple(x.shape[1:-1])
        effective_level = _effective_level_wno(input_size, training_size, self.level)

        # Pad spatial dims to a multiple of 2^level (operates on NHWC).
        x, pad_sizes = _pad_to_level_multiple(
            x, input_size, effective_level, normalized_mode
        )

        subband_codes = _subband_codes(ndim)
        approx_code = (0,) * ndim
        mode_sizes = _mode_sizes(
            training_size, self.level, self.wavelet, normalized_mode, ndim
        )

        # Subband weights are always float32, independent of the model's
        # param_dtype knob: jaxwt's filter banks are float64, and keeping the
        # weights at float32 (rather than e.g. bfloat16) avoids compounding
        # precision loss when coefficients are cast down in _apply_weight_nchw.
        scale = jnp.float32(1.0 / (self.in_channels * self.out_channels))
        init_fn = lambda key, shape: scale * jax.random.uniform(
            key, shape, dtype=jnp.float32
        )
        weights = {
            code: self.param(
                _subband_name(code, ndim),
                init_fn,
                (self.in_channels, self.out_channels) + mode_sizes,
            )
            for code in subband_codes
        }

        # --- Single transpose to NCHW at entry ---
        x_nchw = _to_jaxwt_layout(x, ndim)

        if self.level == 0:
            res_nchw = _apply_weight_nchw(x_nchw, weights[approx_code])
            res = _from_jaxwt_array(res_nchw, ndim).astype(compute_dtype)
            return _crop_spatial(res, input_size) if any(pad_sizes) else res

        if effective_level == 0:
            raise ValueError(
                f"Inference resolution {input_size} is too small relative to "
                f"the training size {training_size} at level={self.level}: the "
                "effective decomposition level clamps to 0, which would "
                "silently skip the wavelet transform and apply the trained "
                "subband weight as a raw pointwise mix instead. Use a larger "
                "input resolution or a smaller configured level."
            )

        # Decompose - jaxwt works natively in NCHW.
        if ndim == 1:
            coeffs_nchw = jaxwt.wavedec(
                x_nchw,
                self.wavelet,
                mode=normalized_mode,
                level=effective_level,
                precision=self.precision,
            )
        elif ndim == 2:
            coeffs_nchw = jaxwt.wavedec2(
                x_nchw,
                self.wavelet,
                mode=normalized_mode,
                level=effective_level,
                precision=self.precision,
            )
        else:
            coeffs_nchw = jaxwt.wavedec3(
                x_nchw,
                self.wavelet,
                mode=normalized_mode,
                level=effective_level,
                precision=self.precision,
            )

        # Apply weights in NCHW - no intermediate transposes.
        weighted_approx = _apply_weight_nchw(coeffs_nchw[0], weights[approx_code])
        weighted_detail = _weighted_detail_nchw(coeffs_nchw[1], weights, ndim)

        recon_coeffs = [weighted_approx, weighted_detail]
        for detail_nchw in coeffs_nchw[2:]:
            recon_coeffs.append(
                _zero_detail_nchw(
                    detail_nchw,
                    self.out_channels,
                    ndim,
                    dtype=weighted_approx.dtype,
                )
            )

        # Reconstruct in NCHW, then single transpose back to NHWC.
        if ndim == 1:
            res_nchw = jaxwt.waverec(
                recon_coeffs,
                self.wavelet,
                precision=self.precision,
            )
        elif ndim == 2:
            res_nchw = jaxwt.waverec2(
                recon_coeffs,
                self.wavelet,
                precision=self.precision,
            )
        else:
            res_nchw = jaxwt.waverec3(
                recon_coeffs,
                self.wavelet,
                precision=self.precision,
            )

        # waverec uses float64 filter banks so the output may be float64 even
        # when all inputs were float32. Cast back to the original compute dtype.
        res = _from_jaxwt_array(res_nchw, ndim).astype(compute_dtype)

        if any(pad_sizes):
            res = _crop_spatial(res, input_size)

        return res
