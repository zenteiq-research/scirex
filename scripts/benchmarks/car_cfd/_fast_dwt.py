# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Separable drop-ins for ``jaxwt.wavedec3``/``jaxwt.waverec3``.

jaxwt computes each 3-D level as one dense convolution with eight
``L x L x L`` outer-product filters on a single input channel (every feature
channel folded into the batch), which XLA runs poorly on GPU: on this
benchmark's 64^3 latent grid it is ~95% of a GIWO training step. Those
filters are separable, so the same transform is three 1-D passes, one per
axis -- jaxwt's own single-level ``wavedec``/``waverec`` along that axis --
at ``3L`` rather than ``L^3`` taps per output.

Same contract as jaxwt's: input ``(..., D, H, W)``, coefficients
``[lll, {aad, ..., ddd} (coarsest), ..., {...} (finest)]``, the key letters
naming the filter on (D, H, W) in that order. ``test_car_cfd_benchmark``
checks both functions against jaxwt.
"""

from __future__ import annotations

import jaxwt


def _split(x, wavelet, mode, axis, precision):
    return jaxwt.wavedec(x, wavelet, mode=mode, level=1, axis=axis, precision=precision)


def _merge(lo, hi, wavelet, axis, precision):
    return jaxwt.waverec([lo, hi], wavelet, axis=axis, precision=precision)


def wavedec3(data, wavelet, mode="symmetric", level=1, precision="highest"):
    """Separable ``jaxwt.wavedec3`` over the last three axes."""
    details = []
    lll = data
    for _ in range(level):
        bands = {"": lll}
        for axis in (-3, -2, -1):
            bands = {
                k + tag: band
                for k, x in bands.items()
                for tag, band in zip("ad", _split(x, wavelet, mode, axis, precision))
            }
        lll = bands.pop("aaa")
        details.append(bands)
    return [lll, *reversed(details)]


def waverec3(coeffs, wavelet, precision="highest"):
    """Separable ``jaxwt.waverec3``; like jaxwt, a reconstruction one sample
    longer than the next finer level's coefficients is cropped at the end."""
    lll = coeffs[0]
    for pos, detail in enumerate(coeffs[1:]):
        bands = {"aaa": lll, **detail}
        for axis in (-1, -2, -3):
            # Merge the last letter (this axis) of every key.
            keys = sorted({k[:-1] for k in bands})
            bands = {
                k: _merge(bands[k + "a"], bands[k + "d"], wavelet, axis, precision)
                for k in keys
            }
        lll = bands[""]
        if pos + 2 < len(coeffs):
            target = coeffs[pos + 2]["aad"].shape[-3:]
            lll = lll[..., : target[0], : target[1], : target[2]]
    return lll


class _JaxwtShim:
    """``jaxwt`` with the 3-D transforms swapped for the separable ones."""

    wavedec3 = staticmethod(wavedec3)
    waverec3 = staticmethod(waverec3)

    def __getattr__(self, name):
        return getattr(jaxwt, name)


def use_separable_3d_transforms() -> None:
    """Points :class:`scirex.layers.WaveletConv`'s 3-D
    path at this module's transforms, for the rest of the process.

    Numerically the same transform (to float32 rounding), so a WNO's
    parameters and outputs are unchanged; only its speed differs. 1-D and
    2-D WNOs are untouched.
    """
    from scirex.layers.blocks import wavelet_conv

    wavelet_conv.jaxwt = _JaxwtShim()
