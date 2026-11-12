# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Resolution-aware domain padding for non-periodic inputs."""

import logging

import flax.linen as nn
import jax
import jax.numpy as jnp

from scirex._internal.scaling import validate_scaling_factor

logger = logging.getLogger(__name__)


class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution.

    Parameters
    ----------
    domain_padding : float or list
        Typically, between zero and one, percentage of padding to use
        if a list, make sure if matches the dim of (d1, ..., dN)
    resolution_scaling_factor : int, optional
        Resolution scaling factor, by default 1

    Notes
    -----
    This class works for any input resolution. Expects inputs of shape
    (batch-size, channels, d1, ...., dN)
    """

    domain_padding: float | list
    resolution_scaling_factor: int | list[int] | None = 1

    def setup(self) -> None:
        if self.resolution_scaling_factor is None:
            # mirror the None-guard from the PyTorch __init__
            object.__setattr__(self, "resolution_scaling_factor", 1)

        # dict(f'{resolution}'=pad_width) such that padded = jnp.pad(x, pad_width)
        object.__setattr__(self, "_padding", dict())

        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        object.__setattr__(self, "_unpad_indices", dict())

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        forward pass: pad the input
        """
        return self.pad(x)

    def pad(self, x: jax.Array, verbose: bool = False) -> jax.Array:
        """Take an input and pad it by the desired fraction

        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        # if domain_padding is list, then to pass on
        domain_padding = self.domain_padding
        if isinstance(domain_padding, float | int):
            domain_padding = [float(domain_padding)] * len(resolution)

        assert len(domain_padding) == len(resolution), (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, ch)"
        )

        resolution_scaling_factor: list[float] | list[int] | None
        if isinstance(self.resolution_scaling_factor, list):
            resolution_scaling_factor = self.resolution_scaling_factor
        else:
            resolution_scaling_factor = validate_scaling_factor(
                self.resolution_scaling_factor, len(resolution), n_layers=None
            )
        # setup() replaces a None resolution_scaling_factor with 1, so this
        # can only be None here if a caller bypassed setup() entirely.
        assert resolution_scaling_factor is not None

        try:
            pad_width = self._padding[f"{resolution}"]
            return jnp.pad(x, pad_width, mode="constant")

        except KeyError:
            padding = [round(p * r) for (p, r) in zip(domain_padding, resolution)]

            if verbose:
                logger.info(
                    f"Padding inputs of resolution={resolution} with "
                    f"padding={padding}, symmetric"
                )

            output_pad = padding

            output_pad = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_pad)
            ]

            # jnp.pad pad_width: no padding on (batch, channel), symmetric on spatial.
            # PyTorch F.pad takes a reversed flat list; jnp.pad takes a list of
            # (before, after) pairs in forward spatial order - mathematically identical.
            pad_width = [(0, 0), (0, 0)] + [(p, p) for p in padding]

            unpad_list = list()
            for p in output_pad:
                if p == 0:
                    padding_end = None
                    padding_start = None
                else:
                    padding_end = p
                    padding_start = -p
                unpad_list.append(slice(padding_end, padding_start, None))
            unpad_indices = (Ellipsis, *tuple(unpad_list))

            self._padding[f"{resolution}"] = pad_width

            padded = jnp.pad(x, pad_width, mode="constant")

            output_shape = padded.shape[2:]

            scaled_output_shape = [
                round(i * j) for (i, j) in zip(resolution_scaling_factor, output_shape)
            ]

            self._unpad_indices[f"{list(scaled_output_shape)}"] = unpad_indices

            return padded

    def unpad(self, x: jax.Array) -> jax.Array:
        """Remove the padding from padding inputs"""
        # Flax creates a fresh module on every apply(), so the _unpad_indices cache
        # populated during pad() is not available here.  Recompute slices directly.
        resolution = x.shape[2:]
        domain_padding = self.domain_padding
        if isinstance(domain_padding, float | int):
            domain_padding = [float(domain_padding)] * len(resolution)

        resolution_scaling_factor: list[float] | list[int] | None
        if isinstance(self.resolution_scaling_factor, list):
            resolution_scaling_factor = self.resolution_scaling_factor
        else:
            resolution_scaling_factor = validate_scaling_factor(
                self.resolution_scaling_factor, len(resolution), n_layers=None
            )
        # setup() replaces a None resolution_scaling_factor with 1, so this
        # can only be None here if a caller bypassed setup() entirely.
        assert resolution_scaling_factor is not None

        unpad_list = []
        for s, p_frac, sf in zip(resolution, domain_padding, resolution_scaling_factor):
            if p_frac == 0.0:
                unpad_list.append(slice(None))
            elif sf == 1:
                # No resolution scaling between pad() and unpad(): s is exactly
                # orig + 2*round(p_frac*orig), which round(s/(1+2*p_frac)) only
                # approximates (off by at most 1, since round()'s own error is
                # <1/2). Search that small neighborhood for the exact inverse so
                # unpad always recovers the resolution pad() actually padded from.
                naive = round(s / (1 + 2 * p_frac))
                orig = next(
                    c
                    for c in range(naive - 2, naive + 3)
                    if c > 0 and c + 2 * round(p_frac * c) == s
                )
                pad = (s - orig) // 2
                unpad_list.append(slice(pad, pad + orig))
            else:
                # reverse: orig ≈ s / (1 + 2*p_frac*sf), pad = (s - orig) // 2
                orig = round(s / (1 + 2 * p_frac * sf))
                pad = (s - orig) // 2
                unpad_list.append(slice(pad, pad + orig))

        index: tuple[slice, ...] = (slice(None), slice(None), *unpad_list)
        return x[index]
