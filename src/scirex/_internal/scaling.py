# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Validation of spectral-layer scaling factors."""

from typing import cast, overload

Number = float | int


@overload
def validate_scaling_factor(
    scaling_factor: None,
    n_dim: int,
    n_layers: int | None = None,
) -> None: ...
@overload
def validate_scaling_factor(
    scaling_factor: Number | list[Number],
    n_dim: int,
    n_layers: None = None,
) -> list[float]: ...
@overload
def validate_scaling_factor(
    scaling_factor: list[list[Number]],
    n_dim: int,
    n_layers: None = None,
) -> list[list[float]]: ...
@overload
def validate_scaling_factor(
    scaling_factor: Number | list[Number] | list[list[Number]],
    n_dim: int,
    n_layers: int,
) -> list[list[float]]: ...
def validate_scaling_factor(
    scaling_factor: None | Number | list[Number] | list[list[Number]],
    n_dim: int,
    n_layers: int | None = None,
) -> None | list[float] | list[list[float]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    # Flax frozen dataclasses serialise list fields as tuples; normalise here.
    if isinstance(scaling_factor, tuple):
        scaling_factor = list(scaling_factor)
    if isinstance(scaling_factor, float | int):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all(isinstance(s, float | int) for s in scaling_factor)
    ):
        if n_layers is None:
            if len(scaling_factor) != n_dim:
                raise ValueError(
                    f"Got a flat scaling_factor={scaling_factor} of length "
                    f"{len(scaling_factor)} but n_dim={n_dim} and n_layers=None; "
                    "a flat scaling factor must have exactly one entry per "
                    "spatial dimension."
                )
            # this is a dim-wise scaling
            flat_factor = cast(list[Number], scaling_factor)
            return [float(s) for s in flat_factor]
        flat_factor = cast(list[Number], scaling_factor)
        return [[float(s)] * n_dim for s in flat_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all(isinstance(s, list) for s in scaling_factor)
    ):
        nested_factor = cast(list[list[Number]], scaling_factor)
        s_sub_pass = True
        for s in nested_factor:
            if all([isinstance(s_sub, int | float) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return [[float(s_sub) for s_sub in s] for s in nested_factor]

    return None
