# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit tests for ``validate_scaling_factor``."""

import pytest

from scirex._internal.scaling import validate_scaling_factor

# ---------------------------------------------------------------------------
# validate_scaling_factor (JAX-only utility)
# ---------------------------------------------------------------------------


def test_validate_scaling_factor_none() -> None:
    """None scaling factor passes through as None."""
    assert validate_scaling_factor(None, n_dim=2) is None


def test_validate_scaling_factor_scalar() -> None:
    """A scalar is broadcast to one value per dim (or per layer when given)."""
    assert validate_scaling_factor(2, n_dim=3) == [2.0, 2.0, 2.0]
    assert validate_scaling_factor(2, n_dim=2, n_layers=2) == [[2.0, 2.0], [2.0, 2.0]]


def test_validate_scaling_factor_dimwise_list() -> None:
    """A length-n_dim list is treated as a per-dim scaling."""
    assert validate_scaling_factor([2, 3], n_dim=2) == [2.0, 3.0]


# ---------------------------------------------------------------------------
# validate_scaling_factor - missing branches (source lines 196-212)
# ---------------------------------------------------------------------------


def test_validate_scaling_factor_flat_list_length_mismatch_raises() -> None:
    """A flat list whose length differs from n_dim is invalid when n_layers is
    None: there's no way to tell a mistyped dim-wise list from an intended
    per-layer list, so this must raise rather than silently reinterpreting it
    as list-of-lists (see PR review discussion on the SpectralConv crash this
    used to cause)."""
    with pytest.raises(ValueError, match="n_layers=None"):
        validate_scaling_factor([0.5, 1.0, 2.0], n_dim=2)


def test_validate_scaling_factor_list_with_n_layers() -> None:
    """List with n_layers set takes the layer-wise return (line 196)."""
    result = validate_scaling_factor([2.0, 3.0], n_dim=2, n_layers=2)
    assert result == [[2.0, 2.0], [3.0, 3.0]]


def test_validate_scaling_factor_list_of_lists() -> None:
    """A list-of-lists is validated and returned as-is (lines 198-210)."""
    lol = [[2.0, 3.0], [4.0, 5.0]]
    result = validate_scaling_factor(lol, n_dim=2)
    assert result == lol


def test_validate_scaling_factor_invalid_returns_none() -> None:
    """An unsupported type falls through and returns None (line 212)."""
    # deliberately-wrong type, to exercise the unsupported-input fallback
    assert validate_scaling_factor("bad_input", n_dim=2) is None  # type: ignore[call-overload]
