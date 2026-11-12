# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit tests for the project path helpers."""

from scirex._internal.paths import get_project_root


def test_get_project_root_points_at_repo() -> None:
    """get_project_root resolves to a directory containing the scirex package."""
    root = get_project_root()
    assert (root / "scirex").is_dir()
