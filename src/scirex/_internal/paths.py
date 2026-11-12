# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Project path helpers."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the directory that contains the top-level `scirex` package."""
    root = Path(__file__).resolve().parents[2]
    return root
