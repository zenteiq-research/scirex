# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit tests for the Weights & Biases helpers.

The reference's ``test_ArgparseConfig`` (``wandb_login``) requires a live
``wandb`` install/login and is reproduced as a gated, skipped test.
"""

from pathlib import Path

import pytest

from scirex._internal.wandb_utils import get_wandb_api_key

# ---------------------------------------------------------------------------
# get_wandb_api_key
# ---------------------------------------------------------------------------


def test_get_wandb_api_key_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no env var, the key is read from the given file."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    key_file = tmp_path / "key.txt"
    key_file.write_text("my_secret_key")
    assert get_wandb_api_key(key_file.as_posix()) == "my_secret_key"


def test_get_wandb_api_key_env_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env var overrides the file (and a wrong path is ignored)."""
    key_file = tmp_path / "key.txt"
    key_file.write_text("my_secret_key")
    monkeypatch.setenv("WANDB_API_KEY", "key_from_env")
    assert get_wandb_api_key(key_file.as_posix()) == "key_from_env"
    assert get_wandb_api_key("wrong_path") == "key_from_env"


@pytest.mark.skip(
    reason="Mirrors test_ArgparseConfig, which exercises wandb_login against a live "
    "wandb install/login. wandb is not a CI dependency and a real login cannot run "
    "in CI; get_wandb_api_key (the testable part) is covered above."
)
def test_wandb_login() -> None:
    pass
