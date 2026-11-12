# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Weights & Biases login helpers (optional dependency)."""

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False


def wandb_login(
    api_key_file: str = "../config/wandb_api_key.txt", key: str | None = None
) -> None:
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file: str = "../config/wandb_api_key.txt") -> None:
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file) as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file: str = "../config/wandb_api_key.txt") -> str:
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file) as f:
            key = f.read()
        return key.strip()
