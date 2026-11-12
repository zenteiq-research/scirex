# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""msgpack helpers for checkpoint and metadata files."""

from pathlib import Path
from typing import Any

import flax.serialization


def to_serializable(obj: Any) -> Any:
    """Recursively converts tuples to lists.

    msgpack (via `flax.serialization`) has no tuple type and raises on
    one with `strict_types=True`; this makes arbitrary nested
    dict/list/tuple payloads safe to pass to `save_msgpack`. Tuples
    come back as lists after a round trip through `load_msgpack`.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_serializable(v) for v in obj]
    return obj


def save_msgpack(payload: Any, path: str | Path) -> None:
    """Serializes `payload` to `path` with msgpack.

    Safe alternative to `pickle` for checkpoint/metadata files: handles
    nested dicts/lists, Python scalars, and jax/numpy arrays, and never
    executes code on load (unlike `pickle.load`).
    """
    Path(path).write_bytes(
        flax.serialization.msgpack_serialize(to_serializable(payload))
    )


def load_msgpack(path: str | Path) -> Any:
    """Inverse of `save_msgpack`."""
    return flax.serialization.msgpack_restore(Path(path).read_bytes())
