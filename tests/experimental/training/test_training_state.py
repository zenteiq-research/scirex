# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for save_training_state / load_training_state (JAX).

neuraloperator's equivalent is exercised indirectly through the trainer
checkpoint tests; here the save/load helpers are tested directly (Phase 1
``layers/tests`` structure). Objects exposing ``state_dict()`` are persisted via
that method, plain objects are serialized as-is, and the manifest carries the
epoch. These helpers are pure filesystem/msgpack code (not pickle -- see VA
finding F-03), so they are device independent.
"""

from pathlib import Path
from typing import Any

import pytest

from scirex._internal.serialization import load_msgpack
from scirex.experimental.training.training_state import (
    load_training_state,
    save_training_state,
)


class _WithStateDict:
    """Object whose persisted form comes from ``state_dict()``."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def state_dict(self) -> Any:
        return self._payload


# ---------------------------------------------------------------------------
# save_training_state
# ---------------------------------------------------------------------------


def test_save_writes_expected_files(tmp_path: Path) -> None:
    """Saving model/optimizer/scheduler/regularizer writes their state
    files + manifest."""
    save_training_state(
        tmp_path,
        "model",
        model=_WithStateDict({"w": [1, 2, 3]}),
        optimizer=_WithStateDict({"lr": 0.1}),
        scheduler=_WithStateDict({"step": 5}),
        regularizer=_WithStateDict({"l": 1e-4}),
        epoch=7,
    )
    for name in (
        "model_state_dict.msgpack",
        "optimizer.msgpack",
        "scheduler.msgpack",
        "regularizer.msgpack",
        "manifest.msgpack",
    ):
        assert (tmp_path / name).exists()


def test_save_uses_state_dict_when_available(tmp_path: Path) -> None:
    """An object with state_dict() is persisted via its state_dict payload."""
    save_training_state(tmp_path, "model", model=_WithStateDict({"w": 42}))
    assert load_msgpack(tmp_path / "model_state_dict.msgpack") == {"w": 42}


def test_save_pickles_plain_object_directly(tmp_path: Path) -> None:
    """A plain object (no state_dict()) is serialized as-is."""
    save_training_state(tmp_path, "model", model={"raw": [1, 2]})
    assert load_msgpack(tmp_path / "model_state_dict.msgpack") == {"raw": [1, 2]}


def test_save_creates_missing_directory(tmp_path: Path) -> None:
    """save_training_state creates a nested save_dir if it does not exist."""
    nested = tmp_path / "a" / "b" / "c"
    save_training_state(nested, "model", model={"x": 1}, epoch=1)
    assert (nested / "manifest.msgpack").exists()


# ---------------------------------------------------------------------------
# load_training_state
# ---------------------------------------------------------------------------


def test_load_round_trips_optimizer_and_epoch(tmp_path: Path) -> None:
    """Saved optimizer state and epoch are recovered by load_training_state."""
    save_training_state(
        tmp_path,
        "model",
        optimizer=_WithStateDict({"lr": 0.1}),
        scheduler=_WithStateDict({"step": 3}),
        epoch=9,
    )
    # pass placeholder objects so the loader knows to read those files back
    _, optimizer, scheduler, _, epoch = load_training_state(
        tmp_path,
        "model",
        optimizer=_WithStateDict({}),
        scheduler=_WithStateDict({}),
    )
    assert epoch == 9
    assert optimizer == {"lr": 0.1}
    assert scheduler == {"step": 3}


def test_load_accepts_str_path(tmp_path: Path) -> None:
    """A string save_dir is accepted (converted to Path internally)."""
    save_training_state(
        tmp_path, "model", optimizer=_WithStateDict({"lr": 0.2}), epoch=1
    )
    _, optimizer, _, _, epoch = load_training_state(
        str(tmp_path), "model", optimizer=_WithStateDict({})
    )
    assert epoch == 1
    assert optimizer == {"lr": 0.2}


def test_load_missing_optimizer_keeps_placeholder(
    tmp_path: Path,
) -> None:
    """Requesting optimizer load when none was saved warns and keeps the input."""
    save_training_state(tmp_path, "model", model={"x": 1})  # no optimizer saved
    placeholder = _WithStateDict({})
    with pytest.warns(UserWarning, match="optimizer"):
        _, optimizer, _, _, _ = load_training_state(
            tmp_path, "model", optimizer=placeholder
        )
    assert optimizer is placeholder


def test_load_without_manifest_returns_none_epoch(tmp_path: Path) -> None:
    """With no manifest on disk, the recovered epoch is None."""
    _, _, _, _, epoch = load_training_state(tmp_path, "model")
    assert epoch is None


# ---------------------------------------------------------------------------
# Additional load_training_state coverage
# ---------------------------------------------------------------------------


class _WithLoadStateDict:
    """Object that exposes load_state_dict() so the loader assigns into it."""

    def __init__(self) -> None:
        self.loaded = None

    def load_state_dict(self, state_dict: Any) -> None:
        self.loaded = state_dict


def test_load_model_without_load_state_dict_returns_raw_dict(tmp_path: Path) -> None:
    """When the model has no load_state_dict(), the raw loaded dict is
    returned (lines 78-78)."""
    save_training_state(tmp_path, "model", model={"weights": [1, 2, 3]})
    # Pass a plain dict as the model placeholder - it has no load_state_dict
    model, _, _, _, _ = load_training_state(
        tmp_path, "model", model={"placeholder": None}
    )
    assert model == {"weights": [1, 2, 3]}


def test_load_model_with_load_state_dict_calls_it(tmp_path: Path) -> None:
    """When the model has load_state_dict(), it is called with the loaded
    params (lines 74-75)."""
    save_training_state(tmp_path, "model", model={"weights": [4, 5, 6]})
    m = _WithLoadStateDict()
    _returned_model, _, _, _, _ = load_training_state(tmp_path, "model", model=m)
    assert m.loaded == {"weights": [4, 5, 6]}


def test_load_missing_model_prints_warning(
    tmp_path: Path,
) -> None:
    """Requesting model load when none was saved warns and keeps the input
    (lines 79-82)."""
    # Nothing saved in tmp_path
    placeholder = {"placeholder": None}
    with pytest.warns(UserWarning, match="model"):
        model, _, _, _, _ = load_training_state(tmp_path, "model", model=placeholder)
    assert model is placeholder


def test_load_missing_scheduler_prints_warning(
    tmp_path: Path,
) -> None:
    """Requesting scheduler load when none was saved warns (line 101)."""
    # Save only the manifest so it exists but no scheduler file
    save_training_state(tmp_path, "model", epoch=3)
    placeholder = _WithStateDict({})
    with pytest.warns(UserWarning, match="scheduler"):
        _, _, scheduler, _, epoch = load_training_state(
            tmp_path, "model", scheduler=placeholder
        )
    assert scheduler is placeholder
    assert epoch == 3


def test_load_regularizer_round_trips(tmp_path: Path) -> None:
    """Regularizer saved and loaded end-to-end (lines 106-109)."""
    save_training_state(tmp_path, "model", regularizer=_WithStateDict({"lambda": 1e-4}))
    _, _, _, reg, _ = load_training_state(
        tmp_path, "model", regularizer=_WithStateDict({})
    )
    assert reg == {"lambda": 1e-4}


def test_load_missing_regularizer_prints_warning(
    tmp_path: Path,
) -> None:
    """Requesting regularizer load when none was saved warns (lines 110-111)."""
    save_training_state(tmp_path, "model", epoch=1)
    placeholder = _WithStateDict({})
    with pytest.warns(UserWarning, match="regularizer"):
        _, _, _, reg, _ = load_training_state(
            tmp_path, "model", regularizer=placeholder
        )
    assert reg is placeholder


# ---------------------------------------------------------------------------
# Additional save_training_state coverage
# ---------------------------------------------------------------------------


def test_save_accepts_string_save_dir(tmp_path: Path) -> None:
    """save_training_state converts a str save_dir to Path internally (line 148)."""
    str_path = str(tmp_path / "checkpoints")
    save_training_state(str_path, "model", model={"x": 1}, epoch=2)
    from pathlib import Path

    assert (Path(str_path) / "manifest.msgpack").exists()
    assert (Path(str_path) / "model_state_dict.msgpack").exists()


def test_save_regularizer_via_state_dict(tmp_path: Path) -> None:
    """Regularizer with state_dict() is persisted via that method (line 197)."""
    save_training_state(tmp_path, "model", regularizer=_WithStateDict({"lambda": 1e-3}))
    saved = load_msgpack(tmp_path / "regularizer.msgpack")
    assert saved == {"lambda": 1e-3}
