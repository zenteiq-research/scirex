# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_notebooks.py``: passes the repo, bites on bad input."""

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_notebooks", ROOT / "scripts" / "dev" / "check_notebooks.py"
)
cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn)


def _nb(outputs=None, count=None, metadata=None, source="x = 1") -> dict:
    cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": outputs or [],
        "execution_count": count,
    }
    return {
        "cells": [cell],
        "metadata": metadata or {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _repo(tmp_path: Path, nb: dict) -> Path:
    (tmp_path / "a.ipynb").write_text(json.dumps(nb))
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_repository_passes() -> None:
    assert cn.check(ROOT) == []


def test_clean_notebook_passes(tmp_path) -> None:
    assert cn.check(_repo(tmp_path, _nb())) == []


def test_outputs_and_counts_are_reported(tmp_path) -> None:
    out = [{"output_type": "stream", "name": "stdout", "text": ["hello\n"]}]
    problems = cn.check(_repo(tmp_path, _nb(outputs=out, count=3)))
    assert any("has outputs" in p for p in problems)
    assert any("execution_count" in p for p in problems)


def test_leaks_in_remaining_outputs_are_reported(tmp_path) -> None:
    out = [{"output_type": "stream", "name": "stdout", "text": ["/home/erin/data\n"]}]
    repo = _repo(tmp_path, _nb(outputs=out))
    problems = cn.check(repo, allow=("a.ipynb",))  # outputs allowed, leaks are not
    assert problems and all("personal path" in p for p in problems)


def test_widget_state_is_reported(tmp_path) -> None:
    assert any(
        "widgets" in p for p in cn.check(_repo(tmp_path, _nb(metadata={"widgets": {}})))
    )


def test_builtin_strip_removes_outputs() -> None:
    out = [{"output_type": "stream", "name": "stdout", "text": ["x\n"]}]
    stripped = cn.strip_notebook(_nb(outputs=out, count=2, metadata={"widgets": {}}))
    cell = stripped["cells"][0]
    assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "widgets" not in stripped["metadata"]
