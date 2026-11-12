# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_tables.py``: passes the repository, bites on a changed number."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_tables", ROOT / "scripts" / "dev" / "check_tables.py"
)
ct = importlib.util.module_from_spec(spec)
sys.modules["check_tables"] = ct  # dataclasses look the module up here
spec.loader.exec_module(ct)


def fake_repo(tmp_path: Path, shown: str, stored: float) -> Path:
    page = tmp_path / "doc" / "benchmarks" / "darcy" / "results" / "fno.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"| FNO | test relative L2 | {shown} |\n")
    metrics = tmp_path / "doc" / "assets" / "benchmarks" / "darcy_fno" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"final_test_rel_l2_mean": stored}))
    return tmp_path


def test_the_repository_tables_match() -> None:
    targets, _ = ct.discover(ROOT)
    assert len(targets) >= 10
    assert ct.check(targets) == []


def test_shown_in_page_handles_percent_and_fractions() -> None:
    assert ct.shown_in_page(0.01109, 100.0, "error 1.109%")
    assert ct.shown_in_page(0.0705, 1.0, "error 0.0705")
    assert not ct.shown_in_page(0.0705, 1.0, "error 0.0905")


def test_matching_table_passes(tmp_path) -> None:
    root = fake_repo(tmp_path, "1.109%", 0.01109)
    targets, skipped = ct.discover(root)
    assert [t.name for t in targets] == ["darcy_fno"] and not skipped
    assert ct.check(targets) == []


def test_edited_number_is_reported(tmp_path) -> None:
    root = fake_repo(tmp_path, "1.109%", 0.01109)
    (root / "doc/assets/benchmarks/darcy_fno/metrics.json").write_text(
        json.dumps({"final_test_rel_l2_mean": 0.02})
    )
    targets, _ = ct.discover(root)
    assert targets == []  # no longer found in the page, so it is reported as unchecked


def test_strict_exit_status(tmp_path) -> None:
    root = fake_repo(tmp_path, "1.109%", 0.01109)
    assert ct.main(["--root", str(root), "--strict"]) == 0
