# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_naming.py``: passes the repo, bites on bad input."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_naming", ROOT / "scripts" / "dev" / "check_naming.py"
)
cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn)


def _tree(tmp_path: Path, files: dict[str, str], bench=()) -> Path:
    for name, text in files.items():
        p = tmp_path / "src" / "scirex" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for b in bench:
        (tmp_path / "scripts" / "benchmarks" / b).mkdir(parents=True)
    return tmp_path


def test_the_repository_passes() -> None:
    assert cn.check(ROOT) == []


def test_good_names_pass(tmp_path) -> None:
    root = _tree(
        tmp_path,
        {
            "fno_block.py": "class FNOBlock:\n    def apply_fn(self): ...\n",
            "config/models.py": "class FNOSmall2d:\n    n_modes: int = 4\n",
        },
        bench=["chip_cooling"],
    )
    assert cn.check(root) == []


def test_bad_names_are_reported(tmp_path) -> None:
    root = _tree(
        tmp_path,
        {
            "BadModule.py": "class bad_class:\n    def BadFunc(self): ...\n",
            "config/models.py": "class Cfg:\n    nModes: int = 4\n",
        },
        bench=["ChipCooling"],
    )
    text = "\n".join(cn.check(root))
    for needle in ("BadModule", "bad_class", "BadFunc", "nModes", "ChipCooling"):
        assert needle in text
