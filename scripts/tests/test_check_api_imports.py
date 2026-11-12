# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for scripts/dev/check_api_imports.py (the static parts)."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "dev" / "check_api_imports.py"
_spec = importlib.util.spec_from_file_location("check_api_imports", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cai = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cai)

STABLE = {"scirex": ["FNO"], "scirex.models": ["FNO"], "scirex.losses": ["LpLoss"]}


def _units(code: str) -> list[tuple[str, str, int]]:
    return [("x.py", code, 0)]


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        ("scirex", "public"),
        ("scirex.models", "public"),
        ("scirex.models.fno", "deep"),
        ("scirex.losses.data_losses", "deep"),
        ("scirex.experimental.giwo", "experimental"),
        ("scirex._internal.web_utils", "private"),
        ("scirex.models._legacy_refs", "private"),
    ],
)
def test_classify(module: str, expected: str) -> None:
    assert cai.classify(cai.Imp("x.py", 1, module, "n"), STABLE) == expected


def test_collect_reads_from_imports_and_plain_imports() -> None:
    code = (
        "from scirex.models import FNO, GINO\n"
        "import scirex.losses\n"
        "import numpy\n"
        "from . import sibling\n"
    )
    imports, bad = cai.collect(_units(code))
    assert bad == 0
    assert [(i.module, i.name) for i in imports] == [
        ("scirex.models", "FNO"),
        ("scirex.models", "GINO"),
        ("scirex.losses", ""),
    ]


def test_markdown_blocks_are_read_and_bad_examples_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text(
        "text\n\n```python\n"
        "from scirex.models import FNO            # yes\n"
        "from scirex.models.fno import FNO        # works today, not guaranteed\n"
        "```\n"
        "```python\nthis is ( not python\n```\n",
        encoding="utf-8",
    )
    imports, bad = cai.collect(cai.python_sources(tmp_path))
    assert [(i.module, i.path, i.line) for i in imports] == [
        ("scirex.models", "guide.md", 4)
    ]
    assert bad == 1


def test_strict_fails_on_a_deep_import_in_user_facing_docs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "api_snapshot.json").write_text(json.dumps(STABLE))
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text(
        "```python\nfrom scirex.losses.data_losses import LpLoss\n```\n"
    )
    argv = ["--root", str(tmp_path), "--static"]
    assert cai.main(argv) == 0
    assert cai.main([*argv, "--strict"]) == 1
    out = capsys.readouterr().out
    assert "LpLoss is public: import it from its package" in out


def test_deep_imports_inside_tests_are_not_a_strict_failure(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "api_snapshot.json").write_text(json.dumps(STABLE))
    (tmp_path / "tests" / "test_a.py").write_text(
        "from scirex.losses.data_losses import LpLoss\n"
    )
    assert cai.main(["--root", str(tmp_path), "--static", "--strict"]) == 0
