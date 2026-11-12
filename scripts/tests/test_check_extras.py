# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_extras.py``: the ``all`` extra covers every import."""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_extras", ROOT / "scripts" / "dev" / "check_extras.py"
)
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)

PYPROJECT = """[project]
dependencies = [
    "numpy>=1.26",
    "scikit-learn>=1.7",
]

[project.optional-dependencies]
cuda12 = ["jax[cuda12]>=0.6"]  # hardware
dev = [
    "pytest>=8",
]
extra = [
    "ray[tune]>=2.5",
]
all = [
    "scirex[dev,extra]",
]

[build-system]
requires = ["hatchling"]
"""


def repo(tmp_path: Path, source: str, pyproject: str = PYPROJECT) -> Path:
    (tmp_path / "pyproject.toml").write_text(pyproject)
    (tmp_path / "a.py").write_text(source)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_repository_is_covered() -> None:
    assert ce.check(ROOT) == []


def test_parse_reads_core_and_extras() -> None:
    core, extras = ce.parse_pyproject(PYPROJECT)
    assert core == ["numpy>=1.26", "scikit-learn>=1.7"]
    assert set(extras) == {"cuda12", "dev", "extra", "all"}
    assert extras["extra"] == ["ray[tune]>=2.5"]


def test_closure_follows_the_all_extra() -> None:
    core, extras = ce.parse_pyproject(PYPROJECT)
    names = ce.closure(core, extras, ["all"])
    assert {"numpy", "scikit-learn", "pytest", "ray"} <= names
    assert "jax" not in names  # hardware extra is not part of all


def test_covered_imports_pass(tmp_path) -> None:
    src = "import numpy\nimport sklearn.cluster\nimport ray\nimport pytest\nimport os\n"
    assert ce.check(repo(tmp_path, src)) == []


def test_uncovered_import_is_reported(tmp_path) -> None:
    found = ce.check(repo(tmp_path, "import pandas\nfrom vtk import x\n"))
    assert any("'pandas' is imported by a.py" in p for p in found)
    assert any("'vtk'" in p for p in found)


def test_extra_left_out_of_all_is_reported(tmp_path) -> None:
    bad = PYPROJECT.replace('"scirex[dev,extra]"', '"scirex[dev]"')
    found = ce.check(repo(tmp_path, "import os\n", bad))
    assert any("'extra' extra" in p for p in found)


def test_missing_all_extra_is_reported(tmp_path) -> None:
    bad = PYPROJECT.replace("all = [", "everything = [")
    assert ce.check(repo(tmp_path, "import os\n", bad)) == [
        "pyproject.toml has no 'all' extra"
    ]


def test_installed_mode_finds_missing_modules(tmp_path) -> None:
    src = "import numpy\nimport ray\n"
    found = ce.check(repo(tmp_path, src), installed=True)
    assert not any("'numpy'" in p for p in found)


def test_strict_exit_status(tmp_path) -> None:
    root = repo(tmp_path, "import pandas\n")
    assert ce.main(["--root", str(root), "--strict"]) == 1
    assert ce.main(["--root", str(root)]) == 0
