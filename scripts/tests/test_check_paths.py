# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_paths.py``: passes the repo, bites on stale references."""

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_paths", ROOT / "scripts" / "dev" / "check_paths.py"
)
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def repo(tmp_path: Path, **files: str) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "real.py").write_text("")
    for name, text in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def run(tmp_path: Path, **files: str) -> list[str]:
    return cp.check(repo(tmp_path, **files))


def test_the_repository_passes() -> None:
    assert cp.check(ROOT) == []


def test_good_references_pass(tmp_path) -> None:
    assert run(tmp_path, **{"a.md": "run `scripts/real.py` or see [x](a.md)\n"}) == []


def test_missing_path_is_reported(tmp_path) -> None:
    found = run(tmp_path, **{"a.md": "run scripts/old/gone.py now\n"})
    assert any("a.md:1: path does not exist: scripts/old/gone.py" in p for p in found)


def test_runtime_outputs_and_placeholders_are_allowed(tmp_path) -> None:
    text = (
        "see scripts/benchmarks/checkpoints/x and doc/assets/benchmarks/<name>/\n"
        "e.g. src/scirex/experimental/point_mlp.py\n"
        "scripts/new/real.py  paths: ok\n"
    )
    assert run(tmp_path, **{"a.md": text}) == []


def test_missing_import_module_is_reported(tmp_path) -> None:
    text = (
        "```python\nfrom scirex.old_module import thing\nimport scirex.nothing\n```\n"
    )
    found = run(tmp_path, **{"a.md": text})
    assert any("scirex.old_module does not exist" in p for p in found)
    assert any("scirex.nothing does not exist" in p for p in found)


def test_missing_imported_name_is_reported(tmp_path) -> None:
    found = run(tmp_path, **{"a.md": "from scirex.models import NoSuchModel\n"})
    assert any("no name 'NoSuchModel'" in p for p in found)


def test_good_import_passes(tmp_path) -> None:
    assert run(tmp_path, **{"a.md": "from scirex.models import FNO\n"}) == []


def test_notebook_cells_are_checked(tmp_path) -> None:
    nb = {"cells": [{"cell_type": "code", "source": ["from scirex.gone import x\n"]}]}
    found = run(tmp_path, **{"n.ipynb": json.dumps(nb)})
    assert any("n.ipynb" in p and "scirex.gone" in p for p in found)


def test_broken_markdown_link_is_reported(tmp_path) -> None:
    found = run(
        tmp_path, **{"a.md": "[x](missing.md) [y](a.md) [z](https://x.org/q)\n"}
    )
    assert [p for p in found if "link does not resolve" in p] == [
        "a.md:1: link does not resolve: missing.md"
    ]


def test_dockerfile_copy_source_is_checked(tmp_path) -> None:
    found = run(tmp_path, **{"docker__Dockerfile": "COPY gone.txt ./\nCOPY . .\n"})
    assert any("COPY source missing: gone.txt" in p for p in found)


def test_mkdocs_nav_page_is_checked(tmp_path) -> None:
    yml = "docs_dir: doc\nnav:\n  - Home: index.md\n  - Gone: gone.md\n"
    found = run(tmp_path, **{"mkdocs.yml": yml, "doc__index.md": "# x\n"})
    assert any("nav page missing: gone.md" in p for p in found)
    assert not any("index.md" in p for p in found)


def test_workflow_script_is_checked(tmp_path) -> None:
    wf = (
        "steps:\n  - run: python scripts/gone.py --x\n  - run: python scripts/real.py\n"
    )
    found = run(tmp_path, **{".github__workflows__ci.yml": wf})
    assert any("script missing: scripts/gone.py" in p for p in found)
    assert not any("scripts/real.py" in p for p in found)


def test_strict_exit_status(tmp_path) -> None:
    root = repo(tmp_path, **{"a.md": "scripts/gone.py\n"})
    assert cp.main(["--root", str(root), "--strict"]) == 1
    assert cp.main(["--root", str(root)]) == 0
