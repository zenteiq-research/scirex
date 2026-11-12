# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_contributing.py``: passes the repo, bites on bad input."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_contributing", ROOT / "scripts" / "dev" / "check_contributing.py"
)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

FILES = [
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CLA.md",
    "CLA-CORPORATE.md",
    "SECURITY.md",
    "Makefile",
    "pytest.ini",
    "doc/dev_guide/contributing.md",
    "doc/tutorials/NewExample/Overview.md",
    "doc/tutorials/NewExample/PR_Checklist.md",
    "doc/dev_guide/jax_patterns.md",
    "doc/dev_guide/naming.md",
    "doc/dev_guide/api_policy.md",
    "GOVERNANCE.md",
    "THIRD_PARTY_NOTICES.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    # files and folders that CONTRIBUTING.md links to
    ".github/CODEOWNERS",
    "scripts/reproduce.py",
    "scripts/tests/__init__.py",
    "src/scirex/__init__.py",
    "tests/conftest.py",
    "tests/tolerances.py",
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small copy of the repository with only the files the checker reads."""
    for name in FILES:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, dest)
    (tmp_path / "a.py").write_text("# " + cc.HEADER + "\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def edit(repo: Path, name: str, old: str, new: str) -> None:
    path = repo / name
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_repository_passes() -> None:
    assert cc.check(ROOT) == []


def test_copy_passes(repo) -> None:
    assert cc.check(repo) == []


def test_missing_spdx_header_is_reported(repo) -> None:
    (repo / "b.py").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert any("b.py: missing SPDX" in p for p in cc.check(repo))


def test_missing_section_is_reported(repo) -> None:
    edit(repo, "CONTRIBUTING.md", "## Licensing of Contributions", "## Other")
    assert any("licence: section" in p for p in cc.check(repo))


def test_undescribed_marker_is_reported(repo) -> None:
    edit(repo, "pytest.ini", "gpu: requires", "gpu: requires")
    text = (repo / "pytest.ini").read_text()
    (repo / "pytest.ini").write_text(
        text.replace("markers =", "markers =\n    slow: slow")
    )
    assert any("'slow'" in p for p in cc.check(repo))


def test_unknown_make_target_is_reported(repo) -> None:
    edit(repo, "CONTRIBUTING.md", "make test-unit", "make test-nothing")
    assert any("make test-nothing" in p for p in cc.check(repo))


def test_broken_link_is_reported(repo) -> None:
    edit(repo, "CONTRIBUTING.md", "](CODE_OF_CONDUCT.md)", "](NOPE.md)")
    assert any("broken link NOPE.md" in p for p in cc.check(repo))


def test_stale_docs_copy_is_reported(repo) -> None:
    (repo / "doc/dev_guide/contributing.md").write_text("old\n")
    assert any("out of date" in p for p in cc.check(repo))


def test_ci_job_must_run_the_documented_commands(repo) -> None:
    edit(repo, ".github/workflows/ci.yml", "--commands setup", "--commands other")
    assert any("setup commands" in p for p in cc.check(repo))


def test_missing_ci_job_is_reported(repo) -> None:
    edit(repo, ".github/workflows/ci.yml", "  contributing:\n", "  other:\n")
    assert any("no `contributing` job" in p for p in cc.check(repo))


def test_commands_come_from_the_marked_blocks() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    setup = cc.commands(text, "setup")
    assert 'pip install -e ".[dev]"' in setup
    assert "pre-commit install" in setup
    assert any(c.startswith("pytest ") for c in cc.commands(text, "check"))


def test_docs_copy_rewrites_links_that_leave_doc() -> None:
    page = cc.docs_copy(
        "[a](SECURITY.md) [b](doc/x/y.md) [c](./doc/dev_guide/z.md) [d](Makefile)"
    )
    assert "](../project/security.md)" in page  # a root file the site publishes
    assert "](../../Makefile)" in page  # the docs build links it to the repository
    assert "](../x/y.md)" in page
    assert "](z.md)" in page


def test_strict_exit_status(repo) -> None:
    (repo / "CONTRIBUTING.md").write_text("empty\n")
    assert cc.main(["--root", str(repo), "--strict"]) == 1
    assert cc.main(["--root", str(repo)]) == 0


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("x.yml", "name: x\n"),
        ("Makefile", "all:\n"),
        ("page.md", "# Page\n"),
    ],
)
def test_other_file_types_need_a_header(repo, name, text) -> None:
    (repo / name).write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert any(f"{name}: missing SPDX" in p for p in cc.check(repo))


def test_markdown_accepts_cc_by_and_mixed_notice(repo) -> None:
    head = "<!--\nSPDX-License-Identifier: CC-BY-4.0\n-->\n"
    mixed = "<!--\nLicense notice: text CC BY 4.0, code LGPL-3.0-only\n-->\n"
    (repo / "a.md").write_text(head + "# A\n")
    (repo / "b.md").write_text(mixed + "# B\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    assert cc.check(repo) == []
