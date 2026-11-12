# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_contacts.py``: passes the repo, bites on bad input."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_contacts", ROOT / "scripts" / "dev" / "check_contacts.py"
)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

GOOD_PYPROJECT = """[project]
authors = [
    { name = "A", email = "research@zenteiq.ai" },
]
maintainers = [
    { name = "B", email = "research@zenteiq.ai" },
]
"""
GOOD_SECURITY = """Do not use public GitHub issues.
https://github.com/o/r/security/advisories/new
Fallback: research@zenteiq.ai
"""


def _repo(tmp_path: Path, pyproject=GOOD_PYPROJECT, security=GOOD_SECURITY, **files):
    (tmp_path / "pyproject.toml").write_text(pyproject)
    (tmp_path / "SECURITY.md").write_text(security)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_repository_passes() -> None:
    assert cc.check(ROOT) == []


def test_good_tree_passes(tmp_path) -> None:
    assert cc.check(_repo(tmp_path)) == []


def test_author_without_mailbox_is_reported(tmp_path) -> None:
    bad = GOOD_PYPROJECT.replace(
        ', email = "research@zenteiq.ai" },\n]\nmaint', " },\n]\nmaint", 1
    )
    assert any("authors entry" in p for p in cc.check(_repo(tmp_path, pyproject=bad)))


def test_missing_github_link_is_reported(tmp_path) -> None:
    sec = "Do not use public GitHub issues.\nMail research@zenteiq.ai\n"
    assert any("GitHub private" in p for p in cc.check(_repo(tmp_path, security=sec)))


def test_mailbox_before_github_link_is_reported(tmp_path) -> None:
    sec = (
        "Do not use public GitHub issues.\nresearch@zenteiq.ai\n"
        "https://github.com/o/r/security/advisories/new\n"
    )
    assert any("come after" in p for p in cc.check(_repo(tmp_path, security=sec)))


@pytest.mark.parametrize("addr", ["someone@gmail.com", "x.y@university.edu"])
def test_personal_address_is_reported(tmp_path, addr) -> None:
    repo = _repo(tmp_path, **{"a.md": f"mail {addr}\n"})
    assert any(addr in p for p in cc.check(repo))


def test_project_and_noreply_addresses_are_allowed(tmp_path) -> None:
    repo = _repo(
        tmp_path, **{"a.md": "info@zenteiq.com 123+x@users.noreply.github.com\n"}
    )
    assert cc.check(repo) == []


def test_listed_names_are_reported_case_insensitively(tmp_path) -> None:
    repo = _repo(tmp_path, **{"a.md": "written by Jane Doe\n"})
    assert cc.check(repo) == []
    found = cc.check(repo, names=("jane doe",))
    assert any("a.md:1" in p and "jane doe" in p for p in found)
