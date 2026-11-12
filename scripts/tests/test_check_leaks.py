# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/dev/check_leaks.py``: passes the repo, bites on bad input."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_leaks", ROOT / "scripts" / "dev" / "check_leaks.py"
)
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def _repo(tmp_path: Path, **files: str) -> Path:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_repository_passes() -> None:
    assert cl.check(ROOT) == []


@pytest.mark.parametrize(
    ("line", "category"),
    [
        ("data in /home/alice/projects/x", "personal path"),
        ("see /Users/bob/Documents", "personal path"),
        (r"C:\Users\carol\data", "personal path"),
        ("clone https://dev.azure.com/org/proj/_git/repo", "internal host"),
        ("ssh git@build01.corp:repo", "internal host"),
        ("curl http://wiki.internal/page", "internal host"),
        ("server at 10.20.30.40", "IP address"),
        ("host 8.8.4.4 is up", "IP address"),
        ("token ghp_" + "a" * 36, "secret"),
        ("AKIA" + "A" * 16, "secret"),
        ("-----BEGIN RSA PRIVATE KEY-----", "secret"),
        ('api_key = "' + "a1" * 12 + '"', "secret"),
        ("this is bullshit", "profanity"),
    ],
)
def test_bad_lines_are_found(line, category) -> None:
    assert any(category in c for c, _ in cl.scan_line(line)), line


@pytest.mark.parametrize(
    "line",
    [
        "x = self.internal  # an attribute, not a host",
        "bind to 127.0.0.1 or 0.0.0.0",
        "example 192.0.2.10 and 203.0.113.7",
        "jax==0.6.2 and numpy>=1.26.4.1",
        "paths like /home/runner/work and /home/$USER",
        "assess the class and the passage",
        "wandb_api_key = os.environ['WANDB_API_KEY']",
        "ok /home/alice/x  # leaks: ok",
    ],
)
def test_clean_lines_pass(line) -> None:
    assert cl.scan_line(line) == [], line


def test_listed_terms_and_hosts(tmp_path) -> None:
    assert cl.scan_line("Acme Corp order") == []
    assert cl.scan_line("Acme Corp order", terms=("acme corp",))
    assert cl.scan_line("see build.example.org", hosts=("build.example.org",))


def test_check_reports_file_and_line(tmp_path) -> None:
    repo = _repo(tmp_path, **{"a.md": "fine\nuses /home/dave/data\n"})
    assert cl.check(repo) == ["a.md:2: personal path: /home/dave"]
