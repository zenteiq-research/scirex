#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check the project's contact details.

* ``pyproject.toml`` lists the project mailbox for every author and maintainer;
* ``SECURITY.md`` points to GitHub private vulnerability reporting first and gives the
  mailbox as the fallback;
* no tracked text file contains an e-mail address outside the project domains
  (``zenteiq.ai``, ``zenteiq.com``) or a ``noreply`` address, so no personal address
  is published in code, docs or notebooks;
* with ``--name NAME`` (repeatable, case-insensitive), no tracked text file mentions
  that name. The names are passed on the command line and never stored in the repo, so
  this script does not itself publish them.

Usage::

    python scripts/dev/check_contacts.py [--root REPO] [--mailbox ADDRESS]
        [--name NAME ...] [--strict]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MAILBOX = "research@zenteiq.ai"
PROJECT_DOMAINS = ("zenteiq.ai", "zenteiq.com")
ADVISORY_URL = re.compile(
    r"https://github\.com/[\w.-]+/[\w.-]+/security/advisories/new"
)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
TEXT_SUFFIXES = {
    ".py", ".md", ".ipynb", ".toml", ".cff", ".txt", ".yml", ".yaml", ".cfg",
    ".tex", ".rst", ".json", ".sh", ".ini", "",
}  # fmt: skip
# this script and its tests contain example addresses on purpose
# The checkers and their tests contain deliberately bad strings as fixtures.
SELF = {
    "scripts/dev/check_contacts.py",
    "scripts/tests/test_check_contacts.py",
    "scripts/tests/test_check_leaks.py",
    "scripts/tests/test_check_notebooks.py",
    "scripts/tests/test_check_package.py",
    "scripts/tests/test_check_no_committers.py",
}


def _tracked(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [root / f for f in out if f and f not in SELF]


def _table_block(text: str, key: str) -> str:
    m = re.search(rf"^{key}\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    return m.group(1) if m else ""


def check(root: Path, mailbox: str = MAILBOX, names: tuple[str, ...] = ()) -> list[str]:
    problems: list[str] = []
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for key in ("authors", "maintainers"):
        block = _table_block(pyproject, key)
        entries = re.findall(r"\{[^}]*\}", block)
        if not entries:
            problems.append(f"pyproject.toml: no {key} listed")
        for entry in entries:
            if mailbox not in entry:
                problems.append(
                    f"pyproject.toml: {key} entry without {mailbox}: {entry}"
                )

    sec = (root / "SECURITY.md").read_text(encoding="utf-8")
    adv = ADVISORY_URL.search(sec)
    if not adv:
        problems.append("SECURITY.md: no GitHub private vulnerability reporting link")
    if mailbox not in sec:
        problems.append(f"SECURITY.md: fallback mailbox {mailbox} missing")
    elif adv and sec.index(mailbox) < adv.start():
        problems.append("SECURITY.md: the mailbox must come after the GitHub link")
    if "public GitHub issues" not in sec:
        problems.append("SECURITY.md: lost the 'do not report publicly' warning")

    for path in _tracked(root):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for name in names:
                if name.lower() in line.lower():
                    problems.append(
                        f"{path.relative_to(root)}:{line_no}: mentions '{name}'"
                    )
            for addr in EMAIL.findall(line):
                domain = addr.rsplit("@", 1)[1].lower()
                if "noreply" in addr.lower() or domain.endswith(PROJECT_DOMAINS):
                    continue
                problems.append(
                    f"{path.relative_to(root)}:{line_no}: e-mail address {addr}"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--mailbox", default=MAILBOX)
    ap.add_argument(
        "--name", action="append", default=[], help="a name that must not appear"
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on problems")
    args = ap.parse_args()
    problems = check(args.root, args.mailbox, tuple(args.name))
    for p in problems:
        print(p)
    print(f"\n{len(problems)} problem(s).")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
