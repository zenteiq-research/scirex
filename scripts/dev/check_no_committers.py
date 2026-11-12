#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that no committer's name, handle or e-mail address is in the release.

The public identity of the project is ``ZenteiQ AiTech Innovations Pvt. Ltd.`` with the
mailbox ``research@zenteiq.ai`` (licensing standard, section 4.6). This script reads
every author and committer identity from ``git log --all`` and fails if any of them
appears:

* in a tracked text file (code, docs, notebooks, metadata, JSON results);
* inside a built sdist or wheel (``--dist DIR``);
* in the identities of a public snapshot repository (``--history PATH``): only the
  allowed identity may appear.

Names are matched case-insensitively as whole words; the e-mail address and the part
before ``@`` are matched too. The names are read from git at run time and never written
anywhere, so this script does not publish them.

Usage::

    python scripts/dev/check_no_committers.py [--root REPO] [--dist DIR]
        [--history PATH] [--allow-name TEXT ...] [--strict]

``--allow-name`` lists a word that is also an ordinary term or a cited author, so it is
not a finding. Exit status 1 with ``--strict`` when a problem is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_EMAILS = {"research@zenteiq.ai"}
# The project mailbox may also have authored commits under another domain; it is not a
# person to search the files for. A public snapshot may still carry only ALLOWED_EMAILS.
PROJECT_MAILBOX_NAME = "research"
ALLOWED_NAMES = {"zenteiq open source", "zenteiq aitech innovations pvt. ltd."}
SELF = {
    "scripts/dev/check_no_committers.py",
    "scripts/tests/test_check_no_committers.py",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cff", ".yml", ".yaml", ".json", ".cfg", ".ini",
    ".sh", ".tex", ".ipynb", ".properties", ".bib", "",
}  # fmt: skip
MAX_BYTES = 3_000_000


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def identities(root: Path) -> set[tuple[str, str]]:
    """``(name, email)`` of every author and committer in any ref."""
    out = git(root, "log", "--all", "--format=%an\t%ae\n%cn\t%ce")
    found = set()
    for line in out.splitlines():
        name, _, email = line.partition("\t")
        if name or email:
            found.add((name.strip(), email.strip().lower()))
    return found


def needles(ids: set[tuple[str, str]], allow: set[str]) -> dict[str, str]:
    """Lower-case search term -> what it is (for the message)."""
    terms: dict[str, str] = {}
    for name, email in ids:
        if (
            email in ALLOWED_EMAILS
            or email.split("@")[0] == PROJECT_MAILBOX_NAME
            or name.lower() in ALLOWED_NAMES
        ):
            continue
        if "noreply" in email:  # a service account such as the GitHub web editor
            continue
        if email:
            terms[email] = "e-mail address"
            local = email.split("@")[0]
            if len(local) >= 4 and local not in allow:
                terms[local] = "e-mail name"
        if len(name) >= 4 and name.lower() not in allow:
            terms[name.lower()] = "committer name"
    return {t: kind for t, kind in terms.items() if t not in allow}


def pattern(terms: dict[str, str]) -> re.Pattern | None:
    if not terms:
        return None
    parts = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"(?<![\w.-])(" + "|".join(re.escape(t) for t in parts) + r")(?![\w-])",
        re.I,
    )


def scan_text(label: str, text: str, pat: re.Pattern, terms: dict[str, str]):
    problems = []
    for no, line in enumerate(text.splitlines(), 1):
        for m in pat.finditer(line):
            kind = terms.get(m.group(1).lower(), "committer")
            problems.append(f"{label}:{no}: {kind}: {m.group(1)}")
    return problems


def check_tree(root: Path, pat: re.Pattern, terms: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for rel in git(root, "ls-files").splitlines():
        path = root / rel
        if (
            rel in SELF
            or path.suffix.lower() not in TEXT_SUFFIXES
            or not path.is_file()
        ):
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        problems += scan_text(rel, text, pat, terms)
    return problems


def check_dist(dist: Path, pat: re.Pattern, terms: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for wheel in sorted(dist.glob("*.whl")):
        with zipfile.ZipFile(wheel) as z:
            for info in z.infolist():
                data = z.read(info) if info.file_size < MAX_BYTES else b""
                problems += scan_text(
                    f"{wheel.name}:{info.filename}",
                    data.decode("utf-8", "ignore"),
                    pat,
                    terms,
                )
    for sdist in sorted(dist.glob("*.tar.gz")):
        with tarfile.open(sdist) as t:
            for info in t.getmembers():
                if info.isfile() and info.size < MAX_BYTES:
                    data = t.extractfile(info).read()
                    problems += scan_text(
                        f"{sdist.name}:{info.name}",
                        data.decode("utf-8", "ignore"),
                        pat,
                        terms,
                    )
    return problems


def check_history(path: Path) -> list[str]:
    """Only the project identity may appear in a public snapshot repository."""
    problems = []
    for name, email in sorted(identities(path)):
        if email not in ALLOWED_EMAILS or name.lower() not in ALLOWED_NAMES:
            problems.append(f"{path}: history has the identity {name!r} <{email}>")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--dist", type=Path, help="folder with the built sdist and wheel")
    ap.add_argument("--history", type=Path, help="public snapshot repository to check")
    ap.add_argument("--allow-name", action="append", default=[], metavar="TEXT")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    allow = {a.lower() for a in args.allow_name}

    ids = identities(root)
    terms = needles(ids, allow)
    pat = pattern(terms)
    print(f"{len(ids)} identities in the history, {len(terms)} search terms")
    problems: list[str] = []
    if pat:
        problems += check_tree(root, pat, terms)
        if args.dist:
            problems += check_dist(args.dist, pat, terms)
    if args.history:
        problems += check_history(args.history.resolve())
    for p in problems:
        print("PROBLEM", p)
    print(f"check_no_committers: {len(problems)} problem(s)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
