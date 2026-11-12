#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that CITATION.cff matches the release: same version, a date from the release.

* ``version`` in CITATION.cff equals ``version`` in ``pyproject.toml``;
* ``date-released`` is a valid ISO date;
* with ``--commit REV`` (default ``HEAD``), the date is the date of that commit, give or
  take ``--days`` days (default 1), so a date left over from an earlier draft is caught.

Set both with ``scripts/release/set_release_date.py``. ``publish-pypi.yml`` runs this on the
tagged commit. Usage::

    python scripts/dev/check_release_metadata.py [--root REPO] [--commit REV] [--days N] [--strict]

Exit status 1 with ``--strict`` when a problem is found.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def field(text: str, key: str) -> str:
    m = re.search(rf'^{key}:\s*"?([^"\n]+?)"?\s*$', text, re.M)
    return m.group(1) if m else ""


def check(root: Path, commit: str = "HEAD", days: int = 1) -> list[str]:
    problems: list[str] = []
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    version = m.group(1) if m else ""
    if field(cff, "version") != version:
        problems.append(
            f"CITATION.cff version {field(cff, 'version')!r} differs from "
            f"pyproject.toml {version!r}"
        )
    try:
        released = datetime.date.fromisoformat(field(cff, "date-released"))
    except ValueError:
        return [*problems, "CITATION.cff date-released is missing or not YYYY-MM-DD"]
    done = subprocess.run(
        ["git", "log", "-1", "--format=%cs", commit],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0 or not done.stdout.strip():
        return [*problems, f"cannot read the date of commit {commit!r}"]
    committed = datetime.date.fromisoformat(done.stdout.strip())
    if abs((released - committed).days) > days:
        problems.append(
            f"CITATION.cff date-released {released} is not the date of {commit} "
            f"({committed}); run scripts/release/set_release_date.py and commit"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    problems = check(args.root.resolve(), args.commit, args.days)
    for p in problems:
        print("PROBLEM", p)
    print(f"check_release_metadata: {len(problems)} problem(s)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
