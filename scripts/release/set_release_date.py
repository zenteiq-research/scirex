#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Set ``version`` and ``date-released`` in CITATION.cff for a release.

Run it on the freeze commit, commit the change as ``chore(release): vX.Y.Z`` and tag that
commit. ``scripts/dev/check_release_metadata.py`` (also run by ``publish-pypi.yml``) fails if the
date in CITATION.cff is not the date of the tagged commit.

Usage::

    python scripts/release/set_release_date.py [--date YYYY-MM-DD] [--root REPO]

The date defaults to today (UTC). The version is read from ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_version(root: Path) -> str:
    m = re.search(
        r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M
    )
    if not m:
        raise SystemExit("no version in pyproject.toml")
    return m.group(1)


def update(text: str, version: str, date: str) -> str:
    for key, value in (("version", version), ("date-released", date)):
        pattern = re.compile(rf"^{key}:.*$", re.M)
        if not pattern.search(text):
            raise SystemExit(f"CITATION.cff has no '{key}:' line")
        text = pattern.sub(f'{key}: "{value}"', text, count=1)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, UTC)")
    args = ap.parse_args(argv)
    date = args.date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        print(f"not a date: {date!r} (use YYYY-MM-DD)", file=sys.stderr)
        return 2
    root = args.root.resolve()
    path = root / "CITATION.cff"
    version = project_version(root)
    path.write_text(
        update(path.read_text(encoding="utf-8"), version, date), encoding="utf-8"
    )
    print(f"CITATION.cff: version {version}, date-released {date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
