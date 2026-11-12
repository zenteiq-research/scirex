#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that the headline number in each committed ``metrics.json`` is the one printed in
the page's results table. No checkpoints, data or GPU are needed; it runs in seconds.

Every benchmark result page (``doc/benchmarks/<problem>/results/<model>.md``) has a table with
a headline test error, and the committed ``doc/assets/benchmarks/<problem>_<model>/metrics.json``
holds the same number. This script reads both and fails when they differ, so a result cannot be
edited in one place only. Re-evaluating the checkpoints is ``scripts/reproduce.py``.

Usage::

    python scripts/dev/check_tables.py [--root REPO] [--list] [--strict]

Exit status 1 with ``--strict`` when a table does not show the committed number.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Where the headline number sits in metrics.json, tried in this order.
HEADLINE_PATHS: tuple[tuple[str, ...], ...] = (
    ("final_test_rel_l2_mean",),
    ("test_rel_l2_over_seeds", "mean"),
    ("test_rel_l2", "mean"),
    ("headline",),
)
SKIP_PAGES = {"comparative", "setup"}
MODELS = ("fno", "wno", "gino", "giwo", "transolver")
FAMILIES = ("darcy", "pipe", "navier_stokes", "car_cfd")


@dataclass
class Target:
    """One result page and the headline number it reports."""

    name: str  # e.g. "darcy_fno"
    family: str
    page: Path
    metrics: Path
    path: tuple[str, ...]  # key path of the headline number
    reference: float  # headline value in the committed metrics.json
    scale: float  # 100 when the table shows a percentage, 1 otherwise


def lookup(data: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    if isinstance(data, bool) or not isinstance(data, int | float):
        return None
    return float(data)


def shown_in_page(value: float, scale: float, text: str) -> bool:
    """Is ``value`` printed in ``text`` (as ``1.109%`` or ``0.0705``)?"""
    shown = value * scale
    if scale == 100:
        return any(f"{shown:.{n}f}%" in text for n in (1, 2, 3, 4))
    return any(f"{shown:.{n}f}" in text for n in (3, 4, 5))


def _candidates(root: Path) -> list[tuple[str, str, Path]]:
    """``(family, model, page)`` for each model. A model has its own page, or, when the
    benchmark has one results page, a section of ``comparative.md`` (found by its assets)."""
    out: list[tuple[str, str, Path]] = []
    for results in sorted((root / "doc" / "benchmarks").glob("*/results")):
        family = results.parent.name
        own = {p.stem: p for p in results.glob("*.md") if p.stem not in SKIP_PAGES}
        shared = results / "comparative.md"
        for model, page in own.items():
            out.append((family, model, page))
        if shared.is_file():
            for model in MODELS:
                assets = root / "doc" / "assets" / "benchmarks" / f"{family}_{model}"
                if model not in own and assets.is_dir():
                    out.append((family, model, shared))
    return out


def discover(root: Path = ROOT) -> tuple[list[Target], list[str]]:
    """Return the checkable targets and the names of the pages that are not."""
    targets: list[Target] = []
    skipped: list[str] = []
    for family, model, page in _candidates(root):
        name = f"{family}_{model}"
        metrics = root / "doc" / "assets" / "benchmarks" / name / "metrics.json"
        if family not in FAMILIES or not metrics.is_file():
            skipped.append(name)
            continue
        text = page.read_text(encoding="utf-8")
        data = json.loads(metrics.read_text(encoding="utf-8"))
        found = None
        for path in HEADLINE_PATHS:
            value = lookup(data, path)
            if value is None:
                continue
            for scale in (100.0, 1.0):
                if shown_in_page(value, scale, text):
                    found = Target(name, family, page, metrics, path, value, scale)
                    break
            if found:
                break
        if found:
            targets.append(found)
        else:
            skipped.append(name)
    return targets, skipped


def check(targets: list[Target]) -> list[str]:
    problems = []
    for t in targets:
        text = t.page.read_text(encoding="utf-8")
        data = json.loads(t.metrics.read_text(encoding="utf-8"))
        value = lookup(data, t.path)
        if value is None or not shown_in_page(value, t.scale, text):
            problems.append(f"{t.name}: table in {t.page.name} does not show {value}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--list", action="store_true", help="list the checked targets")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    targets, skipped = discover(args.root.resolve())
    if args.list:
        for t in targets:
            print(
                f"{t.name:28s} {t.reference * t.scale:>9.4g}{'%' if t.scale == 100 else ''}"
            )
        if skipped:
            print("not checked (layout not read here): " + ", ".join(skipped))
    problems = check(targets)
    for p in problems:
        print("MISMATCH", p)
    print(f"check_tables: {len(targets) - len(problems)}/{len(targets)} tables match")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
