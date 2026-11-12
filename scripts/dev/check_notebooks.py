#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that tracked Jupyter notebooks carry no outputs and leak nothing.

For every tracked ``*.ipynb`` file:

* no cell has outputs or an ``execution_count`` (what ``nbstripout`` removes);
* no widget state, and no ``metadata.path`` / ``metadata.papermill`` style local data;
* anything that is left (outputs you chose to keep with ``--allow-outputs``, metadata,
  sources) holds no personal path, internal host, IP address, secret or profanity, using
  the same rules as ``scripts/dev/check_leaks.py``.

``--fix`` strips outputs in place with ``nbstripout`` when it is installed, otherwise
with a small built-in version of the same operation.

Usage::

    python scripts/dev/check_notebooks.py [--root REPO] [--fix]
        [--allow-outputs FILE ...] [--strict]

The repository has no notebooks today, so the check passes trivially; it exists so a
notebook added later cannot ship outputs. The ``nbstripout`` pre-commit hook enforces
the same rule at commit time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_leaks():
    spec = importlib.util.spec_from_file_location(
        "check_leaks", HERE / "check_leaks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_leaks", mod)
    spec.loader.exec_module(mod)
    return mod


leaks = _load_leaks()
BAD_METADATA_KEYS = (
    "widgets",
    "papermill",
    "path",
    "application/vnd.jupyter.widget-state+json",
)


def tracked_notebooks(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return [root / f for f in out if f]


def strip_notebook(nb: dict) -> dict:
    """Built-in equivalent of ``nbstripout`` (outputs, counts, widget state)."""
    for key in ("widgets",):
        nb.get("metadata", {}).pop(key, None)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    return nb


def _strings(obj):
    if isinstance(obj, str):
        yield from obj.splitlines()
    elif isinstance(obj, list):
        for item in obj:
            yield from _strings(item)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)


def check_notebook(path: Path, allow_outputs: bool = False) -> list[str]:
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path}: cannot read notebook ({exc})"]
    meta = nb.get("metadata", {})
    for key in BAD_METADATA_KEYS:
        if key in meta:
            problems.append(f"{path}: notebook metadata has '{key}'")
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if not allow_outputs:
            if cell.get("outputs"):
                problems.append(f"{path}: cell {i} has outputs")
            if cell.get("execution_count") is not None:
                problems.append(f"{path}: cell {i} has an execution_count")
        for line in _strings(cell.get("outputs", [])):
            for category, hit in leaks.scan_line(line):
                problems.append(f"{path}: cell {i} output {category}: {hit}")
    for line in _strings(nb):
        for category, hit in leaks.scan_line(line):
            problems.append(f"{path}: {category}: {hit}")
    return sorted(set(problems))


def fix(path: Path) -> None:
    if shutil.which("nbstripout"):
        subprocess.run(["nbstripout", str(path)], check=True)
        return
    nb = strip_notebook(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def check(root: Path, allow: tuple[str, ...] = ()) -> list[str]:
    problems: list[str] = []
    for nb in tracked_notebooks(root):
        rel = str(nb.relative_to(root))
        problems += check_notebook(nb, allow_outputs=rel in allow)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=HERE.parents[1])
    ap.add_argument("--fix", action="store_true", help="strip outputs in place")
    ap.add_argument("--allow-outputs", action="append", default=[], metavar="FILE")
    ap.add_argument("--strict", action="store_true", help="exit 1 on problems")
    args = ap.parse_args()
    notebooks = tracked_notebooks(args.root)
    print(f"{len(notebooks)} tracked notebook(s).")
    if args.fix:
        for nb in notebooks:
            if str(nb.relative_to(args.root)) not in args.allow_outputs:
                fix(nb)
    problems = check(args.root, tuple(args.allow_outputs))
    for p in problems:
        print(p)
    print(f"\n{len(problems)} problem(s).")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
