#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check the naming rules in ``doc/dev_guide/naming.md``.

Rules checked (all on the stable and experimental package, ``src/scirex``):

* module and package names are ``snake_case``;
* function and method names are ``snake_case``;
* class names are ``CapWords`` (acronyms allowed: ``FNOBlock``, ``GINODarcy2d``);
* config keys (attributes of the classes in ``scirex/config``) are ``snake_case``;
* benchmark folders under ``scripts/benchmarks`` are ``snake_case``.

Model names follow the paper (``FNO``, not ``Fno``), so a class may start with an
acronym. The names in ``ALLOWED`` are existing exceptions, each with its reason.

Usage::

    python scripts/dev/check_naming.py [--root REPO] [--strict]

Exit status is 1 with ``--strict`` when a violation is found.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

SNAKE = re.compile(r"^_{0,2}[a-z][a-z0-9]*(_[a-z0-9]+)*_{0,2}$")
# CapWords with acronyms and digits: starts upper case, no underscores in the middle.
CAPWORDS = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
DUNDER = re.compile(r"^__[a-z0-9_]+__$")

# name -> reason. Keep this list short; fix the name when you can.
ALLOWED: dict[str, str] = {
    "__init__": "dunder",
}


def _is_snake(name: str) -> bool:
    return bool(SNAKE.match(name) or DUNDER.match(name) or name in ALLOWED)


def _is_capwords(name: str) -> bool:
    return bool(CAPWORDS.match(name) or name in ALLOWED)


def check(root: Path) -> list[str]:
    problems: list[str] = []
    src = root / "src" / "scirex"
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(root)
        for part in path.relative_to(src).with_suffix("").parts:
            if not _is_snake(part):
                problems.append(
                    f"{rel}: module/package name '{part}' is not snake_case"
                )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:  # pragma: no cover
            problems.append(f"{rel}: cannot parse ({e})")
            continue
        in_config = "config" in path.relative_to(src).parts[:1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not _is_capwords(node.name):
                    problems.append(
                        f"{rel}:{node.lineno}: class '{node.name}' is not CapWords"
                    )
                if in_config:
                    for stmt in node.body:
                        keys = []
                        if isinstance(stmt, ast.AnnAssign) and isinstance(
                            stmt.target, ast.Name
                        ):
                            keys = [stmt.target.id]
                        elif isinstance(stmt, ast.Assign):
                            keys = [
                                t.id for t in stmt.targets if isinstance(t, ast.Name)
                            ]
                        for key in keys:
                            if not _is_snake(key):
                                problems.append(
                                    f"{rel}:{stmt.lineno}: config key '{key}' of "
                                    f"{node.name} is not snake_case"
                                )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if not _is_snake(node.name):
                    problems.append(
                        f"{rel}:{node.lineno}: function '{node.name}' is not snake_case"
                    )
    bench = root / "scripts" / "benchmarks"
    if bench.is_dir():
        for d in sorted(p for p in bench.iterdir() if p.is_dir()):
            if d.name != "__pycache__" and not _is_snake(d.name):
                problems.append(
                    f"{d.relative_to(root)}: benchmark folder not snake_case"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--strict", action="store_true", help="exit 1 on violations")
    args = ap.parse_args()
    problems = check(args.root)
    for p in problems:
        print(p)
    print(f"\n{len(problems)} naming violation(s).")
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
