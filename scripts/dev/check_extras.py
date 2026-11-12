#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that ``pip install -e ".[all]"`` installs everything the repository imports.

The ``all`` extra of ``pyproject.toml`` is meant to be the one command that makes every file
in the repository runnable on the CPU. This script keeps that true:

* it reads every third-party import of every tracked ``.py`` file (source, scripts, tests,
  ``reproduce.py``) and fails if one is not provided by the core dependencies or by an extra
  that ``all`` includes;
* it fails if ``all`` leaves out an extra that is not hardware specific (``cuda12`` and
  ``tpu`` are chosen by the user: ``pip install -e ".[all,cuda12]"``);
* with ``--installed`` it also checks that each of those modules can be found in the
  running interpreter, which proves an environment built with ``.[all]`` is complete.

Usage::

    python scripts/dev/check_extras.py [--root REPO] [--installed] [--strict]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = {
    "scirex",
    "scripts",
    "tests",
    "conftest",
    "reproduce",
    "reproduce_paper_map",
    "reproduce_benchmark",
    "reproduce_benchmark_table",
    "inventory_checkpoints",
    "make_checkpoint_cards",
    "list_datasets",
    "train",  # scripts/train.py, imported by scripts/predict.py and its test
    "predict",
    "make_benchmark_matrices",  # scripts/make_benchmark_matrices.py, used by make_figures.py
}
# import name -> distribution name, where they differ
ALIASES = {
    "docx": "python-docx",
    "sklearn": "scikit-learn",
    "opt_einsum": "opt-einsum",
    "mpl_toolkits": "matplotlib",
    "yaml": "pyyaml",
    "PIL": "pillow",
}
# provided by a declared dependency, not named in pyproject.toml
TRANSITIVE = {"msgpack": "flax", "mkdocs": "mkdocs-material"}  # mkdocs: docs hook
# names used as fixtures that must NOT be installed
FIXTURES = {"another_missing_pkg", "definitely_not_installed_pkg"}
# extras the user chooses, not part of "all"
HARDWARE = {"cuda", "cuda12", "tpu"}
# extras that are subsets of others or the aggregate itself
SUBSETS = {"all", "wandb", "open3d"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req: str) -> str:
    return norm(re.split(r"[\[<>=!~; ]", req.strip(), maxsplit=1)[0])


def parse_pyproject(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Core dependencies and the extras of ``pyproject.toml`` (no TOML parser needed)."""
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    core = re.findall(r'"([^"]+)"', deps.group(1)) if deps else []
    block = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.M | re.S
    )
    extras: dict[str, list[str]] = {}
    if block:
        body = re.sub(r"^[ \t]*#[^\n]*$", "", block.group(1), flags=re.M)
        for name, items in re.findall(
            r"^([A-Za-z0-9_-]+)\s*=\s*\[(.*?)\][ \t]*(?:#[^\n]*)?$", body, re.M | re.S
        ):
            extras[name] = re.findall(r'"([^"]+)"', items)
    return core, extras


def closure(core: list[str], extras: dict[str, list[str]], selected: list[str]):
    """Distribution names provided by the core dependencies plus the selected extras."""
    names = {requirement_name(r) for r in core}
    todo = list(selected)
    seen: set[str] = set()
    while todo:
        extra = todo.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for req in extras.get(extra, []):
            if requirement_name(req) == "scirex":
                m = re.search(r"\[([^\]]+)\]", req)
                todo += [e.strip() for e in m.group(1).split(",")] if m else []
            else:
                names.add(requirement_name(req))
    return names


def imported_modules(root: Path) -> dict[str, set[str]]:
    std = set(sys.stdlib_module_names)
    files = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    used: dict[str, set[str]] = {}
    for rel in files:
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in std and top not in LOCAL and top not in FIXTURES:
                    used.setdefault(top, set()).add(rel)
    return used


def check(root: Path, installed: bool = False) -> list[str]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    core, extras = parse_pyproject(text)
    problems: list[str] = []
    if "all" not in extras:
        return ["pyproject.toml has no 'all' extra"]

    provided = closure(core, extras, ["all"])
    for extra in sorted(set(extras) - HARDWARE - SUBSETS):
        if not closure([], extras, [extra]) <= provided:
            problems.append(f"the 'all' extra does not include the '{extra}' extra")

    for module, files in sorted(imported_modules(root).items()):
        dist = norm(ALIASES.get(module, module))
        if dist in provided:
            continue
        via = TRANSITIVE.get(module)
        if via and norm(via) in provided:
            continue
        where = sorted(files)[0] + (
            f" (+{len(files) - 1} more)" if len(files) > 1 else ""
        )
        problems.append(
            f"'{module}' is imported by {where} but no extra in 'all' installs it"
        )

    if installed:
        for module in sorted(imported_modules(root)):
            try:
                found = importlib.util.find_spec(module)
            except (ImportError, ValueError):
                found = None
            if found is None:
                problems.append(f"'{module}' is not installed in this environment")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--installed", action="store_true", help="also import-check")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    problems = check(args.root.resolve(), args.installed)
    for p in problems:
        print("PROBLEM", p)
    print(f"check_extras: {len(problems)} problem(s)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
