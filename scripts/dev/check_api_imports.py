#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Check that every ``scirex`` import in the repository goes through the public API.

The API policy (``doc/dev_guide/api_policy.md``) says to import a public name from its
package (``from scirex.losses import LpLoss``), not from the module that defines it
(``from scirex.losses.data_losses import LpLoss``). This script reads every import of
``scirex`` in Python files and in the Python code blocks of Markdown files, and
reports it as one of:

* ``public``        a stable package, and the name is in the package's ``__all__``
* ``experimental``  ``scirex.experimental.*`` (allowed, no stability guarantee)
* ``deep``          a module inside a stable package, such as ``scirex.losses.galerkin``
* ``private``       ``scirex._internal`` or any underscore module
* ``unresolved``    the module or the name does not exist
* ``deprecated``    resolves, but through an old path that emits a ``DeprecationWarning``

It also checks that every name in the stable packages' ``__all__`` can be imported, and
that the live ``__all__`` lists match ``tests/api_snapshot.json``.

Usage::

    python scripts/dev/check_api_imports.py            # report
    python scripts/dev/check_api_imports.py --strict   # exit 1 on problems (see below)
    python scripts/dev/check_api_imports.py --static   # do not import anything

``--strict`` fails on: an unresolved import, a deprecated path, a snapshot mismatch, or a
``deep``/``private`` import in a user-facing file (README, ``doc/``, ``scripts/tutorials``).
Run it with a Python that has the scirex dependencies installed (unless ``--static``).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "tests" / "api_snapshot.json"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "site", "build"}
USER_FACING = ("README.md", "doc/", "docs/", "scripts/tutorials/")
MAINTAINER_DOCS = ("doc/dev_guide/",)  # may show private helpers on purpose
INTERNAL_DIRS = ("src/", "tests/")  # deep imports are expected here
# a trailing comment such as "# works today, not guaranteed" marks a bad example
COUNTER_EXAMPLE = re.compile(r"#.*(not guaranteed|do not|don't|avoid)", re.I)
FENCE = re.compile(r"^```(?:python|py)\s*\n(.*?)^```", re.S | re.M)


class Imp:
    """One import statement member: ``from module import name`` (name may be empty)."""

    def __init__(self, path: str, line: int, module: str, name: str) -> None:
        self.path, self.line, self.module, self.name = path, line, module, name

    def where(self) -> str:
        return f"{self.path}:{self.line}"

    def text(self) -> str:
        return (
            f"from {self.module} import {self.name}"
            if self.name
            else f"import {self.module}"
        )


def python_sources(root: Path) -> list[tuple[str, str, int]]:
    """Return ``(relative path, source text, first line offset)`` for every code unit."""
    units: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or set(path.relative_to(root).parts) & SKIP_DIRS:
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".py":
            units.append((rel, path.read_text(encoding="utf-8", errors="replace"), 0))
        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in FENCE.finditer(text):
                offset = text.count("\n", 0, match.start(1))
                units.append((rel, match.group(1), offset))
    return units


def collect(units: list[tuple[str, str, int]]) -> tuple[list[Imp], int]:
    imports: list[Imp] = []
    unparsable = 0
    for rel, source, offset in units:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            unparsable += 1
            continue
        lines = source.splitlines()
        for node in ast.walk(tree):
            line = node.lineno + offset if hasattr(node, "lineno") else 0
            if hasattr(node, "lineno") and COUNTER_EXAMPLE.search(
                lines[node.lineno - 1]
            ):
                continue  # a documented "do not do this" example
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module == "scirex" or node.module.startswith("scirex."):
                    for alias in node.names:
                        imports.append(Imp(rel, line, node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scirex" or alias.name.startswith("scirex."):
                        imports.append(Imp(rel, line, alias.name, ""))
    return imports, unparsable


def classify(imp: Imp, stable: dict[str, list[str]]) -> str:
    parts = imp.module.split(".")
    if any(p.startswith("_") for p in parts[1:]):
        return "private"
    if imp.module.startswith("scirex.experimental"):
        return "experimental"
    if imp.module in stable or imp.module == "scirex":
        return "public"
    return "deep"


def resolve(imp: Imp) -> tuple[str, str]:
    """Return ``(status, detail)``: ok, deprecated or unresolved."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            module = importlib.import_module(imp.module)
            if imp.name and imp.name != "*" and not hasattr(module, imp.name):
                try:  # a submodule that is not an attribute yet
                    importlib.import_module(f"{imp.module}.{imp.name}")
                except ImportError:
                    return "unresolved", f"{imp.module} has no {imp.name!r}"
        except ModuleNotFoundError as err:
            missing = err.name or ""
            if not missing.startswith("scirex"):
                return "optional", f"needs {missing!r}, which is not installed"
            return "unresolved", f"{type(err).__name__}: {err}"[:120]
        except ImportError as err:
            if "requires" in str(err) and "not installed" in str(err):
                return "optional", str(err)[:100]
            return "unresolved", f"{type(err).__name__}: {err}"[:120]
        except Exception as err:  # - report whatever the import raises
            return "unresolved", f"{type(err).__name__}: {err}"[:120]
    dep = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "ScireX" in str(w.message)
    ]
    if dep:
        return "deprecated", str(dep[0].message)[:140]
    return "ok", ""


def check_surface(stable: dict[str, list[str]], import_ok: bool) -> list[str]:
    """Every name in each stable package's ``__all__`` imports, and ``__all__`` matches."""
    problems: list[str] = []
    for package, names in sorted(stable.items()):
        if not import_ok:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                module = importlib.import_module(package)
            except Exception as err:
                problems.append(
                    f"{package}: cannot import ({type(err).__name__}: {err})"
                )
                continue
            live = sorted(getattr(module, "__all__", []))
            if live != sorted(names):
                added = sorted(set(live) - set(names))
                removed = sorted(set(names) - set(live))
                problems.append(
                    f"{package}: __all__ differs from the snapshot "
                    f"(new: {added}, missing: {removed})"
                )
            for name in live:
                try:
                    getattr(module, name)
                except Exception as err:
                    problems.append(f"{package}.{name}: cannot be imported ({err})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true", help="exit 1 on problems")
    parser.add_argument("--static", action="store_true", help="do not import anything")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    parser.add_argument(
        "--show-all", action="store_true", help="list public imports too"
    )
    parser.add_argument(
        "--all-scopes",
        action="store_true",
        help="also list deep/private imports inside src/ and tests/",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(args.root / "src"))
    snapshot = json.loads((args.root / "tests" / "api_snapshot.json").read_text())
    stable = {k: v for k, v in snapshot.items()}
    public_names = {n for names in stable.values() for n in names}

    units = python_sources(args.root)
    imports, unparsable = collect(units)
    by_kind: dict[str, list[Imp]] = defaultdict(list)
    details: dict[int, tuple[str, str]] = {}
    cache: dict[tuple[str, str], tuple[str, str]] = {}
    for imp in imports:
        by_kind[classify(imp, stable)].append(imp)
        if not args.static:
            key = (imp.module, imp.name)
            if key not in cache:
                cache[key] = resolve(imp)
            details[id(imp)] = cache[key]

    def user_facing(imp: Imp) -> bool:
        return imp.path.startswith(USER_FACING) and not imp.path.startswith(
            MAINTAINER_DOCS
        )

    def shown(imp: Imp) -> bool:
        return args.all_scopes or not imp.path.startswith(INTERNAL_DIRS)

    print(
        f"Scanned {len(units)} code units; {len(imports)} scirex imports "
        f"({unparsable} Markdown blocks were not valid Python and were skipped).\n"
    )
    print("By kind:")
    for kind in ("public", "experimental", "deep", "private"):
        total = len(by_kind[kind])
        ui = sum(user_facing(i) for i in by_kind[kind])
        print(f"  {kind:<13}{total:>5}   (in README/docs/tutorials: {ui})")

    problems = 0
    for kind in ("deep", "private"):
        rows = [i for i in by_kind[kind] if shown(i)]
        hidden = len(by_kind[kind]) - len(rows)
        if not rows:
            continue
        note = (
            f"; {hidden} more in src/ and tests/ hidden, see --all-scopes"
            if hidden
            else ""
        )
        print(
            f"\n== {kind} imports in scripts and docs (user-facing files marked *{note})"
        )
        grouped: dict[str, list[Imp]] = defaultdict(list)
        for imp in rows:
            grouped[imp.module].append(imp)
        for module, items in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            names = sorted({i.name for i in items if i.name})
            hint = ""
            use = [n for n in names if n in public_names]
            if use and kind == "deep":
                hint = f"   -> {', '.join(use)} is public: import it from its package"
            print(f"  {module}  ({len(items)}){hint}")
            for imp in items[:3]:
                star = "*" if user_facing(imp) else " "
                print(f"     {star} {imp.where()}: {imp.text()}")
            if len(items) > 3:
                print(f"       ... {len(items) - 3} more")
            problems += sum(user_facing(i) for i in items)

    if not args.static:
        bad = [(i, details[id(i)]) for i in imports if details[id(i)][0] != "ok"]
        optional = [(i, d) for i, d in bad if d[0] == "optional"]
        bad = [(i, d) for i, d in bad if d[0] != "optional"]
        if optional:
            missing = sorted({d[1] for _, d in optional})
            print(
                f"\n{len(optional)} import(s) not checked because an optional package is missing:"
            )
            for line in missing[:6]:
                print(f"  {line}")
        for status in ("unresolved", "deprecated"):
            rows = [(i, d) for i, d in bad if d[0] == status]
            if rows:
                print(f"\n== {status} imports: {len(rows)}")
                for imp, (_, why) in rows[:25]:
                    print(f"  {imp.where()}: {imp.text()}\n      {why}")
                problems += len(rows)
        if not bad:
            print("\nAll imports resolve, and none goes through a deprecated path.")

    surface = check_surface(stable, not args.static)
    if surface:
        print("\n== public surface problems")
        for line in surface:
            print(f"  {line}")
        problems += len(surface)
    elif not args.static:
        count = sum(len(v) for v in stable.values())
        print(
            f"Public surface: {count} names in {len(stable)} packages import and match the snapshot."
        )

    if args.show_all:
        print("\n== public imports")
        for imp in by_kind["public"]:
            print(f"  {imp.where()}: {imp.text()}")

    print(
        f"\n{'FAIL' if problems else 'OK'}: {problems} problem(s) that --strict counts."
    )
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
