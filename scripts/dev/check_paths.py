#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that imports, configs, scripts, notebooks, Dockerfiles and docs point at real paths.

After files move, old references survive in places nobody greps. This script reads every
tracked text file and checks:

* **imports**: each ``import scirex...`` / ``from scirex... import name`` (in ``.py`` files,
  Markdown code blocks, notebooks, YAML and Dockerfiles) names a module that exists, and the
  imported names exist in it (modules are imported, so run it in an environment with scirex
  installed);
* **repository paths**: each ``scripts/...``, ``src/scirex/...``, ``doc/...``, ``docker/...``,
  ``tests/...`` or ``.github/...`` path written in a file exists (outputs that are created
  at run time, such as checkpoints or results folders, are on an allow list);
* **Dockerfile** ``COPY``/``ADD`` sources, **compose** build contexts and volumes, **workflow**
  ``run:`` lines naming a script, **Makefile**, **pyproject.toml** and **mkdocs.yml**
  (nav pages, ``docs_dir``) point at files that exist;
* **relative Markdown links** resolve to a file.

Usage::

    python scripts/dev/check_paths.py [--root REPO] [--strict]

Exit status 1 with ``--strict`` when a problem is found. Add ``paths: ok`` to a line to
exempt it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_MARK = "paths: ok"
TEXT_SUFFIXES = {".py", ".md", ".ipynb", ".yml", ".yaml", ".toml", ".ini", ".sh"}
TEXT_NAMES = {"Makefile", "Dockerfile"}
SELF = {"scripts/dev/check_paths.py", "scripts/tests/test_check_paths.py"}
TEMPLATES = {"doc/dev_guide/problem_card_template.md"}  # placeholders, not paths
# Folders written at run time or kept outside git.
RUNTIME = (
    "checkpoints", "results", "logs", "outputs", "output", "data", "runs", "figures",
    "mlruns", "wandb", "vtk_outputs", "release-report", "dist", "build", "site", "htmlcov",
)  # fmt: skip
PATH_RE = re.compile(
    r"(?<![\w/.-])((?:scripts|src/scirex|doc|docker|tests|\.github)/[\w./-]+)"
)
IMPORT_RE = re.compile(
    r"^[ \t>]*(?:from\s+(scirex[\w.]*)\s+import\s+([\w, \t*()]+)|import\s+(scirex[\w.]*))",
    re.M,
)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
GENERATED = ("doc/api/",)
# The test suites exercise their own imports and use made-up paths as fixtures; imports
# inside the package are exercised by every import of it.
NO_IMPORT_CHECK = ("src/scirex/", "tests/", "scripts/tests/")
NO_PATH_CHECK = ("tests/", "scripts/tests/")
# Pages that doc/_hooks/project_files.py adds to the site from the root policy files.
SITE_HOOK = "doc/_hooks/project_files.py"


def site_pages(root: Path) -> set[str]:
    """The ``doc/...`` pages the MkDocs hook generates, which are not files on disk."""
    hook = root / SITE_HOOK
    if not hook.is_file():
        return set()
    pages = re.findall(r'"(project/[\w-]+\.md)"', hook.read_text(encoding="utf-8"))
    return {f"doc/{page}" for page in pages}


def tracked(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [
        root / f
        for f in out
        if f
        and f not in SELF | TEMPLATES
        and ((root / f).suffix in TEXT_SUFFIXES or (root / f).name in TEXT_NAMES)
    ]


def is_runtime(path: str) -> bool:
    parts = Path(path).parts
    return any(p in RUNTIME for p in parts) or path.endswith(("/", ".", ".env"))


def exists(root: Path, rel: str) -> bool:
    rel = rel.rstrip("/.,;:)'\"`")
    if not rel or any(c in rel for c in "*<>{}$|"):
        return True
    if is_runtime(rel):
        return True
    # a module path written without the .py suffix, or a folder
    candidate = root / rel
    return candidate.exists() or candidate.with_suffix(".py").exists()


def generated_page(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    return rel in site_pages(root)


def resolve_import(module: str, names: list[str]) -> list[str]:
    """Problems for ``from module import names`` (names may be empty for ``import``)."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "scirex":
            return [f"module {module} does not exist ({exc})"]
        return []  # a third-party dependency is not installed here
    except Exception as exc:
        return [f"module {module} fails to import: {type(exc).__name__}: {exc}"]
    problems = []
    for name in names:
        if name == "*":
            continue
        try:
            if hasattr(mod, name):
                continue
        except ImportError:
            continue  # a lazy name whose optional dependency is not installed
        try:
            importlib.import_module(f"{module}.{name}")
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] == "scirex":
                problems.append(f"{module} has no name {name!r}")
        except Exception:
            pass
    return problems


def text_of(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    cells = json.loads(raw).get("cells", [])
    return "\n".join("".join(c.get("source", [])) for c in cells)


def check_file(root: Path, path: Path) -> list[str]:
    rel = path.relative_to(root)
    try:
        text = text_of(path)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    problems: list[str] = []
    lines = text.splitlines()

    def skipped(offset: int) -> bool:
        line_no = text.count("\n", 0, offset)
        if line_no >= len(lines):
            return False
        line = lines[line_no]
        # a placeholder or an example name, not a real path
        return SKIP_MARK in line or "e.g." in line or "<" in line

    posix = rel.as_posix()
    for m in IMPORT_RE.finditer("" if posix.startswith(NO_IMPORT_CHECK) else text):
        if skipped(m.start()):
            continue
        module = m.group(1) or m.group(3)
        names = []
        if m.group(2):
            names = [n.strip() for n in re.split(r"[,\s()]+", m.group(2)) if n.strip()]
            names = [n for n in names if n != "as" and n.isidentifier() or n == "*"]
        for problem in resolve_import(module, names):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(f"{rel}:{line}: import: {problem}")

    for m in PATH_RE.finditer("" if posix.startswith(NO_PATH_CHECK) else text):
        if skipped(m.start()):
            continue
        target = m.group(1)
        if target.endswith(("_", "-")):  # a prefix of a name built at run time
            continue
        if rel.as_posix().startswith(GENERATED) or exists(root, target):
            continue
        line = text.count("\n", 0, m.start()) + 1
        problems.append(f"{rel}:{line}: path does not exist: {target.rstrip('.,;:)')}")

    if path.suffix == ".md":
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http", "mailto:", "/")) or skipped(m.start()):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists() and not generated_page(root, resolved):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{rel}:{line}: link does not resolve: {target}")
    return problems


def check_structured(root: Path) -> list[str]:
    problems: list[str] = []
    dockerfiles = [p for p in tracked(root) if p.name == "Dockerfile"]
    for df in dockerfiles:
        context = root
        for no, line in enumerate(df.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"\s*(?:COPY|ADD)\s+(?:--\S+\s+)*(.+?)\s+\S+\s*$", line)
            if not m or "--from" in line:
                continue
            for src in m.group(1).split():
                if src in (".", "./") or any(c in src for c in "*$"):
                    continue
                if not (context / src).exists() and not is_runtime(src):
                    rel = df.relative_to(root)
                    problems.append(f"{rel}:{no}: COPY source missing: {src}")

    mkdocs = root / "mkdocs.yml"
    if mkdocs.is_file():
        text = mkdocs.read_text(encoding="utf-8")
        docs_dir = re.search(r"^docs_dir:\s*(\S+)", text, re.M)
        base = root / (docs_dir.group(1) if docs_dir else "docs")
        if not base.is_dir():
            problems.append(f"mkdocs.yml: docs_dir {base.name} does not exist")
        for page in re.findall(r"(?::|^\s*-)\s+([\w./-]+\.md)\s*$", text, re.M):
            generated = f"doc/{page}" in site_pages(root)
            if not (base / page).is_file() and not generated:
                problems.append(f"mkdocs.yml: nav page missing: {page}")

    for wf in sorted((root / ".github" / "workflows").glob("*.yml")):
        for no, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(
                r"python3?\s+(scripts/[\w./-]+\.py|[\w-]+\.py)\b", line
            ):
                if SKIP_MARK not in line and not (root / m.group(1)).is_file():
                    rel = wf.relative_to(root)
                    problems.append(f"{rel}:{no}: script missing: {m.group(1)}")
    return problems


def check(root: Path) -> list[str]:
    root = root.resolve()
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    problems: list[str] = []
    for path in tracked(root):
        problems += check_file(root, path)
    problems += check_structured(root)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    problems = check(args.root)
    for p in problems:
        print(p)
    print(f"check_paths: {len(problems)} problem(s)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
