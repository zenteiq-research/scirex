#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that CONTRIBUTING.md is complete, true and in sync with the repository.

Checks
------
* every tracked source file (``.py``, YAML, TOML, Makefile, Dockerfile, ...) starts with
  the LGPL SPDX header and every Markdown page with the CC-BY-4.0 or mixed notice
  (the licence-notice rules of CONTRIBUTING.md);
* ``CONTRIBUTING.md`` has the required parts: development setup, test commands with the
  markers of ``pytest.ini``, style rules, SPDX rule, pull-request process (issue, tests,
  docs, CHANGELOG, review), the contribution licence section, and links to the
  new-example guide, issue templates, ``SECURITY.md`` and ``CODE_OF_CONDUCT.md``;
* every relative link in it points to a file that exists;
* every ``make`` target it names exists in the ``Makefile``;
* the documentation copy ``doc/dev_guide/contributing.md`` is the generated copy of
  ``CONTRIBUTING.md`` (``--write-docs`` regenerates it);
* ``.github/workflows/ci.yml`` has a ``contributing`` job that runs the documented
  setup and check commands taken from ``CONTRIBUTING.md`` (``--commands``).

Usage::

    python scripts/dev/check_contributing.py [--root REPO] [--strict]
    python scripts/dev/check_contributing.py --commands setup|check   # print the commands
    python scripts/dev/check_contributing.py --write-docs             # refresh the docs copy

Exit status 1 with ``--strict`` when a problem is found.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER = "SPDX-License-Identifier: LGPL-3.0-only"
DOC_HEADERS = ("SPDX-License-Identifier: CC-BY-4.0", "License notice:")
# Licensing standard v1.0, section 3: "#"-comment files carry the LGPL SPDX lines,
# Markdown carries CC-BY-4.0 or the mixed notice. Formats without comments, legal texts
# and generated or ignore files are exempt (section 3.7).
HASH_SUFFIXES = (".py", ".yml", ".yaml", ".toml", ".ini", ".properties", ".sh")
HASH_NAMES = ("Makefile", "Dockerfile")
REPO_URL = "https://github.com/zenteiq-research/scirex"
# Root files that doc/_hooks/project_files.py publishes as pages of the site.
SITE_PAGES = {
    "CODE_OF_CONDUCT.md": "project/code_of_conduct.md",
    "GOVERNANCE.md": "project/governance.md",
    "SECURITY.md": "project/security.md",
    "CLA.md": "project/cla.md",
    "CLA-CORPORATE.md": "project/cla_corporate.md",
    "CHANGELOG.md": "project/changelog.md",
    "THIRD_PARTY_NOTICES.md": "project/third_party_notices.md",
    "CITATION.cff": "project/citation.md",
    "NOTICE": "project/notice.md",
    "LICENSE": "project/license_lgpl.md",
    "COPYING": "project/license_gpl.md",
}
DOCS_PAGE = Path("doc/dev_guide/contributing.md")
DOCS_HEAD = """<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.

GENERATED from CONTRIBUTING.md by scripts/dev/check_contributing.py --write-docs.
Edit CONTRIBUTING.md, not this file.
-->

"""

# Phrases CONTRIBUTING.md must contain: (what, regex).
REQUIRED = [
    ("setup: clone", r"git clone "),
    ("setup: venv", r"python3 -m venv "),
    ("setup: editable install", r'pip install -e "\.\[dev\]"'),
    ("setup: pre-commit install", r"pre-commit install"),
    ("tests: package suite", r'pytest tests/ -m "not gpu"'),
    ("tests: script suite", r'pytest scripts/tests/ -m "not gpu"'),
    ("tests: separate processes", r"separate pytest processes"),
    ("style: ruff", r"ruff format"),
    ("style: pre-commit run", r"pre-commit run --all-files"),
    ("style: SPDX header", r"SPDX-License-Identifier: LGPL-3\.0-only"),
    ("PR: issue first", r"(?i)issue first"),
    ("PR: tests", r"\*\*Tests\*\*"),
    ("PR: docs", r"\*\*Docs\*\*"),
    ("PR: changelog", r"\*\*CHANGELOG\*\*"),
    ("PR: review", r"(?i)maintainer approves"),
    ("licence: section", r"## Licensing of Contributions"),
    ("licence: code", r"LGPL-3\.0-only"),
    ("licence: docs", r"CC BY 4\.0"),
    ("licence: CLA", r"Contributor Licence Agreement"),
]
# Files CONTRIBUTING.md must link to.
REQUIRED_LINKS = [
    "doc/tutorials/NewExample/Overview.md",
    "doc/tutorials/NewExample/PR_Checklist.md",
    ".github/ISSUE_TEMPLATE",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CLA.md",
]
LINK = re.compile(r"\]\(([^)\s]+)\)")
BLOCK = re.compile(r"<!-- contributing:(\w+) -->\s*```(?:bash)?\n(.*?)```", re.S)


def tracked(root: Path, pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    return [root / line for line in out if line]


def commands(text: str, name: str) -> list[str]:
    """The commands of the ``<!-- contributing:NAME -->`` block, one per line."""
    for block, body in BLOCK.findall(text):
        if block == name:
            return [ln.strip() for ln in body.splitlines() if ln.strip()]
    return []


def make_targets(makefile: str) -> set[str]:
    return set(re.findall(r"^([A-Za-z0-9_-]+):", makefile, re.M))


def markers(pytest_ini: str) -> list[str]:
    block = re.search(r"markers\s*=\s*\n((?:[ \t]+.+\n)+)", pytest_ini)
    return re.findall(r"^[ \t]+(\w+):", block.group(1), re.M) if block else []


def docs_copy(text: str) -> str:
    """CONTRIBUTING.md as a documentation page: links that leave ``doc/`` are relative paths."""

    def fix(match: re.Match) -> str:
        target = match.group(1)
        if target.startswith(("http", "#", "mailto:")):
            return match.group(0)
        path = target.removeprefix("./")
        if path.startswith("doc/dev_guide/"):
            path = path.removeprefix("doc/dev_guide/")
        elif path.startswith("doc/"):
            path = "../" + path.removeprefix("doc/")
        elif path.split("#")[0] in SITE_PAGES:
            name, _, anchor = path.partition("#")
            path = "../" + SITE_PAGES[name] + ("#" + anchor if anchor else "")
        else:
            path = "../../" + path  # the docs build links it to the repository
        return f"]({path})"

    text = re.sub(r"\A<!--.*?-->\s*", "", text, count=1, flags=re.S)
    text = LINK.sub(fix, text).replace(REPO_URL, "{{ repo_url }}")
    return DOCS_HEAD + text


def check(root: Path) -> list[str]:
    problems: list[str] = []

    for path in tracked(root, "*"):
        if not path.is_file():
            continue
        if path.suffix in HASH_SUFFIXES or path.name in HASH_NAMES:
            wanted = (HEADER,)
        elif path.suffix == ".md":
            wanted = DOC_HEADERS
        else:
            continue
        try:
            head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12])
        except UnicodeDecodeError:
            continue
        if not any(w in head for w in wanted):
            problems.append(f"{path.relative_to(root)}: missing SPDX header")

    contributing = root / "CONTRIBUTING.md"
    if not contributing.is_file():
        return [*problems, "CONTRIBUTING.md is missing"]
    text = contributing.read_text(encoding="utf-8")

    for what, pattern in REQUIRED:
        if not re.search(pattern, text):
            problems.append(f"CONTRIBUTING.md: missing {what}")

    for name in ("setup", "check"):
        if not commands(text, name):
            problems.append(f"CONTRIBUTING.md: no <!-- contributing:{name} --> block")

    ini = root / "pytest.ini"
    if ini.is_file():
        for marker in markers(ini.read_text(encoding="utf-8")):
            if f"`{marker}`" not in text:
                problems.append(
                    f"CONTRIBUTING.md: pytest marker {marker!r} not described"
                )

    for target in REQUIRED_LINKS:
        if f"]({target}" not in text and f"]({'./' + target}" not in text:
            problems.append(f"CONTRIBUTING.md: no link to {target}")

    for target in LINK.findall(text):
        if target.startswith(("http", "#", "mailto:")):
            continue
        if not (root / target.split("#")[0]).exists():
            problems.append(f"CONTRIBUTING.md: broken link {target}")

    makefile = root / "Makefile"
    targets = (
        make_targets(makefile.read_text(encoding="utf-8"))
        if makefile.is_file()
        else set()
    )
    for name in sorted(set(re.findall(r"`?make ([a-z][a-z-]*)", text))):
        if name not in targets:
            problems.append(f"CONTRIBUTING.md: `make {name}` is not a Makefile target")

    page = root / DOCS_PAGE
    if not page.is_file() or page.read_text(encoding="utf-8") != docs_copy(text):
        problems.append(f"{DOCS_PAGE} is out of date; run --write-docs")

    ci = root / ".github" / "workflows" / "ci.yml"
    ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""
    if not re.search(r"^  contributing:\s*$", ci_text, re.M):
        problems.append("ci.yml: no `contributing` job")
    else:
        for name in ("setup", "check"):
            if f"check_contributing.py --commands {name}" not in ci_text:
                problems.append(
                    f"ci.yml: contributing job does not run the {name} commands"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    ap.add_argument("--commands", choices=["setup", "check"], help="print a block")
    ap.add_argument("--write-docs", action="store_true", help="refresh the docs copy")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    text = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")

    if args.commands:
        cmds = commands(text, args.commands)
        if not cmds:
            print(f"no {args.commands} block in CONTRIBUTING.md", file=sys.stderr)
            return 2
        print("\n".join(cmds))
        return 0
    if args.write_docs:
        (root / DOCS_PAGE).write_text(docs_copy(text), encoding="utf-8")
        print(f"wrote {DOCS_PAGE}")
        return 0

    problems = check(root)
    for p in problems:
        print("PROBLEM", p)
    print(f"check_contributing: {len(problems)} problem(s)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
