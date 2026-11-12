# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""MkDocs hook: publish the policy files at the repository root as pages of the site.

CODE_OF_CONDUCT.md, GOVERNANCE.md, SECURITY.md, the CLAs, the changelog, the notices,
the licences and CITATION.cff live at the root, where GitHub expects them. This hook
adds each one to the site under ``project/`` when the site is built, so the docs link to
a page of the site instead of to a GitHub file. Links between the root files, and from
them into ``doc/``, are rewritten to the matching page.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files

ROOT = Path(__file__).resolve().parents[2]

# root file -> (page in the site, title, how to show it: "md" or a code language)
PAGES = {
    "CODE_OF_CONDUCT.md": ("project/code_of_conduct.md", "Code of Conduct", "md"),
    "GOVERNANCE.md": ("project/governance.md", "Governance", "md"),
    "SECURITY.md": ("project/security.md", "Security policy", "md"),
    "CLA.md": ("project/cla.md", "Contributor Licence Agreement", "md"),
    "CLA-CORPORATE.md": ("project/cla_corporate.md", "Corporate CLA", "md"),
    "CHANGELOG.md": ("project/changelog.md", "Changelog", "md"),
    "THIRD_PARTY_NOTICES.md": (
        "project/third_party_notices.md",
        "Third-party notices",
        "md",
    ),
    "CITATION.cff": ("project/citation.md", "CITATION.cff", "yaml"),
    "NOTICE": ("project/notice.md", "NOTICE", "text"),
    "LICENSE": ("project/license_lgpl.md", "GNU LGPL v3 (LICENSE)", "text"),
    "COPYING": ("project/license_gpl.md", "GNU GPL v3 (COPYING)", "text"),
}
# root files that already have a page in doc/
EXISTING = {"CONTRIBUTING.md": "dev_guide/contributing.md"}
TARGETS = {name: page for name, (page, _, _) in PAGES.items()} | EXISTING

LINK = re.compile(r"(\]\()([^)\s]+)(\))")


def _rewrite(text: str, page: str) -> str:
    """Point the relative links of a root file at pages of the site."""
    here = posixpath.dirname(page)

    def fix(m: re.Match[str]) -> str:
        target, _, anchor = m.group(2).partition("#")
        if not target or re.match(r"[a-z]+:", target):
            return m.group(0)
        target = posixpath.normpath(target)
        suffix = f"#{anchor}" if anchor else ""
        if target in TARGETS:
            new = posixpath.relpath(TARGETS[target], here)
        elif target.startswith("doc/"):
            new = posixpath.relpath(target.removeprefix("doc/"), here)
        else:  # a file the site does not publish, for example .github/CODEOWNERS
            new = (
                "../" * (len(here.split("/")) + 1) + target
            )  # the repository, linked at build
        return f"{m.group(1)}{new}{suffix}{m.group(3)}"

    return LINK.sub(fix, text)


def _page(name: str, page: str, title: str, kind: str) -> str:
    text = (ROOT / name).read_text(encoding="utf-8")
    if kind == "md":
        return _rewrite(text, page)
    return f"# {title}\n\n```{kind}\n{text.rstrip()}\n```\n"


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add a page for each root policy file that exists."""
    for name, (page, title, kind) in PAGES.items():
        if (ROOT / name).exists() and files.get_file_from_path(page) is None:
            files.append(
                File.generated(config, page, content=_page(name, page, title, kind))
            )
    return files


def on_serve(server: Any, config: MkDocsConfig, builder: Any) -> Any:
    """Rebuild the site when a root policy file changes."""
    for name in PAGES:
        server.watch(str(ROOT / name))
    return server
