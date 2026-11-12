# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""MkDocs hook: let pages link to repository files with relative paths.

A page under ``doc/`` links to a source file, a script or a folder outside ``doc/``
the way it links to any other file, for example ``../../../scripts/reproduce.py``.
The path works in an editor and on GitHub, and a typo is caught here: a relative link
that leaves ``doc/`` must point at a file or folder that exists, or the build stops.
When the site is built, each such link becomes a link to that path on GitHub, under
``repo_url`` of ``mkdocs.yml`` and the branch in ``BRANCH``, so the host and the
branch are set in one place; the placeholders ``{{ repo_url }}`` and ``{{ org_url }}``
in a page give the repository and organisation addresses. Text that a page includes
from another file (``--8<--``) is rewritten as well, because the links are changed
after the page is parsed.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

from markdown import Extension, Markdown
from markdown.treeprocessors import Treeprocessor
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.exceptions import PluginError

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "main"
SCHEME = re.compile(r"[a-z][a-z0-9+.-]*:")
_state = {"page": "", "repo": ""}


class _RepoLinks(Treeprocessor):
    def run(self, root: Element) -> None:
        here = posixpath.dirname(_state["page"])
        for a in root.iter("a"):
            href = a.get("href") or ""
            target, hash_, anchor = href.partition("#")
            if not target or SCHEME.match(target) or target.startswith("/"):
                continue
            path = posixpath.normpath(posixpath.join(here, target))
            if not path.startswith("../"):  # inside doc/: MkDocs checks it
                continue
            repo_path = posixpath.normpath(posixpath.join("doc", path))
            if repo_path.startswith("../") or not (ROOT / repo_path).exists():
                raise PluginError(
                    f"{_state['page']}: the link {href!r} points to {repo_path}, "
                    "which is not in the repository"
                )
            kind = "tree" if (ROOT / repo_path).is_dir() else "blob"
            a.set(
                "href", f"{_state['repo']}/{kind}/{BRANCH}/{repo_path}{hash_}{anchor}"
            )


class _Extension(Extension):
    def extendMarkdown(self, md: Markdown) -> None:
        # before MkDocs checks relative links (priority 0)
        md.treeprocessors.register(_RepoLinks(md), "zq_repo_links", 20)


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    """Add the extension and remember the repository URL."""
    _state["repo"] = config["repo_url"].rstrip("/")
    config["markdown_extensions"].append(_Extension())
    return config


def on_page_markdown(markdown: str, page: Any, config: MkDocsConfig, files: Any) -> str:
    """Remember which page is being converted; fill in the repository address.

    ``{{ repo_url }}`` is ``repo_url`` of ``mkdocs.yml`` (a page uses it for the issue
    tracker, discussions, ``git clone`` and the like, which are not repository files)
    and ``{{ org_url }}`` is its parent, the organisation.
    """
    _state["page"] = page.file.src_uri
    repo = _state["repo"]
    return markdown.replace("{{ repo_url }}", repo).replace(
        "{{ org_url }}", repo.rsplit("/", 1)[0]
    )
