# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""MkDocs hook: give pages that share a navigation title their own browser-tab title.

The navigation shows short titles ("Overview", "Results", "FNO") under each problem,
and Material uses the navigation title as the title of the browser tab. A reader with
several tabs open, and a search engine, then see the same title on many pages. This
hook sets the title of such a page to its parent section followed by its own title,
for example "Darcy Flow: Results" or "Darcy Flow: FNO tutorial". A page that sets
``title`` in its front matter keeps it.
"""

from __future__ import annotations

from typing import Any

from mkdocs.config.defaults import MkDocsConfig

SHARED = {"Overview", "Results", "Setup", "FNO", "WNO", "GINO", "Transolver", "GIWO"}
MODELS = {"FNO", "WNO", "GINO", "Transolver", "GIWO"}


def on_page_markdown(markdown: str, page: Any, config: MkDocsConfig, files: Any) -> str:
    """Set ``page.meta["title"]`` for a page whose navigation title is shared."""
    if page.meta.get("title") or page.title not in SHARED or page.parent is None:
        return markdown
    ancestors = []
    node = page.parent
    while node is not None:
        ancestors.append(node.title)
        node = node.parent
    suffix = " tutorial" if page.title in MODELS and "Learn" in ancestors else ""
    page.meta["title"] = f"{page.parent.title}: {page.title}{suffix}"
    return markdown
