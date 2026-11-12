# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""MkDocs hook: tag each page with the top-level tab it belongs to.

The ``<body>`` of every page gets ``data-zq-section="install"`` (or ``learn``,
``benchmarks`` and so on), so ``stylesheets/vibrant.css`` can give each section its
own accent gradient.
"""

from __future__ import annotations

import re
from typing import Any

from mkdocs.config.defaults import MkDocsConfig


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def on_post_page(output: str, page: Any, config: MkDocsConfig) -> str:
    """Add the section of the page as a data attribute on <body>."""
    ancestors = getattr(page, "ancestors", None) or []
    top = ancestors[-1].title if ancestors else page.title
    section = _slug(top or "home") or "home"
    return output.replace("<body ", f'<body data-zq-section="{section}" ', 1)
