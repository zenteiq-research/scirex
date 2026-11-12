# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The documentation site loads only pinned scripts from an allowed host."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

# Hosts that may serve scripts. polyfill.io was taken over in 2024 and served malware.
ALLOWED_HOSTS = {"cdn.jsdelivr.net", "cdnjs.cloudflare.com", "unpkg.com"}
BLOCKED_HOSTS = {"polyfill.io", "cdn.polyfill.io", "bootcss.com", "bootcdn.net"}


def external_urls() -> list[str]:
    return re.findall(r"^\s+- (https?://\S+)", CONFIG, re.M)


def test_there_is_a_math_script() -> None:
    assert any("mathjax" in url for url in external_urls())


def test_scripts_come_from_allowed_hosts() -> None:
    for url in external_urls():
        host = url.split("/")[2]
        assert host not in BLOCKED_HOSTS, url
        assert host in ALLOWED_HOSTS, f"{url}: host not in the allowed list"


def test_scripts_are_pinned_to_an_exact_version() -> None:
    for url in external_urls():
        assert re.search(r"@\d+\.\d+\.\d+/", url), f"{url} is not pinned"


def test_no_polyfill_anywhere_in_the_docs_config() -> None:
    assert "polyfill" not in CONFIG.lower()
    css = (ROOT / "doc" / "stylesheets" / "extra.css").read_text(encoding="utf-8")
    assert "http" not in css  # no remote imports in the stylesheet
