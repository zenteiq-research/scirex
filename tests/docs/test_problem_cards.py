# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Enforce the problem-card template and the cards written from it.

Every retained benchmark has a card at ``doc/benchmarks/<name>/README.md``. The
template lives in ``doc/dev_guide/problem_card_template.md``. These tests keep
the template from drifting, check that each card follows it, and check that each
card folder matches a script folder and a navigation entry.
"""

import re
from pathlib import Path

import pytest

from tests.public_api_helpers import REPO

# In template order. A heading in OPTIONAL_HEADINGS may be left out of a card.
REQUIRED_HEADINGS = [
    "## At a glance",
    "## Method",
    "## Training and evaluation",
    "## Notes on the data",
    "## Reproduce",
    "## Notes on the results",
    "## References",
]
OPTIONAL_HEADINGS = {"## Method", "## Notes on the data", "## Notes on the results"}

TEMPLATE_PATH = REPO / "doc" / "dev_guide" / "problem_card_template.md"
BENCHMARK_DOCS = REPO / "doc" / "benchmarks"
BENCHMARK_SCRIPTS = REPO / "scripts" / "benchmarks"
MKDOCS = REPO / "mkdocs.yml"

PLACEHOLDER_TITLE = "# <Problem name>"
UNFILLED_MARKER = "TODO(owner)"
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

CARDS = sorted(BENCHMARK_DOCS.glob("*/README.md"))
CARD_PARAMS = (
    [pytest.param(card, id=card.parent.name) for card in CARDS]
    if CARDS
    else [pytest.param(None, marks=pytest.mark.skip(reason="no problem cards yet"))]
)
FOLDER_PARAMS = (
    [pytest.param(card.parent, id=card.parent.name) for card in CARDS]
    if CARDS
    else [pytest.param(None, marks=pytest.mark.skip(reason="no problem cards yet"))]
)


def _headings(lines: list[str]) -> list[str]:
    """Return the H2 headings of a markdown document, ignoring code fences.

    Parameters
    ----------
    lines : list of str
        Lines of the markdown document.

    Returns
    -------
    list of str
        Lines starting with ``## `` that are outside fenced code blocks, in
        document order, with trailing whitespace removed.
    """
    found = []
    fence = None
    for line in lines:
        marker = re.match(r"^(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker.group(1)
            elif line.startswith(fence):
                fence = None
            continue
        if fence is None and line.startswith("## "):
            found.append(line.rstrip())
    return found


def _template_block() -> list[str]:
    """Return the lines inside the four-backtick fenced block of the template.

    Returns
    -------
    list of str
        Lines between the opening ```` ````markdown ```` fence and the closing
        ```` ```` ```` fence.
    """
    lines = TEMPLATE_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("````markdown"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "````")
    return lines[start + 1 : end]


def _status_line(text: str) -> str | None:
    """Return the first ``Status:`` line of a card, or None when absent.

    Parameters
    ----------
    text : str
        Full text of the card.

    Returns
    -------
    str or None
        The matching line, stripped, or None.
    """
    for line in text.splitlines():
        if line.startswith("Status:"):
            return line.strip()
    return None


def test_template_headings_match_required_list():
    """The template's H2 headings equal the fixed list, in order."""
    assert _headings(_template_block()) == REQUIRED_HEADINGS


def test_template_has_title_and_status_placeholders():
    """The template starts with the placeholder title and a status line."""
    block = _template_block()
    assert block[0] == PLACEHOLDER_TITLE
    assert _status_line("\n".join(block)) is not None


@pytest.mark.parametrize("card", CARD_PARAMS)
def test_card_follows_template(card: Path):
    """Each card has the required headings in order and a real title.

    A ``Status:`` line is optional; when it says a reviewer checked the card, no
    ``TODO(owner)`` marker may remain.
    """
    text = card.read_text(encoding="utf-8")
    lines = text.splitlines()
    first = next((line for line in lines if line.startswith("# ")), None)
    assert first is not None, f"{card}: no H1 title"
    assert first.strip() != PLACEHOLDER_TITLE, f"{card}: title is still the placeholder"
    found = _headings(lines)
    expected = [
        h for h in REQUIRED_HEADINGS if h in found or h not in OPTIONAL_HEADINGS
    ]
    assert found == expected, f"{card}: headings differ"
    status = _status_line(text)
    if (
        status is not None
        and "checked" in status.lower()
        and "draft" not in status.lower()
    ):
        assert UNFILLED_MARKER not in text, f"{card}: checked but has {UNFILLED_MARKER}"


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_card_folder_matches_scripts(folder: Path):
    """Each card folder is snake_case and has a matching scripts folder."""
    name = folder.name
    assert SNAKE_CASE.match(name), f"{name} is not snake_case"
    assert (BENCHMARK_SCRIPTS / name).is_dir(), f"missing scripts/benchmarks/{name}/"


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_card_listed_in_mkdocs(folder: Path):
    """Each card is reachable from the mkdocs navigation."""
    entry = f"benchmarks/{folder.name}/README.md"
    nav = MKDOCS.read_text(encoding="utf-8")
    assert entry in nav, f"{entry} missing from mkdocs.yml"
