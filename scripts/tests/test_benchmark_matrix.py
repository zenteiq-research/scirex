# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every number of the model-comparison matrix is on the page it links to."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "doc" / "benchmarks" / "comparison.md"
ROW = re.compile(
    r"^\| \[(?P<name>[^\]]+)\]\((?P<link>[^)]+)\) \| (?P<metric>[^|]+) \|(?P<cells>.+)\|$"
)


def _rows() -> list[tuple[str, str, list[str]]]:
    rows = []
    for line in PAGE.read_text().splitlines():
        match = ROW.match(line)
        if match:
            cells = [c.strip().strip("*") for c in match["cells"].split("|")]
            rows.append((match["name"], match["link"], cells))
    return rows


def test_matrix_has_the_nine_problems() -> None:
    assert len(_rows()) == 9


@pytest.mark.parametrize(
    "name,link,cells",
    _rows(),
    ids=lambda v: v if isinstance(v, str) and len(v) < 30 else "",
)
def test_numbers_are_on_the_comparison_page(
    name: str, link: str, cells: list[str]
) -> None:
    page = (PAGE.parent / link).read_text()
    # every number on the page; a headline in % may be written there as a
    # fraction (7.05 % as 0.0705, 0.197 % of the 1 m/s lid speed as 1.97e-3)
    stated = [float(s) for s in re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", page)]
    assert len(cells) == 4  # FNO, WNO, GINO, Transolver (GIWO is on its own page)
    for value in cells:
        if value == "–":
            continue
        x = float(value.replace("%", "").strip())
        found = any(abs(s - c) <= 0.006 * c for s in stated for c in (x, x / 100) if c)
        assert found, f"{name}: {value} is not on {link}"


def test_every_figure_exists() -> None:
    text = PAGE.read_text()
    paths = re.findall(r"\]\((\.\./assets/[^)]+\.png)\)", text)
    assert len(paths) >= 20
    for path in paths:
        assert (PAGE.parent / path).resolve().exists(), path
