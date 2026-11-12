# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Keep the problem classification table in step with the benchmark scripts.

``doc/benchmarks/classification.md`` has one row per benchmark problem. These tests
check that every folder under ``scripts/benchmarks/`` has a row, that each row links
to the problem's problem card, and that no cell is empty.
"""

import re

from scripts.dev.check_paths import RUNTIME
from tests.public_api_helpers import REPO

BENCHMARK_DOCS = REPO / "doc" / "benchmarks"
PAGE = BENCHMARK_DOCS / "classification.md"
BENCHMARK_SCRIPTS = REPO / "scripts" / "benchmarks"
COLUMNS = [
    "Benchmark",
    "Domain",
    "Dimension",
    "Geometry",
    "Grid",
    "Input to output",
    "Supervision",
    "Benchmark/Target",
    "Metric",
    "Models run",
]
SUPERVISION = {"DD", "Physics loss", "H"}
ROW_LINK = re.compile(r"^\[[^\]]+\]\((?P<name>[a-z0-9_]+)/README\.md\)$")


def _table() -> tuple[list[str], list[list[str]]]:
    """Return the header cells and the data rows of the table."""
    lines = [
        ln for ln in PAGE.read_text(encoding="utf-8").splitlines() if ln.startswith("|")
    ]
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return cells[0], cells[2:]


def _problem_folders() -> set[str]:
    """The benchmark problems: folders of ``scripts/benchmarks/`` that are not private
    (``_x``, ``.x``) and not a run-time output folder such as ``vtk_outputs``, which
    ``infer.py`` writes there (gitignored) and which is not a problem."""
    return {
        p.name
        for p in BENCHMARK_SCRIPTS.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and p.name not in RUNTIME
    }


def test_columns():
    header, _ = _table()
    assert header == COLUMNS


def test_no_empty_cell():
    _, rows = _table()
    for cells in rows:
        assert len(cells) == len(COLUMNS), cells[0]
        assert all(cells), f"empty cell in row {cells[0]}"


def test_supervision_values():
    _, rows = _table()
    for cells in rows:
        assert cells[COLUMNS.index("Supervision")] in SUPERVISION, cells[0]


def test_rows_match_benchmark_folders():
    _, rows = _table()
    names = []
    for cells in rows:
        match = ROW_LINK.match(cells[0])
        assert match, f"first cell must link to <problem>/README.md: {cells[0]}"
        name = match["name"]
        assert (BENCHMARK_DOCS / name / "README.md").is_file(), name
        names.append(name)
    assert len(names) == len(set(names)), "duplicate problem"
    assert set(names) == _problem_folders()
