# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Dataset register: generator on toy folders, and the committed register."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import list_datasets as ld

OUT = Path(__file__).resolve().parents[2] / "doc" / "benchmarks"


def test_measure_skips_archive_copies(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.npz").write_bytes(b"x" * 10)
    (tmp_path / "d" / "sub").mkdir()
    (tmp_path / "d" / "sub" / "b.npz").write_bytes(b"x" * 5)
    (tmp_path / "d" / "d.tar.gz").write_bytes(b"x" * 100)
    (tmp_path / "d" / "d.zip").write_bytes(b"x" * 100)
    assert ld.measure(tmp_path / "d") == 15
    assert ld.measure(tmp_path / "d" / "a.npz") == 10
    assert ld.measure(tmp_path / "missing") is None


def test_human():
    assert ld.human(3 * (1 << 30)) == "3.0 GiB"
    assert ld.human(5 << 20) == "5.0 MiB"
    assert ld.human(12) == "12 B"
    assert ld.human(None) == ld.UNKNOWN


def test_guide_facts(tmp_path):
    (tmp_path / "g.md").write_text(
        "- **Dataset name:** Toy flow\n"
        "- **Version / release date:** Doe 2020, zenodo 1\n"
    )
    assert ld.guide_facts(tmp_path, "g.md") == ("Toy flow", "Doe 2020, zenodo 1")
    assert ld.guide_facts(tmp_path, "none.md") == (ld.UNKNOWN, ld.UNKNOWN)


def test_build_links_the_guide_and_measures_the_size(tmp_path):
    (tmp_path / "g.md").write_text("- **Dataset name:** Toy\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.npz").write_bytes(b"x" * 2048)
    ds = [ld.Dataset("g.md", "Toy", (tmp_path / "data",), host="Zenodo")]
    (row,) = ld.build(tmp_path, ds)
    assert "licence" not in row  # licences are on Hugging Face, not in the register
    assert row["host"] == "Zenodo"
    assert row["download"] == "doc/dataset_guide/g.md#get-the-data"
    assert row["size"] == "2.0 KiB"


def test_missing_location_reads_unknown(tmp_path):
    ds = [ld.Dataset("g.md", "Toy", (tmp_path / "nope",))]
    assert ld.build(tmp_path, ds)[0]["size"] == ld.UNKNOWN


def test_every_benchmark_dataset_is_listed():
    guides = {d.guide for d in ld.DATASETS}
    assert len(ld.DATASETS) == 8
    assert {
        "darcy_2d.md",
        "pipe.md",
        "navier_stokes.md",
        "cavity.md",
        "chip_cooling.md",
        "airfrans.md",
        "shapenetcar.md",
        "car_cfd.md",
    } == guides


@pytest.mark.skipif(not (OUT / "datasets.csv").is_file(), reason="not generated")
def test_committed_register_is_complete():
    with open(OUT / "datasets.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for r in rows:
        assert all(r[c] for c in ld.COLUMNS if c != "alternate_location"), r["dataset"]
        assert r["download"].endswith("#get-the-data"), r["dataset"]
        assert r["size"] != ld.UNKNOWN, r["dataset"]


def test_regenerating_without_the_data_keeps_the_committed_sizes(tmp_path) -> None:
    committed = tmp_path / ld.CSV_NAME
    committed.write_text("dataset,size\nA,3.2 GiB\nB,UNKNOWN\n")
    rows = [
        {"dataset": "A", "size": ld.UNKNOWN},
        {"dataset": "B", "size": ld.UNKNOWN},
        {"dataset": "C", "size": "1.0 GiB"},
    ]
    ld.keep_known_sizes(rows, committed)
    assert [r["size"] for r in rows] == ["3.2 GiB", ld.UNKNOWN, "1.0 GiB"]
