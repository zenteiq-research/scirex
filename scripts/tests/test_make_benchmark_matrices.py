# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark matrices are written from metrics.json and are up to date."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "make_benchmark_matrices", ROOT / "scripts" / "make_benchmark_matrices.py"
)
mm = importlib.util.module_from_spec(spec)
sys.modules["make_benchmark_matrices"] = mm  # dataclasses look the module up here
spec.loader.exec_module(mm)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    doc, assets = tmp_path / "doc", tmp_path / "assets"
    (doc / "darcy" / "results").mkdir(parents=True)
    (doc / "darcy" / "results" / "fno.md").write_text("page")
    (doc / "comparison.md").write_text("no table here")
    (assets / "darcy_fno").mkdir(parents=True)
    (assets / "darcy_fno" / "metrics.json").write_text(
        json.dumps(
            {
                "final_test_rel_l2_mean": 0.01109,
                "n_params_real": 2836281,
                "epochs": 500,
                "batch_size": 20,
                "total_time_s": 250.0,
            }
        )
    )
    return doc, assets


def test_cells_come_from_metrics_json_and_link_to_the_page(tmp_path):
    doc, assets = _tree(tmp_path)
    pages = mm.build_pages(assets, doc)
    assert "[1.109%](darcy/results/fno.md)" in pages["accuracy.md"]
    assert "[2,836,281](darcy/results/fno.md)" in pages["compute_cost.md"]
    assert (
        "[0.50](darcy/results/fno.md)" in pages["compute_cost.md"]
    )  # 250 s over 500 epochs
    assert "[500 / 20](darcy/results/fno.md)" in pages["setup.md"]


def test_a_missing_value_is_a_dash_and_has_no_link(tmp_path):
    doc, assets = _tree(tmp_path)
    row = next(
        l
        for l in mm.build_pages(assets, doc)["accuracy.md"].splitlines()
        if "Pipe Flow" in l
    )
    assert "(" not in row.split("|", 2)[2]  # no link in the cells
    assert row.count(mm.DASH) == len(mm.MODELS)


def test_check_detects_a_stale_page(tmp_path):
    doc, assets = _tree(tmp_path)
    args = ["--doc", str(doc), "--assets", str(assets)]
    assert mm.main(args + ["--check"]) == 1  # nothing written yet
    assert mm.main(args) == 0
    assert mm.main(args + ["--check"]) == 0
    (doc / "setup.md").write_text("edited by hand")
    assert mm.main(args + ["--check"]) == 1


def test_committed_pages_are_up_to_date():
    assert mm.main(["--check"]) == 0
