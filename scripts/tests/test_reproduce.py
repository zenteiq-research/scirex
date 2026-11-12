# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ``scripts/reproduce.py`` and its cell map (no checkpoints or GPU needed)."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("reproduce_t3", SCRIPTS / "reproduce.py")
rp = importlib.util.module_from_spec(spec)
sys.modules["reproduce_t3"] = rp
spec.loader.exec_module(rp)
import reproduce_paper_map as tmap  # noqa: E402


def test_dig_follows_dotted_keys() -> None:
    assert rp.dig({"a": {"b": 3}}, "a.b") == 3
    with pytest.raises(KeyError):
        rp.dig({"a": {}}, "a.b")


def test_cell_map_is_consistent() -> None:
    assert tmap.CELLS
    for cell in tmap.CELLS:
        assert cell.model in tmap.MODELS, cell
        assert cell.ckpt, f"{cell.row}/{cell.model} has no checkpoint path"
        assert not hasattr(
            cell, "paper"
        )  # the cells carry no printed value to compare with


def test_every_runner_group_has_an_infer_script() -> None:
    groups = {c.group for c in tmap.CELLS}
    for group in groups - {"airfrans_gino"}:
        folder = group if (SCRIPTS / "benchmarks" / group).is_dir() else None
        if folder:
            assert (SCRIPTS / "benchmarks" / folder / "infer.py").is_file(), group


def test_rows_are_unique_and_ordered() -> None:
    rows = tmap.rows()
    assert len(rows) == len(set(rows))
    assert rows[0] == tmap.CELLS[0].row


def test_list_prints_every_cell_and_exits_zero(capsys) -> None:
    assert rp.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == len(tmap.CELLS)


def test_only_selects_matching_rows(capsys) -> None:
    assert rp.main(["--list", "--only", "Pipe"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines
    assert all("Pipe" in ln for ln in lines)


def test_train_mode_is_refused() -> None:
    with pytest.raises(SystemExit) as exc:
        rp.main(["--mode", "train", "--only", "Pipe"])
    assert "not implemented" in str(exc.value)


def test_cells_without_checkpoints_are_reported_missing_not_crashed(
    tmp_path, capsys
) -> None:
    code = rp.main(
        [
            "--only", "Pipe",
            "--ckpt-root", str(tmp_path / "nothing"),
            "--data-root", str(tmp_path / "data"),
            "--out-dir", str(tmp_path / "out"),
            "--device", "cpu",
        ]
    )  # fmt: skip
    assert code == 0  # MISSING is not a failure unless --strict-missing
    results = tmp_path / "out" / "results.csv"
    assert results.is_file()
    with results.open(newline="") as f:
        statuses = {row["status"] for row in csv.DictReader(f)}
    assert statuses == {rp.MISSING}


def test_strict_missing_makes_missing_cells_fail(tmp_path) -> None:
    code = rp.main(
        [
            "--only", "Pipe",
            "--ckpt-root", str(tmp_path / "nothing"),
            "--data-root", str(tmp_path / "data"),
            "--out-dir", str(tmp_path / "out"),
            "--device", "cpu",
            "--strict-missing",
        ]
    )  # fmt: skip
    assert code == 1


def test_outputs_are_written(tmp_path) -> None:
    rp.main(
        [
            "--only", "Pipe",
            "--ckpt-root", str(tmp_path / "x"),
            "--out-dir", str(tmp_path / "out"),
            "--device", "cpu",
        ]
    )  # fmt: skip
    for name in ("results.csv", "results_table.csv", "results_table.tex"):
        assert (tmp_path / "out" / name).is_file(), name


def test_hf_paths_are_unique_per_checkpoint_and_follow_the_layout() -> None:
    from reproduce_paper_map import CELLS, PROBLEMS, hf_path

    paths = {hf_path(c) for c in CELLS if c.ckpt}
    assert len(paths) == 37
    for p in paths:
        model, problem, *_ = p.split("/")
        assert model in {"fno", "wno", "gino", "transolver", "giwo"}
        assert problem in PROBLEMS


def test_model_and_problem_filters_combine_with_and() -> None:
    cells = rp.select(None, None, model="fno", problem="pipe")
    assert [(c.row, c.model) for c in cells] == [("Pipe", "FNO")]
    assert rp.select(None, None, problem="darcy_s85")
    assert rp.select(None, None, model="giwo", problem="pipe") == []


def test_missing_roots_are_reported_not_crashed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SCIREX_CKPT_ROOT", raising=False)
    monkeypatch.delenv("SCIREX_DATA_ROOT", raising=False)
    code = rp.main(["--model", "fno", "--problem", "pipe", "--out-dir", str(tmp_path)])
    assert code == 0
    assert "SCIREX_CKPT_ROOT" in (tmp_path / "results.csv").read_text()


def test_hf_layout_checkpoint_is_found(tmp_path) -> None:
    (tmp_path / "ckpt" / "fno" / "pipe").mkdir(parents=True)
    cell = rp.select(None, None, model="fno", problem="pipe")[0]
    a = rp.argparse.Namespace(ckpt_root=str(tmp_path / "ckpt"), layout="hf")
    assert rp.ckpt_dir(a, cell).exists()
    a.layout = "archive"
    assert not rp.ckpt_dir(a, cell).exists()


def test_hf_repositories_are_one_per_model_and_problem() -> None:
    from reproduce_paper_map import CELLS, HF_ORG, hf_repo_of

    assert hf_repo_of("FNO", "pipe/fno") == f"{HF_ORG}/scirex-fno-pipe"
    assert hf_repo_of("GIWO", "car_cfd_giwo") == f"{HF_ORG}/scirex-giwo-car-cfd"
    assert (
        hf_repo_of("GINO", "darcy_s85/data_driven/gino/seed0")
        == f"{HF_ORG}/scirex-gino-darcy"
    )
    assert (
        hf_repo_of("WNO", "darcy_s85/physics_only/wno/fields_1000")
        == f"{HF_ORG}/scirex-wno-darcy-physics"
    )
    assert (
        hf_repo_of("WNO", "darcy_s85/physics_only/wno/fields_20000")
        == f"{HF_ORG}/scirex-wno-darcy-physics-extra-fields"
    )
    assert (
        hf_repo_of("GINO", "lid_driven_cavity_gino/gino_3to1") is None
    )  # Cavity is not public
    assert hf_repo_of("FNO", None) is None
    repos = {hf_repo_of(c.model, c.ckpt) for c in CELLS if c.ckpt} - {None}
    assert len(repos) == 33  # the 33 repositories published on Hugging Face
