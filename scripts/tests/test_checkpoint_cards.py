# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Checkpoint cards: generator on a toy archive, and the committed cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_checkpoint_cards as mc
from reproduce_paper_map import CELLS

CARDS = Path(__file__).resolve().parents[2] / "doc" / "benchmarks" / "cards"
BENCHMARKS = {mc.benchmark_of(c.ckpt) for c in CELLS if c.ckpt}


def _toy(root: Path, name: str, cfg: dict | None, files=("params.msgpack",)):
    d = root / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"x")
    if cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg))


def test_seed_only_from_config():
    assert mc.seed_of({"training": {"seed": 3}}) == "3"
    assert (
        mc.seed_of({"training": {"seed": 1, "seeds": [0, 1]}}) == "1 (of seeds [0, 1])"
    )
    assert mc.seed_of({}) == mc.UNKNOWN


def test_epoch_rule():
    cavity = {"training": {"best_epoch": 287, "epochs_run": 300}}
    assert "best validation epoch (287 of 300" in mc.epoch_rule("cavity", cavity, [])
    assert mc.epoch_rule("chip_cooling", {}, []).startswith(mc.UNKNOWN)
    assert "final" in mc.epoch_rule("darcy_s85", {}, ["a_final.msgpack"])
    assert "final" in mc.epoch_rule("pipe", {"training": {}}, ["params.msgpack"])


def test_describe_reads_config(tmp_path):
    import numpy as np
    from flax import serialization

    _toy(
        tmp_path,
        "pipe/fno",
        {"run": {"arch": {"w": 1}}, "n_params": 5, "training": {"seed": 0}},
    )
    (tmp_path / "pipe/fno/params.msgpack").write_bytes(
        serialization.msgpack_serialize({"w": np.zeros(5, np.float32)})
    )
    info = mc.describe(tmp_path, "pipe/fno", None)
    assert info["seed"] == "0" and info["n_params"] == "5"
    assert info["commit"] == mc.UNKNOWN


def test_weights_hashes_accept_paths_with_and_without_the_archive_prefix(tmp_path):
    inv = tmp_path / "inv.csv"
    inv.write_text(
        "path,weights_sha256\n"
        "Checkpoints_Benchmarks/pipe/fno,aaa\n"
        "darcy_s85/data_driven/fno/seed0,bbb\n"
        "airfrans/wno,\n"
    )
    assert mc.weights_hashes(inv) == {
        "pipe/fno": "aaa",
        "darcy_s85/data_driven/fno/seed0": "bbb",
    }


def test_weights_hashes_keep_the_newest_run_of_a_path(tmp_path):
    inv = tmp_path / "inv.csv"
    inv.write_text(
        "path,weights_sha256,created_utc\n"
        "Checkpoints_Benchmarks/pipe/transolver,old,2026-10-26T09:05:26+00:00\n"
        "pipe/transolver,new,2026-11-04T10:24:59+00:00\n"
        "pipe/fno,same,\n"
        "Checkpoints_Benchmarks/pipe/fno,same,\n"
    )
    assert mc.weights_hashes(inv) == {"pipe/transolver": "new", "pipe/fno": "same"}


def test_stale_when_weights_newer_than_run(tmp_path):
    diff = tmp_path / "d.csv"
    diff.write_text("row,model\n")
    assert mc._stale("2999-01-01T00:00:00+00:00", diff)
    assert not mc._stale("2000-01-01T00:00:00+00:00", diff)
    assert not mc._stale("", diff)


HEAD = "| Model | Checkpoint | Parameters | SHA-256 | Reproduced |"
SEP = "|---|---|---|---|---|"


def test_strip_measured_ignores_run_columns():
    a = f"{HEAD}\n{SEP}\n| FNO | `p` | 5 | `abc` | 0.4 |"
    b = f"{HEAD}\n{SEP}\n| FNO | `p` | 5 | `abc` | not run |"
    assert mc._strip_measured(a) == mc._strip_measured(b)
    c = f"{HEAD}\n{SEP}\n| FNO | `p` | 6 | `abc` | 0.4 |"
    assert mc._strip_measured(a) != mc._strip_measured(
        c
    )  # a changed parameter count shows


def _stub_info(ckpt, diff_path):
    return {
        "path": ckpt, "weights": ["params.msgpack"], "n_params": "1234567",
        "arch": "w=1", "train": "epochs=3", "seed": "1 (of seeds [0, 1, 2])",
        "epoch_rule": "final epoch", "commit": "abc123", "created": "2026-01-01",
        "created_label": "Created", "stale": False, "converted_from": "",
        "recorded": "",
    }  # fmt: skip


def test_gather_and_render_one_page_with_a_table_per_benchmark():
    diff = {
        (c.row, c.model): {"ours": "1.0", "status": "PASS"} for c in CELLS if c.ckpt
    }
    hashes = {c.ckpt: "ab" * 32 for c in CELLS if c.ckpt}
    benches = list(dict.fromkeys(mc.benchmark_of(c.ckpt) for c in CELLS if c.ckpt))
    sections = [
        mc.gather(b, None, diff, None, hashes, lambda root, ck, dp: _stub_info(ck, dp))
        for b in benches
    ]
    page = mc.render(sections)
    for b in benches:
        assert f"## {mc.TITLES[b]}" in page
    released = [b for b in benches if b not in mc.UNRELEASED]
    assert page.count('??? note "Configuration and training"') == len(released)
    assert "| 1,234,567 |" in page and "1 of 0, 1, 2" not in page  # no Seed column
    assert "| Seed |" not in page and "| Result |" not in page and "PASS" not in page
    assert "`abababababab`" in page  # 12 characters of the hash in the table
    assert "ab" * 32 in page  # the full hash in the details
    assert "| Download |" not in page


def _rows(*specs):
    return [
        {"model": "FNO", "ckpt": "c", "params": "5", "seed": "0", "sha": "ab" * 32,
         "row": row, "metric": metric, "unit": unit, "paper": 1, "ours": "1",
         "result": "PASS"}
        for row, metric, unit in specs
    ]  # fmt: skip


def test_metric_column_uses_short_labels_and_states_a_shared_unit_once():
    sec = {
        "bench": "cavity",
        "rows": _rows(
            ("Cavity, 1 step", "RMSE", "1e-3 m/s"),
            ("Cavity, 10 steps", "RMSE", "1e-3 m/s"),
        ),
        "details": [],
    }
    page = mc.render([sec])
    assert "| Metric |" in page and "| Cell |" not in page
    assert "| 1 step |" in page and "| 10 steps |" in page
    assert "Metric: RMSE (1e-3 m/s)." in page
    assert "m/s)" not in page.split("| Metric |")[1]  # the unit is not repeated


def test_paper_row_label_carries_the_unit_when_the_rows_differ():
    sec = {
        "bench": "airfrans",
        "rows": _rows(
            ("AirfRANS (volume MSE)", "Volume MSE", "1e-3"),
            ("AirfRANS (C_L)", "C_L rel. error", "%"),
        ),
        "details": [],
    }
    page = mc.render([sec])
    assert "| Volume MSE (1e-3) |" in page and "| C_L error (%) |" in page
    assert "Metric:" not in page


def test_a_row_without_a_short_label_keeps_its_full_label():
    sec = {
        "bench": "pipe",
        "rows": _rows(("Some new row", "rel. L2", "%"), ("Another", "rel. L2", "%")),
        "details": [],
    }
    page = mc.render([sec])
    assert "| Some new row |" in page and "| Another |" in page


def test_render_adds_a_download_column_for_a_hugging_face_repo():
    sec = {
        "bench": "pipe",
        "rows": [
            {"model": "FNO", "ckpt": "pipe/fno", "params": "5", "seed": "0",
             "sha": "cd" * 32, "row": "Pipe", "metric": "rel. L2", "unit": "%",
             "paper": 0.9, "ours": "0.9", "result": "PASS"},
        ],
        "details": [],
    }  # fmt: skip
    page = mc.render([sec], "myorg")
    assert "(https://huggingface.co/myorg/scirex-fno-pipe)" in page
    assert "| Download |" in page


@pytest.mark.skipif(not CARDS.is_dir(), reason="cards not generated")
def test_the_committed_page_covers_every_benchmark_and_checkpoint():
    text = (CARDS / "index.md").read_text()
    assert len(BENCHMARKS) == 9
    for b in BENCHMARKS:
        assert f"## {mc.TITLES[b]}" in text, b
    released = [b for b in BENCHMARKS if b not in mc.UNRELEASED]
    assert text.count('??? note "Configuration and training"') == len(released)
    assert "Note (" not in text  # the cards carry no per-model notes
    experimental = (CARDS / "experimental.md").read_text()
    for c in CELLS:
        if (
            c.ckpt
            and c.row not in mc.SKIP_ROWS
            and mc.benchmark_of(c.ckpt) not in mc.UNRELEASED
        ):
            page = experimental if c.model in mc.EXPERIMENTAL_MODELS else text
            assert f"`{c.ckpt}`" in page, c.ckpt
    assert "GIWO |" not in text and "| GIWO |" in experimental
    for field in ("Epoch rule", "SHA-256", "Architecture"):
        assert field in text, field
    for internal in (
        "Training commit",
        "Copied to archive",
        "Converted from",
        "Recorded at training",
    ):
        assert internal not in text, internal
    assert "TODO" not in text and "None" not in text
    assert "nan" not in text.lower().split()
    assert "LGPL-3.0-only" in text and "dataset guide" in text
    # every checkpoint has a hash in the table (a repeated cell of a checkpoint is blank)
    column = None
    for line in text.splitlines():
        if line.startswith("| Model |"):
            heads = [h.strip() for h in line.strip().strip("|").split("|")]
            column = heads.index("SHA-256") if "SHA-256" in heads else None
        elif (
            column is not None and line.startswith("| ") and not line.startswith("|---")
        ):
            sha = line.strip().strip("|").split("|")[column].strip()
            assert sha != mc.UNKNOWN, line  # blank when it repeats the row above
        elif not line.startswith("|"):
            column = None
    assert sorted(p.name for p in CARDS.glob("*.md")) == ["experimental.md", "index.md"]


def test_main_writes_one_page_and_removes_the_old_per_benchmark_pages(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "cards"
    out.mkdir()
    old = out / "pipe.md"
    old.write_text(
        "# Pipe Flow: Checkpoint Card\n<!-- generated by scripts/make_checkpoint_cards.py -->\n"
    )
    mine = out / "notes.md"  # a page someone wrote by hand is left alone
    mine.write_text("# Notes\n")
    monkeypatch.setattr(mc, "build", lambda root, diff: {"index.md": "# page\n"})
    args = ["--root", str(tmp_path), "--out-dir", str(out)]
    assert mc.main([*args, "--check"]) == 1  # the page is missing and an old one exists
    assert "pipe.md" in capsys.readouterr().out
    assert mc.main(args) == 0
    assert (out / "index.md").read_text() == "# page\n"
    assert not old.exists() and mine.exists()
    assert mc.main([*args, "--check"]) == 0


def test_count_params_counts_complex_as_two(tmp_path):
    import numpy as np
    from flax import serialization

    tree = {"a": np.zeros((3, 4), np.float32), "b": {"w": np.zeros(5, np.complex64)}}
    f = tmp_path / "params.msgpack"
    f.write_bytes(serialization.msgpack_serialize(tree))
    assert mc.count_params(f) == 12 + 10
    assert mc.params_of(tmp_path, ["params.msgpack"], 22) == "22"
    assert "the archive records 7" in mc.params_of(tmp_path, ["params.msgpack"], 7)
    assert mc.count_params(tmp_path / "missing.msgpack") is None


def test_legacy_config_from_script_arguments_and_metrics():
    rep = {"script_arguments": {"epochs": 500, "modes": 20, "n_layers": 4}}
    arch, train = mc.legacy_config(rep, {"hidden_channels": 40, "n_params": 1})
    assert arch == {"hidden_channels": 40, "modes": 20, "n_layers": 4}
    assert train == {"epochs": 500}


def test_legacy_config_from_command_line():
    rep = {
        "command": "python run.py --data-dir /d --epochs 500 --latent-res 128 64 "
        "--in-radius 0.16 --latent-sdf --fno-modes 21 16"
    }
    arch, train = mc.legacy_config(rep, {})
    assert arch == {
        "latent_res": [128, 64],
        "in_radius": 0.16,
        "latent_sdf": True,
        "fno_modes": [21, 16],
    }
    assert train == {"epochs": 500}
    assert "REPRODUCE" not in mc.config_of({}, rep)[0]


def test_an_unreleased_benchmark_shows_results_only():
    sec = {
        "bench": "cavity",
        "rows": _rows(("Cavity, 1 step", "RMSE", "1e-3 m/s")),
        "details": [
            {"ckpt": "pipe/fno", "model": "FNO", "weights": ["w"], "sha": "ab" * 32}
        ],
        "notes": [],
    }
    page = mc.render([sec], "myorg")
    assert "its checkpoints are not released" in page
    assert "| Checkpoint |" not in page and "| SHA-256 |" not in page
    assert "| Download |" not in page and "huggingface.co" not in page
    assert '??? note "Configuration and training"' not in page
    assert "| Model | Parameters |" in page


def test_car_cfd_training_line_uses_the_run_not_the_protocol_default():
    protocol = {"learning_rate": 2.5e-4, "weight_decay": 0.0, "n_epochs": 100}
    giwo = {
        "run": {
            "learning_rate": 0.008,
            "lr_step": 50,
            "lr_gamma": 0.5,
            "weight_decay": 0.0,
            "n_epochs": 100,
            "schedule": {"kind": "warmup_cosine", "warmup_epochs": 1},
        },
        "training": protocol,
    }
    line = mc.config_of(giwo, {}, None, "car_cfd")[1]
    assert "learning_rate=0.008" in line and "0.00025" not in line
    assert "warmup_cosine" in line and "lr_step" not in line
    gino = {
        "run": {
            "learning_rate": 2.5e-4,
            "lr_step": 50,
            "lr_gamma": 0.5,
            "schedule": None,
        },
        "training": protocol,
    }
    line = mc.config_of(gino, {}, None, "car_cfd")[1]
    assert "lr_step=50" in line and "lr_gamma=0.5" in line and "schedule" not in line


def test_a_placeholder_epoch_count_is_left_out_of_the_training_line():
    cfg = {"run": {}, "training": {"epochs": "0 (no training)", "lr": 0.001}}
    line = mc.config_of(cfg, {})[1]
    assert "epochs" not in line and "lr=0.001" in line
