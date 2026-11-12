# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every benchmark infer.py takes --checkpoint, --data-dir and --out (no GPU needed)."""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmarks._cli import add_standard_args, checkpoint_dir, select_run

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = [
    "airfrans",
    "car_cfd",
    "cavity",
    "chip_cooling",
    "darcy",
    "navier_stokes",
    "pipe",
    "shapenetcar",
]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_infer_help_lists_standard_flags(problem):
    r = subprocess.run(
        [sys.executable, "-m", f"scripts.benchmarks.{problem}.infer", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr[-300:]
    for flag in ("--checkpoint", "--data-dir", "--out"):
        assert flag in r.stdout, f"{problem} lacks {flag}"


def _parser():
    ap = argparse.ArgumentParser()
    add_standard_args(ap, out_flag="--out-dir", out_default=Path("default"))
    ap.add_argument("--ckpt-root", type=Path, default=Path("root"))
    return ap


def test_out_and_legacy_flag_fill_the_same_attribute():
    ap = _parser()
    assert ap.parse_args(["--out", "a"]).out_dir == Path("a")
    assert ap.parse_args(["--out-dir", "b"]).out_dir == Path("b")
    assert ap.parse_args([]).out_dir == Path("default")


def test_select_run_uses_directory_name_or_run_flag(tmp_path):
    ck = tmp_path / "fno"
    ck.mkdir()
    ap = _parser()
    a = ap.parse_args(["--checkpoint", str(ck)])
    assert select_run(ap, a, ["fno", "wno"]) == "fno"
    assert checkpoint_dir(a, "fno") == ck
    assert checkpoint_dir(a, "wno") == Path("root/wno")
    odd = tmp_path / "odd"
    odd.mkdir()
    a = ap.parse_args(["--checkpoint", str(odd)])
    with pytest.raises(SystemExit):
        select_run(ap, a, ["fno"])
    a = ap.parse_args(["--checkpoint", str(odd), "--run", "fno"])
    assert select_run(ap, a, ["fno"]) == "fno"


def test_without_checkpoint_every_run_is_evaluated():
    ap = _parser()
    a = ap.parse_args([])
    assert select_run(ap, a, ["fno"]) is None
    assert checkpoint_dir(a, "fno") == Path("root/fno")
