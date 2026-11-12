# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""scripts/train.py and scripts/predict.py: data checks, every model end to end
on a CPU, prediction matching training, and data parallelism over 4 devices."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import train as tr  # noqa: E402

CASES = [
    ("fno", "grid"),
    ("wno", "grid"),
    ("transolver", "mesh"),
    ("gino", "mesh"),
    ("transolver", "grid"),
]


def make_data(folder: Path) -> dict[str, Path]:
    """A 16 x 16 grid problem and a 64-point mesh problem, 40 samples each."""
    rng = np.random.default_rng(0)
    g = np.linspace(0, 1, 16)
    a = rng.standard_normal((40, 3))[:, :, None, None] * np.sin(
        np.pi * np.arange(1, 4)[:, None, None] * g[:, None]
    )
    x = a.sum(1)[..., None] * np.ones((1, 1, 16, 1))
    paths = {"grid": folder / "grid.npz", "mesh": folder / "mesh.npz"}
    np.savez(
        paths["grid"], x=x.astype(np.float32), y=(0.5 * x + x**2).astype(np.float32)
    )
    coords = rng.random((40, 64, 2)).astype(np.float32)
    xm = np.sin(np.pi * coords[..., :1]) * rng.uniform(1, 2, (40, 1, 1))
    np.savez(
        paths["mesh"],
        coords=coords,
        x=xm.astype(np.float32),
        y=(xm * coords[..., 1:]).astype(np.float32),
    )
    return paths


def run(script: str, *args: str, devices: int = 1) -> subprocess.CompletedProcess:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args, "--device", "cpu"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_data(tmp_path_factory.mktemp("data"))


# -- data checks (no JAX) ------------------------------------------------------


def _save(tmp_path, **arrays) -> Path:
    path = tmp_path / "d.npz"
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize(
    "arrays, message",
    [
        (dict(x=np.ones((4, 8, 8, 1))), "no 'y'"),
        (dict(y=np.ones((4, 8, 8, 1))), "needs 'x'"),
        (
            dict(x=np.ones((4, 8, 8, 1)), y=np.ones((5, 8, 8, 1))),
            "different numbers of samples",
        ),
        (dict(x=np.ones((4, 8, 8, 1)), y=np.ones((4, 8, 6, 1))), "different grids"),
        (dict(x=np.ones((4, 1, 32, 32)), y=np.ones((4, 1, 32, 32))), "channels last"),
        (dict(x=np.ones((4, 32)), y=np.ones((4, 32))), "add the axis"),
        (dict(x=np.full((4, 8, 8, 1), np.nan), y=np.ones((4, 8, 8, 1))), "NaN"),
        (
            dict(
                coords=np.ones((4, 10, 5)), x=np.ones((4, 10, 1)), y=np.ones((4, 10, 1))
            ),
            "'coords' has shape",
        ),
        (
            dict(coords=np.ones((12, 2)), x=np.ones((4, 10, 1)), y=np.ones((4, 10, 1))),
            "coords has 12 points",
        ),
    ],
)
def test_bad_data_is_refused_with_a_clear_message(tmp_path, arrays, message):
    with pytest.raises(tr.DataError, match=message):
        tr.load_data(_save(tmp_path, **arrays))


def test_grid_and_mesh_are_told_apart(tmp_path, data):
    grid = tr.load_data(data["grid"])
    mesh = tr.load_data(data["mesh"])
    assert (grid["kind"], grid["spatial"], grid["c_in"], grid["c_out"]) == (
        "grid",
        (16, 16),
        1,
        1,
    )
    assert (mesh["kind"], mesh["spatial"], mesh["c_in"]) == ("mesh", (64,), 1)
    geometry_only = tr.load_data(
        _save(tmp_path, coords=np.ones((10, 3)), y=np.ones((4, 10, 1)))
    )
    assert geometry_only["c_in"] == 0


def test_split_is_seeded_and_disjoint(data):
    d = tr.load_data(data["grid"])
    a, b = tr.split(d, 0.25, seed=3)
    a2, b2 = tr.split(d, 0.25, seed=3)
    assert (len(a["y"]), len(b["y"])) == (30, 10)
    np.testing.assert_array_equal(b["y"], b2["y"])
    assert not np.isin(b["y"][:, 5, 5, 0], a["y"][:, 5, 5, 0]).any()


def test_set_values_are_parsed():
    assert tr.parse_set(
        ["n_modes=24,24", "slice_num=64", "lr=3e-4", "flag=true", "wavelet=db6"]
    ) == dict(n_modes=(24, 24), slice_num=64, lr=3e-4, flag=True, wavelet="db6")


def test_model_settings_fit_the_grid():
    info = dict(spatial=(16, 16), d=2, c_in=1, c_out=1)
    assert tr.model_spec("fno", "large", info, {})["n_modes"] == (
        8,
        8,
    )  # at most half the grid
    assert (
        tr.model_spec("wno", "medium", info, {})["level"] == 2
    )  # coarsest level keeps 4 points
    assert tr.model_spec("fno", "small", info, {"n_layers": 6})["n_layers"] == 6


# -- end to end on a CPU (subprocesses: devices are set before JAX starts) -------


@pytest.mark.parametrize("model, kind", CASES, ids=lambda c: c)
def test_train_then_predict(tmp_path, data, model, kind):
    out = tmp_path / "run"
    res = run(
        "train.py",
        "--model",
        model,
        "--data",
        str(data[kind]),
        "--smoke",
        "--out",
        str(out),
    )
    assert res.returncode == 0, res.stderr[-3000:]
    assert "Saved" in res.stdout
    files = sorted(p.name for p in out.iterdir())
    assert files == [
        "config.json",
        "history.json",
        "metrics.json",
        "model_metadata.msgpack",
        "model_params.bytes",
    ]
    metrics = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(metrics["test_rel_l2_mean"]) and metrics["n_params"] > 0

    # predict.py on the held-out samples reproduces the error train.py reported
    cfg = json.loads((out / "config.json").read_text())
    d = tr.load_data(data[kind], smoke=True)
    _, test = tr.split(d, cfg["options"]["test_fraction"], cfg["options"]["seed"])
    test_file = tmp_path / "test.npz"
    np.savez(test_file, **{k: v for k, v in test.items() if v is not None})
    pred = run(
        "predict.py",
        "--run",
        str(out),
        "--data",
        str(test_file),
        "--out",
        str(tmp_path / "p.npz"),
    )
    assert pred.returncode == 0, pred.stderr[-3000:]
    y_pred = np.load(tmp_path / "p.npz")["y_pred"]
    assert y_pred.shape == test["y"].shape
    err = np.linalg.norm(
        (y_pred - test["y"]).reshape(len(y_pred), -1), axis=1
    ) / np.linalg.norm(test["y"].reshape(len(y_pred), -1), axis=1)
    assert err.mean() == pytest.approx(metrics["test_rel_l2_mean"], rel=1e-4)


def test_device_cpu_overrides_jax_platform_name(tmp_path, data, monkeypatch):
    """A shell that sets JAX_PLATFORM_NAME=gpu must not break ``--device cpu``."""
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    res = run(
        "train.py",
        "--model",
        "fno",
        "--data",
        str(data["mesh"]),
        "--smoke",
        "--out",
        str(tmp_path / "r"),
    )
    assert "Unknown backend" not in res.stderr
    assert "needs data on a regular grid" in res.stderr


def test_grid_model_refuses_a_mesh(tmp_path, data):
    res = run(
        "train.py",
        "--model",
        "fno",
        "--data",
        str(data["mesh"]),
        "--smoke",
        "--out",
        str(tmp_path / "r"),
    )
    assert res.returncode != 0
    assert "needs data on a regular grid" in res.stderr


def test_four_devices_train_like_one(tmp_path, data):
    """Splitting each batch over 4 devices gives the same model as 1 device."""
    errs = []
    for n in (1, 4):
        out = tmp_path / f"run{n}"
        res = run(
            "train.py",
            "--model",
            "fno",
            "--data",
            str(data["grid"]),
            "--smoke",
            "--batch-size",
            "8",
            "--out",
            str(out),
            devices=n,
        )
        assert res.returncode == 0, res.stderr[-3000:]
        assert f"Device: {n} x" in res.stdout
        errs.append(json.loads((out / "metrics.json").read_text())["test_rel_l2_mean"])
    assert errs[0] == pytest.approx(errs[1], rel=1e-4)


def test_fno_predicts_at_a_new_resolution(tmp_path, data):
    out = tmp_path / "run"
    assert (
        run(
            "train.py",
            "--model",
            "fno",
            "--data",
            str(data["grid"]),
            "--smoke",
            "--out",
            str(out),
        ).returncode
        == 0
    )
    x = np.load(data["grid"])["x"][:3]
    fine = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)  # 32 x 32
    np.savez(tmp_path / "fine.npz", x=fine)
    res = run(
        "predict.py",
        "--run",
        str(out),
        "--data",
        str(tmp_path / "fine.npz"),
        "--out",
        str(tmp_path / "p.npz"),
    )
    assert res.returncode == 0, res.stderr[-3000:]
    assert np.load(tmp_path / "p.npz")["y_pred"].shape == (3, 32, 32, 1)
