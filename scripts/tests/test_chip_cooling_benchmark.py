# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/chip_cooling/.

Everything runs on a few synthetic samples on the CPU, so no dataset or GPU
is needed: the data split and normalisation, the size of the published runs,
and the train -> checkpoint -> infer round trip of every model.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.data.datasets.chipcooling import (  # noqa: E402
    COORD_KEY,
    FIELD_KEY,
)
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.chip_cooling import infer  # noqa: E402
from scripts.benchmarks.chip_cooling._checkpoint import is_complete  # noqa: E402
from scripts.benchmarks.chip_cooling._config import PROTOCOL, RUNS  # noqa: E402
from scripts.benchmarks.chip_cooling._data import (  # noqa: E402
    grid_split,
    load_raw,
    mesh_split,
)
from scripts.benchmarks.chip_cooling._models import get_spec  # noqa: E402
from scripts.benchmarks.chip_cooling.train import train_one  # noqa: E402

GRID = (16, 32)
SMALL = dict(PROTOCOL, ntrain=4, ntest=2, grid=GRID)
N_NODES = 400
TINY_ARCH = {
    "fno": dict(n_modes=(4, 4), hidden_channels=4),
    "wno": dict(
        RUNS["wno"].arch,
        hidden_channels=4,
        n_layers=1,
        level=1,
        wavelet="db4",
        projection_hidden_channels=8,
    ),
    "gino": dict(
        RUNS["gino"].arch,
        latent_h=8,
        latent_w=16,
        in_gno_radius=0.3,
        out_gno_radius=0.3,
        in_gno_channel_mlp_hidden_layers=(8,),
        out_gno_channel_mlp_hidden_layers=(8,),
        fno_n_modes=(4, 4),
        fno_hidden_channels=4,
        fno_n_layers=1,
        fno_lifting_channel_ratio=1,
    ),
    "transolver": dict(
        RUNS["transolver"].arch, n_layers=1, n_hidden=8, n_head=2, slice_num=4
    ),
}
# The published runs' sizes (doc/benchmarks/chip_cooling/results/comparative.md).
PUBLISHED_PARAMS = dict(
    fno=1_829_496, wno=1_920_481, gino=1_821_028, transolver=1_807_561
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Six synthetic simulations in the model_setup.json / sim_*.npz layout:
    a 7 cm x 20 cm rectangle with a hole, on N_NODES scattered nodes."""
    d = tmp_path_factory.mktemp("chip")
    rng = np.random.default_rng(0)
    n = SMALL["ntrain"] + SMALL["ntest"]
    pts = rng.random((4 * N_NODES, 2)) * [0.07, 0.2]
    hole = (pts[:, 0] > 0.04) & (np.abs(pts[:, 1] - 0.1) < 0.03)
    pts = pts[~hole][:N_NODES]
    coords = np.column_stack([np.zeros(len(pts)), pts]).astype(np.float32)
    doe = {
        "chip1": {"thermal.heat_flux": list(rng.uniform(1e5, 1e6, n))},
        "chip2": {"thermal.heat_flux": list(rng.uniform(1e5, 1e6, n))},
        "inlet1": {
            "momentum.mass_flow_rate": list(rng.uniform(1e-3, 1e-2, n)),
            "thermal.total_temperature": list(rng.uniform(290, 340, n)),
        },
        "inlet2": {
            "momentum.mass_flow_rate": list(rng.uniform(1e-3, 1e-2, n)),
            "thermal.total_temperature": list(rng.uniform(290, 340, n)),
        },
    }
    (d / "model_setup.json").write_text(json.dumps({"doe_configuration": doe}))
    for i in range(n):
        temp = 300 + 100 * rng.random() * np.sin(pts[:, 1] * 30) + 50 * pts[:, 0]
        np.savez(
            d / f"sim_{i}.npz",
            **{FIELD_KEY: temp.astype(np.float32), COORD_KEY: coords},
        )
    return d


@pytest.fixture(scope="module")
def raw(data_dir: Path) -> dict:
    return load_raw(data_dir)


def test_grid_split_masks_and_normalises(raw: dict, tmp_path: Path) -> None:
    data = grid_split(raw, SMALL, tmp_path)
    mask = data["mask"]
    assert data["X_train"].shape == (4, 7, *GRID)
    assert data["Y_test"].shape == (2, 1, *GRID)
    assert 0 < mask.sum() < mask.size
    # Channel 6 is the in-domain mask, untouched by the normalisation.
    np.testing.assert_array_equal(
        data["X_train"][:, 6], np.broadcast_to(mask, (4, *GRID))
    )
    # Targets are zero outside the domain and z-scored inside it.
    assert np.all(data["Y_train"][:, 0][:, ~mask] == 0)
    inside = data["Y_train"][:, 0][:, mask]
    assert abs(inside.mean()) < 1e-4 and abs(inside.std() - 1) < 1e-3


def test_mesh_split_uses_training_statistics(raw: dict) -> None:
    data = mesh_split(raw, SMALL)
    norm = data["normalisation"]
    assert norm["t_mean"] == pytest.approx(float(raw["fields"][:4].mean()))
    np.testing.assert_allclose(data["params_n"][:4].mean(axis=0), 0, atol=1e-5)
    np.testing.assert_array_equal(data["fields_test"], raw["fields"][-2:])


@pytest.mark.parametrize("name", sorted(RUNS))
def test_published_runs_have_their_published_size(
    name: str, raw: dict, tmp_path
) -> None:
    run = RUNS[name]
    spec = get_spec(run.model)
    protocol = dict(SMALL, grid=PROTOCOL["grid"])
    data = (
        grid_split(raw, protocol, tmp_path)
        if spec.kind == "grid"
        else mesh_split(raw, protocol)
    )
    prepared = spec.prepare(run.arch, data)
    params = spec.init(spec.build(run.arch, protocol["grid"]), run.arch, prepared, 0)
    assert int(count_model_params(params)) == PUBLISHED_PARAMS[name]


@pytest.mark.parametrize("model", sorted(RUNS))
def test_train_checkpoint_infer_round_trip(
    model: str, raw: dict, tmp_path: Path
) -> None:
    run = replace(
        RUNS[model],
        arch=TINY_ARCH[model],
        epochs=2,
        opt=dict(RUNS[model].opt, warmup_epochs=1),
    )
    run_dir = tmp_path / "ckpt" / model
    train_one(run, raw, run_dir, SMALL, tmp_path / "cache")
    assert is_complete(run_dir)

    ck = infer.load_run(run_dir, raw, SMALL, tmp_path / "cache")
    assert ck["run"] == run
    assert len(ck["history"]["val_loss"]) == 2
    pred = ck["spec"].predict(ck["model"], ck["params"], ck["prepared"], ck["data"])
    if ck["spec"].kind == "grid":
        assert pred.shape == (2, 1, *GRID)
    else:
        assert pred.shape == (2, N_NODES)
    metrics = infer.test_metrics(ck, pred)
    assert np.isfinite(metrics["MAE"]) and len(metrics["MAE_per_sample"]) == 2

    out = tmp_path / "assets"
    infer.write_assets(ck, raw, pred, out, metrics, run_dir, sample=1)
    assert sorted(p.name for p in out.iterdir()) == [
        "compute_cost.png",
        "contour.png",
        "error_curve.png",
        "metrics.json",
    ]
    written = json.loads((out / "metrics.json").read_text())
    assert {k: written[k] for k in ck["history"]} == ck["history"]
    assert written["test"]["MAE"] == metrics["MAE"]
    assert written["inference_provenance"]["checkpoints"][0]["sha256"]["params.msgpack"]


def test_checkpoint_refuses_a_different_split(raw: dict, tmp_path: Path) -> None:
    run = replace(RUNS["fno"], arch=TINY_ARCH["fno"], epochs=1)
    run_dir = tmp_path / "ckpt"
    train_one(run, raw, run_dir, SMALL, tmp_path / "cache")
    with pytest.raises(ValueError, match="normalisation"):
        infer.load_run(run_dir, raw, dict(SMALL, ntrain=3, ntest=3), tmp_path / "cache")
