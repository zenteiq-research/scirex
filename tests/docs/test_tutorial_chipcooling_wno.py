# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the chip cooling WNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on a small synthetic dataset in the real file
layout, and the script is checked against the benchmark's own code: the same
inputs (with the grid channels the benchmark appends through a private
helper), size and recipe, and the same prediction and loss for the same
weights. The test that reads the real dataset runs only when
``CHIPCOOLING_DATA_DIR`` points at it.
"""

import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances
from tests.docs.chipcooling_tutorial_helpers import (
    benchmark_raw,
    fake_chip_cooling,
    load_script,
)

NTRAIN, NTEST, GRID = 6, 3, (32, 64)
SMALL = dict(
    hidden_channels=4,
    n_layers=1,
    level=2,
    wavelet="db4",
    padding=1,
    projection_hidden_channels=8,
)


def _splits(tmp_path: Path):
    from scripts.benchmarks.chip_cooling._data import grid_split

    tutorial = load_script("wno")
    folder = fake_chip_cooling(tmp_path / "data")
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    grid = tutorial.rasterise(
        data["coords"], np.concatenate([data["T_train"], data["T_test"]]), GRID
    )
    grid["fields_train"], grid["fields_test"] = (
        grid["fields"][:NTRAIN],
        grid["fields"][NTRAIN:],
    )
    norm = tutorial.normalise(data, grid)
    protocol = dict(ntrain=NTRAIN, ntest=NTEST, grid=GRID)
    ref = grid_split(benchmark_raw(folder), protocol, tmp_path / "cache")
    return tutorial, grid, norm, ref


def test_append_grid_matches_the_benchmark_helper() -> None:
    from scirex._internal.wno_train_utils import append_grid_nhwc

    tutorial = load_script("wno")
    x = np.random.default_rng(0).normal(size=(2, 5, 7, 3)).astype(np.float32)
    np.testing.assert_allclose(
        tutorial.append_grid(x), append_grid_nhwc(jnp.asarray(x), (1.0, 1.0)), atol=1e-7
    )


def test_inputs_match_the_benchmark(tmp_path: Path) -> None:
    from scripts.benchmarks.chip_cooling._config import RUNS
    from scripts.benchmarks.chip_cooling._models import get_spec

    tutorial, _grid, norm, ref = _splits(tmp_path)
    prepared = get_spec("wno").prepare(RUNS["wno"].arch, ref)
    np.testing.assert_allclose(
        tutorial.to_wno_inputs(norm["x_train"]),
        prepared["x_train"],
        atol=tolerances.atol(1e-5),
    )
    np.testing.assert_allclose(
        tutorial.to_wno_inputs(norm["x_test"]),
        prepared["x_test"],
        atol=tolerances.atol(1e-5),
    )
    np.testing.assert_allclose(
        norm["y_test"].transpose(0, 2, 3, 1),
        prepared["y_test"],
        atol=tolerances.atol(1e-5),
    )


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.chip_cooling._config import PROTOCOL, RUNS

    tutorial = load_script("wno")
    run = RUNS["wno"]
    extra = dict(compute_dtype="bfloat16", param_dtype="float32", grid_range=(1.0, 1.0))
    assert dict(tutorial.SIZES["full"], **extra) == run.arch
    assert tutorial.SEED == PROTOCOL["seed"]
    for key in ("ntrain", "ntest", "grid", "epochs"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    assert run.opt["optimizer"] == "adam + add_decayed_weights"
    assert tutorial.PROTOCOL["learning_rate"] == run.opt["lr"]
    assert tutorial.PROTOCOL["weight_decay"] == run.opt["weight_decay"]
    assert tutorial.PROTOCOL["batch_size"] == run.opt["batch_size"]
    _model, params = tutorial.build_model(
        tutorial.SIZES["full"], PROTOCOL["grid"], np.zeros((*PROTOCOL["grid"], 9)), 0
    )
    assert count_model_params(params) == 1_920_481


def test_prediction_and_loss_match_the_benchmark(tmp_path: Path) -> None:
    from scripts.benchmarks.chip_cooling._models import get_spec
    from scripts.benchmarks.chip_cooling.train import relative_l2

    tutorial, grid, norm, _ref = _splits(tmp_path)
    x_test = tutorial.to_wno_inputs(norm["x_test"])
    y_test = norm["y_test"].transpose(0, 2, 3, 1)
    model, params = tutorial.build_model(SMALL, GRID, x_test[0], seed=42)
    spec = get_spec("wno")
    arch = dict(
        SMALL, compute_dtype="bfloat16", param_dtype="float32", grid_range=(1, 1)
    )
    ref_model = spec.build(arch, GRID)
    ref_params = spec.init(ref_model, arch, dict(x_train=x_test), 42)
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, ref_params)
    pred = model.apply({"params": params}, jnp.asarray(x_test))
    np.testing.assert_array_equal(
        pred, ref_model.apply({"params": params}, jnp.asarray(x_test))
    )
    mask = jnp.asarray(grid["mask"])[None, :, :, None]
    loss = tutorial.make_loss(model, grid["mask"])(
        params, jnp.asarray(x_test), jnp.asarray(y_test)
    )
    expected = relative_l2(pred, jnp.asarray(y_test), mask)
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)


def test_schedule_matches_the_benchmark() -> None:
    import optax

    tutorial = load_script("wno")
    steps = 500 * math.ceil(1600 / 8)
    _, schedule = tutorial.make_optimizer(tutorial.PROTOCOL, steps)
    ref = optax.cosine_decay_schedule(1e-3, decay_steps=steps, alpha=0.0)
    for step in (0, 1, steps // 2, steps - 1, steps):
        assert float(schedule(step)) == pytest.approx(float(ref(step)), abs=1e-12)


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_chip_cooling(tmp_path / "data")),
            "--smoke",
            "--n-train", "4",
            "--n-test", "2",
            "--epochs", "1",
            "--grid", "32", "64",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["model"]["MAE"])
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("CHIPCOOLING_DATA_DIR"),
    reason="set CHIPCOOLING_DATA_DIR to the folder with model_setup.json and sim_*.npz",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "8", "--n-test", "2",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["CHIPCOOLING_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["model"]["MAE"])
