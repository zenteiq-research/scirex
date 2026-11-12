# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the chip cooling FNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on a small synthetic dataset in the real file
layout, and the script is checked against the benchmark's own code: the same
split, grid, inputs, size and recipe, and the same prediction and loss for
the same weights. The test that reads the real dataset runs only when
``CHIPCOOLING_DATA_DIR`` points at it.
"""

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


def _tutorial_split(tutorial, folder: Path):
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    grid = tutorial.rasterise(
        data["coords"], np.concatenate([data["T_train"], data["T_test"]]), GRID
    )
    grid["fields_train"], grid["fields_test"] = (
        grid["fields"][:NTRAIN],
        grid["fields"][NTRAIN:],
    )
    return data, grid, tutorial.normalise(data, grid)


def _benchmark_split(folder: Path, cache: Path) -> dict:
    from scripts.benchmarks.chip_cooling._data import grid_split

    protocol = dict(ntrain=NTRAIN, ntest=NTEST, grid=GRID)
    return grid_split(benchmark_raw(folder), protocol, cache)


def test_load_takes_the_benchmark_split(tmp_path: Path) -> None:
    tutorial = load_script("fno")
    folder = fake_chip_cooling(tmp_path / "data")
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    raw = benchmark_raw(folder)
    np.testing.assert_array_equal(data["params_train"], raw["parameters"][:NTRAIN])
    np.testing.assert_array_equal(data["params_test"], raw["parameters"][-NTEST:])
    np.testing.assert_array_equal(data["T_train"], raw["fields"][:NTRAIN])
    np.testing.assert_array_equal(data["T_test"], raw["fields"][-NTEST:])
    np.testing.assert_array_equal(data["coords"], raw["coords"])
    with pytest.raises(ValueError, match="need ntrain"):
        tutorial.load_chip_cooling(folder, 10, 3)


def test_grid_and_inputs_match_the_benchmark(tmp_path: Path) -> None:
    tutorial = load_script("fno")
    folder = fake_chip_cooling(tmp_path / "data")
    _, grid, norm = _tutorial_split(tutorial, folder)
    ref = _benchmark_split(folder, tmp_path / "cache")
    np.testing.assert_array_equal(grid["mask"], ref["mask"])
    assert 0 < grid["mask"].mean() < 1
    np.testing.assert_allclose(
        norm["x_train"],
        ref["X_train"],
        rtol=tolerances.rtol(1e-6),
        atol=tolerances.atol(1e-6),
    )
    np.testing.assert_allclose(
        norm["x_test"],
        ref["X_test"],
        rtol=tolerances.rtol(1e-6),
        atol=tolerances.atol(1e-6),
    )
    np.testing.assert_allclose(
        norm["y_train"],
        ref["Y_train"],
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )
    np.testing.assert_allclose(
        norm["y_test"],
        ref["Y_test_n"],
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )
    np.testing.assert_allclose(
        grid["fields_test"], ref["Y_test"][:, 0], rtol=tolerances.rtol(1e-6)
    )


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.chip_cooling._config import PROTOCOL, RUNS

    tutorial = load_script("fno")
    run = RUNS["fno"]
    assert tutorial.ARCH == run.arch
    assert tutorial.SEED == PROTOCOL["seed"]
    for key in ("ntrain", "ntest", "grid", "epochs"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    assert run.opt["optimizer"] == "adamw_cx"
    assert tutorial.PROTOCOL["learning_rate"] == run.opt["lr"]
    assert tutorial.PROTOCOL["weight_decay"] == run.opt["weight_decay"]
    assert tutorial.PROTOCOL["batch_size"] == run.opt["batch_size"]
    _model, params = tutorial.build_model(
        tutorial.ARCH, np.zeros((7, *PROTOCOL["grid"]), np.float32), seed=0
    )
    assert count_model_params(params) == 1_829_496


def test_prediction_and_loss_match_the_benchmark(tmp_path: Path) -> None:
    from scripts.benchmarks.chip_cooling._models import get_spec
    from scripts.benchmarks.chip_cooling.train import relative_l2

    tutorial = load_script("fno")
    folder = fake_chip_cooling(tmp_path / "data")
    _, grid, norm = _tutorial_split(tutorial, folder)
    arch = dict(n_modes=(8, 8), hidden_channels=4)
    model, params = tutorial.build_model(arch, norm["x_train"][0], seed=42)
    spec = get_spec("fno")
    ref_model = spec.build(arch, GRID)
    ref_params = spec.init(ref_model, arch, dict(x_train=norm["x_train"]), 42)
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, ref_params)
    x, y = jnp.asarray(norm["x_test"]), jnp.asarray(norm["y_test"])
    pred = model.apply(params, x=x)
    np.testing.assert_array_equal(pred, ref_model.apply(params, x=x))
    mask = jnp.asarray(grid["mask"])[None, None]
    loss = tutorial.make_loss(model, grid["mask"])(params, x, y)
    assert float(loss) == pytest.approx(float(relative_l2(pred, y, mask)), rel=1e-6)


@pytest.mark.parametrize("optimizer", ["adamw_cx", "adamw"])
def test_optimizer_keeps_dtypes_and_conjugates_complex_gradients(optimizer) -> None:
    tutorial = load_script("fno")
    tx = tutorial.make_optimizer(tutorial.PROTOCOL, optimizer)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))
    loss = lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2)  # noqa: E731
    state = tx.init(params)
    assert all(not jnp.iscomplexobj(v) for v in state.hyperparams.values())
    updates, _ = tx.update(jax.grad(loss)(params), state, params)
    assert updates["w"].dtype == jnp.complex64
    assert updates["b"].dtype == params["b"].dtype
    step = complex(updates["w"][0])
    if optimizer == "adamw_cx":
        assert step.real < 0 and step.imag < 0  # towards 0 along -(1 + 1j)
    else:
        assert step.real < 0 and step.imag > 0  # plain optax: imaginary part ascends


def test_cosine_lr_matches_the_benchmark() -> None:
    import math

    tutorial = load_script("fno")
    for epoch in (0, 1, 250, 499):
        expected = 1e-3 * (1 + math.cos(math.pi * epoch / 500)) / 2
        assert tutorial.cosine_lr(tutorial.PROTOCOL, epoch, 500) == pytest.approx(
            expected
        )


def test_metrics_match_definition() -> None:
    tutorial = load_script("fno")
    true = np.array([[300.0, 310.0, 320.0], [305.0, 315.0, 330.0]])
    m = tutorial.field_metrics(true, true)
    assert m["R2"] == 1.0 and m["RMSE"] == 0.0 and m["max_AE"] == 0.0
    m = tutorial.field_metrics(true + 2.0, true)
    assert m["MAE"] == pytest.approx(2.0) and m["RMSE"] == pytest.approx(2.0)


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("fno")
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
    assert np.isfinite(metrics["baseline"]["R2"])
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("CHIPCOOLING_DATA_DIR"),
    reason="set CHIPCOOLING_DATA_DIR to the folder with model_setup.json and sim_*.npz",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = load_script("fno")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "8", "--n-test", "2",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["CHIPCOOLING_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["model"]["MAE"])
