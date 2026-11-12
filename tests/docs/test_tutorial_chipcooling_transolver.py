# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the chip cooling Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on a small synthetic dataset in the real file
layout, and the script is checked against the benchmark's own code: the same
tokens, targets, size, schedule and recipe, and the same prediction and loss
for the same weights. The test that reads the real dataset runs only when
``CHIPCOOLING_DATA_DIR`` points at it.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from tests import tolerances
from tests.docs.chipcooling_tutorial_helpers import (
    benchmark_raw,
    fake_chip_cooling,
    load_script,
)

NTRAIN, NTEST = 6, 3


def _setup(tmp_path: Path):
    from scripts.benchmarks.chip_cooling._data import mesh_split
    from scripts.benchmarks.chip_cooling._models import get_spec

    tutorial = load_script("transolver")
    folder = fake_chip_cooling(tmp_path / "data", n_nodes=400)
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    norm = tutorial.normalise(data)
    split = mesh_split(benchmark_raw(folder), dict(ntrain=NTRAIN, ntest=NTEST))
    arch = tutorial.SIZES["smoke"]
    prepared = get_spec("transolver").prepare(arch, split)
    return tutorial, norm, arch, prepared


def test_tokens_and_targets_match_the_benchmark(tmp_path: Path) -> None:
    tutorial, norm, _, prepared = _setup(tmp_path)
    for name in ("train", "test"):
        x, fx = tutorial.tokens(norm["nodes"], norm[f"p_{name}"])
        np.testing.assert_allclose(x, prepared[f"x_{name}"], atol=1e-7)
        np.testing.assert_allclose(
            fx, prepared[f"fx_{name}"], atol=tolerances.atol(1e-6)
        )
        np.testing.assert_allclose(
            norm[f"y_{name}"], prepared[f"y_{name}"], atol=tolerances.atol(1e-5)
        )


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.chip_cooling._config import PROTOCOL, RUNS

    tutorial = load_script("transolver")
    run = RUNS["transolver"]
    assert tutorial.SIZES["full"] == run.arch
    assert tutorial.SEED == PROTOCOL["seed"]
    for key in ("ntrain", "ntest", "epochs"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    for key in ("weight_decay", "grad_clip", "warmup_epochs", "lr_floor", "batch_size"):
        assert tutorial.PROTOCOL[key] == run.opt[key], key
    assert tutorial.PROTOCOL["learning_rate"] == run.opt["lr"]
    nodes = np.random.default_rng(0).random((50, 2)).astype(np.float32)
    _, params = tutorial.build_model(
        tutorial.SIZES["full"], nodes, np.zeros((1, 6), np.float32), 0
    )
    assert count_model_params(params) == 1_807_561


def test_full_schedule_matches_the_benchmark() -> None:
    tutorial = load_script("transolver")
    steps_per_epoch = 1600 // 4
    _, schedule = tutorial.make_optimizer(tutorial.PROTOCOL, 500, steps_per_epoch)
    warmup, decay = 50 * steps_per_epoch, 450 * steps_per_epoch
    ref = optax.join_schedules(
        [
            optax.linear_schedule(0.0, 1e-4, transition_steps=warmup),
            optax.cosine_decay_schedule(1e-4, decay_steps=decay, alpha=1e-2),
        ],
        boundaries=[warmup],
    )
    for step in (0, 1, warmup - 1, warmup, warmup + decay // 2, warmup + decay):
        assert float(schedule(step)) == pytest.approx(float(ref(step)), abs=1e-14)


def test_prediction_and_loss_match_the_benchmark(tmp_path: Path) -> None:
    from scripts.benchmarks.chip_cooling._models import get_spec
    from scripts.benchmarks.chip_cooling.train import relative_l2

    tutorial, norm, arch, prepared = _setup(tmp_path)
    model, params = tutorial.build_model(arch, norm["nodes"], norm["p_train"], seed=42)
    spec = get_spec("transolver")
    ref_model = spec.build(arch, None)
    ref_params = spec.init(ref_model, arch, prepared, 42)
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, ref_params)
    x, fx = (jnp.asarray(a) for a in tutorial.tokens(norm["nodes"], norm["p_test"]))
    y = jnp.asarray(norm["y_test"])
    pred = model.apply(params, x=x, fx=fx)
    ref_pred = ref_model.apply(
        params, x=jnp.asarray(prepared["x_test"]), fx=jnp.asarray(prepared["fx_test"])
    )
    np.testing.assert_allclose(
        pred, ref_pred, rtol=tolerances.rtol(1e-5), atol=tolerances.atol(1e-5)
    )
    loss = tutorial.make_loss(model)(params, x, fx, y)
    expected = relative_l2(ref_pred, jnp.asarray(prepared["y_test"]))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("transolver")
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_chip_cooling(tmp_path / "data", n_nodes=400)),
            "--smoke",
            "--n-train", "4",
            "--n-test", "2",
            "--epochs", "2",
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
    tutorial = load_script("transolver")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "4", "--n-test", "2",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["CHIPCOOLING_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["model"]["MAE"])
