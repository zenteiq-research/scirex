# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Darcy WNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on small synthetic ``.mat`` files, and the
script is checked against the benchmark: the same size and recipe, and the
same prediction and loss for the same weights. The test that reads the real
dataset runs only when ``DARCY_DATA_DIR`` points at the two ``.mat`` files.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances
from tests.docs.darcy_tutorial_helpers import benchmark_split, fake_darcy, load_script


def test_load_darcy_takes_the_benchmark_split(tmp_path: Path) -> None:
    from scipy.io import loadmat

    tutorial = load_script("wno")
    data = tutorial.load_darcy(fake_darcy(tmp_path), n_train=6, n_test=3)
    raw = loadmat(tmp_path / "piececonst_r421_N1024_smooth2.mat")
    assert data["a_train"].shape == (6, 22, 22)
    assert data["u_test"].shape == (3, 22, 22)
    np.testing.assert_allclose(
        data["u_test"], raw["sol"][:3, ::5, ::5], rtol=tolerances.rtol(1e-6)
    )
    with pytest.raises(ValueError, match="only 12 samples"):
        tutorial.load_darcy(tmp_path, n_train=13, n_test=1)


def test_pixel_normalizer_round_trip() -> None:
    tutorial = load_script("wno")
    v = np.random.default_rng(2).normal(3.0, 2.0, size=(6, 5, 5, 1)).astype(np.float32)
    norm = tutorial.PixelNormalizer(v)
    np.testing.assert_allclose(
        norm.encode(v).mean(axis=0), 0.0, atol=tolerances.atol(1e-5)
    )
    np.testing.assert_allclose(
        norm.decode(norm.encode(v)),
        v,
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )


def test_rel_l2_matches_definition() -> None:
    tutorial = load_script("wno")
    true = jnp.array([[3.0, 4.0]])
    assert float(tutorial.rel_l2_per_sample(true, true)[0]) == 0.0
    assert float(
        tutorial.rel_l2_per_sample(jnp.zeros((1, 2)), true)[0]
    ) == pytest.approx(1.0)


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scripts.benchmarks.darcy._config import PROTOCOL, RUNS

    tutorial = load_script("wno")
    run = RUNS["wno"]
    assert tutorial.SIZES["full"] == run.arch
    assert tutorial.SEED == run.seed
    for key in ("train_file", "test_file", "n_train", "n_test", "subsample_rate"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    r = tutorial.RECIPE
    assert run.opt["optimizer"] == "adamw" and run.opt["schedule"] == "step"
    assert r["learning_rate"] == run.opt["lr"]
    for key in ("weight_decay", "decay_epochs", "decay_rate", "batch_size", "epochs"):
        assert r[key] == run.opt[key], key
    assert run.opt["grad_clip"] is None and run.opt["grad_loss_weight"] == 0.0


def test_prediction_and_loss_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy._config import RUNS
    from scripts.benchmarks.darcy._data import batched_rel_l2_loss
    from scripts.benchmarks.darcy._models import prepare

    tutorial = load_script("wno")
    split = benchmark_split()
    prep = prepare("wno", RUNS["wno"].arch, split)
    a_norm = tutorial.PixelNormalizer(split["train_a"][..., None])
    u_norm = tutorial.PixelNormalizer(split["train_u"][..., None])
    x = a_norm.encode(split["train_a"][..., None])
    np.testing.assert_array_equal(x, prep.train_x)
    model, params, _ = tutorial.build_model(tutorial.SIZES["full"], x[0], seed=0)
    assert count_model_params(params) == 2_829_601
    weights = prep.init(jax.random.PRNGKey(0))
    ours = u_norm.decode(model.apply(weights, jnp.asarray(x))).reshape(4, -1)
    theirs = prep.predict(weights, jnp.asarray(prep.train_x))
    np.testing.assert_allclose(ours, theirs, rtol=tolerances.rtol(1e-6), atol=1e-9)
    u = jnp.asarray(split["train_u"][..., None])
    loss = tutorial.make_loss(model, u_norm)(weights, jnp.asarray(x), u)
    assert float(loss) == pytest.approx(
        float(batched_rel_l2_loss(theirs, u.reshape(4, -1))), rel=1e-6
    )


def test_optimizer_is_plain_adamw_on_a_step_schedule() -> None:
    tutorial = load_script("wno")
    tx = tutorial.make_optimizer(tutorial.RECIPE, steps_per_epoch=5)
    params = dict(b=jnp.ones(3))
    state = tx.init(params)
    assert float(state.hyperparams["weight_decay"]) == pytest.approx(1e-6)
    updates, _ = tx.update(
        jax.grad(lambda p: jnp.sum(p["b"] ** 2))(params), state, params
    )
    assert updates["b"].dtype == params["b"].dtype
    assert bool(jnp.all(updates["b"] < 0))


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_darcy(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "2",
            "--epochs", "2",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
    assert metrics["worst"] >= metrics["median"]
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("DARCY_DATA_DIR"),
    reason="set DARCY_DATA_DIR to the folder with the real .mat files",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "20", "--n-test", "4",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
