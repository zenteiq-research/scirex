# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Darcy Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on small synthetic ``.mat`` files, and the
script is checked against the benchmark: the same size, recipe, seed and
normalisation, and the same prediction and loss for the same weights. The
test that reads the real dataset runs only when ``DARCY_DATA_DIR`` points at
the two ``.mat`` files.
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
    tutorial = load_script("transolver")
    data = tutorial.load_darcy(fake_darcy(tmp_path), n_train=6, n_test=3)
    assert data["a_train"].shape == (6, 22, 22)
    assert data["u_test"].shape == (3, 22, 22)


def test_global_normalizer_is_one_scalar_per_field() -> None:
    tutorial = load_script("transolver")
    v = np.random.default_rng(3).normal(5.0, 2.0, size=(6, 49)).astype(np.float32)
    norm = tutorial.GlobalNormalizer(v)
    assert np.ndim(norm.mean) == 0 and np.ndim(norm.std) == 0
    assert float(norm.encode(v).mean()) == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(
        norm.decode(norm.encode(v)),
        v,
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scripts.benchmarks.darcy._config import PROTOCOL, RUNS

    tutorial = load_script("transolver")
    run = RUNS["transolver"]
    assert {**tutorial.SIZES["full"], "kernel": 3} == run.arch
    assert tutorial.SEED == run.seed == 2
    for key in ("train_file", "test_file", "n_train", "n_test", "subsample_rate"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    r = tutorial.RECIPE
    assert run.opt["optimizer"] == "adamw" and run.opt["schedule"] == "onecycle"
    assert r["learning_rate"] == run.opt["lr"]
    assert r["grad_clip"] == run.opt["grad_clip"]
    for key in ("weight_decay", "grad_loss_weight", "batch_size", "epochs"):
        assert r[key] == run.opt[key], key


def test_prediction_and_loss_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy._config import RUNS
    from scripts.benchmarks.darcy._data import (
        batched_rel_l2_loss,
        gradient_regularization_loss,
    )
    from scripts.benchmarks.darcy._models import prepare

    tutorial = load_script("transolver")
    n_side, n = 85, 4
    split = benchmark_split(n_side, n_train=n)
    prep = prepare("transolver", RUNS["transolver"].arch, split)
    flat = {k: v.reshape(len(v), -1) for k, v in split.items()}
    a_norm = tutorial.GlobalNormalizer(flat["train_a"])
    u_norm = tutorial.GlobalNormalizer(flat["train_u"])
    x = a_norm.encode(flat["train_a"])[..., None].astype(np.float32)
    np.testing.assert_allclose(
        x, prep.train_x, rtol=tolerances.rtol(1e-6), atol=tolerances.atol(1e-6)
    )
    model, params, _ = tutorial.build_model(tutorial.SIZES["full"], x[0], n_side, 2)
    assert count_model_params(params) == 2_826_945
    weights = prep.init(jax.random.PRNGKey(0))
    ours = u_norm.decode(tutorial.apply_model(model, weights, jnp.asarray(x)))
    theirs = prep.predict(weights, jnp.asarray(prep.train_x))
    np.testing.assert_allclose(ours, theirs, rtol=tolerances.rtol(1e-5), atol=1e-8)
    u = jnp.asarray(flat["train_u"])
    loss_fn = tutorial.make_loss(model, u_norm, n_side, 0.1)
    expected = batched_rel_l2_loss(theirs, u) + 0.1 * gradient_regularization_loss(
        theirs.reshape(n, n_side, n_side, 1),
        u.reshape(n, n_side, n_side, 1),
        1 / n_side,
    )
    assert float(loss_fn(weights, jnp.asarray(x), u)) == pytest.approx(
        float(expected), rel=1e-5
    )


def test_gradient_loss_is_zero_for_the_truth_with_a_zero_border() -> None:
    tutorial = load_script("transolver")
    s = np.linspace(0.0, 1.0, 16)
    u = jnp.asarray(np.outer(np.sin(np.pi * s), np.sin(np.pi * s))[None])
    u = u.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    u = u.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    assert float(tutorial.gradient_loss(u, u, 1 / 16)) == pytest.approx(0.0, abs=1e-6)


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("transolver")
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
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("DARCY_DATA_DIR"),
    reason="set DARCY_DATA_DIR to the folder with the real .mat files",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = load_script("transolver")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "8", "--n-test", "4",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
