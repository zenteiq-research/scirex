# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Darcy GINO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on small synthetic ``.mat`` files, and the
script is checked against the benchmark: the same size, recipe and neighbour
lists, and the same prediction and loss for the same weights. The test that
reads the real dataset runs only when ``DARCY_DATA_DIR`` points at the two
``.mat`` files.
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
    tutorial = load_script("gino")
    data = tutorial.load_darcy(fake_darcy(tmp_path), n_train=6, n_test=3)
    assert data["a_train"].shape == (6, 22, 22)
    assert data["u_test"].shape == (3, 22, 22)


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scripts.benchmarks.darcy._config import PROTOCOL, RUNS

    tutorial = load_script("gino")
    run = RUNS["gino"]
    assert tutorial.SIZES["full"] == run.arch
    assert tutorial.SEED == run.seed
    for key in ("train_file", "test_file", "n_train", "n_test", "subsample_rate"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    r = tutorial.RECIPE
    assert run.opt["optimizer"] == "adamw_cx" and run.opt["schedule"] == "onecycle"
    assert r["learning_rate"] == run.opt["lr"]
    assert r["grad_clip"] == run.opt["grad_clip"]
    for key in ("weight_decay", "grad_loss_weight", "batch_size", "epochs"):
        assert r[key] == run.opt[key], key


def test_geometry_matches_the_benchmark() -> None:
    from scripts.benchmarks.darcy._models import (
        _build_neighbors,
        build_grid_geometry,
        gno_radii,
    )

    tutorial = load_script("gino")
    n_side = 22
    assert tutorial.gno_radii(n_side) == gno_radii({}, n_side)
    geometry = tutorial.build_geometry(n_side)
    xy = build_grid_geometry(n_side)
    np.testing.assert_array_equal(np.asarray(geometry["input_geom"][0]), xy)
    for name, radius in zip(
        ("neighbors_in", "neighbors_out"), gno_radii({}, n_side), strict=True
    ):
        expected = _build_neighbors(xy, xy, radius)
        for key, value in expected.items():
            np.testing.assert_array_equal(np.asarray(geometry[name][key]), value)


def test_prediction_and_loss_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy._config import RUNS
    from scripts.benchmarks.darcy._data import (
        batched_rel_l2_loss,
        gradient_regularization_loss,
    )
    from scripts.benchmarks.darcy._models import prepare

    tutorial = load_script("gino")
    n_side, n = 85, 4
    split = benchmark_split(n_side, n_train=n)
    prep = prepare("gino", RUNS["gino"].arch, split)
    flat = {k: v.reshape(len(v), -1) for k, v in split.items()}
    a_norm = tutorial.PixelNormalizer(flat["train_a"])
    u_norm = tutorial.PixelNormalizer(flat["train_u"])
    x = a_norm.encode(flat["train_a"])
    np.testing.assert_array_equal(x, prep.train_x)
    geometry = tutorial.build_geometry(n_side)
    model, params, _ = tutorial.build_model(
        tutorial.SIZES["full"], geometry, n_side, seed=0
    )
    assert count_model_params(params) == 2_801_340
    weights = prep.init(jax.random.PRNGKey(0))
    out = model.apply(weights, x=jnp.asarray(x)[..., None], **geometry)[..., 0]
    theirs = prep.predict(weights, jnp.asarray(prep.train_x))
    np.testing.assert_allclose(
        u_norm.decode(out), theirs, rtol=tolerances.rtol(1e-6), atol=1e-9
    )
    u = jnp.asarray(flat["train_u"])
    loss_fn = tutorial.make_loss(model, geometry, u_norm, n_side, 0.1)
    expected = batched_rel_l2_loss(theirs, u) + 0.1 * gradient_regularization_loss(
        theirs.reshape(n, n_side, n_side, 1),
        u.reshape(n, n_side, n_side, 1),
        1 / n_side,
    )
    assert float(loss_fn(weights, jnp.asarray(x), u)) == pytest.approx(
        float(expected), rel=1e-6
    )


def test_optimizer_keeps_hyperparameters_real_and_updates_dtypes() -> None:
    tutorial = load_script("gino")
    tx = tutorial.make_optimizer(tutorial.RECIPE, total_steps=10)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))
    grads = jax.grad(lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2))(
        params
    )
    state = tx.init(params)
    assert all(not jnp.iscomplexobj(v) for v in state[1].hyperparams.values())
    updates, _ = tx.update(grads, state, params)
    assert updates["b"].dtype == params["b"].dtype
    step = complex(updates["w"][0])
    assert step.real < 0 and step.imag < 0  # towards 0, the minimum of |w|^2


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("gino")
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
    tutorial = load_script("gino")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "16", "--n-test", "4",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
