# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Car-CFD Transolver tutorial script on the CPU and check it against
the benchmark's own code.

The page checks are in ``test_tutorial_pages.py``. Here the script runs on a
small fake dataset in the real file layout (``carcfd_tutorial_helpers``), and
its inputs, model, loss and optimiser steps are compared with
``scripts/benchmarks/car_cfd`` for the same cars and the same weights. The
test that reads the real dataset runs only when ``CARCFD_DATA_DIR`` points at
the extracted ``processed-car-pressure-data`` folder.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")

import jax
import jax.numpy as jnp

from tests.docs.carcfd_tutorial_helpers import fake_car_cfd, load_script, max_diff


@pytest.fixture(scope="module")
def tutorial():
    return load_script("carcfd_transolver")


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("car_cfd")
    fake_car_cfd(root)
    return root


@pytest.fixture(scope="module")
def both(tutorial, fake_root):
    """The tutorial's split and the benchmark's data module, model and processor."""
    from scripts.benchmarks.car_cfd._config import RUNS, build_config
    from scripts.benchmarks.car_cfd._data import load_data_module
    from scripts.benchmarks.car_cfd._models import build

    data = tutorial.load_cars(fake_root / "processed-car-pressure-data", 4, 2)
    run = replace(
        RUNS["transolver_val"], model=dict(tutorial.SIZES["smoke"]), n_epochs=3
    )
    dm = load_data_module(fake_root, run.sdf_query_resolution, 4, 2, download=False)
    model_b, proc = build(run, build_config(run, 4, 2), dm, fake_root / "logs")
    return data, dm, model_b, proc


def test_full_size_and_protocol_match_the_benchmark(tutorial) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.car_cfd._config import PROTOCOL, RUNS
    from scripts.benchmarks.car_cfd._models import build_module

    p, run = tutorial.PROTOCOL, RUNS["transolver_val"]
    assert tutorial.SIZES["full"] == run.model
    assert (p["n_train"], p["n_test"], p["n_epochs"]) == (
        PROTOCOL["n_train"],
        PROTOCOL["n_test"],
        run.n_epochs,
    )
    assert p["learning_rate"] == run.learning_rate
    assert run.schedule == dict(kind="warmup_cosine", warmup_epochs=1, end_factor=0.0)
    assert p["warmup_epochs"] == run.schedule["warmup_epochs"]
    assert (p["weight_decay"], p["adam_b2"], p["adam_eps"]) == (
        run.weight_decay,
        run.adam_b2,
        run.adam_eps,
    )
    assert run.batch_size == 1 and run.grad_clip is None

    x = jnp.zeros((1, 3586, 6))
    ours = jax.eval_shape(
        lambda: tutorial.build_model(
            tutorial.SIZES["full"],
            dict(vertices=np.zeros((3586, 3)), normals=np.zeros((3586, 3))),
            seed=0,
        )[1]
    )
    theirs = jax.eval_shape(
        lambda: build_module(run, None).init(jax.random.PRNGKey(0), x=x)
    )
    shapes = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), ours)
    assert shapes == jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), theirs)
    assert count_model_params(ours) == 75_481_841


def test_inputs_match_the_benchmark(tutorial, both) -> None:
    data, dm, _, proc = both
    bench = dm.train_data.data_list + dm.test_data.data_list
    for b, car in zip(bench, data["train"] + data["test"]):
        sample = {
            k: np.asarray(v)[None] for k, v in b.items() if not isinstance(v, dict)
        }
        batch = proc.preprocess(sample)
        np.testing.assert_array_equal(
            batch["x"], np.asarray(tutorial.transolver_input(car))
        )
        assert max_diff(batch["y"].ravel(), car["press"]) < 1e-5


def test_model_loss_and_optimiser_match_the_benchmark(tutorial, both) -> None:
    import optax

    from scirex.losses import LpLoss

    data, dm, model_b, proc = both
    batch = proc.preprocess(next(iter(dm.train_loader(batch_size=1, shuffle=False))))
    model_b.init(batch)
    params = model_b.params
    model_t, params_t = tutorial.build_model(
        tutorial.SIZES["smoke"], data["train"][0], 0
    )
    # Same seed, same initial weights.
    for a, b in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(params_t)
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    car = data["train"][0]
    x, press = tutorial.transolver_input(car), jnp.asarray(car["press"])
    out_b = model_b.module.apply(params, x=jnp.asarray(batch["x"]))
    np.testing.assert_array_equal(
        np.asarray(out_b), np.asarray(model_t.apply(params, x=x))
    )
    loss_b = LpLoss(d=2, p=2)(out_b, jnp.asarray(batch["y"]))
    loss_fn = tutorial.make_loss(model_t)
    assert float(loss_fn(params, x, press)) == pytest.approx(float(loss_b), rel=1e-5)

    tx, schedule = tutorial.make_optimizer(tutorial.PROTOCOL, 4, 3)
    state, p_t = tx.init(params), params
    for _ in range(3):
        grads = jax.grad(loss_fn)(p_t, x, press)
        updates, state = tx.update(grads, state, p_t)
        p_t = optax.apply_updates(p_t, updates)
        model_b.train_step(batch, LpLoss(d=2, p=2))
    for a, b in zip(
        jax.tree_util.tree_leaves(model_b.params), jax.tree_util.tree_leaves(p_t)
    ):
        assert max_diff(a, b) < 1e-5
    lr_b = float(model_b.opt_state.hyperparams["learning_rate"])
    assert float(state.hyperparams["learning_rate"]) == pytest.approx(lr_b)
    assert float(schedule(4)) == pytest.approx(tutorial.PROTOCOL["learning_rate"])


def test_rel_l2_matches_definition(tutorial) -> None:
    true = jnp.array([3.0, 4.0])
    assert float(tutorial.rel_l2(true, true)) == 0.0
    assert float(tutorial.rel_l2(jnp.zeros(2), true)) == pytest.approx(1.0, abs=1e-6)


def test_smoke_main_on_fake_data(tutorial, fake_root, tmp_path: Path) -> None:
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_root / "processed-car-pressure-data"),
            "--smoke",
            "--n-train", "3",
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
    not os.environ.get("CARCFD_DATA_DIR"),
    reason="set CARCFD_DATA_DIR to the real processed-car-pressure-data folder",
)
def test_smoke_main_on_real_data(tutorial, tmp_path: Path) -> None:
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["CARCFD_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
