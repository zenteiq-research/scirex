# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Car-CFD GINO tutorial script on the CPU and check it against the
benchmark's own code.

The page checks are in ``test_tutorial_pages.py``. Here the script runs on a
small fake dataset in the real file layout (``carcfd_tutorial_helpers``), and
its loader, neighbour lists, model, loss and optimiser step are compared with
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

from tests.docs.carcfd_tutorial_helpers import (
    fake_car_cfd,
    load_script,
    max_diff,
)

RES = 16  # latent grid of the comparisons (the smoke size's)


@pytest.fixture(scope="module")
def tutorial():
    return load_script("carcfd_gino")


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

    size = tutorial.SIZES["smoke"]
    data = tutorial.load_cars(fake_root / "processed-car-pressure-data", 4, 2, RES)
    tutorial.attach_neighbors(
        data["train"] + data["test"], data["latent_queries"], size["out_gno_radius"]
    )
    run = replace(
        RUNS["paper"],
        sdf_query_resolution=RES,
        model={
            **RUNS["paper"].model,
            "fno_n_modes": list(size["fno_n_modes"]),
            "fno_hidden_channels": size["fno_hidden_channels"],
            "fno_n_layers": size["fno_n_layers"],
            "out_gno_radius": size["out_gno_radius"],
            "out_gno_channel_mlp_hidden_layers": list(
                size["out_gno_channel_mlp_hidden_layers"]
            ),
        },
    )
    dm = load_data_module(fake_root, RES, 4, 2, download=False)
    model_b, proc = build(run, build_config(run, 4, 2), dm, fake_root / "logs")
    return data, dm, model_b, proc


def test_full_size_and_protocol_match_the_benchmark(tutorial) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.car_cfd._config import PROTOCOL, RUNS, build_config
    from scripts.benchmarks.car_cfd._models import build_module

    p, run = tutorial.PROTOCOL, RUNS["paper"]
    assert (p["n_train"], p["n_test"], p["n_epochs"]) == (
        PROTOCOL["n_train"],
        PROTOCOL["n_test"],
        run.n_epochs,
    )
    assert (p["learning_rate"], p["lr_step"], p["lr_gamma"]) == (
        run.learning_rate,
        run.lr_step,
        run.lr_gamma,
    )
    assert (p["weight_decay"], p["adam_b2"], p["adam_eps"]) == (
        run.weight_decay,
        run.adam_b2,
        run.adam_eps,
    )
    assert run.schedule is None and run.batch_size == 1 and run.grad_clip is None
    size = tutorial.SIZES["full"]
    assert size["latent_res"] == run.sdf_query_resolution

    r, n_edges = size["latent_res"], 1000
    car = dict(
        vertices=np.zeros((3586, 3), np.float32),
        sdf=np.zeros((r, r, r, 1), np.float32),
        neighbors_out=dict(
            neighbors_index=np.zeros(n_edges, np.int32),
            segment_ids=np.zeros(n_edges, np.int32),
            counts=np.zeros(3586, np.int32),
        ),
    )
    q = np.zeros((r, r, r, 3), np.float32)
    ours = jax.eval_shape(lambda: tutorial.build_model(size, car, q, seed=0)[1])
    theirs = jax.eval_shape(
        lambda: build_module(run, build_config(run, 500, 111)).init(
            jax.random.PRNGKey(0), **tutorial.gino_inputs(car, q)
        )
    )
    shapes = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), ours)
    assert shapes == jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), theirs)
    assert count_model_params(ours) == 75_917_953


def test_loader_and_neighbours_match_the_benchmark(both) -> None:
    data, dm, _, _ = both
    np.testing.assert_array_equal(
        np.asarray(dm.train_data.constant["query_points"]), data["latent_queries"]
    )
    bench = dm.train_data.data_list + dm.test_data.data_list
    assert len(bench) == len(data["train"]) + len(data["test"]) == 6
    for b, car in zip(bench, data["train"] + data["test"]):
        np.testing.assert_array_equal(np.asarray(b["vertices"]), car["vertices"])
        np.testing.assert_array_equal(np.asarray(b["distance"]), car["sdf"])
        # The benchmark's normaliser streams the statistics in float32.
        assert max_diff(np.asarray(b["press"]).ravel(), car["press"]) < 1e-5
        for key in ("neighbors_index", "segment_ids", "counts"):
            np.testing.assert_array_equal(
                np.asarray(b["neighbors_out"][key]), car["neighbors_out"][key]
            )
    norm = dm.normalizers["press"]
    assert data["press_mean"] == pytest.approx(float(norm.mean.ravel()[0]), rel=1e-6)
    assert data["press_scale"] == pytest.approx(
        float(norm.std.ravel()[0] + norm.eps), rel=1e-6
    )


def test_model_loss_and_metric_match_the_benchmark(tutorial, both) -> None:
    from scirex.losses import LpLoss
    from scripts.benchmarks.car_cfd._models import model_inputs

    data, dm, model_b, proc = both
    sample = next(iter(dm.test_loader(batch_size=1, shuffle=False)))
    batch = proc.preprocess(sample)
    model_b.init(batch)
    model_t, params_t = tutorial.build_model(
        tutorial.SIZES["smoke"], data["train"][0], data["latent_queries"], seed=0
    )
    shapes = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), model_b.params)
    assert shapes == jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), params_t)

    car = data["test"][0]
    inputs = tutorial.gino_inputs(car, data["latent_queries"])
    out_b = model_b.module.apply(model_b.params, **model_inputs(batch))
    out_t = model_t.apply(model_b.params, **inputs)
    np.testing.assert_array_equal(np.asarray(out_b), np.asarray(out_t))

    loss_b = LpLoss(d=2, p=2)(out_b, jnp.asarray(batch["y"]))
    loss_t = tutorial.make_loss(model_t)(
        model_b.params, inputs, jnp.asarray(car["press"])
    )
    assert float(loss_t) == pytest.approx(float(loss_b), rel=1e-5)

    proc.training = False
    pred_b, sample_b = proc.postprocess(out_b, dict(batch))
    metric_b = LpLoss(d=2, p=2)(pred_b, sample_b["y"])
    metric_t = tutorial.evaluate(
        tutorial.make_predict(model_t), model_b.params, dict(data, test=[car])
    )["test_rel_l2"]
    assert metric_t == pytest.approx(float(metric_b), rel=1e-5)


def test_optimiser_matches_the_benchmark_and_keeps_dtypes(tutorial, both) -> None:
    """Two Adam steps from the same weights give the same weights; the
    injected learning rate stays real next to the complex spectral weights."""
    import optax

    from scirex.losses import LpLoss

    data, dm, model_b, proc = both
    sample = next(iter(dm.train_loader(batch_size=1, shuffle=False)))
    batch = proc.preprocess(sample)
    model_b.init(batch)
    params = model_b.params
    assert any(jnp.iscomplexobj(a) for a in jax.tree_util.tree_leaves(params))

    model_t, _ = tutorial.build_model(
        tutorial.SIZES["smoke"], data["train"][0], data["latent_queries"], seed=0
    )
    car = data["train"][0]
    inputs = tutorial.gino_inputs(car, data["latent_queries"])
    tx = tutorial.make_optimizer(tutorial.PROTOCOL)
    state = tx.init(params)
    assert all(
        not jnp.iscomplexobj(v) for v in state.hyperparams.values()
    ), "injected hyperparameters must stay real"
    loss_fn = tutorial.make_loss(model_t)
    p_t = params
    for _ in range(2):
        grads = jax.grad(loss_fn)(p_t, inputs, jnp.asarray(car["press"]))
        updates, state = tx.update(grads, state, p_t)
        for u, w in zip(
            jax.tree_util.tree_leaves(updates), jax.tree_util.tree_leaves(p_t)
        ):
            assert u.dtype == w.dtype
        p_t = optax.apply_updates(p_t, updates)
        model_b.train_step(batch, LpLoss(d=2, p=2))
    for a, b in zip(
        jax.tree_util.tree_leaves(model_b.params), jax.tree_util.tree_leaves(p_t)
    ):
        assert max_diff(a, b) < 1e-5


def test_step_lr_halves_at_epoch_50(tutorial) -> None:
    p = tutorial.PROTOCOL
    assert tutorial.step_lr(p, 0) == tutorial.step_lr(p, 49) == p["learning_rate"]
    assert tutorial.step_lr(p, 50) == pytest.approx(p["learning_rate"] / 2)


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
