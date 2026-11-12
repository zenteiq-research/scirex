# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Car-CFD GIWO tutorial script on the CPU and check it against the
benchmark's own GIWO.

The page checks are in ``test_tutorial_pages.py``. Here the script runs on a
small fake dataset in the real file layout (``carcfd_tutorial_helpers``), and
its model, loss and optimiser steps are compared with
``scripts/benchmarks/car_cfd`` for the same cars and the same weights. The
benchmark's GIWO computes its 3-D wavelet transform with a separable
implementation; the tutorial uses the wavelet library's own one, so the two
agree to float32 rounding. The test that reads the real dataset runs only
when ``CARCFD_DATA_DIR`` points at the extracted
``processed-car-pressure-data`` folder.
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
    return load_script("carcfd_giwo")


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("car_cfd")
    fake_car_cfd(root)
    return root


def _bench_run(size: dict):
    """The benchmark's giwo_val run at ``size``."""
    from scripts.benchmarks.car_cfd._config import RUNS

    run = RUNS["giwo_val"]
    return replace(
        run,
        sdf_query_resolution=size["latent_res"],
        model={
            **run.model,
            "fno_hidden_channels": size["width"],
            "wno_level": size["wno_level"],
            "wno_n_layers": size["wno_n_layers"],
            "wno_projection_hidden_channels": size["wno_projection_hidden_channels"],
            "out_gno_radius": size["out_gno_radius"],
            "out_gno_channel_mlp_hidden_layers": list(
                size["out_gno_channel_mlp_hidden_layers"]
            ),
        },
    )


def test_full_size_and_protocol_match_the_benchmark(tutorial) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.car_cfd._config import PROTOCOL, RUNS, build_config
    from scripts.benchmarks.car_cfd._giwo import GIWO
    from scripts.benchmarks.car_cfd._models import build_module

    p, run, size = tutorial.PROTOCOL, RUNS["giwo_val"], tutorial.SIZES["full"]
    assert (p["n_train"], p["n_test"], p["n_epochs"]) == (
        PROTOCOL["n_train"],
        PROTOCOL["n_test"],
        run.n_epochs,
    )
    assert p["learning_rate"] == run.learning_rate
    assert run.schedule == dict(kind="warmup_cosine", warmup_epochs=1, end_factor=0.0)
    assert (p["weight_decay"], p["adam_b2"], p["adam_eps"]) == (
        run.weight_decay,
        run.adam_b2,
        run.adam_eps,
    )
    assert size["latent_res"] == run.sdf_query_resolution
    assert size["width"] == run.model["fno_hidden_channels"]
    assert size["wno_level"] == run.model["wno_level"]
    defaults = GIWO.__dataclass_fields__
    assert size["wno_n_layers"] == defaults["wno_n_layers"].default
    assert (
        size["wno_projection_hidden_channels"]
        == defaults["wno_projection_hidden_channels"].default
    )

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
    assert count_model_params(ours) == 74_197_232


def test_model_loss_and_optimiser_match_the_benchmark(tutorial, fake_root) -> None:
    import optax

    from scirex.losses import LpLoss
    from scripts.benchmarks.car_cfd._config import build_config
    from scripts.benchmarks.car_cfd._data import load_data_module
    from scripts.benchmarks.car_cfd._models import build, model_inputs

    size = tutorial.SIZES["smoke"]
    data = tutorial.load_cars(
        fake_root / "processed-car-pressure-data", 4, 2, size["latent_res"]
    )
    tutorial.attach_neighbors(
        data["train"] + data["test"], data["latent_queries"], size["out_gno_radius"]
    )
    run = _bench_run(size)
    dm = load_data_module(fake_root, size["latent_res"], 4, 2, download=False)
    model_b, proc = build(run, build_config(run, 4, 2), dm, fake_root / "logs")
    batch = proc.preprocess(next(iter(dm.train_loader(batch_size=1, shuffle=False))))
    model_b.init(batch)
    params = model_b.params
    model_t, params_t = tutorial.build_model(
        size, data["train"][0], data["latent_queries"], seed=0
    )
    shapes = jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), params)
    assert shapes == jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), params_t)

    car = data["train"][0]
    inputs, press = (
        tutorial.gino_inputs(car, data["latent_queries"]),
        jnp.asarray(car["press"]),
    )
    out_b = model_b.module.apply(params, **model_inputs(batch))
    assert max_diff(out_b, model_t.apply(params, **inputs)) < 1e-6
    loss_b = LpLoss(d=2, p=2)(out_b, jnp.asarray(batch["y"]))
    loss_fn = tutorial.make_loss(model_t)
    assert float(loss_fn(params, inputs, press)) == pytest.approx(
        float(loss_b), rel=1e-5
    )

    tx, _ = tutorial.make_optimizer(tutorial.PROTOCOL, 4, 3)
    state, p_t = tx.init(params), params
    for _ in range(3):
        grads = jax.grad(loss_fn)(p_t, inputs, press)
        updates, state = tx.update(grads, state, p_t)
        p_t = optax.apply_updates(p_t, updates)
        model_b.train_step(batch, LpLoss(d=2, p=2))
    for a, b in zip(
        jax.tree_util.tree_leaves(model_b.params), jax.tree_util.tree_leaves(p_t)
    ):
        assert max_diff(a, b) < 1e-4


def test_lifting_and_fno_layers_own_no_parameters(tutorial, fake_root) -> None:
    size = tutorial.SIZES["smoke"]
    data = tutorial.load_cars(
        fake_root / "processed-car-pressure-data", 2, 1, size["latent_res"]
    )
    tutorial.attach_neighbors(
        data["train"], data["latent_queries"], size["out_gno_radius"]
    )
    _, params = tutorial.build_model(size, data["train"][0], data["latent_queries"], 0)
    assert set(params["params"]) == {"wno", "gno_out", "projection"}


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
