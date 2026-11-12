# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the chip cooling GINO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on a small synthetic dataset in the real file
layout, and the script is checked against the benchmark's own code: the same
graphs, latent features, targets, size and recipe, and the same prediction
and loss for the same weights. The test that reads the real dataset runs only
when ``CHIPCOOLING_DATA_DIR`` points at it.
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

NTRAIN, NTEST = 6, 3


def _small_arch(tutorial) -> dict:
    return dict(
        tutorial.SIZES["smoke"],
        latent_h=8,
        latent_w=16,
        in_gno_radius=0.15,
        out_gno_radius=0.15,
        in_gno_channel_mlp_hidden_layers=(8,),
        out_gno_channel_mlp_hidden_layers=(8,),
        fno_n_modes=(4, 4),
        fno_hidden_channels=4,
        fno_n_layers=1,
    )


def _setup(tmp_path: Path):
    from scripts.benchmarks.chip_cooling._data import mesh_split
    from scripts.benchmarks.chip_cooling._models import get_spec

    tutorial = load_script("gino")
    folder = fake_chip_cooling(tmp_path / "data", n_nodes=600)
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    norm = tutorial.normalise(data)
    arch = _small_arch(tutorial)
    geometry = tutorial.build_geometry(norm["nodes"], arch)
    lf = dict(
        train=tutorial.latent_features(norm["p_train"], arch),
        test=tutorial.latent_features(norm["p_test"], arch),
    )
    split = mesh_split(benchmark_raw(folder), dict(ntrain=NTRAIN, ntest=NTEST))
    prepared = get_spec("gino").prepare(arch, split)
    return tutorial, data, norm, arch, geometry, lf, prepared


def test_load_takes_the_benchmark_split(tmp_path: Path) -> None:
    tutorial = load_script("gino")
    folder = fake_chip_cooling(tmp_path / "data")
    data = tutorial.load_chip_cooling(folder, NTRAIN, NTEST)
    raw = benchmark_raw(folder)
    np.testing.assert_array_equal(data["params_test"], raw["parameters"][-NTEST:])
    np.testing.assert_array_equal(data["T_train"], raw["fields"][:NTRAIN])


def test_graphs_features_and_targets_match_the_benchmark(tmp_path: Path) -> None:
    _tutorial, _, norm, _, geometry, lf, prepared = _setup(tmp_path)
    ref = prepared["geometry"]
    for key in ("input_geom", "latent_queries", "output_queries"):
        np.testing.assert_allclose(geometry[key], ref[key], atol=1e-7)
    for graph in ("neighbors_in", "neighbors_out"):
        for key in ("neighbors_index", "segment_ids", "counts"):
            np.testing.assert_array_equal(geometry[graph][key], ref[graph][key])
    assert len(geometry["neighbors_in"]["neighbors_index"]) > 0
    np.testing.assert_allclose(
        lf["train"], prepared["lf_train"], atol=tolerances.atol(1e-6)
    )
    np.testing.assert_allclose(
        lf["test"], prepared["lf_test"], atol=tolerances.atol(1e-6)
    )
    for i in range(NTEST):
        np.testing.assert_allclose(
            norm["y_test"][i],
            prepared["y_test"][i][0, :, 0],
            atol=tolerances.atol(1e-5),
        )


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.chip_cooling._config import PROTOCOL, RUNS

    tutorial = load_script("gino")
    run = RUNS["gino"]
    assert tutorial.SIZES["full"] == run.arch
    assert tutorial.SEED == PROTOCOL["seed"]
    for key in ("ntrain", "ntest", "epochs"):
        assert tutorial.PROTOCOL[key] == PROTOCOL[key], key
    assert run.opt["optimizer"] == "adamw_cx" and run.opt["batch_size"] == 1
    assert tutorial.PROTOCOL["learning_rate"] == run.opt["lr"]
    assert tutorial.PROTOCOL["weight_decay"] == run.opt["weight_decay"]
    rng = np.random.default_rng(0)
    nodes = rng.random((300, 2)).astype(np.float32)
    arch = tutorial.SIZES["full"]
    geometry = tutorial.build_geometry(nodes, arch)
    lf = tutorial.latent_features(rng.normal(size=(1, 6)).astype(np.float32), arch)
    _, params = tutorial.build_model(arch, geometry, lf, 0)
    assert count_model_params(params) == 1_821_028


def test_prediction_and_loss_match_the_benchmark(tmp_path: Path) -> None:
    from scripts.benchmarks.chip_cooling._models import get_spec
    from scripts.benchmarks.chip_cooling.train import relative_l2

    tutorial, _, norm, arch, geometry, lf, prepared = _setup(tmp_path)
    model, params = tutorial.build_model(arch, geometry, lf["train"][:1], seed=42)
    spec = get_spec("gino")
    ref_model = spec.build(arch, None)
    ref_params = spec.init(ref_model, arch, prepared, 42)
    jax.tree_util.tree_map(np.testing.assert_array_equal, params, ref_params)
    pred = model.apply(params, **geometry, latent_features=lf["test"][:1])
    ref_pred = ref_model.apply(params, **spec.inputs(prepared, prepared["lf_test"][:1]))
    np.testing.assert_allclose(
        pred, ref_pred, rtol=tolerances.rtol(1e-5), atol=tolerances.atol(1e-5)
    )
    loss = tutorial.make_loss(model, geometry)(
        params, lf["test"][:1], norm["y_test"][0]
    )
    expected = relative_l2(ref_pred, jnp.asarray(prepared["y_test"][0]))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)


def test_optimizer_keeps_dtypes_and_cosine_matches_the_benchmark() -> None:
    tutorial = load_script("gino")
    tx = tutorial.make_optimizer(tutorial.PROTOCOL)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))
    state = tx.init(params)
    assert all(not jnp.iscomplexobj(v) for v in state.hyperparams.values())
    grads = jax.grad(lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2))(
        params
    )
    updates, _ = tx.update(grads, state, params)
    assert updates["b"].dtype == jnp.float32
    assert complex(updates["w"][0]).imag < 0  # conjugated: towards 0
    for epoch in (1, 250, 500):  # stepped after each epoch, as in train.py
        expected = 1e-3 * (1 + math.cos(math.pi * epoch / 500)) / 2
        assert tutorial.cosine_lr(tutorial.PROTOCOL, epoch, 500) == pytest.approx(
            expected
        )


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("gino")
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_chip_cooling(tmp_path / "data", n_nodes=600)),
            "--smoke",
            "--n-train", "3",
            "--n-test", "2",
            "--epochs", "1",
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
    tutorial = load_script("gino")
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "4", "--n-test", "2",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["CHIPCOOLING_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["model"]["MAE"])
