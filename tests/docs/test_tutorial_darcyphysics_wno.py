# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the physics-only Darcy WNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
physics is checked against the benchmark's own code, its full size and
recipe against ``train_wno.py``, and its ``main`` runs on small synthetic
``.mat`` files. The test that reads the real dataset runs only when
``DARCY_DATA_DIR`` points at the folder with the two files.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances
from tests.docs.darcyphysics_tutorial_helpers import (
    check_physics_matches_benchmark,
    fake_darcy,
    load_script,
)


def test_physics_matches_the_benchmark() -> None:
    check_physics_matches_benchmark(load_script("wno"))


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy_physics import train_wno as bench

    tutorial = load_script("wno")
    sample = jnp.zeros((1, 85, 85, 1))
    params = jax.eval_shape(
        lambda: bench.build_model(85).init(jax.random.PRNGKey(0), sample)
    )
    model = tutorial.WNO(
        in_channels=1,
        out_channels=1,
        size=(85, 85),
        use_grid=True,
        compute_dtype=jnp.float32,
        **tutorial.SIZES["full"],
    )
    ours = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), sample))
    assert count_model_params(ours) == count_model_params(params) == 2_829_601
    r = tutorial.RECIPE
    assert (r["epochs"], r["batch_size"], r["lr_decay_epochs"]) == (4000, 20, 400)
    assert (r["learning_rate"], r["lr_decay_rate"], r["weight_decay"]) == (
        1e-3,
        0.5,
        1e-6,
    )
    assert r["adam_eps"] == 1e-8


def test_prediction_and_loss_match_the_benchmark() -> None:
    """Same weights, inputs and cells: same nodal prediction and Ritz energy as
    ``train_wno.py``."""
    from scirex.losses import Q4Galerkin
    from scirex.losses.darcy_piecewise_coefficient import (
        BatchedEnergyFunctionalLossQ4,
        kappa_fn,
        source_fn,
    )
    from scripts.benchmarks.darcy_physics import train_wno as bench

    tutorial = load_script("wno")
    size = tutorial.SIZES["smoke"]
    xy, quad, _ = tutorial.build_q4_mesh(21)
    with jax.default_matmul_precision("highest"):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    rng = np.random.default_rng(0)
    a_enc = jnp.asarray(rng.normal(size=(2, 22, 22, 1)), dtype=jnp.float32)
    cells = jnp.asarray(np.where(rng.normal(size=(2, 21, 21)) > 0, 12.0, 3.0))
    factor = jnp.asarray(tutorial.sin_boundary_factor(xy), dtype=jnp.float32)
    model, params = tutorial.build_model(size, np.asarray(a_enc[0]), seed=0)

    ours = tutorial.make_predict_nodes(model, 0.004, 0.006, factor)
    theirs = bench.make_predict_nodes(model, 0.004, 0.006, factor)
    np.testing.assert_allclose(
        ours(params, a_enc), theirs(params, a_enc), rtol=tolerances.rtol(1e-6)
    )

    loss = tutorial.make_loss(ours, tutorial.make_ritz_energy(galerkin))
    bench_loss = bench.make_loss_fn(
        theirs, BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    )
    np.testing.assert_allclose(
        float(loss(params, a_enc, cells)),
        float(bench_loss(params, a_enc, cells)),
        rtol=tolerances.rtol(1e-5),
    )


def test_optimizer_is_plain_adamw_with_real_hyperparameters() -> None:
    tutorial = load_script("wno")
    tx = tutorial.make_optimizer(tutorial.RECIPE, steps_per_epoch=5)
    params = dict(w=jnp.ones((3,)), b=jnp.ones(1))
    state = tx.init(params)
    assert not any(jnp.iscomplexobj(v) for v in state.hyperparams.values())
    assert float(state.hyperparams["weight_decay"]) == pytest.approx(1e-6)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, state, params)
    assert all(float(u.max()) < 0 for u in jax.tree_util.tree_leaves(updates))


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(fake_darcy(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "4",
            "--epochs", "2",
            "--eval-every", "1",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
    assert np.isfinite(metrics["test_rel_l2_nodes"])
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("DARCY_DATA_DIR"),
    reason="set DARCY_DATA_DIR to the folder with the real .mat files",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = load_script("wno")
    metrics = tutorial.main(
        ["--smoke", "--n-train", "20", "--n-test", "4", "--epochs", "1",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
