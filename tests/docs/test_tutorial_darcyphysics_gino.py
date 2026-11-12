# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the physics-only Darcy GINO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
physics is checked against the benchmark's own code, its full size and
recipe against ``train_gino.py``, and its ``main`` runs on small synthetic
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
    check_physics_matches_benchmark(load_script("gino"))


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy_physics import train_gino as bench

    tutorial = load_script("gino")
    model, kwargs, r_in, r_out = bench.build_model(85)
    x0 = jnp.zeros((1, 85 * 85, 1))
    theirs = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), x=x0, **kwargs))
    xy, _, _ = tutorial.build_q4_mesh(84)
    geometry = tutorial.build_geometry(xy, 85)
    _, ours = tutorial.build_model(tutorial.SIZES["full"], geometry, 85, seed=0)
    assert count_model_params(ours) == count_model_params(theirs) == 2_801_340
    assert tutorial.gno_radii(85) == pytest.approx((r_in, r_out))
    r = tutorial.RECIPE
    assert (r["epochs"], r["batch_size"], r["grad_clip"]) == (2000, 16, 1.0)
    assert (r["learning_rate"], r["weight_decay"], r["adam_eps"]) == (1e-3, 1e-4, 1e-8)


def test_prediction_and_loss_match_the_benchmark() -> None:
    """Same weights, inputs and cells: same nodal prediction and Ritz energy as
    ``train_gino.py``, each with its own geometry."""
    from scirex.losses import Q4Galerkin
    from scirex.losses.darcy_piecewise_coefficient import (
        BatchedEnergyFunctionalLossQ4,
        kappa_fn,
        source_fn,
    )
    from scripts.benchmarks.darcy_physics import train_gino as bench

    tutorial = load_script("gino")
    xy, quad, _ = tutorial.build_q4_mesh(21)
    with jax.default_matmul_precision("highest"):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    rng = np.random.default_rng(0)
    a_enc = jnp.asarray(rng.normal(size=(2, 22 * 22)), dtype=jnp.float32)
    cells = jnp.asarray(np.where(rng.normal(size=(2, 21, 21)) > 0, 12.0, 3.0))
    factor = jnp.asarray(tutorial.sin_boundary_factor(xy), dtype=jnp.float32)
    geometry = tutorial.build_geometry(xy, 22)
    model, params = tutorial.build_model(tutorial.SIZES["full"], geometry, 22, seed=0)
    bench_model, bench_kwargs, _, _ = bench.build_model(22)

    ours = tutorial.make_predict_nodes(model, geometry, 0.004, 0.006, factor)
    theirs = bench.make_predict_nodes(bench_model, bench_kwargs, 0.004, 0.006, factor)
    np.testing.assert_allclose(
        ours(params, a_enc),
        theirs(params, a_enc),
        rtol=tolerances.rtol(1e-5),
        atol=1e-9,
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


def test_optimizer_clips_and_keeps_hyperparameters_real() -> None:
    tutorial = load_script("gino")
    tx = tutorial.make_optimizer(tutorial.RECIPE, total_steps=10)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))
    state = tx.init(params)
    hyper = state[1].hyperparams
    assert not any(jnp.iscomplexobj(v) for v in hyper.values())
    grads = jax.grad(lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2))(
        params
    )
    updates, _ = tx.update(grads, state, params)
    assert updates["b"].dtype == params["b"].dtype
    step = complex(updates["w"][0])
    assert step.real < 0 and step.imag < 0  # descends towards 0


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("gino")
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
    tutorial = load_script("gino")
    metrics = tutorial.main(
        ["--smoke", "--n-train", "20", "--n-test", "4", "--epochs", "1",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
