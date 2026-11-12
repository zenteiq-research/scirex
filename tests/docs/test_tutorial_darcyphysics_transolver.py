# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the physics-only Darcy Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
physics is checked against the benchmark's own code, its full size and
recipe against ``train_transolver.py``, and its ``main`` runs on small synthetic
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
    check_physics_matches_benchmark(load_script("transolver"))


def test_full_size_and_recipe_match_the_benchmark() -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.darcy_physics import train_transolver as bench

    tutorial = load_script("transolver")
    n = 85 * 85
    theirs = jax.eval_shape(
        lambda: bench.build_model(85).init(
            jax.random.PRNGKey(0), x=jnp.zeros((1, n, 2)), fx=jnp.zeros((1, n, 1))
        )
    )
    _, ours = tutorial.build_model(tutorial.SIZES["full"], 85, seed=0)
    assert count_model_params(ours) == count_model_params(theirs) == 2_826_945
    r = tutorial.RECIPE
    assert (r["epochs"], r["batch_size"], r["max_grad_norm"]) == (500, 4, 0.1)
    assert (r["learning_rate"], r["weight_decay"]) == (1e-3, 1e-5)


def test_normalizer_matches_the_benchmark() -> None:
    from scripts.benchmarks.darcy._data import GlobalNormalizer

    tutorial = load_script("transolver")
    a = np.where(np.random.default_rng(0).normal(size=(5, 22, 22)) > 0, 12.0, 3.0)
    a = a.astype(np.float32)
    np.testing.assert_allclose(
        tutorial.GlobalNormalizer(a).encode(a)[..., None],
        GlobalNormalizer(a[..., None]).encode(a[..., None]),
        rtol=tolerances.rtol(1e-6),
    )


def test_prediction_and_loss_match_the_benchmark() -> None:
    """Same weights, inputs and cells: same nodal prediction and Ritz energy as
    ``train_transolver.py``."""
    from scirex.losses import Q4Galerkin
    from scirex.losses.darcy_piecewise_coefficient import (
        BatchedEnergyFunctionalLossQ4,
        kappa_fn,
        source_fn,
    )
    from scripts.benchmarks.darcy_physics import train_transolver as bench

    tutorial = load_script("transolver")
    xy, quad, _ = tutorial.build_q4_mesh(21)
    with jax.default_matmul_precision("highest"):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    rng = np.random.default_rng(0)
    a_enc = jnp.asarray(rng.normal(size=(2, 22 * 22, 1)), dtype=jnp.float32)
    cells = jnp.asarray(np.where(rng.normal(size=(2, 21, 21)) > 0, 12.0, 3.0))
    factor = jnp.asarray(tutorial.sin_boundary_factor(xy), dtype=jnp.float32)
    model, params = tutorial.build_model(tutorial.SIZES["smoke"], 22, seed=0)

    ours = tutorial.make_predict_nodes(model, 0.004, 0.006, factor)
    theirs = bench.make_predict_nodes(model, 22 * 22, 0.004, 0.006, factor)
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


def test_optimizer_descends_with_real_weights() -> None:
    tutorial = load_script("transolver")
    tx = tutorial.make_optimizer(tutorial.RECIPE, total_steps=10)
    params = dict(w=jnp.ones((3,)))
    updates, _ = tx.update(dict(w=jnp.full((3,), 1e6)), tx.init(params), params)
    assert updates["w"].dtype == params["w"].dtype
    assert float(updates["w"].max()) < 0


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = load_script("transolver")
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
    tutorial = load_script("transolver")
    metrics = tutorial.main(
        ["--smoke", "--n-train", "20", "--n-test", "4", "--epochs", "1",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["DARCY_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
