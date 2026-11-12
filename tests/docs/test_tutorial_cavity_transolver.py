# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Cavity Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script runs on
synthetic data in the benchmark's layout and is compared with the benchmark's
own code (``scripts/benchmarks/cavity``): the cell grid, the split, the
model, the loss and the baseline. The test that reads the real data runs
only when ``CAVITY_CACHE_DIR`` points at the benchmark's cache, the folder with
the nine ``<case>.npz`` files.
"""

import os
from pathlib import Path

import jax
import numpy as np
import pytest

from tests.docs import cavity_tutorial_helpers as helpers

NAME = "cavity_transolver"


@pytest.fixture(scope="module")
def tutorial():
    return helpers.load_script(NAME)


def test_corners_to_cells_matches_benchmark(tutorial, tmp_path: Path) -> None:
    helpers.check_corners_to_cells_matches_benchmark(tutorial, tmp_path)


def test_prepare_case_writes_the_benchmark_cache(
    tutorial, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tutorial, "N_STEPS", 2)
    helpers.write_fake_csvs(tmp_path / "raw", "04k", n_steps=2)
    out = tutorial.prepare_case(tmp_path / "raw", "04k", tmp_path / "cache")
    grid = tutorial.load_case(tmp_path / "cache", "04k")
    assert out.name == "04k.npz" and grid.shape == (2, 128, 128, 3)
    with np.load(out) as npz:
        assert float(npz["u_lid"]) == pytest.approx(0.4)


def test_split_matches_benchmark(tutorial, tmp_path: Path) -> None:
    helpers.check_split_matches_benchmark(
        tutorial, helpers.write_fake_cache(tmp_path, 12)
    )


def test_protocol_matches_benchmark(tutorial) -> None:
    helpers.check_protocol_matches_benchmark(tutorial)


def test_model_and_loss_match_benchmark(tutorial) -> None:
    helpers.check_model_and_loss_match_benchmark(
        tutorial, "transolver", "transolver_3to1", 1_093_851
    )


def test_train_selects_the_best_epoch_and_baseline_matches(
    tutorial, tmp_path: Path
) -> None:
    data = tutorial.build_split(
        helpers.write_fake_cache(tmp_path, 12), tutorial.PROTOCOL, 0, n_train=8, n_val=4
    )
    model, params = tutorial.build_model(
        tutorial.SIZES["smoke"], 1, data["x_train"][0], 0
    )
    params, history = tutorial.train(model, params, data, 2, seed=0)
    assert [h["epoch"] for h in history] == [1, 2]
    assert all(
        np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in history
    )
    helpers.check_baseline_matches_benchmark(
        tutorial, model, params, data["normalizer"]
    )


def test_divergence_of_a_solenoidal_field_is_small(tutorial) -> None:
    s = (np.arange(128) + 0.5) / 128 * 0.1
    yy, xx = np.meshgrid(s, s, indexing="ij")
    vel = np.stack(
        [np.sin(20 * xx) * np.cos(20 * yy), -np.cos(20 * xx) * np.sin(20 * yy)], -1
    )
    div = np.asarray(
        tutorial.divergence(jax.numpy.asarray(vel, dtype=jax.numpy.float32))
    )
    assert np.abs(div[2:-2, 2:-2]).max() < 1e-2


def test_smoke_main_on_synthetic_cache(tutorial, tmp_path: Path) -> None:
    cache = helpers.write_fake_cache(tmp_path / "cache", 16)
    out = tmp_path / "out"
    metrics = tutorial.main(
        ["--cache-dir", str(cache), "--smoke", "--n-train", "8", "--n-val", "4",
         "--epochs", "1", "--out-dir", str(out)]
    )  # fmt: skip
    assert np.isfinite(metrics["rmse"]) and np.isfinite(metrics["rollout_rmse"])
    for name in ("prediction.png", "rollout.png", "loss.png"):
        assert (out / name).stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("CAVITY_CACHE_DIR"),
    reason="set CAVITY_CACHE_DIR to the folder with the nine <case>.npz files",
)
def test_smoke_main_on_real_data(tutorial, tmp_path: Path) -> None:
    """One epoch on the benchmark's cache; its copy-last-frame baseline on
    Re 10,000 is the one in doc/assets/benchmarks/cavity_*/metrics.json."""
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--out-dir", str(tmp_path / "out"),
         "--cache-dir", os.environ["CAVITY_CACHE_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["rmse"]) and np.isfinite(metrics["rollout_rmse"])
    assert metrics["baseline_rmse"] == pytest.approx(4.3995e-3, rel=1e-4)
    assert metrics["baseline_rollout_rmse"] == pytest.approx(1.4958e-2, rel=1e-4)
