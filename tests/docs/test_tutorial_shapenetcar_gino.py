# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the ShapeNetCar GINO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions run on synthetic cars, and every piece the script writes out
itself (the latent grid, the padded neighbour lists, the model wrapper, the
loss) is checked against the benchmark's own code in
``scripts/benchmarks/shapenetcar``. The test that reads the real dataset runs
only when ``vtk`` is installed and ``SHAPENETCAR_DATA`` points at
``training_data``.
"""

import copy
import importlib.util
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "shapenetcar_gino.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("shapenetcar_gino_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # The script defines a Flax module (a dataclass), which needs its module
    # registered while the class is created.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_cars(n_cars: int, n_points: int = 60, n_surf: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    cars = []
    for _ in range(n_cars):
        surf = np.zeros(n_points, dtype=bool)
        surf[:n_surf] = True
        x = rng.normal(size=(n_points, 7)).astype(np.float32)
        x[:, :3] *= 0.6  # keep the points inside the fixed latent domain
        cars.append(
            dict(
                x=x,
                y=rng.normal(size=(n_points, 4)).astype(np.float32),
                pos=x[:, :3].copy(),
                surf=surf,
            )
        )
    return cars


def test_settings_match_the_benchmark() -> None:
    from scripts.benchmarks.shapenetcar._config import PROTOCOL as BENCH
    from scripts.benchmarks.shapenetcar._config import RUNS

    tutorial = _load_script()
    assert tutorial.SIZES["full"] == RUNS["gino_wide_radius"].arch
    for key, value in tutorial.PROTOCOL.items():
        assert BENCH[key] == value, key


def test_latent_grid_and_neighbors_match_the_benchmark() -> None:
    from scripts.benchmarks.shapenetcar import _gino_geometry

    tutorial = _load_script()
    np.testing.assert_array_equal(tutorial.DOMAIN_MIN, _gino_geometry._DOMAIN_MIN)
    np.testing.assert_array_equal(tutorial.DOMAIN_MAX, _gino_geometry._DOMAIN_MAX)
    for g in (6, 10):
        np.testing.assert_array_equal(
            tutorial.build_latent_grid(g), _gino_geometry.build_latent_queries(g)
        )
    ours, theirs = _fake_cars(3), _fake_cars(3)
    size = dict(tutorial.SIZES["full"])
    tutorial.attach_neighbors(ours, size)
    _gino_geometry.attach_neighbors(
        theirs, size["latent_grid"], size["in_gno_radius"], size["out_gno_radius"]
    )
    for a, b in zip(ours, theirs, strict=True):
        for key in ("neighbors_in", "neighbors_out"):
            for field in ("neighbors_index", "segment_ids", "counts"):
                np.testing.assert_array_equal(
                    np.asarray(a[key][field]), np.asarray(b[key][field])
                )


@pytest.mark.parametrize("size_name", ["full", "smoke"])
def test_model_and_loss_match_the_benchmark(size_name: str) -> None:
    from scripts.benchmarks.shapenetcar._data import MODEL_KWARGS
    from scripts.benchmarks.shapenetcar._models import build_model
    from scripts.benchmarks.shapenetcar._physics import make_loss

    tutorial = _load_script()
    size = tutorial.SIZES[size_name]
    cars = _fake_cars(2)
    tutorial.attach_neighbors(cars, size)
    model, params = tutorial.build_model(size, cars[0], seed=0)
    bench = build_model("gino", size, MODEL_KWARGS)
    x, y, surf, nbr_in, nbr_out = tutorial.as_arrays(cars[1])
    bench_params = bench.init(
        jax.random.PRNGKey(0), x=x, fx=None, neighbors_in=nbr_in, neighbors_out=nbr_out
    )
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        bench_params
    )
    if size_name == "full":
        from scirex.utils import count_model_params

        assert count_model_params(params) == 3_852_494

    ours = model.apply(params, x, nbr_in, nbr_out)[0]
    theirs = bench.apply(
        params, x=x, fx=None, neighbors_in=nbr_in, neighbors_out=nbr_out
    )[0]
    np.testing.assert_allclose(
        np.asarray(ours),
        np.asarray(theirs),
        rtol=tolerances.rtol(1e-6),
        atol=tolerances.atol(1e-6),
    )

    loss_ours = tutorial.make_loss(model, 0.5)(params, x, y, surf, nbr_in, nbr_out)
    loss_theirs = make_loss(bench, 0.5)(params, x, y, surf, nbr_in, nbr_out)
    np.testing.assert_allclose(
        np.asarray(jax.tree_util.tree_leaves(loss_ours)),
        np.asarray(jax.tree_util.tree_leaves(loss_theirs)),
        rtol=tolerances.rtol(1e-6),
    )


def test_padding_drops_no_edge_and_adds_no_contribution() -> None:
    tutorial = _load_script()
    cars = _fake_cars(3)
    size = tutorial.SIZES["smoke"]
    raw = copy.deepcopy(cars)
    tutorial.attach_neighbors(cars, size)
    model, params = tutorial.build_model(size, cars[0], seed=0)
    # Unpadded lists of car 1 alone give the same prediction as the padded ones.
    latent = tutorial.build_latent_grid(size["latent_grid"]).reshape(-1, 3)
    pos = raw[1]["x"][:, :3]
    nbr_in = tutorial.neighbors_to_segments(
        tutorial.kdtree_neighbor_search(
            data=pos, queries=latent, radius=size["in_gno_radius"]
        )
    )
    nbr_out = tutorial.neighbors_to_segments(
        tutorial.kdtree_neighbor_search(
            data=latent, queries=pos, radius=size["out_gno_radius"]
        )
    )
    x = jnp.asarray(raw[1]["x"])[None]
    padded = model.apply(params, x, cars[1]["neighbors_in"], cars[1]["neighbors_out"])
    unpadded = model.apply(params, x, nbr_in, nbr_out)
    np.testing.assert_allclose(
        np.asarray(padded),
        np.asarray(unpadded),
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-6),
    )
    n_edges = {c["neighbors_out"]["neighbors_index"].shape[0] for c in cars}
    assert len(n_edges) == 1  # one shape for every car, so one compile


def test_train_evaluate_plot_on_synthetic_cars(tmp_path: Path) -> None:
    tutorial = _load_script()
    train_cars, val_cars = _fake_cars(3), _fake_cars(2, seed=1)
    size = tutorial.SIZES["smoke"]
    tutorial.attach_neighbors(train_cars + val_cars, size)
    model, params = tutorial.build_model(size, train_cars[0], seed=0)
    params, history = tutorial.train(model, params, train_cars, val_cars, 2, seed=0)
    assert [h["epoch"] for h in history] == [0, 1]
    assert all(
        np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in history
    )

    coef_norm = (np.zeros(7), np.ones(7), np.zeros(4), np.ones(4))
    metrics = tutorial.evaluate(model, params, val_cars, coef_norm)
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    tutorial.plot_surface_pressure(
        val_cars[0], metrics["predictions"][0], coef_norm, tmp_path / "p.png"
    )
    tutorial.plot_loss(history, tmp_path / "loss.png")
    assert (tmp_path / "p.png").stat().st_size > 0
    assert (tmp_path / "loss.png").stat().st_size > 0


def test_smoke_main_on_synthetic_vtk_cars(tmp_path: Path) -> None:
    pytest.importorskip("vtk")
    pytest.importorskip("sklearn")
    from tests.data.datasets.test_shapenet_car_dataset import _make_sample

    src = tmp_path / "raw"
    for fold in (0, 1):
        for s in range(3):
            _make_sample(src / f"param{fold}" / f"car_{s}", seed=fold * 10 + s)
    out = tmp_path / "out"
    metrics = _load_script().main(
        [
            "--data-dir", str(src),
            "--smoke",
            "--n-train", "2",
            "--n-val", "1",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    assert (out / "surface_pressure.png").exists()
    assert (out / "loss.png").exists()


@pytest.mark.skipif(
    not os.environ.get("SHAPENETCAR_DATA"),
    reason="set SHAPENETCAR_DATA to the real training_data folder to run this test",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    pytest.importorskip("vtk")
    metrics = _load_script().main(
        ["--smoke", "--epochs", "1", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["SHAPENETCAR_DATA"]]
    )  # fmt: skip
    assert np.isfinite(metrics["rel_l2_volume"])
