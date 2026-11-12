# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared fixtures and checks for the Cavity_<Model> tutorial tests.

Every check runs a tutorial script against the benchmark's own code in
``scripts/benchmarks/cavity`` on synthetic data, so nothing is downloaded:
the same cell grid from the same corners, the same windows, split and
normalisation, the same parameters and outputs for the same weights, the same
loss and the same copy-last-frame baseline.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests import tolerances
from tests.public_api_helpers import REPO

N = 128  # cells per side, fixed by the benchmark
CASES = ("01k", "02k", "04k", "06k", "14k", "16k", "18k", "20k", "10k")


def load_script(name: str):
    path = REPO / "scripts" / "tutorials" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    # Flax builds a module's dataclass from sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_grid(case: str, n_steps: int = 101, seed: int = 0) -> np.ndarray:
    """A smooth ``(n_steps, 128, 128, 3)`` field that changes in time."""
    rng = np.random.default_rng(seed + int(case[:2]))
    s = (np.arange(N, dtype=np.float32) + 0.5) / N
    yy, xx = np.meshgrid(s, s, indexing="ij")
    t = np.arange(n_steps, dtype=np.float32)[:, None, None] / n_steps
    lid = int(case[:2]) / 10.0
    a, b = rng.normal(size=2).astype(np.float32)
    ux = lid * yy**4 * (1.0 + 0.3 * np.sin(np.pi * (xx + t + a)))
    uy = 0.2 * lid * np.sin(np.pi * xx) * np.cos(np.pi * (yy + t + b))
    p = 0.05 * lid**2 * np.cos(np.pi * (xx - yy + t))
    return np.stack([p, ux, uy], axis=-1).astype(np.float32)


def write_fake_cache(cache_dir: Path, n_steps: int = 101) -> Path:
    """``<cache_dir>/<case>.npz`` for all nine cases, in the benchmark's format."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        np.savez(
            cache_dir / f"{case}.npz",
            grid=fake_grid(case, n_steps),
            steps=np.arange(n_steps, dtype=np.int32),
            u_lid=np.float32(int(case[:2]) / 10.0),
        )
    return cache_dir


def write_fake_csvs(raw_dir: Path, case: str, n_steps: int, seed: int = 0) -> None:
    """``<raw_dir>/<case>/Data<case>0.<step>.csv`` with the export's columns.

    The corners come in a shuffled order, on two z layers, with a little
    round-trip noise on the coordinates, as in the real export.
    """
    rng = np.random.default_rng(seed)
    folder = raw_dir / case
    folder.mkdir(parents=True, exist_ok=True)
    h = 0.1 / N
    iy, ix = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    order = rng.permutation((N + 1) ** 2)
    for step in range(n_steps):
        vals = rng.normal(size=((N + 1) ** 2, 3)).astype(np.float32)
        rows = []
        for z in (0.0, 0.01):
            x = ix.ravel() * h + rng.normal(scale=1e-9, size=(N + 1) ** 2)
            y = iy.ravel() * h + rng.normal(scale=1e-9, size=(N + 1) ** 2)
            rows.append(
                np.column_stack(
                    [vals[:, 0], vals[:, 1], vals[:, 2], np.zeros((N + 1) ** 2),
                     x, y, np.full((N + 1) ** 2, z)]
                )[order]
            )  # fmt: skip
        np.savetxt(
            folder / f"Data{case}0.{step}.csv",
            np.concatenate(rows),
            delimiter=",",
            header='"p","U:0","U:1","U:2","Points:0","Points:1","Points:2"',
            comments="",
            fmt="%.9g",
        )


def check_corners_to_cells_matches_benchmark(tutorial, tmp_path: Path) -> None:
    from scripts.benchmarks.cavity import _data

    write_fake_csvs(tmp_path, "04k", n_steps=1)
    pos, fields = tutorial.read_snapshot(tmp_path / "04k" / "Data04k0.0.csv")
    assert pos.shape == ((N + 1) ** 2, 2)
    order = _data.vertex_sort_index(pos)
    expected = _data.vertex_to_cell(fields[order].reshape(N + 1, N + 1, 3))
    np.testing.assert_allclose(
        tutorial.corners_to_cells(pos, fields), expected, rtol=tolerances.rtol(1e-6)
    )


def check_split_matches_benchmark(tutorial, cache_dir: Path) -> None:
    from scripts.benchmarks.cavity._config import PROTOCOL as BENCH
    from scripts.benchmarks.cavity._data import build_train_val

    ours = tutorial.build_split(cache_dir, tutorial.PROTOCOL, seed=BENCH["seed"])
    ref = build_train_val(
        cache_dir, 3, 1, val_frac=BENCH["val_frac"], seed=BENCH["seed"]
    )
    for ours_key, ref_key in (("x_train", "X_train"), ("y_train", "Y_train"),
                              ("x_val", "X_val"), ("y_val", "Y_val")):  # fmt: skip
        np.testing.assert_allclose(
            ours[ours_key],
            ref[ref_key],
            rtol=tolerances.rtol(1e-5),
            atol=tolerances.atol(1e-5),
        )


def check_protocol_matches_benchmark(tutorial) -> None:
    from scripts.benchmarks.cavity._config import PROTOCOL as BENCH

    p = tutorial.PROTOCOL
    assert p["learning_rate"] == BENCH["lr"]
    for key in ("lr_factor", "lr_patience", "early_stop_patience", "epochs",
                "batch_size", "mc_weight", "val_frac"):  # fmt: skip
        assert p[key] == BENCH[key], key
    assert (p["n_in"], p["n_out"]) == (3, 1)
    assert BENCH["optimizer"] == "adam"


def check_model_and_loss_match_benchmark(
    tutorial, model_name: str, run_name: str, n_params: int
) -> None:
    """Same parameter tree, count, output and loss as the benchmark's model."""
    from scirex.utils import count_model_params
    from scripts.benchmarks.cavity._config import RUNS
    from scripts.benchmarks.cavity._data import fit_normalizer
    from scripts.benchmarks.cavity._models import build_model
    from scripts.benchmarks.cavity._physics import make_loss

    run = RUNS[run_name]
    assert tutorial.SIZES["full"] == run.arch
    ours = tutorial.build_model(tutorial.SIZES["full"], 1, np.zeros((3, N, N, 3)), 0)[0]
    ref = build_model(model_name, n_out=1, **run.arch)
    x = jnp.asarray(np.stack([fake_grid("10k")[i : i + 3] for i in (5, 40)]))
    y = jnp.asarray(np.stack([fake_grid("10k")[i + 3 : i + 4] for i in (5, 40)]))
    params = ref.init(jax.random.PRNGKey(0), x)
    assert count_model_params(params) == n_params
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        ours.init(jax.random.PRNGKey(0), x)
    )
    np.testing.assert_allclose(
        ours.apply(params, x),
        ref.apply(params, x),
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-6),
    )
    norm = fit_normalizer(np.asarray(x))
    a = tutorial.make_loss(ours, norm, 0.1)(params, x, y)
    b = make_loss(ref, norm, 0.1)(params, x, y)
    np.testing.assert_allclose(a[0], b[0], rtol=tolerances.rtol(1e-5))
    np.testing.assert_allclose(a[1][1], b[1][1], rtol=tolerances.rtol(1e-5))


def check_baseline_matches_benchmark(tutorial, model, params, normalizer) -> None:
    from scripts.benchmarks.cavity.infer import persistence

    grid = fake_grid("10k", n_steps=20)
    m = tutorial.evaluate(model, params, grid, normalizer, 3, horizon=4)
    base = persistence(grid, 3, 4)
    np.testing.assert_allclose(
        m["baseline_rmse"], base["single_step"]["rmse"], rtol=tolerances.rtol(1e-5)
    )
    np.testing.assert_allclose(
        m["baseline_rollout_per_step"],
        base["rollout"]["per_step_rmse"],
        rtol=tolerances.rtol(1e-5),
    )
    np.testing.assert_allclose(
        m["baseline_rollout_rmse"], base["rollout"]["rmse"], rtol=tolerances.rtol(1e-5)
    )
    assert len(m["rollout_per_step"]) == 4
    assert np.isfinite(m["rmse"]) and np.isfinite(m["rollout_rmse"])
