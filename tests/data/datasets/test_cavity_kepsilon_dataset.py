# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the Transient k-epsilon lid-driven-cavity dataset loader
(src/scirex/data/datasets/cavity_kepsilon.py).

Covers:
  - lid_velocity() case-string parsing
  - get_case_steps() sorted output and exclusion of backup/lock files
    (regression test for the unanchored-filename-regex bug)
  - _read_snapshot() column-order independence, missing-column error, and
    z-layer tolerance (regression tests for the positional-column-slicing
    and exact-float-equality bugs), plus the wrong-point-count error path
  - _read_case() npz caching round-trip and the no-snapshots error path
  - load_cavity_kepsilon() end-to-end on a small synthetic dataset
    (N_POINTS monkeypatched down for speed), including the
    mesh-mismatch-across-cases error path
  - load_cavity_kepsilon(cases=...) subset selection: only the requested
    cases are loaded, the returned ``cases`` tuple is verbatim (order
    preserved), and meta ``case_index`` indexes into it rather than the
    module-level CASES
  - grid_index_maps() / to_grid_arrays() correctness on a small grid

Run with:
    pytest tests/data/datasets/test_cavity_kepsilon_dataset.py -v
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from scirex.data.datasets import cavity_kepsilon as ck
from scirex.data.datasets.cavity_kepsilon import (
    CASES,
    _read_case,
    _read_snapshot,
    get_case_steps,
    grid_index_maps,
    lid_velocity,
    load_cavity_kepsilon,
    to_grid_arrays,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh(n_points, seed=0):
    """A fixed (x, y) point mesh, shared across every snapshot/case that
    should be treated as "the same mesh" (mirrors the real dataset's
    invariant that every snapshot shares one 16,641-point mesh)."""
    rng = np.random.default_rng(seed)
    x = rng.random(n_points).astype(np.float32)
    y = rng.random(n_points).astype(np.float32)
    return x, y


def _write_snapshot_csv(path, x, y, header=None, z0=True, seed=0):
    """Writes one synthetic OpenFOAM-style CSV: the given (x, y) mesh, each
    point duplicated at z=0 and z=0.01 (mirroring the real front/back
    patch), with independently-seeded field values."""
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    p = rng.random(n).astype(np.float32)
    u0 = rng.random(n).astype(np.float32)
    u1 = rng.random(n).astype(np.float32)
    u2 = np.zeros(n, dtype=np.float32)
    z0v = np.zeros(n, dtype=np.float32) if z0 else np.full(n, 1e-8, dtype=np.float32)
    z1v = np.full(n, 0.01, dtype=np.float32)

    cols = {
        "p": np.concatenate([p, p]),
        "U:0": np.concatenate([u0, u0]),
        "U:1": np.concatenate([u1, u1]),
        "U:2": np.concatenate([u2, u2]),
        "Points:0": np.concatenate([x, x]),
        "Points:1": np.concatenate([y, y]),
        "Points:2": np.concatenate([z0v, z1v]),
    }
    header = header or list(cols.keys())
    with open(path, "w") as f:
        f.write(",".join(f'"{h}"' for h in header) + "\n")
        for row in zip(*(cols[h] for h in header)):
            f.write(",".join(str(v) for v in row) + "\n")
    return p, np.stack([u0, u1], axis=-1)


def _write_case(case_dir, case, steps, x, y):
    case_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        _write_snapshot_csv(case_dir / f"Data{case}0.{step}.csv", x, y, seed=step)


@pytest.fixture
def small_n_points(monkeypatch):
    """Shrinks N_POINTS so synthetic CSVs/datasets stay fast to write/read."""
    monkeypatch.setattr(ck, "N_POINTS", 9)
    return 9


# ---------------------------------------------------------------------------
# lid_velocity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,expected",
    [("01k", 0.1), ("02k", 0.2), ("04k", 0.4), ("14k", 1.4), ("16k", 1.6)],
)
def test_lid_velocity(case, expected):
    assert lid_velocity(case) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_case_steps
# ---------------------------------------------------------------------------


def test_get_case_steps_sorted(tmp_path):
    case_dir = tmp_path / "14k"
    case_dir.mkdir()
    for step in [50, 10, 30]:
        (case_dir / f"Data14k0.{step}.csv").touch()
    assert get_case_steps(tmp_path, "14k") == [10, 30, 50]


def test_get_case_steps_excludes_backup_and_lock_files(tmp_path):
    """Regression test: get_case_steps must reject trailing-suffix backup
    files (.bak, ~, .orig) and leading-dot editor lock files alike, not
    just leading-dot ones (the unanchored `.match()` bug)."""
    case_dir = tmp_path / "14k"
    case_dir.mkdir()
    for name in [
        "Data14k0.123.csv",
        "Data14k0.123.csv.bak",
        "Data14k0.123.csv~",
        "Data14k0.123.csv.orig",
        ".Data14k0.124.csv.swp",
        "Data14k0.124.csv",
    ]:
        (case_dir / name).touch()
    assert get_case_steps(tmp_path, "14k") == [123, 124]


def test_get_case_steps_ignores_other_cases(tmp_path):
    case_dir = tmp_path / "14k"
    case_dir.mkdir()
    (case_dir / "Data14k0.5.csv").touch()
    (case_dir / "Data16k0.5.csv").touch()  # wrong case, same directory
    assert get_case_steps(tmp_path, "14k") == [5]


# ---------------------------------------------------------------------------
# _read_snapshot
# ---------------------------------------------------------------------------


def test_read_snapshot_standard_order(tmp_path, small_n_points):
    n = small_n_points
    x, y = _make_mesh(n, seed=1)
    path = tmp_path / "std.csv"
    p, u = _write_snapshot_csv(path, x, y, seed=1)

    pos_out, p_out, u_out = _read_snapshot(path)
    assert pos_out.shape == (n, 2)
    assert p_out.shape == (n,)
    assert u_out.shape == (n, 2)
    assert np.allclose(p_out, p)
    assert np.allclose(u_out, u)
    assert np.allclose(pos_out, np.stack([x, y], axis=-1))


def test_read_snapshot_reordered_columns(tmp_path, small_n_points):
    """Regression test: column order must not affect parsing (the
    positional-slicing bug)."""
    n = small_n_points
    x, y = _make_mesh(n, seed=2)
    path = tmp_path / "reordered.csv"
    header = ["Points:0", "Points:1", "Points:2", "p", "U:0", "U:1", "U:2"]
    p, u = _write_snapshot_csv(path, x, y, header=header, seed=2)

    pos_out, p_out, u_out = _read_snapshot(path)
    assert np.allclose(p_out, p)
    assert np.allclose(u_out, u)
    assert np.allclose(pos_out, np.stack([x, y], axis=-1))


def test_read_snapshot_missing_column_raises(tmp_path, small_n_points):
    n = small_n_points
    x, y = _make_mesh(n, seed=3)
    path = tmp_path / "bad.csv"
    header = ["p", "U:0", "U:1", "U:2", "Points:0", "Points:2"]  # missing Points:1
    _write_snapshot_csv(path, x, y, header=header, seed=3)

    with pytest.raises(ValueError, match="unexpected CSV header"):
        _read_snapshot(path)


def test_read_snapshot_z_tolerance(tmp_path, small_n_points):
    """Regression test: the z=0 layer may carry tiny float noise instead of
    an exact 0.0 and must still be selected correctly."""
    n = small_n_points
    x, y = _make_mesh(n, seed=4)
    path = tmp_path / "noisy_z.csv"
    p, u = _write_snapshot_csv(path, x, y, z0=False, seed=4)

    pos_out, p_out, u_out = _read_snapshot(path)
    assert pos_out.shape == (n, 2)
    assert np.allclose(p_out, p)
    assert np.allclose(u_out, u)


def test_read_snapshot_wrong_point_count_raises(tmp_path, small_n_points):
    path = tmp_path / "short.csv"
    with open(path, "w") as f:
        f.write('"p","U:0","U:1","U:2","Points:0","Points:1","Points:2"\n')
        f.write("1.0,0.0,0.0,0.0,0.0,0.0,0.0\n")  # only one z=0 row, not 9

    with pytest.raises(ValueError, match="expected .* points"):
        _read_snapshot(path)


# ---------------------------------------------------------------------------
# _read_case (npz caching)
# ---------------------------------------------------------------------------


def test_read_case_caches_to_npz(tmp_path, small_n_points):
    n = small_n_points
    x, y = _make_mesh(n, seed=5)
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    _write_case(data_dir / "14k", "14k", [1, 2], x, y)

    result1 = _read_case(data_dir, "14k", cache_dir, use_cache=True)
    assert result1["P"].shape == (2, n)
    assert result1["U"].shape == (2, n, 2)
    assert (cache_dir / "14k.npz").exists()

    # Remove the source CSVs; a second call must still succeed, from cache.
    shutil.rmtree(data_dir / "14k")
    result2 = _read_case(data_dir, "14k", cache_dir, use_cache=True)
    assert np.array_equal(result1["P"], result2["P"])
    assert np.array_equal(result1["pos"], result2["pos"])


def test_read_case_no_snapshots_raises(tmp_path, small_n_points):
    data_dir = tmp_path / "data"
    (data_dir / "14k").mkdir(parents=True)
    with pytest.raises(ValueError, match="No snapshots"):
        _read_case(data_dir, "14k", None, use_cache=False)


# ---------------------------------------------------------------------------
# load_cavity_kepsilon (end-to-end)
# ---------------------------------------------------------------------------


def test_load_cavity_kepsilon_end_to_end(tmp_path, small_n_points):
    n = small_n_points
    x, y = _make_mesh(n, seed=0)
    data_dir = tmp_path / "data"
    steps = [1, 2, 3, 4, 5]
    for case in CASES:
        _write_case(data_dir / case, case, steps, x, y)

    data = load_cavity_kepsilon(
        data_dir, cache_dir=tmp_path / "cache", n_test_frac=0.2, seed=0, verbose=False
    )

    n_total = len(CASES) * len(steps)
    assert data["X_train"].shape[1:] == (n, 4)
    assert data["Y_train"].shape[1:] == (n, 3)
    assert data["X_train"].shape[0] + data["X_test"].shape[0] == n_total
    assert data["meta_train"].shape[1] == 2
    assert set(data["model_kwargs"].keys()) == {
        "space_dim",
        "fun_dim",
        "out_dim",
        "unified_pos",
        "geometry",
    }
    # Train-split normalization should be ~zero-mean, ~unit-std.
    x_train_flat = data["X_train"].reshape(-1, 4)
    assert np.allclose(x_train_flat.mean(axis=0), 0.0, atol=1e-3)
    assert np.allclose(x_train_flat.std(axis=0), 1.0, atol=1e-2)


def test_load_cavity_kepsilon_mesh_mismatch_raises(tmp_path, small_n_points):
    n = small_n_points
    x, y = _make_mesh(n, seed=0)
    x_bad, y_bad = _make_mesh(n, seed=999)  # a genuinely different mesh
    data_dir = tmp_path / "data"
    for case in CASES[:-1]:
        _write_case(data_dir / case, case, [1], x, y)
    _write_case(data_dir / CASES[-1], CASES[-1], [1], x_bad, y_bad)

    with pytest.raises(ValueError, match="Mesh mismatch"):
        load_cavity_kepsilon(
            data_dir,
            cache_dir=None,
            use_cache=False,
            n_test_frac=0.2,
            seed=0,
            verbose=False,
        )


# ---------------------------------------------------------------------------
# load_cavity_kepsilon (cases subset)
# ---------------------------------------------------------------------------


def test_load_cavity_kepsilon_cases_subset(tmp_path, small_n_points):
    """Passing ``cases`` loads only that subset (not the full CASES), and
    returns the requested tuple verbatim with order preserved."""
    n = small_n_points
    x, y = _make_mesh(n, seed=0)
    data_dir = tmp_path / "data"
    # Write every canonical case on disk, but request only two of them.
    for case in CASES:
        _write_case(data_dir / case, case, [1, 2], x, y)

    subset = ("04k", "01k")  # deliberately not in module-CASES order
    data = load_cavity_kepsilon(
        data_dir,
        cache_dir=None,
        use_cache=False,
        n_test_frac=0.2,
        seed=0,
        verbose=False,
        cases=subset,
    )

    assert data["cases"] == subset
    # Only the two requested cases were loaded, not all five.
    n_total = len(subset) * 2
    assert data["X_train"].shape[0] + data["X_test"].shape[0] == n_total
    all_idx = np.concatenate([data["meta_train"][:, 0], data["meta_test"][:, 0]])
    assert set(all_idx.tolist()) <= set(range(len(subset)))


def test_load_cavity_kepsilon_case_index_maps_into_returned_cases(
    tmp_path, small_n_points
):
    """Regression: meta ``case_index`` must index into the *returned*
    ``cases`` tuple, not the module-level CASES. Each case gets a disjoint
    set of step numbers, so a sample's case_index can be cross-checked
    against the step value it carries."""
    n = small_n_points
    x, y = _make_mesh(n, seed=0)
    data_dir = tmp_path / "data"
    steps_by_case = {"04k": [1, 2], "01k": [7, 8, 9]}
    for case, steps in steps_by_case.items():
        _write_case(data_dir / case, case, steps, x, y)

    subset = ("04k", "01k")
    data = load_cavity_kepsilon(
        data_dir,
        cache_dir=None,
        use_cache=False,
        n_test_frac=0.2,
        seed=0,
        verbose=False,
        cases=subset,
    )

    meta = np.concatenate([data["meta_train"], data["meta_test"]], axis=0)
    for case_idx, step in meta:
        case = data["cases"][case_idx]
        assert step in steps_by_case[case], (
            f"case_index {case_idx} -> {case!r} but step {step} is not one of "
            f"that case's steps {steps_by_case[case]}"
        )


# ---------------------------------------------------------------------------
# grid_index_maps / to_grid_arrays
# ---------------------------------------------------------------------------


def test_grid_index_maps_simple():
    # A 2x2 grid: (x, y) in {0, 1} x {0, 2}.
    pos = np.array([[0, 0], [0, 2], [1, 0], [1, 2]], dtype=np.float32)
    gi, gj, H, W = grid_index_maps(pos)
    assert (H, W) == (2, 2)
    assert list(gi) == [0, 0, 1, 1]
    assert list(gj) == [0, 1, 0, 1]


def test_to_grid_arrays_roundtrip():
    pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    n, n_pts, C = 2, 4, 1
    flat = np.arange(n * n_pts * C, dtype=np.float32).reshape(n, n_pts, C)
    data = {
        "pos": pos,
        "X_train": flat,
        "Y_train": flat,
        "X_test": flat,
        "Y_test": flat,
    }

    out = to_grid_arrays(data)
    assert out["H"] == 2 and out["W"] == 2
    assert out["X_train_grid"].shape == (n, C, 2, 2)

    # Scatter then gather (by the same index map) must recover the original
    # flat per-point values.
    gi, gj, _, _ = grid_index_maps(pos)
    recovered = out["X_train_grid"][:, 0][:, gi, gj]
    assert np.allclose(recovered, flat[:, :, 0])
