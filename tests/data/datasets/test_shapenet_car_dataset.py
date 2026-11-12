# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the Car-Design-ShapeNetCar dataset loader (JAX).

The real dataset (Nobuyuki Umetani's raw ``mlcfd_data.zip``, ~889 samples) is
a multi-gigabyte external download, so these tests instead build small
synthetic ``quadpress_smpl.vtk``/``hexvelo_smpl.vtk`` fixtures with real VTK
cell connectivity (quads for the surface mesh, so ``vtkPolyDataNormals``
produces meaningful normals) and exercise the full read -> SDF/normal ->
feature-assembly -> normalization pipeline against them.

Requires ``vtk`` and ``scikit-learn``; both tests and the module they cover
are skipped entirely if either is missing.
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")
pytest.importorskip("sklearn")

from vtk.util.numpy_support import numpy_to_vtk  # noqa: E402

from scirex.data.datasets.shapenet_car import (  # noqa: E402
    get_samples,
    load_shapenet_car,
)


@pytest.fixture(autouse=True)
def _capture_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The library logs progress at INFO; capture it in every test of this module."""
    caplog.set_level(logging.INFO)


def _make_sample(
    sample_dir: Path, seed: int, nu: int = 6, nv: int = 5, n_extra: int = 12
) -> int:
    """Writes one synthetic quadpress_smpl.vtk/hexvelo_smpl.vtk pair.
    Returns the surface point count (nu * nv)."""
    rng = np.random.default_rng(seed)
    sample_dir.mkdir(parents=True, exist_ok=True)

    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(-1.0, 1.0, nv)
    pts = np.array(
        [
            [np.cos(u) * (1.0 + 0.05 * seed), np.sin(u) * (1.0 + 0.05 * seed), v]
            for v in vs
            for u in us
        ],
        dtype=np.float64,
    )

    vtk_points = vtk.vtkPoints()
    for p in pts:
        vtk_points.InsertNextPoint(*p)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vtk_points)

    def idx(iu: int, iv: int) -> int:
        return iv * nu + (iu % nu)

    for iv in range(nv - 1):
        for iu in range(nu):
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, idx(iu, iv))
            quad.GetPointIds().SetId(1, idx(iu + 1, iv))
            quad.GetPointIds().SetId(2, idx(iu + 1, iv + 1))
            quad.GetPointIds().SetId(3, idx(iu, iv + 1))
            ug.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    press_vtk = numpy_to_vtk(
        rng.normal(size=(pts.shape[0],)).astype(np.float64), deep=True
    )
    press_vtk.SetName("p")
    ug.GetPointData().SetScalars(press_vtk)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(sample_dir / "quadpress_smpl.vtk"))
    writer.SetInputData(ug)
    writer.Write()

    extra_pts = rng.uniform(-2, 2, size=(n_extra, 3)).astype(np.float64)
    velo_pts = np.concatenate([pts, extra_pts], axis=0)
    vtk_points2 = vtk.vtkPoints()
    for p in velo_pts:
        vtk_points2.InsertNextPoint(*p)
    ug2 = vtk.vtkUnstructuredGrid()
    ug2.SetPoints(vtk_points2)
    for i in range(velo_pts.shape[0]):
        vert = vtk.vtkVertex()
        vert.GetPointIds().SetId(0, i)
        ug2.InsertNextCell(vert.GetCellType(), vert.GetPointIds())
    velo_vtk = numpy_to_vtk(
        rng.normal(size=(velo_pts.shape[0], 3)).astype(np.float64), deep=True
    )
    velo_vtk.SetName("U")
    ug2.GetPointData().SetVectors(velo_vtk)

    writer2 = vtk.vtkUnstructuredGridWriter()
    writer2.SetFileName(str(sample_dir / "hexvelo_smpl.vtk"))
    writer2.SetInputData(ug2)
    writer2.Write()

    return nu * nv


FixtureRoot = tuple[Path, int]


@pytest.fixture
def fixture_root(tmp_path: Path) -> FixtureRoot:
    n_surf = None
    for fold in range(2):
        for s in range(2):
            n = _make_sample(
                tmp_path / f"param{fold}" / f"sample_{s:03d}", seed=fold * 10 + s
            )
            n_surf = n_surf or n
    assert n_surf is not None
    return tmp_path, n_surf


def test_get_samples_enumerates_9_folds_and_finds_only_populated_ones(
    fixture_root: FixtureRoot,
) -> None:
    root, _ = fixture_root
    samples = get_samples(root)
    assert len(samples) == 9
    assert [len(f) for f in samples] == [2, 2, 0, 0, 0, 0, 0, 0, 0]


def test_load_shapenet_car_shapes_and_split(fixture_root: FixtureRoot) -> None:
    root, n_surf = fixture_root
    n_extra = 12
    n_total = n_surf + n_extra

    data = load_shapenet_car(root, fold_id=0)

    assert len(data["train"]) == 2  # param1
    assert len(data["val"]) == 2  # param0 (held out)

    for split in ("train", "val"):
        for sample in data[split]:
            assert sample["x"].shape == (n_total, 7)
            assert sample["y"].shape == (n_total, 4)
            assert sample["pos"].shape == (n_total, 3)
            assert sample["surf"].shape == (n_total,)
            assert sample["surf"].sum() == n_surf
            assert sample["x"].dtype == np.float32

    assert data["model_kwargs"] == dict(
        space_dim=7, fun_dim=0, out_dim=4, unified_pos=False, geometry="irregular"
    )


def test_load_shapenet_car_skips_incomplete_sample(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    n_surf = None
    for fold in range(2):
        for s in range(2):
            n = _make_sample(
                tmp_path / f"param{fold}" / f"sample_{s:03d}", seed=fold * 10 + s
            )
            n_surf = n_surf or n

    # Real-world observed case: a sample dir with only param1.txt/param2.txt,
    # no VTK files at all (not corruption -- some raw simulation runs produce
    # no output). Should be skipped, not crash.
    incomplete_dir = tmp_path / "param1" / "sample_incomplete"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "param1.txt").write_text("0\n")
    (incomplete_dir / "param2.txt").write_text("0\n")

    samples = get_samples(tmp_path)
    assert (
        len(samples[1]) == 3
    )  # 2 complete + 1 incomplete -- enumeration doesn't filter

    data = load_shapenet_car(tmp_path, fold_id=0)
    assert len(data["val"]) == 2  # param0, unaffected
    assert (
        len(data["train"]) == 2
    )  # param1's 2 complete samples; incomplete one skipped

    captured = caplog
    assert "skipped 1 incomplete train sample" in captured.text


def test_load_shapenet_car_normalizes_using_train_stats_only(
    fixture_root: FixtureRoot,
) -> None:
    root, _ = fixture_root
    data = load_shapenet_car(root, fold_id=0)

    mean_in, std_in, mean_out, std_out = data["coef_norm"]
    assert mean_in.shape == (7,)
    assert std_in.shape == (7,)
    assert mean_out.shape == (4,)
    assert std_out.shape == (4,)

    # Train split's x, re-assembled, should have ~zero mean per channel
    # (population stats fit directly on this same data).
    x_train_all = np.concatenate([s["x"] for s in data["train"]], axis=0)
    np.testing.assert_allclose(x_train_all.mean(axis=0), np.zeros(7), atol=1e-4)


def test_invalid_fold_id_raises(fixture_root: FixtureRoot) -> None:
    root, _ = fixture_root
    with pytest.raises(ValueError, match="fold_id"):
        load_shapenet_car(root, fold_id=9)


def test_load_shapenet_car_cache_round_trip(
    fixture_root: FixtureRoot, tmp_path: Path
) -> None:
    root, _ = fixture_root
    cache_dir = tmp_path / "_cache"

    data1 = load_shapenet_car(root, fold_id=0, cache_dir=cache_dir)
    assert any(cache_dir.iterdir())

    data2 = load_shapenet_car(root, fold_id=0, cache_dir=cache_dir, use_cache=True)
    np.testing.assert_allclose(data1["train"][0]["x"], data2["train"][0]["x"])
    np.testing.assert_allclose(data1["train"][0]["y"], data2["train"][0]["y"])


def test_load_shapenet_car_feeds_transolver_forward_pass(
    fixture_root: FixtureRoot,
) -> None:
    import jax
    import jax.numpy as jnp

    from scirex.models.transolver import Transolver

    root, _ = fixture_root
    data = load_shapenet_car(root, fold_id=0)
    sample = data["train"][0]

    model = Transolver(
        n_hidden=8,
        n_layers=1,
        n_head=2,
        mlp_ratio=1,
        slice_num=4,
        **data["model_kwargs"],
    )
    x = jnp.asarray(sample["x"])[None, :, :]
    params = model.init(jax.random.PRNGKey(0), x=x, fx=None)
    out = cast(jax.Array, model.apply(params, x=x, fx=None))

    assert out.shape == (1, sample["x"].shape[0], 4)
