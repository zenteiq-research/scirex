# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for ``shapenet_car_boundary`` (JAX): the ShapeNetCar-specific
adapter producing ``(boundary_points, boundary_normals)`` for
:mod:`~scirex.experimental.losses.domain_membership`.

Fixture generator similar in spirit to ``test_shapenet_car_dataset.py``'s
own (a cylinder-like quad-mesh surface, radius ``1.0 + 0.05 * seed``), but
capped at both ends into a closed (watertight) surface -- required by this
module's own watertightness check, see ``_make_sample``'s docstring.

* ``test_boundary_matches_pos_surf_slice`` -- cross-checks the extracted
  boundary points against ``sample["pos"][sample["surf"]]`` (already
  trusted/tested elsewhere), confirming point identity and order.
* ``test_boundary_normals_are_unit_length`` -- the property
  ``domain_membership.signed_distance`` requires and that a *normalized*
  ``sample["x"]``'s normal columns would NOT have (the reason this adapter
  re-reads the raw VTK instead of slicing the loaded/normalized sample).
* ``test_boundary_composes_with_domain_membership`` -- the actual
  end-to-end claim: feeding this adapter's output into
  ``domain_membership.signed_distance`` correctly classifies a point
  clearly inside the synthetic cylinder's tube vs. clearly outside it.
* ``test_boundary_rejects_non_watertight_mesh`` -- an open tube (no caps)
  is rejected outright, rather than silently handed to
  ``vtkPolyDataNormals`` with no guarantee its orientation is trustworthy.
* ``test_boundary_rejects_missing_file`` / ``test_boundary_rejects_empty_file``
  -- a missing or unreadable ``quadpress_smpl.vtk`` fails at the point
  where the bad path is actually known, not several calls later inside
  ``domain_membership``'s ``cKDTree`` construction.
* ``test_culled_patches_drops_interior_patches_and_keeps_exterior_ones`` --
  ``shapenet_car_culled_patches`` against the real synthetic cylinder: a
  hand-picked patch entirely inside the cylinder's radius is dropped, one
  entirely outside it is kept, and the total count shrinks versus the full
  uniform grid -- the same guarantee ``test_domain_membership.py``
  checks on a synthetic circle, now checked end-to-end through the actual
  ShapeNetCar-specific adapter.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances

vtk = pytest.importorskip("vtk")
pytest.importorskip("sklearn")

from vtk.util.numpy_support import numpy_to_vtk  # noqa: E402

from scirex.data.datasets.shapenet_car import (  # noqa: E402
    load_shapenet_car,
)
from scirex.data.datasets.shapenet_car_boundary import (  # noqa: E402
    shapenet_car_boundary,
    shapenet_car_culled_patches,
)
from scirex.experimental.losses.domain_membership import signed_distance  # noqa: E402
from scirex.experimental.losses.quadrature import Patch, make_patches  # noqa: E402


def _make_sample(
    sample_dir: Path, seed: int, nu: int = 8, nv: int = 6, n_extra: int = 12
) -> None:
    """Writes one synthetic quadpress_smpl.vtk/hexvelo_smpl.vtk pair: a
    capped cylinder, radius 1.0 + 0.05*seed, z in [-1, 1] -- a closed
    (watertight) surface, required by ``shapenet_car_boundary``'s
    watertightness check (see that module's docstring): VTK's
    auto-orient-normals guarantee only holds for a closed mesh, so an open
    tube (this fixture's shape before the two end caps were added) would
    (correctly) be rejected rather than exercising the intended case.

    Similar in spirit to ``test_shapenet_car_dataset.py``'s own fixture
    (also a cylinder), but that one is deliberately left open -- it never
    calls ``shapenet_car_boundary`` and so never needs to be watertight."""
    rng = np.random.default_rng(seed)
    sample_dir.mkdir(parents=True, exist_ok=True)
    radius = 1.0 + 0.05 * seed

    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(-1.0, 1.0, nv)
    pts = np.array(
        [[np.cos(u) * radius, np.sin(u) * radius, v] for v in vs for u in us],
        dtype=np.float64,
    )
    bottom_center_idx = pts.shape[0]
    top_center_idx = pts.shape[0] + 1
    pts = np.concatenate([pts, [[0.0, 0.0, vs[0]]], [[0.0, 0.0, vs[-1]]]], axis=0)

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

    # Cap both ends with a triangle fan to a center point -- closes the
    # tube into a watertight surface. Winding order doesn't need to be
    # manually consistent here: vtkPolyDataNormals's consistency +
    # auto-orient pass is exactly what's being exercised.
    for iu in range(nu):
        bottom_tri = vtk.vtkTriangle()
        bottom_tri.GetPointIds().SetId(0, bottom_center_idx)
        bottom_tri.GetPointIds().SetId(1, idx(iu, 0))
        bottom_tri.GetPointIds().SetId(2, idx(iu + 1, 0))
        ug.InsertNextCell(bottom_tri.GetCellType(), bottom_tri.GetPointIds())

        top_tri = vtk.vtkTriangle()
        top_tri.GetPointIds().SetId(0, top_center_idx)
        top_tri.GetPointIds().SetId(1, idx(iu, nv - 1))
        top_tri.GetPointIds().SetId(2, idx(iu + 1, nv - 1))
        ug.InsertNextCell(top_tri.GetCellType(), top_tri.GetPointIds())

    press_vals = rng.normal(size=(pts.shape[0],)).astype(np.float64)
    press_vtk = numpy_to_vtk(press_vals, deep=True)
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


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    _make_sample(tmp_path / "param0" / "sample_000", seed=0)
    _make_sample(tmp_path / "param1" / "sample_000", seed=1)
    return tmp_path


def test_boundary_matches_pos_surf_slice(fixture_root: Path) -> None:
    data = load_shapenet_car(fixture_root, fold_id=0)
    sample = data["train"][0]

    boundary_points, _ = shapenet_car_boundary(fixture_root, sample)
    expected = sample["pos"][sample["surf"]]

    assert boundary_points.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(boundary_points), expected, atol=tolerances.atol(1e-5)
    )


def test_boundary_normals_are_unit_length(fixture_root: Path) -> None:
    data = load_shapenet_car(fixture_root, fold_id=0)
    sample = data["train"][0]

    _, boundary_normals = shapenet_car_boundary(fixture_root, sample)
    norms = np.linalg.norm(np.asarray(boundary_normals), axis=-1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


def test_boundary_rejects_missing_sample_rel_path(fixture_root: Path) -> None:
    with pytest.raises(KeyError, match="sample_rel_path"):
        shapenet_car_boundary(fixture_root, {"pos": np.zeros((3, 3))})


def test_boundary_composes_with_domain_membership(fixture_root: Path) -> None:
    """The actual end-to-end claim: this adapter's output plugs directly
    into domain_membership.signed_distance and gives geometrically
    sane results on the synthetic cylinder (radius 1.0, z in [-1, 1])."""
    data = load_shapenet_car(fixture_root, fold_id=0)
    sample = data["train"][0]  # param1, seed=1, radius=1.05

    boundary_points, boundary_normals = shapenet_car_boundary(fixture_root, sample)

    origin = jnp.asarray([[0.0, 0.0, 0.0]])  # clearly inside the tube
    far_outside = jnp.asarray([[10.0, 10.0, 0.0]])  # clearly outside

    sdf_inside = signed_distance(origin, boundary_points, boundary_normals)
    sdf_outside = signed_distance(far_outside, boundary_points, boundary_normals)

    assert float(sdf_inside[0]) < 0.0
    assert float(sdf_outside[0]) > 0.0


def _write_open_tube_vtk(vtk_path: Path, nu: int = 8, nv: int = 6) -> None:
    """A minimal *uncapped* cylindrical tube -- boundary edges at both ends,
    not watertight. Only ``quadpress_smpl.vtk`` is written; ``shapenet_car_boundary``
    never reads ``hexvelo_smpl.vtk``."""
    us = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vs = np.linspace(-1.0, 1.0, nv)
    pts = np.array(
        [[np.cos(u), np.sin(u), v] for v in vs for u in us], dtype=np.float64
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

    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(vtk_path))
    writer.SetInputData(ug)
    writer.Write()


def test_boundary_rejects_non_watertight_mesh(tmp_path: Path) -> None:
    """An open tube (no end caps) must be rejected outright -- VTK's
    auto-orient-normals guarantee doesn't hold for it, so silently handing
    back its normals could mask the wrong side with no error at all."""
    sample_dir = tmp_path / "sample_000"
    _write_open_tube_vtk(sample_dir / "quadpress_smpl.vtk")

    with pytest.raises(ValueError, match="watertight"):
        shapenet_car_boundary(tmp_path, {"sample_rel_path": "sample_000"})


def test_boundary_rejects_missing_file(tmp_path: Path) -> None:
    """A sample_rel_path that doesn't resolve to a real file must fail with
    the actual path named, not several calls later inside cKDTree."""
    with pytest.raises(FileNotFoundError, match="quadpress_smpl"):
        shapenet_car_boundary(tmp_path, {"sample_rel_path": "does_not_exist"})


def test_boundary_rejects_empty_file(tmp_path: Path) -> None:
    """A quadpress_smpl.vtk that exists but reads back with 0 points (the
    documented vtkUnstructuredGridReader failure mode for a truncated or
    corrupted file) must fail at the point the bad file is known, not
    downstream in cKDTree construction."""
    sample_dir = tmp_path / "sample_000"
    sample_dir.mkdir(parents=True)
    empty_grid = vtk.vtkUnstructuredGrid()
    empty_grid.SetPoints(vtk.vtkPoints())
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(sample_dir / "quadpress_smpl.vtk"))
    writer.SetInputData(empty_grid)
    writer.Write()

    with pytest.raises(ValueError, match="empty mesh"):
        shapenet_car_boundary(tmp_path, {"sample_rel_path": "sample_000"})


def test_culled_patches_drops_interior_patches_and_keeps_exterior_ones(
    fixture_root: Path,
) -> None:
    """seed=0 -> cylinder radius 1.0, z in [-1, 1]. bounds/n_patches_per_axis
    chosen so one patch (center (-0.25, -0.25, -0.5), half_width 0.25/0.25/0.5)
    sits entirely inside the radius-1.0 cylinder -- every one of its
    quadrature points is in the excluded solid interior -- and another
    (center (1.25, 1.25, -0.5)) sits entirely outside it."""
    data = load_shapenet_car(fixture_root, fold_id=0)
    sample = data["train"][0]

    bounds = [(-1.5, 1.5), (-1.5, 1.5), (-1.0, 1.0)]
    n_patches_per_axis = [6, 6, 2]
    n = 3
    half_width = (0.25, 0.25, 0.5)

    all_patches = make_patches(bounds, n_patches_per_axis)
    culled = shapenet_car_culled_patches(
        fixture_root, sample, bounds, n_patches_per_axis, n
    )

    interior_patch = Patch(center=(-0.25, -0.25, -0.5), half_width=half_width)
    exterior_patch = Patch(center=(1.25, 1.25, -0.5), half_width=half_width)
    assert interior_patch in all_patches and exterior_patch in all_patches

    assert len(culled) < len(all_patches)
    assert interior_patch not in culled
    assert exterior_patch in culled
