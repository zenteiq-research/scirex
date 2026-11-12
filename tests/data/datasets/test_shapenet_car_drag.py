# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the Car-Design-ShapeNetCar drag-coefficient port
(``shapenet_car_drag.py``).

Uses a single flat-plate quad cell lying in the z=0 plane (normal along the
flow axis) rather than the multi-quad cylindrical fixture in
``test_shapenet_car_dataset.py``, since a flat plate under constant
pressure and zero velocity has a hand-computable Cd: with cell area 1,
frontal (convex-hull) area 1, and zero shear contribution, ``|F_drag| =
pressure * cell_area``, giving ``Cd = 2 * F_drag / (v^2 * A * rho)``
directly from the module's own default constants.
"""

from pathlib import Path

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

from vtk.util.numpy_support import numpy_to_vtk  # noqa: E402

from scirex.data.datasets.shapenet_car_drag import (  # noqa: E402
    DEFAULT_AIR_DENSITY,
    DEFAULT_FREESTREAM_VELOCITY,
    compute_drag_coefficient,
    mean_relative_drag_error,
    spearman_drag_correlation,
)


def _write_flat_plate(path: Path, pressure_value: float = 5.0) -> None:
    """One unit-square quad cell in the z=0 plane -- its VTK-computed cell
    normal points purely along +/-z (the dataset's flow axis), so the
    convex-hull frontal area (1.0) and cell area (1.0) are both exactly
    computable by hand."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    vtk_points = vtk.vtkPoints()
    for p in pts:
        vtk_points.InsertNextPoint(*p)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vtk_points)

    quad = vtk.vtkQuad()
    for i in range(4):
        quad.GetPointIds().SetId(i, i)
    ug.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    press_vtk = numpy_to_vtk(np.full(4, pressure_value, dtype=np.float64), deep=True)
    press_vtk.SetName("p")
    ug.GetPointData().SetScalars(press_vtk)

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ug)
    writer.Write()


def test_compute_drag_coefficient_matches_hand_calculation(tmp_path: Path) -> None:
    path = tmp_path / "quadpress_smpl.vtk"
    _write_flat_plate(path, pressure_value=5.0)

    cd = compute_drag_coefficient(path, np.full(4, 5.0), np.zeros((4, 3)))

    v, rho = DEFAULT_FREESTREAM_VELOCITY, DEFAULT_AIR_DENSITY
    expected_magnitude = 2.0 * 5.0 / (v**2 * 1.0 * rho)  # cell_area=A=1, no shear term
    assert abs(cd) == pytest.approx(expected_magnitude, rel=1e-6)


def test_compute_drag_coefficient_scales_linearly_with_pressure(tmp_path: Path) -> None:
    path = tmp_path / "quadpress_smpl.vtk"
    _write_flat_plate(path)

    zero_velo = np.zeros((4, 3))
    cd1 = compute_drag_coefficient(path, np.full(4, 5.0), zero_velo)
    cd2 = compute_drag_coefficient(path, np.full(4, 10.0), zero_velo)
    assert cd2 == pytest.approx(2.0 * cd1, rel=1e-6)


def test_compute_drag_coefficient_zero_pressure_and_velocity_is_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quadpress_smpl.vtk"
    _write_flat_plate(path)

    cd = compute_drag_coefficient(path, np.zeros(4), np.zeros((4, 3)))
    assert cd == pytest.approx(0.0, abs=1e-10)


def test_mean_relative_drag_error() -> None:
    err = mean_relative_drag_error([2.0, 4.0], [2.2, 3.6])
    assert err == pytest.approx((0.1 + 0.1) / 2, rel=1e-6)


def test_spearman_drag_correlation_perfect_rank_agreement() -> None:
    rho = spearman_drag_correlation([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == pytest.approx(1.0)


def test_spearman_drag_correlation_perfect_rank_disagreement() -> None:
    rho = spearman_drag_correlation([1, 2, 3, 4], [40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)
