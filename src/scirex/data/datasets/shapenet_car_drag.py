# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Drag coefficient (Cd) evaluation for Car-Design-ShapeNetCar.

Port of the reference PyTorch Transolver's
`Car-Design-ShapeNetCar/utils/drag_coefficient.py` (`cal_coefficient`),
used **only as a post-hoc evaluation metric** -- exactly like the reference,
which calls this solely from `main_evaluation.py`, never from
`train.py`/`main.py` (training loss there, and in this port's
`scripts/benchmarks/shapenetcar/`, is pointwise MSE on pressure and
velocity; Cd is not differentiated through or optimized for).

Formula::

    Cd = 2 * F_drag / (rho * v^2 * A)

where:

- `F_drag` is the pressure ("form") drag plus an approximate viscous/shear
  drag, both projected onto the flow direction (+z axis) and integrated
  (summed) over every surface-mesh quad cell, weighted by cell area. The
  viscous term uses a crude per-cell finite-difference velocity gradient
  across each quad's diagonals -- not a true wall-normal gradient -- ported
  as-is from the reference for fidelity, not because it's the most accurate
  possible estimator.
- `A` is the convex-hull area of the surface points projected onto the x-y
  plane (frontal/silhouette area, since flow runs along +z throughout this
  dataset).
- `rho` (air density, `0.3`) and `v` (freestream velocity, ``72/3.6 =
  20.0`` m/s) are **literal constants hard-coded by the paper's own
  benchmark authors** -- not physical air density/speed, not read from any
  file, not derived per-sample or per-fold. They are reproduced here as
  documented defaults purely to stay faithful to the paper's own evaluation
  protocol; override them only if you have a specific reason to deviate from
  it. Don't confuse this `rho` with `spearman_drag_correlation`'s
  "rho_d" metric name below -- both are conventionally called "rho" in
  fluid dynamics vs. statistics, but are otherwise unrelated quantities.

**Point-order requirement**: `press_surf`/`velo_surf` must be in the
same point order as `quadpress_vtk_path`'s own points. When working with a
sample from `load_shapenet_car` (`shapenet_car.py`), that
loader builds `pos = concatenate([pos_ext, pos_surf])` where
`pos_surf = points_press` verbatim (unreordered) -- so slicing a
denormalized `y`/model-output array with that same sample's `surf`
boolean mask (`y[surf]`) already yields arrays in the correct order; no
extra reordering/matching step is needed. Also denormalize first
(`y * std_out + mean_out`, using `load_shapenet_car`'s returned
`coef_norm`) -- passing the network's raw normalized output here silently
produces a physically meaningless Cd.

Requires `vtk` and `scipy` (both already required elsewhere in this
package for ShapeNetCar).
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

ArrayLike = np.ndarray | list

# The paper's own hard-coded normalization constants (see module docstring) --
# NOT physical air properties. Named here purely for documentation; still
# overridable via compute_drag_coefficient's keyword arguments.
DEFAULT_AIR_DENSITY = 0.3  # kg/m^3 in name only -- the paper's own constant
DEFAULT_FREESTREAM_VELOCITY = 72.0 / 3.6  # = 20.0 m/s (72 km/h), paper's constant
DEFAULT_DYNAMIC_VISCOSITY = 1.8e-5  # Pa.s, ~air viscosity at 15C; shear term only


def _load_unstructured_grid_data(file_path: str | Path) -> Any:
    import vtk

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(file_path))
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    return reader.GetOutput()


def _unstructured_grid_data_to_poly_data(
    unstructured_grid_data: Any,
) -> tuple[Any, Any]:
    import vtk

    filt = vtk.vtkDataSetSurfaceFilter()
    filt.SetInputData(unstructured_grid_data)
    filt.Update()
    return filt.GetOutput(), filt


def _frontal_area(points_surf: np.ndarray) -> float:
    """Convex-hull area of the surface points projected onto the x-y plane.
    Ports `calculate_pos` -- scipy's 2D `ConvexHull.volume` is the hull's
    *area* (its `.area` attribute is the hull's perimeter instead).
    """
    hull = ConvexHull(points_surf[:, :2])
    return float(hull.volume)


def _mesh_cell_areas(unstructured_grid_data: Any) -> np.ndarray:
    """Per-cell area of each quad face. Ports `calculate_mesh_cell_area`."""
    import vtk

    poly_data, _ = _unstructured_grid_data_to_poly_data(unstructured_grid_data)
    points = poly_data.GetPoints()
    cells = poly_data.GetPolys()

    cell_areas = np.zeros(cells.GetNumberOfCells())
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    cell_id = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() == 4:
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            area = 0.5 * (
                np.linalg.norm(np.cross(p2 - p1, p3 - p1))
                + np.linalg.norm(np.cross(p3 - p1, p4 - p1))
            )
            cell_areas[cell_id] += area
            cell_id += 1
    return cell_areas


def _cell_velocity_gradient(
    unstructured_grid_data: Any, velocity: np.ndarray
) -> np.ndarray:
    """Crude per-quad finite-difference velocity gradient across each cell's
    diagonals -- an approximate wall-shear-stress proxy, not a true
    wall-normal gradient. Ports `calculate_cell_velocity_gradient` exactly,
    including that approximation (see module docstring).
    """
    from vtk.util.numpy_support import numpy_to_vtk

    velocity_c = np.ascontiguousarray(velocity, dtype=np.float64)
    velocity_arr = numpy_to_vtk(velocity_c, deep=True)
    velocity_arr.SetName("Velocity")
    unstructured_grid_data.GetPointData().AddArray(velocity_arr)

    poly_data, _ = _unstructured_grid_data_to_poly_data(unstructured_grid_data)
    points = poly_data.GetPoints()
    velocity_data = poly_data.GetPointData().GetArray("Velocity")

    import vtk

    cells = poly_data.GetPolys()
    grad_u = np.zeros((poly_data.GetNumberOfCells(), 3))
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    cell_id = 0
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() == 4:
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            u1 = np.array(velocity_data.GetTuple(cell.GetId(0)))
            u2 = np.array(velocity_data.GetTuple(cell.GetId(1)))
            u3 = np.array(velocity_data.GetTuple(cell.GetId(2)))
            u4 = np.array(velocity_data.GetTuple(cell.GetId(3)))

            du_dx = (u2 - u1 + u3 - u4) / (np.linalg.norm(p2 - p1 + p3 - p4) + 1e-8)
            du_dy = (u3 - u1 + u4 - u2) / (np.linalg.norm(p3 - p1 + p4 - p2) + 1e-8)
            du_dz = (u4 - u1 + u2 - u3) / (np.linalg.norm(p4 - p1 + p2 - p3) + 1e-8)

            grad_u[cell_id] += du_dx + du_dy + du_dz
            cell_id += 1
    return grad_u


def _cell_normals(unstructured_grid_data: Any) -> np.ndarray:
    """Per-*cell* surface normals, auto-oriented/consistency-enforced, no
    extra renormalization. Ports `drag_coefficient.py::get_normal` -- a
    different function from `shapenet_car.py`'s `_get_normal`
    (per-*point* normals, used as a model input feature, with an extra
    max-abs+L2 renormalization). The two are not interchangeable.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    poly_data, _ = _unstructured_grid_data_to_poly_data(unstructured_grid_data)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    return vtk_to_numpy(normal_filter.GetOutput().GetCellData().GetNormals())


def _point_values_to_cell(
    unstructured_grid_data: Any, point_values: np.ndarray, array_name: str
) -> np.ndarray:
    """Interpolates a per-point array to per-cell values via VTK's
    `vtkPointDataToCellData`, matching how the reference converts a
    caller-supplied `press_surf` (point data) into the per-cell values that
    `cell_areas`/`normal_surf` are indexed by.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    arr = numpy_to_vtk(np.ascontiguousarray(point_values, dtype=np.float64), deep=True)
    arr.SetName(array_name)
    unstructured_grid_data.GetPointData().AddArray(arr)

    c2p = vtk.vtkPointDataToCellData()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    return vtk_to_numpy(c2p.GetOutput().GetCellData().GetArray(array_name))


def load_surface_mesh(quadpress_vtk_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Surface mesh geometry for plotting: point coordinates and quad-cell
    connectivity (vertex indices into the returned points), in the same
    point order `compute_drag_coefficient` expects `press_surf`/
    `velo_surf` to be in (see module docstring's point-order requirement).
    Used by `scripts/benchmarks/_plot_utils.py`'s
    `plot_car_surface_pressure` to render the true CFD mesh instead of a
    2D projection/triangulation of the point cloud.

    Returns
    -------
    points : array, shape (N_points, 3)
    quads : array, shape (N_cells, 4), dtype int64
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    ug = _load_unstructured_grid_data(quadpress_vtk_path)
    poly_data, _ = _unstructured_grid_data_to_poly_data(ug)
    points = vtk_to_numpy(poly_data.GetPoints().GetData())

    cells = poly_data.GetPolys()
    quads = []
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    while cells.GetNextCell(cell):
        if cell.GetNumberOfIds() == 4:
            quads.append([cell.GetId(i) for i in range(4)])
    return points, np.asarray(quads, dtype=np.int64)


def _drag_force(
    cell_areas: np.ndarray,
    normal_z: np.ndarray,
    pressure: np.ndarray,
    velocity_gradient_z: np.ndarray,
    dynamic_viscosity: float,
) -> float:
    """Pressure (form) drag plus an approximate viscous/shear drag, both
    projected onto the flow (z) axis and summed over all surface cells.
    Ports `calculate_drag_force` exactly.
    """
    pressure_component = -np.dot(
        pressure.flatten() * cell_areas.flatten(), normal_z.flatten()
    )
    shear_component = (
        -np.dot(
            velocity_gradient_z.flatten() * cell_areas.flatten(), normal_z.flatten()
        )
        * dynamic_viscosity
    )
    return float(pressure_component + shear_component)


def compute_drag_coefficient(
    quadpress_vtk_path: str | Path,
    press_surf: np.ndarray,
    velo_surf: np.ndarray,
    rho: float = DEFAULT_AIR_DENSITY,
    freestream_velocity: float = DEFAULT_FREESTREAM_VELOCITY,
    dynamic_viscosity: float = DEFAULT_DYNAMIC_VISCOSITY,
) -> float:
    """Drag coefficient for one car mesh: `Cd = 2*F_drag / (rho*v^2*A)`.

    Geometry (cell connectivity, per-cell area/normal, frontal-area convex
    hull) always comes from `quadpress_vtk_path` on disk -- pass the
    *ground-truth* file for both a predicted and a ground-truth Cd (only
    `press_surf`/`velo_surf` differ between the two calls), matching the
    reference's own evaluation protocol.

    Parameters
    ----------
    quadpress_vtk_path : path
        the sample's `quadpress_smpl.vtk`.
    press_surf : array, shape (N_surf,) or (N_surf, 1)
        per-point pressure, denormalized, in the same point order as
        `quadpress_vtk_path`'s points (see module docstring).
    velo_surf : array, shape (N_surf, 3)
        per-point surface velocity, same requirements as `press_surf`.
    rho, freestream_velocity, dynamic_viscosity : float
        the paper's own hard-coded normalization constants (module
        docstring) -- override only to deviate deliberately from the
        reference's evaluation protocol.

    Returns
    -------
    float
        the drag coefficient for this sample.
    """
    ug_press = _load_unstructured_grid_data(quadpress_vtk_path)

    from vtk.util.numpy_support import vtk_to_numpy

    points_surf = vtk_to_numpy(ug_press.GetPoints().GetData())
    normal_surf = _cell_normals(ug_press)
    frontal_area = _frontal_area(points_surf)
    cell_areas = _mesh_cell_areas(ug_press)
    velo_surf_arr = np.asarray(velo_surf, dtype=np.float64)
    velocity_gradient = _cell_velocity_gradient(ug_press, velo_surf_arr)
    press_surf_arr = np.asarray(press_surf, dtype=np.float64).reshape(-1)
    press_cell = _point_values_to_cell(ug_press, press_surf_arr, "my_press")

    drag_force = _drag_force(
        cell_areas,
        normal_surf[:, -1],
        press_cell,
        velocity_gradient[:, -1],
        dynamic_viscosity,
    )

    return (2.0 / (freestream_velocity**2 * frontal_area * rho)) * drag_force


def mean_relative_drag_error(ground_truth: ArrayLike, predicted: ArrayLike) -> float:
    """Mean of `|pred - gt| / |gt|` over samples -- the paper's `c_d`
    evaluation metric (`main_evaluation.py`).
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(np.abs(predicted - ground_truth) / np.abs(ground_truth)))


def spearman_drag_correlation(ground_truth: ArrayLike, predicted: ArrayLike) -> float:
    """Spearman rank correlation between per-sample ground-truth and
    predicted drag coefficients -- the paper's reported "rho_d" metric
    (`main_evaluation.py`: ``scipy.stats.spearmanr(gt_coef_list,
    pred_coef_list)[0]``). Unrelated to the `rho` (air density) constant
    used inside `compute_drag_coefficient` -- see module docstring.
    """
    return float(spearmanr(np.asarray(ground_truth), np.asarray(predicted))[0])
