# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Small synthetic AirfRANS cases for the AirfRANS tutorial tests.

Each case is written in the real layout, so every reader of the tutorials and
of the benchmark can open it: ``<case>/<case>_internal.vtu`` (a quad mesh of
the ring between a circle of radius 0.25 and one of radius 2, with ``U``,
``p``, ``nut`` and ``implicit_distance``), ``<case>_aerofoil.vtp`` (the circle,
with its line cells, ``Normals`` and ``Length``) and ``<case>_freestream.vtp``
(the outer circle, with ``U``), plus ``manifest.json``. The flow is a smooth
made-up field with ``U = 0`` exactly on the wall, so the wall nodes are those
with ``Ux == 0``, as in the real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R_WALL, R_OUT, CENTRE = 0.25, 2.0, (0.5, 0.0)


def _case_id(u_inf: float, aoa: float, k: int) -> str:
    return f"airFoil2D_SST_{u_inf:.3f}_{aoa:.3f}_0.5_1.5_{k}.0_12.0"


def _write(dataset, path: Path) -> None:
    import vtk

    writer = (
        vtk.vtkXMLPolyDataWriter()
        if str(path).endswith(".vtp")
        else vtk.vtkXMLUnstructuredGridWriter()
    )
    writer.SetFileName(str(path))
    writer.SetInputData(dataset)
    writer.Write()


def _array(name: str, values: np.ndarray):
    from vtk.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    arr.SetName(name)
    return arr


def _points(xy: np.ndarray):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = vtk.vtkPoints()
    xyz = np.column_stack([xy, np.zeros(len(xy))]).astype(np.float32)
    pts.SetData(numpy_to_vtk(xyz, deep=True))
    return pts


def _circle(r: float, n: int, closed_lines: bool):
    """Points of a circle around CENTRE, its outward normals and line cells."""
    import vtk

    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([CENTRE[0] + r * np.cos(t), CENTRE[1] + r * np.sin(t)], 1)
    normals = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
    poly = vtk.vtkPolyData()
    poly.SetPoints(_points(xy))
    lines = vtk.vtkCellArray()
    for i in range(n if closed_lines else n - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, (i + 1) % n)
        lines.InsertNextCell(line)
    poly.SetLines(lines)
    return xy.astype(np.float32), normals, poly


def write_case(
    root: Path, u_inf: float, aoa: float, k: int, n_t: int = 48, n_r: int = 20
) -> str:
    """Write one synthetic case under ``root`` and return its id."""
    import vtk

    cid = _case_id(u_inf, aoa, k)
    folder = Path(root) / cid
    folder.mkdir(parents=True, exist_ok=True)
    a = np.deg2rad(aoa)

    # ring mesh: radial index j = 0 is the wall; geometric spacing towards it
    t = np.linspace(0.0, 2 * np.pi, n_t, endpoint=False)
    s = (np.geomspace(1.0, 101.0, n_r) - 1.0) / 100.0
    r = R_WALL + (R_OUT - R_WALL) * s
    rr, tt = np.meshgrid(r, t, indexing="ij")  # (n_r, n_t)
    xy = (
        np.stack([CENTRE[0] + rr * np.cos(tt), CENTRE[1] + rr * np.sin(tt)], -1)
        .reshape(-1, 2)
        .astype(np.float32)
    )
    # The wall ring must be bit-identical to the airfoil points (float32).
    wall_xy, normals, airfoil = _circle(R_WALL, n_t, closed_lines=True)
    xy[:n_t] = wall_xy

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(_points(xy))
    for j in range(n_r - 1):
        for i in range(n_t):
            quad = vtk.vtkQuad()
            ids = (j * n_t + i, j * n_t + (i + 1) % n_t,
                   (j + 1) * n_t + (i + 1) % n_t, (j + 1) * n_t + i)  # fmt: skip
            for m, pid in enumerate(ids):
                quad.GetPointIds().SetId(m, pid)
            grid.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

    rad = np.linalg.norm(xy - np.array(CENTRE, np.float32), axis=1)
    ramp = 1.0 - (R_WALL / np.maximum(rad, R_WALL)) ** 2  # 0 on the wall
    theta = np.arctan2(xy[:, 1] - CENTRE[1], xy[:, 0] - CENTRE[0])
    ux = u_inf * np.cos(a) * ramp * (1.0 + 0.2 * np.sin(theta) ** 2)
    uy = u_inf * np.sin(a) * ramp + 0.05 * u_inf * ramp * np.cos(theta)
    ux[:n_t] = 0.0
    uy[:n_t] = 0.0
    U = np.column_stack([ux, uy, np.zeros_like(ux)]).astype(np.float32)
    p = (0.5 * u_inf**2 * (1.0 - 4 * np.sin(theta + a) ** 2 * (1 - ramp))).astype(
        np.float32
    )
    nut = (1e-4 * u_inf * ramp * np.exp(-rad)).astype(np.float32)
    nut[:n_t] = 0.0
    sdf = (rad - R_WALL).astype(np.float32)
    sdf[:n_t] = 0.0
    for name, values in (("U", U), ("p", p), ("nut", nut), ("implicit_distance", sdf)):
        grid.GetPointData().AddArray(_array(name, values))
    _write(grid, folder / f"{cid}_internal.vtu")

    airfoil.GetPointData().AddArray(_array("Normals", normals.astype(np.float32)))
    length = np.full(n_t, 2 * np.pi * R_WALL / n_t, np.float32)
    airfoil.GetCellData().AddArray(_array("Length", length))
    _write(airfoil, folder / f"{cid}_aerofoil.vtp")

    _, _, far = _circle(R_OUT, n_t, closed_lines=True)
    far_u = np.tile([u_inf * np.cos(a), u_inf * np.sin(a), 0.0], (n_t, 1)).astype(
        np.float32
    )
    far.GetPointData().AddArray(_array("U", far_u))
    _write(far, folder / f"{cid}_freestream.vtp")
    return cid


def write_dataset(root: Path, n_train: int = 6, n_test: int = 3) -> Path:
    """``n_train + n_test`` cases and a ``manifest.json`` with the full task."""
    rng = np.random.default_rng(0)
    ids = [
        write_case(root, float(rng.uniform(30, 90)), float(rng.uniform(-5, 15)), k)
        for k in range(n_train + n_test)
    ]
    manifest = dict(full_train=ids[:n_train], full_test=ids[n_train:])
    (Path(root) / "manifest.json").write_text(json.dumps(manifest))
    return Path(root)
