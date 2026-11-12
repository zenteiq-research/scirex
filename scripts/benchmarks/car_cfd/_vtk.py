# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""VTK export of the Car-CFD surface pressure (truth vs. prediction).

Writes vertex-centered scalar fields (e.g. pressure) onto a surface mesh as
.vtu (VTK XML Unstructured Grid) files that can be opened in ParaView for
interactive 3D visualisation (rotate, zoom, change colour maps). Supports
either triangle cells (GINO's car-CFD mesh, via open3d) or quad cells
(Transolver's Car-Design-ShapeNetCar mesh, ``quadpress_smpl.vtk``) -- cell
type is inferred from the cell array's vertex count (3 or 4 per cell).

Outputs (gitignored; see scripts/benchmarks/vtk_outputs/):
    truth.vtu  - ground-truth scalar field
    pred.vtu   - model-predicted scalar field
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ``vtk`` is imported inside the writer, so a missing ``vtk`` package only
# costs the ParaView export, not the rest of ``infer.py``.


def export_vtk_comparison(
    vertices: np.ndarray,
    cells: np.ndarray,
    truth_scalars: np.ndarray,
    pred_scalars: np.ndarray,
    out_dir: Path | str,
    scalar_name: str = "Pressure",
) -> None:
    """Write truth.vtu and pred.vtu to *out_dir*.

    Parameters
    ----------
    vertices:
        Float array of shape ``(n_vertices, 3)`` - 3D coordinates.
    cells:
        Integer array of shape ``(n_cells, 3)`` (triangles) or
        ``(n_cells, 4)`` (quads) - vertex indices per cell.
    truth_scalars:
        Float array of shape ``(n_vertices,)`` - ground-truth scalar field.
    pred_scalars:
        Float array of shape ``(n_vertices,)`` - predicted scalar field.
    out_dir:
        Directory in which to write the ``.vtu`` files (created if absent).
    scalar_name:
        Name of the point-data array written into the ``.vtu`` files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_vtu(
        vertices,
        cells,
        truth_scalars,
        out_dir / "truth.vtu",
        scalar_name=scalar_name,
    )
    _write_vtu(
        vertices, cells, pred_scalars, out_dir / "pred.vtu", scalar_name=scalar_name
    )
    print(f"[vtk] wrote truth.vtu and pred.vtu to {out_dir}")


def _write_vtu(
    vertices: np.ndarray,
    cells: np.ndarray,
    scalars: np.ndarray,
    path: Path,
    scalar_name: str = "Pressure",
) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    scalars = np.ascontiguousarray(scalars, dtype=np.float64).ravel()

    n_verts_per_cell = cells.shape[1]
    if n_verts_per_cell == 3:
        vtk_cell_type = vtk.VTK_TRIANGLE
    elif n_verts_per_cell == 4:
        vtk_cell_type = vtk.VTK_QUAD
    else:
        raise ValueError(
            f"Unsupported cell size {n_verts_per_cell} (expected 3 for "
            "triangles or 4 for quads)."
        )

    # Points
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(vertices, deep=True))

    # Cell connectivity: [k, i0, ..., i(k-1),  k, i0, ..., i(k-1), ...]
    n_cells = len(cells)
    cell_conn = np.hstack(
        [np.full((n_cells, 1), n_verts_per_cell, dtype=np.int64), cells]
    ).ravel()
    id_arr = numpy_to_vtkIdTypeArray(cell_conn, deep=True)
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(n_cells, id_arr)

    # Unstructured grid with uniform cell type (VTK_TRIANGLE or VTK_QUAD).
    # Use an explicit UnsignedChar array for cell types - SetCells(int, cells)
    # creates an implicit constant array that VTK 9.x ASCII writer cannot iterate.
    types_np = np.full(n_cells, vtk_cell_type, dtype=np.uint8)
    cell_types_arr = numpy_to_vtk(types_np, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(pts)
    ug.SetCells(cell_types_arr, vtk_cells)

    # Scalar field as point-data
    s_arr = numpy_to_vtk(scalars, deep=True)
    s_arr.SetName(scalar_name)
    ug.GetPointData().SetScalars(s_arr)

    # Write XML unstructured grid (.vtu) in ASCII for broad ParaView compatibility
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ug)
    writer.SetDataModeToAscii()
    writer.Write()
