# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""The official AirfRANS scores of the Transolver paper, re-implemented.

Source: ``Airfoil-Design-AirfRANS/utils/metrics.py`` (``Infer_test``,
``Airfoil_test``, ``Compute_coefficients``, ``Results_test``) of
github.com/thuml/Transolver. Per test case:

* **inference** (:func:`cover_predict`): random 32,000-node subsets are
  predicted until every node has been hit at least once, and the predictions
  of each node are averaged;
* **wall values**: on wall nodes ``Ux``, ``Uy`` and ``nut`` are overwritten
  with their exact value 0 before scoring;
* **Volume / Surf** (:func:`field_scores`): MSE per channel in the official
  normalisation, on non-wall / wall nodes, averaged over the 4 channels. With
  the wall values imposed, Surf is in effect the wall-pressure MSE / 4;
* **forces** (:func:`force_coefficients`): wall shear stress from the VTK
  gradient of ``U`` (``nu = 1.56e-5``) plus pressure, integrated over the
  airfoil cells, rotated by the angle of attack: ``C = 2 F / U_inf^2``.

The paper reports the mean of Volume, Surf and ``|C_L - C_L,true| / |C_L,true|``
over the 200 test cases, and Spearman's rho of ``C_L`` over them. pyvista,
which the official code uses, is replaced by the VTK filters it wraps
(``vtkGradientFilter``, ``vtkPointDataToCellData``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from scripts.benchmarks.airfrans._data import NU, _read


def cover_predict(
    predict: Callable[[np.ndarray], np.ndarray],
    n_nodes: int,
    subsample: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Official ``Infer_test``: average of random-subset predictions.

    ``predict(idx)`` returns the prediction ``(len(idx), 4)`` for the nodes
    ``idx``. Returns the per-node average and the number of passes.
    """
    acc, hits, passes = None, np.zeros(n_nodes), 0
    while (hits == 0).any():
        idx = rng.choice(n_nodes, min(subsample, n_nodes), replace=False)
        out = np.asarray(predict(idx), dtype=np.float64)
        if acc is None:
            acc = np.zeros((n_nodes, out.shape[1]))
        acc[idx] += out
        hits[idx] += 1
        passes += 1
    return acc / hits[:, None], passes


def impose_wall(y: np.ndarray, surf: np.ndarray) -> np.ndarray:
    """Exact wall values: ``Ux = Uy = nut = 0`` on wall nodes (physical units)."""
    y = np.array(y, dtype=np.float64, copy=True)
    y[surf, 0] = y[surf, 1] = y[surf, 3] = 0.0
    return y


def field_scores(
    pred: np.ndarray,
    truth: np.ndarray,
    surf: np.ndarray,
    out_mean: np.ndarray,
    out_std: np.ndarray,
) -> dict:
    """Per-channel MSE in the official normalisation on volume / wall nodes.

    ``pred`` must already have the wall values imposed (:func:`impose_wall`).
    """
    z = lambda v: (v - out_mean) / (out_std + 1e-8)  # noqa: E731
    se = (z(pred) - z(truth)) ** 2
    vol, wall = se[~surf].mean(0), se[surf].mean(0)
    return dict(
        volume=float(vol.mean()),
        surf=float(wall.mean()),
        volume_per_channel=vol.tolist(),
        surf_per_channel=wall.tolist(),
    )


def _gradient(ds, name: str) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    g = vtk.vtkGradientFilter()  # pyvista DataSet.compute_derivative
    g.SetInputData(ds)
    g.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, name)
    g.SetResultArrayName("pred_grad")
    g.SetComputeGradient(True)
    g.SetComputeDivergence(False)
    g.SetComputeVorticity(False)
    g.SetComputeQCriterion(False)
    g.SetFasterApproximation(False)
    g.Update()
    return vtk_to_numpy(g.GetOutput().GetPointData().GetArray("pred_grad")).copy()


def _set_array(ds, name: str, values: np.ndarray) -> None:
    from vtk.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    arr.SetName(name)
    ds.GetPointData().RemoveArray(name)
    ds.GetPointData().AddArray(arr)


def _wall_shear_stress(jac: np.ndarray, normals: np.ndarray) -> np.ndarray:
    S = 0.5 * (jac + jac.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    return (2 * NU * S * normals[:, :2].reshape(-1, 1, 2)).sum(axis=2)


class CaseGeometry:
    """The VTK meshes of one case, read once, for force integration."""

    def __init__(self, data_dir: Path, case_id: str):
        from vtk.util.numpy_support import vtk_to_numpy

        folder = Path(data_dir) / case_id
        self.case_id = case_id
        self.internal = _read(folder / f"{case_id}_internal.vtu")
        self.airfoil = _read(folder / f"{case_id}_aerofoil.vtp", poly=True)
        U = vtk_to_numpy(self.internal.GetPointData().GetArray("U"))
        self.U = U.copy()
        wall = np.flatnonzero(U[:, 0] == 0)
        pts = vtk_to_numpy(self.internal.GetPoints().GetData())[:, :2]
        where = {tuple(q): i for i, q in enumerate(pts[wall].tolist())}
        apts = vtk_to_numpy(self.airfoil.GetPoints().GetData())[:, :2]
        # internal node of every airfoil point (the official reorganize)
        self.wall_order = np.array([wall[where[tuple(q)]] for q in apts.tolist()])
        self.u_inf = float(case_id.split("_")[2])
        self.alpha = float(case_id.split("_")[3])

    def outline(self) -> np.ndarray:
        """Airfoil outline as an ordered closed loop ``(n, 2)``, following the
        line cells of the airfoil surface mesh."""
        from vtk.util.numpy_support import vtk_to_numpy

        pts = vtk_to_numpy(self.airfoil.GetPoints().GetData())[:, :2]
        lines = vtk_to_numpy(self.airfoil.GetLines().GetData()).reshape(-1, 3)[:, 1:]
        nxt = dict(zip(lines[:, 0].tolist(), lines[:, 1].tolist(), strict=True))
        order = [int(lines[0, 0])]
        while nxt[order[-1]] != order[0] and len(order) < len(pts):
            order.append(nxt[order[-1]])
        return pts[order]

    def coefficients(self, fields: np.ndarray | None = None) -> np.ndarray:
        """``(C_D, C_L)`` of the truth (``fields=None``) or of physical
        ``(Ux, Uy, p, nut)`` node fields (official ``Compute_coefficients``)."""
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        internal = self.internal
        if fields is not None:
            internal = vtk.vtkUnstructuredGrid()
            internal.DeepCopy(self.internal)
            U = self.U.copy()
            U[:, :2] = fields[:, :2]
            _set_array(internal, "U", U)
            _set_array(internal, "p", fields[:, 2])
            _set_array(internal, "nut", fields[:, 3])
        grad = _gradient(internal, "U").reshape(-1, 3, 3)[self.wall_order, :2, :2]
        surf_p = vtk_to_numpy(internal.GetPointData().GetArray("p"))[self.wall_order]
        wss = _wall_shear_stress(
            grad, -vtk_to_numpy(self.airfoil.GetPointData().GetArray("Normals"))
        )
        airfoil = vtk.vtkPolyData()
        airfoil.DeepCopy(self.airfoil)
        _set_array(airfoil, "wallShearStress", wss)
        _set_array(airfoil, "p", surf_p)
        ptc = vtk.vtkPointDataToCellData()  # pyvista DataSet.ptc(pass_point_data=True)
        ptc.SetInputData(airfoil)
        ptc.SetPassPointData(True)
        ptc.Update()
        cells = ptc.GetOutput().GetCellData()
        p_c = vtk_to_numpy(cells.GetArray("p"))
        n_c = vtk_to_numpy(cells.GetArray("Normals"))
        length = vtk_to_numpy(cells.GetArray("Length")).reshape(-1, 1)
        wss_c = vtk_to_numpy(cells.GetArray("wallShearStress"))
        force = (wss_c * length).sum(0) - (-p_c[:, None] * n_c[:, :2] * length).sum(0)
        a = np.deg2rad(self.alpha)
        rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return 2 * (rot @ force) / self.u_inf**2


def summarise(rows: list[dict]) -> dict:
    """The paper's aggregates over the test cases, plus distributions."""
    from scipy.stats import spearmanr

    g = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    vol, cl_err = g("volume"), g("relerr_cl")
    return dict(
        n_cases=len(rows),
        volume=float(vol.mean()),
        surf=float(g("surf").mean()),
        relerr_cl=float(cl_err.mean()),
        spearman_cl=float(spearmanr(g("cl_true"), g("cl_pred"))[0]),
        relerr_cd=float(g("relerr_cd").mean()),
        spearman_cd=float(spearmanr(g("cd_true"), g("cd_pred"))[0]),
        volume_per_channel=dict(
            zip(
                ("Ux", "Uy", "p", "nut"),
                np.mean([r["volume_per_channel"] for r in rows], 0).tolist(),
            )
        ),
        volume_distribution=dict(
            median=float(np.median(vol)),
            p90=float(np.percentile(vol, 90)),
            max=float(vol.max()),
        ),
        relerr_cl_median=float(np.median(cl_err)),
    )
