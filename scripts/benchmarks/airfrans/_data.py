# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""AirfRANS data: the official node features, the split, and normalisation.

Node features follow the official Transolver ``dataset/dataset.py``
(``Dataset(..., sample=None)``), node by node on the internal mesh, no crop::

    x7 = [pos_x, pos_y, U_inf cos(a), U_inf sin(a), -implicit_distance,
          normal_x, normal_y]

``U_inf`` and the angle of attack ``a`` (degrees) are read from the case
name; the normal is ``-Normals`` of the airfoil surface on wall nodes and 0
elsewhere; wall nodes are those with ``U_x == 0``. Targets are
``(Ux, Uy, p, nut)`` (``p`` is the kinematic gauge pressure).

On top of that, :func:`features` appends ``log10(|wall distance| + 1e-6)``
and ``log10(Re)`` (9 features), and :func:`target_scale` divides the targets
by the inflow: ``(Ux, Uy) / U_inf``, ``p / U_inf^2``, ``nut / U_inf``. Both are
pure input / output transformations of the same data.

Two normalisations exist:

* the model's own (:func:`model_stats`): mean and population std of the 9
  features and the scaled targets over every node of the 720 training cases;
* the official one (:func:`official_output_stats`): of the raw targets over
  the same nodes, computed with ``dataset.py``'s running float64 update. It is
  used only to score predictions on the paper's scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NU = 1.56e-5  # kinematic viscosity of the AirfRANS simulations (official metrics.py)
N_FEATURES = 9


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]


def load_split(data_dir: Path, n_val: int = 80) -> Split:
    """The official split: ``full_train[:-n_val]`` / ``full_train[-n_val:]`` /
    ``full_test`` from ``manifest.json``."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    train = manifest["full_train"]
    return Split(train=train[:-n_val], val=train[-n_val:], test=manifest["full_test"])


def _read(path: Path, poly: bool = False):
    import vtk

    reader = vtk.vtkXMLPolyDataReader() if poly else vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def build_case(data_dir: Path, case_id: str) -> dict[str, np.ndarray]:
    """The official raw features of one case, read from its VTU / VTP files."""
    from vtk.util.numpy_support import vtk_to_numpy

    folder = Path(data_dir) / case_id
    internal = _read(folder / f"{case_id}_internal.vtu")
    airfoil = _read(folder / f"{case_id}_aerofoil.vtp", poly=True)
    pts = vtk_to_numpy(internal.GetPoints().GetData())[:, :2].astype(np.float32)
    fields = internal.GetPointData()
    U = vtk_to_numpy(fields.GetArray("U"))
    p = vtk_to_numpy(fields.GetArray("p"))
    nut = vtk_to_numpy(fields.GetArray("nut"))
    sdf = vtk_to_numpy(fields.GetArray("implicit_distance"))
    surf = U[:, 0] == 0
    u_inf, alpha = (
        float(case_id.split("_")[2]),
        np.deg2rad(float(case_id.split("_")[3])),
    )
    inflow = np.broadcast_to(
        np.array([np.cos(alpha), np.sin(alpha)]) * u_inf, (len(pts), 2)
    )
    normal = np.zeros((len(pts), 2))
    apts = vtk_to_numpy(airfoil.GetPoints().GetData())[:, :2].astype(np.float32)
    anormal = -vtk_to_numpy(airfoil.GetPointData().GetArray("Normals"))[:, :2]
    where = {tuple(q): i for i, q in enumerate(apts.tolist())}  # official reorganize
    wall = np.flatnonzero(surf)
    normal[wall] = anormal[[where[tuple(q)] for q in pts[wall].tolist()]]
    x7 = np.concatenate([pts, inflow, -sdf[:, None], normal], axis=1).astype(np.float32)
    y = np.column_stack([U[:, 0], U[:, 1], p, nut]).astype(np.float32)
    return dict(
        x7=x7, y=y, pos=pts, surf=surf, segments=airfoil_segments(data_dir, case_id)
    )


def airfoil_segments(data_dir: Path, case_id: str) -> np.ndarray:
    """The airfoil outline as line segments ``(S, 2, 2)``, from the line cells
    of ``<case>_aerofoil.vtp`` (used by grid-based models)."""
    from vtk.util.numpy_support import vtk_to_numpy

    airfoil = _read(Path(data_dir) / case_id / f"{case_id}_aerofoil.vtp", poly=True)
    pts = vtk_to_numpy(airfoil.GetPoints().GetData())[:, :2]
    lines = vtk_to_numpy(airfoil.GetLines().GetData()).reshape(-1, 3)[:, 1:]
    return pts[lines].astype(np.float32)


def load_case(data_dir: Path, cache_dir: Path, case_id: str) -> dict[str, np.ndarray]:
    """:func:`build_case`, cached as ``<cache_dir>/<case_id>.npz``.

    A cache file written before ``segments`` existed is completed in place.
    """
    path = Path(cache_dir) / f"{case_id}.npz"
    if path.exists():
        with np.load(path) as f:
            case = {k: f[k] for k in f.files}
        if "segments" not in case:
            case["segments"] = airfoil_segments(data_dir, case_id)
            np.savez(path, **case)
        return case
    case = build_case(data_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **case)
    return case


def inflow_speed(x7: np.ndarray) -> float:
    return float(np.linalg.norm(x7[0, 2:4]))


def features(x7: np.ndarray) -> np.ndarray:
    """The 7 official features plus log10 wall distance and log10 Re."""
    log_d = np.log10(np.abs(x7[:, 4:5]) + 1e-6)
    log_re = np.full((len(x7), 1), np.log10(inflow_speed(x7) / NU))
    return np.concatenate([x7, log_d, log_re], axis=1).astype(np.float32)


def target_scale(u_inf: float) -> np.ndarray:
    """Per-channel divisor of the targets: U_inf, U_inf, U_inf^2, U_inf."""
    return np.array([u_inf, u_inf, u_inf**2, u_inf], np.float32)


@dataclass(frozen=True)
class Stats:
    """Mean and population std of the model's inputs and scaled targets."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def encode_x(self, x9: np.ndarray) -> np.ndarray:
        return ((x9 - self.x_mean) / (self.x_std + 1e-8)).astype(np.float32)

    def encode_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return ((y_scaled - self.y_mean) / (self.y_std + 1e-8)).astype(np.float32)

    def decode_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (self.y_std + 1e-8) + self.y_mean

    def to_dict(self) -> dict:
        return {k: np.asarray(v, np.float64).tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> Stats:
        return Stats(**{k: np.asarray(v, np.float32) for k, v in d.items()})


def model_stats(cases: list[dict]) -> Stats:
    """Float64 mean and population std over every node of ``cases``."""
    n, sx, sxx, sy, syy = 0, 0.0, 0.0, 0.0, 0.0
    for c in cases:
        x = features(c["x7"]).astype(np.float64)
        y = (c["y"] / target_scale(inflow_speed(c["x7"]))).astype(np.float64)
        n += len(x)
        sx, sxx = sx + x.sum(0), sxx + (x**2).sum(0)
        sy, syy = sy + y.sum(0), syy + (y**2).sum(0)
    mx, my = sx / n, sy / n
    # A constant column has zero variance, which E[x^2] - E[x]^2 can round to a
    # tiny negative number; clip it so the std is 0 instead of NaN.
    return Stats(
        x_mean=mx.astype(np.float32),
        x_std=np.sqrt(np.maximum(sxx / n - mx**2, 0.0)).astype(np.float32),
        y_mean=my.astype(np.float32),
        y_std=np.sqrt(np.maximum(syy / n - my**2, 0.0)).astype(np.float32),
    )


def official_output_stats(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """``coef_norm[2:]`` of the official ``dataset.py``: running float64 mean,
    then running float64 variance around the float32-cast mean."""
    n, mean = 0, 0.0
    for c in cases:
        k = len(c["y"])
        mean = mean + (c["y"].sum(0, dtype=np.float64) - k * mean) / (n + k)
        n += k
    mean = np.asarray(mean).astype(np.float32)
    n, var = 0, 0.0
    for c in cases:
        k = len(c["y"])
        var = var + (((c["y"] - mean) ** 2).sum(0, dtype=np.float64) - k * var) / (
            n + k
        )
        n += k
    return mean, np.sqrt(var).astype(np.float32)


def prepare_case(case: dict, stats: Stats) -> dict[str, np.ndarray]:
    """Normalised model inputs and targets of one case, plus what scoring needs."""
    u_inf = inflow_speed(case["x7"])
    scale = target_scale(u_inf)
    return dict(
        x=stats.encode_x(features(case["x7"])),
        y=stats.encode_y(case["y"] / scale),
        pos=case["pos"],
        surf=case["surf"],
        scale=scale,
    )


def to_physical(pred_norm: np.ndarray, scale: np.ndarray, stats: Stats) -> np.ndarray:
    """Model output (normalised scaled targets) to ``(Ux, Uy, p, nut)``."""
    return stats.decode_y(pred_norm) * scale
