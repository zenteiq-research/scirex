# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Cell-by-cell map of the results table to benchmark, model, run, checkpoint and metric.

Single source of truth for ``reproduce.py``. Paths are relative to the checkpoint root of
the original archive; ``hf_path`` gives the Hugging Face path. ``unit`` is the unit of the
table, and ``scale`` converts the metric stored by the benchmark (a fraction, or m/s) to it.
"""

from __future__ import annotations

from dataclasses import dataclass

MODELS = ("FNO", "WNO", "GINO", "Transolver", "GIWO")


@dataclass(frozen=True)
class Cell:
    row: str
    metric: str
    unit: str
    model: str
    group: str  # which runner evaluates it, see reproduce.RUNNERS
    ckpt: str | None = None  # path under the checkpoint root
    run: str | None = None  # run name the benchmark infer.py expects
    key: str | None = None  # metric selector inside the runner's result
    scale: float = 100.0


def _c(row, metric, unit, model, group, **kw):
    return Cell(row, metric, unit, model, group, **kw)


_DD = "darcy_s85/data_driven"
_PO = "darcy_s85/physics_only"
_NAS = "Checkpoints_Benchmarks"

CELLS: list[Cell] = [
    # ---- Darcy, data driven (rel. L2 %) ----
    _c(
        "Darcy",
        "rel. L2",
        "%",
        "FNO",
        "darcy",
        ckpt=f"{_DD}/fno/seed0",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy",
        "rel. L2",
        "%",
        "WNO",
        "darcy",
        ckpt=f"{_DD}/wno/seed0",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy",
        "rel. L2",
        "%",
        "GINO",
        "darcy",
        ckpt=f"{_DD}/gino/seed0",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy",
        "rel. L2",
        "%",
        "Transolver",
        "darcy",
        ckpt=f"{_DD}/transolver/seed2",
        key="final_test_rel_l2_mean",
    ),
    # ---- Darcy, label-free, 10^3 fields ----
    _c(
        "Darcy, label-free, 10^3 fields",
        "rel. L2",
        "%",
        "FNO",
        "darcy",
        ckpt=f"{_PO}/fno/fields_1000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, 10^3 fields",
        "rel. L2",
        "%",
        "WNO",
        "darcy",
        ckpt=f"{_PO}/wno/fields_1000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, 10^3 fields",
        "rel. L2",
        "%",
        "GINO",
        "darcy",
        ckpt=f"{_PO}/gino/fields_1000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, 10^3 fields",
        "rel. L2",
        "%",
        "Transolver",
        "darcy",
        ckpt=f"{_PO}/transolver/fields_1000",
        key="final_test_rel_l2_mean",
    ),
    # ---- Darcy, label-free, best over field counts ----
    _c(
        "Darcy, label-free, best",
        "rel. L2",
        "%",
        "FNO",
        "darcy",
        ckpt=f"{_PO}/fno/fields_10000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, best",
        "rel. L2",
        "%",
        "WNO",
        "darcy",
        ckpt=f"{_PO}/wno/fields_20000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, best",
        "rel. L2",
        "%",
        "GINO",
        "darcy",
        ckpt=f"{_PO}/gino/fields_20000",
        key="final_test_rel_l2_mean",
    ),
    _c(
        "Darcy, label-free, best",
        "rel. L2",
        "%",
        "Transolver",
        "darcy",
        ckpt=f"{_PO}/transolver/fields_10000",
        key="final_test_rel_l2_mean",
    ),
    # ---- Pipe (rel. L2 %) ----
    _c(
        "Pipe",
        "rel. L2",
        "%",
        "FNO",
        "pipe",
        ckpt="pipe/fno",
        run="fno",
        key="seeds.0.test_rel_l2.mean",
    ),
    _c(
        "Pipe",
        "rel. L2",
        "%",
        "WNO",
        "pipe",
        ckpt="pipe/wno",
        run="wno",
        key="seeds.0.test_rel_l2.mean",
    ),
    _c(
        "Pipe",
        "rel. L2",
        "%",
        "GINO",
        "pipe",
        ckpt="pipe/gino",
        run="gino",
        key="seeds.0.test_rel_l2.mean",
    ),
    _c(
        "Pipe",
        "rel. L2",
        "%",
        "Transolver",
        "pipe",
        ckpt="pipe/transolver",
        run="transolver_tuned",
        key="seeds.0.test_rel_l2.mean",
    ),
    # ---- Navier-Stokes (rollout rel. L2 %) ----
    _c(
        "Navier-Stokes",
        "rollout rel. L2",
        "%",
        "FNO",
        "navier_stokes",
        ckpt="navier_stokes/fno",
        run="fno",
        key="test_rel_l2.mean",
    ),
    _c(
        "Navier-Stokes",
        "rollout rel. L2",
        "%",
        "WNO",
        "navier_stokes",
        ckpt="navier_stokes/wno",
        run="wno",
        key="test_rel_l2.mean",
    ),
    _c(
        "Navier-Stokes",
        "rollout rel. L2",
        "%",
        "GINO",
        "navier_stokes",
        ckpt="navier_stokes/gino",
        run="gino",
        key="test_rel_l2.mean",
    ),
    _c(
        "Navier-Stokes",
        "rollout rel. L2",
        "%",
        "Transolver",
        "navier_stokes",
        ckpt="navier_stokes/transolver",
        run="transolver",
        key="test_rel_l2.mean",
    ),
    # ---- Cavity (RMSE 1e-3 m/s) ----
    _c(
        "Cavity, 1 step",
        "RMSE",
        "1e-3 m/s",
        "FNO",
        "cavity",
        ckpt="lid_driven_cavity_fno/fno_3to1",
        run="fno_3to1",
        key="single_step.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 1 step",
        "RMSE",
        "1e-3 m/s",
        "WNO",
        "cavity",
        ckpt="lid_driven_cavity_wno/wno_3to1",
        run="wno_3to1",
        key="single_step.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 1 step",
        "RMSE",
        "1e-3 m/s",
        "GINO",
        "cavity",
        ckpt="lid_driven_cavity_gino/gino_3to1",
        run="gino_3to1",
        key="single_step.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 1 step",
        "RMSE",
        "1e-3 m/s",
        "Transolver",
        "cavity",
        ckpt="lid_driven_cavity_transolver/transolver_3to1",
        run="transolver_3to1",
        key="single_step.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 10 steps",
        "RMSE",
        "1e-3 m/s",
        "FNO",
        "cavity",
        ckpt="lid_driven_cavity_fno/fno_3to1",
        run="fno_3to1",
        key="rollout.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 10 steps",
        "RMSE",
        "1e-3 m/s",
        "WNO",
        "cavity",
        ckpt="lid_driven_cavity_wno/wno_3to1",
        run="wno_3to1",
        key="rollout.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 10 steps",
        "RMSE",
        "1e-3 m/s",
        "GINO",
        "cavity",
        ckpt="lid_driven_cavity_gino/gino_3to1",
        run="gino_3to1",
        key="rollout.rmse",
        scale=1e3,
    ),
    _c(
        "Cavity, 10 steps",
        "RMSE",
        "1e-3 m/s",
        "Transolver",
        "cavity",
        ckpt="lid_driven_cavity_transolver/transolver_3to1",
        run="transolver_3to1",
        key="rollout.rmse",
        scale=1e3,
    ),
    # ---- Chip cooling ----
    _c(
        "Chip cooling (rel. L2)",
        "rel. L2",
        "%",
        "FNO",
        "chip_cooling",
        ckpt="chip_cooling/fno",
        run="fno",
        key="mean_RelL2",
    ),
    _c(
        "Chip cooling (rel. L2)",
        "rel. L2",
        "%",
        "WNO",
        "chip_cooling",
        ckpt="chip_cooling/wno",
        run="wno",
        key="mean_RelL2",
    ),
    _c(
        "Chip cooling (rel. L2)",
        "rel. L2",
        "%",
        "GINO",
        "chip_cooling",
        ckpt="chip_cooling/gino",
        run="gino",
        key="mean_RelL2",
    ),
    _c(
        "Chip cooling (rel. L2)",
        "rel. L2",
        "%",
        "Transolver",
        "archived",
        ckpt="chip_cooling/transolver_full_norm",
        key="mean_RelL2",
    ),
    _c(
        "Chip cooling (MAE)",
        "MAE",
        "K",
        "FNO",
        "chip_cooling",
        ckpt="chip_cooling/fno",
        run="fno",
        key="MAE",
        scale=1.0,
    ),
    _c(
        "Chip cooling (MAE)",
        "MAE",
        "K",
        "WNO",
        "chip_cooling",
        ckpt="chip_cooling/wno",
        run="wno",
        key="MAE",
        scale=1.0,
    ),
    _c(
        "Chip cooling (MAE)",
        "MAE",
        "K",
        "GINO",
        "chip_cooling",
        ckpt="chip_cooling/gino",
        run="gino",
        key="MAE",
        scale=1.0,
    ),
    _c(
        "Chip cooling (MAE)",
        "MAE",
        "K",
        "Transolver",
        "archived",
        ckpt="chip_cooling/transolver_full_norm",
        key="MAE",
        scale=1.0,
    ),
    # ---- AirfRANS ----
    _c(
        "AirfRANS (volume MSE)",
        "Volume MSE",
        "1e-3",
        "WNO",
        "airfrans",
        ckpt="airfrans/wno",
        run="wno",
        key="results.volume",
        scale=1e3,
    ),
    _c(
        "AirfRANS (volume MSE)",
        "Volume MSE",
        "1e-3",
        "GINO",
        "airfrans_gino",
        ckpt="airfrans/data_driven/gino/table3_500ep",
        run="gino",
        key="volume",
        scale=1e3,
    ),
    _c(
        "AirfRANS (volume MSE)",
        "Volume MSE",
        "1e-3",
        "Transolver",
        "airfrans",
        ckpt="airfrans/transolver",
        run="transolver",
        key="results.volume",
        scale=1e3,
    ),
    _c(
        "AirfRANS (C_L)",
        "C_L rel. error",
        "%",
        "WNO",
        "airfrans",
        ckpt="airfrans/wno",
        run="wno",
        key="results.relerr_cl",
    ),
    _c(
        "AirfRANS (C_L)",
        "C_L rel. error",
        "%",
        "GINO",
        "airfrans_gino",
        ckpt="airfrans/data_driven/gino/table3_500ep",
        run="gino",
        key="C_L_rel_err",
    ),
    _c(
        "AirfRANS (C_L)",
        "C_L rel. error",
        "%",
        "Transolver",
        "airfrans",
        ckpt="airfrans/transolver",
        run="transolver",
        key="results.relerr_cl",
    ),
    # ---- ShapeNetCar (surface rel. L2 %) ----
    _c(
        "ShapeNetCar",
        "surface rel. L2",
        "%",
        "GINO",
        "shapenetcar",
        ckpt="shapenetcar_gino",
        run="gino_wide_radius",
        key="rel_l2_surface",
    ),
    _c(
        "ShapeNetCar",
        "surface rel. L2",
        "%",
        "Transolver",
        "shapenetcar",
        ckpt="shapenetcar_transolver",
        run="transolver",
        key="rel_l2_surface",
    ),
    _c(
        "ShapeNetCar",
        "surface rel. L2",
        "%",
        "GIWO",
        "shapenetcar",
        ckpt="shapenetcar_giwo",
        run="giwo_transolver_cost_matched",
        key="rel_l2_surface",
    ),
    # ---- Car-CFD (rel. L2 %) ----
    _c(
        "Car-CFD",
        "rel. L2",
        "%",
        "GINO",
        "car_cfd",
        ckpt="car_cfd_gino",
        run="paper",
        key="final_test_err",
    ),
    _c(
        "Car-CFD",
        "rel. L2",
        "%",
        "Transolver",
        "car_cfd",
        ckpt="car_cfd_transolver",
        run="transolver_val",
        key="final_test_err",
    ),
    _c(
        "Car-CFD",
        "rel. L2",
        "%",
        "GIWO",
        "car_cfd",
        ckpt="car_cfd_giwo",
        run="giwo_val",
        key="final_test_err",
    ),
]

# Cells of the results table marked "-" (model not run for this problem).
NOT_RUN = {
    "AirfRANS (volume MSE)": ("FNO", "GIWO"),
    "AirfRANS (C_L)": ("FNO", "GIWO"),
    "ShapeNetCar": ("FNO", "WNO"),
    "Car-CFD": ("FNO", "WNO"),
}


def rows() -> list[str]:
    seen: list[str] = []
    for c in CELLS:
        if c.row not in seen:
            seen.append(c.row)
    return seen


# ------------------------------------------------- Hugging Face checkpoint layout
# Every checkpoint is its own Hugging Face repository,
# ``<org>/scirex-<model>-<problem>`` (``zenteiq-ai/scirex-fno-pipe``). Downloaded into
# ``<ckpt-root>/<model>/<problem>/`` (and for Darcy ``<model>/darcy_s85/<regime>/``)
# the repositories form the folder layout that ``reproduce.py --layout hf`` reads.
# ``Cell.ckpt`` stays the path in the original archive; these functions give the
# matching layout path and Hugging Face repository.
HF_ORG = "zenteiq-ai"
# problem folder -> repository suffix; Darcy is split by regime below.
# Cavity has no repository.
HF_SLUG = {
    "airfrans": "airfrans",
    "car_cfd": "car-cfd",
    "chip_cooling": "chip-cooling",
    "navier_stokes": "navier-stokes",
    "pipe": "pipe",
    "shapenetcar": "shapenetcar",
}
PROBLEMS = (
    "airfrans",
    "car_cfd",
    "cavity",
    "chip_cooling",
    "darcy_s85",
    "navier_stokes",
    "pipe",
    "shapenetcar",
)


def problem_of(cell: Cell) -> str | None:
    """Problem folder of the cell's checkpoint, or None without a checkpoint."""
    return problem_of_ckpt(cell.ckpt)


def problem_of_ckpt(ckpt: str | None) -> str | None:
    """Problem folder of an archive checkpoint path, or None without one."""
    if not ckpt:
        return None
    top = ckpt.split("/")[0]
    for prefix, problem in (
        ("lid_driven_cavity_", "cavity"),
        ("car_cfd_", "car_cfd"),
        ("shapenetcar_", "shapenetcar"),
    ):
        if top.startswith(prefix):
            return problem
    return top  # airfrans, chip_cooling, darcy_s85, navier_stokes, pipe


def hf_path(cell: Cell) -> str | None:
    """Path of the cell's checkpoint in the Hugging Face layout (``fno/pipe``)."""
    problem = problem_of(cell)
    if problem is None:
        return None
    path = f"{cell.model.lower()}/{problem}"
    if problem == "darcy_s85":
        parts = cell.ckpt.split("/")
        if parts[1] == "data_driven":
            path += "/data_driven"
        else:
            path += (
                "/physics_only_1000"
                if parts[-1] == "fields_1000"
                else "/physics_only_best"
            )
    return path


def hf_repo_of(model: str, ckpt: str | None, org: str = HF_ORG) -> str | None:
    """Hugging Face repository of a checkpoint (``zenteiq-ai/scirex-fno-pipe``).

    ``model`` is the model name (``FNO``) and ``ckpt`` the path in the original archive.
    The Cavity data is not public, so its checkpoints have no repository.
    """
    problem = problem_of_ckpt(ckpt)
    if problem is None:
        return None
    if problem == "darcy_s85":
        parts = ckpt.split("/")
        if parts[1] == "data_driven":
            slug = "darcy"
        else:
            slug = (
                "darcy-physics"
                if parts[-1] == "fields_1000"
                else "darcy-physics-extra-fields"
            )
    else:
        slug = HF_SLUG.get(problem)
        if slug is None:
            return None
    return f"{org}/scirex-{model.lower()}-{slug}"
