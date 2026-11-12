# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Dataset loader and grid utilities for the two-chip electronics-cooling
CFD benchmark in `scripts/benchmarks/chip_cooling/` (FNO, GINO, WNO,
Transolver -- all data-driven).

Pure NumPy/SciPy -- no JAX/Flax import here, so it's safe to use from
config modules as well as training scripts.

Expects a directory layout of::

    <dataset_dir>/
        model_setup.json      # DOE parameter grid (doe_configuration)
        sim_*.npz             # one file per simulation, holding
                               # FIELD_KEY (temperature) and COORD_KEY
                               # (mesh coordinates)

`<dataset_dir>` defaults to `<repo_root>/1_Dataset`, overridable via the
`CHIPCOOLING_DATA_DIR` environment variable (mirrors the
`CAVITY_DATA_DIR` convention used by the lid-driven-cavity suite). This
dataset is not bundled with the repository; set the environment variable to
point at a local copy before running the benchmark's `train.py` or
`infer.py`.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants re-exported so callers can do
# `from scirex.data.datasets.chipcooling import FIELD_KEY`
# ---------------------------------------------------------------------------
FIELD_KEY = "yz-mid|temperature"
COORD_KEY = "yz-mid|coordinates"

TRAIN_SIZE = 1600  # 80:20 split of the 2000-sample pool, matching Curl & Hu
TEST_SIZE = 400

GRID_H = 128
GRID_W = 256

CACHE_VERSION = 3
N_INPUT_CHANNELS = 7

CHIP_AREA = 0.0016
PARAM_IDX = {"Q1": 0, "Q2": 1, "m1": 2, "T_in1": 3, "m2": 4, "T_in2": 5}

# <repo_root>/1_Dataset, overridable via CHIPCOOLING_DATA_DIR.
DEFAULT_DATASET_DIR = os.environ.get(
    "CHIPCOOLING_DATA_DIR",
    str(Path(__file__).resolve().parents[4] / "1_Dataset"),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _sim_sort_key(path):
    """Numeric sort key for `sim_<N>.npz` files.

    `sorted()` on the raw paths sorts lexicographically, which orders
    `sim_10` before `sim_2`. The i-th simulation file must line up with the
    i-th DOE parameter row (see `load_data`), so sort by the integer index
    embedded in the name instead. Falls back to the name when no digits are
    found so the sort stays total and deterministic.
    """
    m = re.search(r"(\d+)", path.stem)
    return (0, int(m.group(1))) if m else (1, path.name)


def load_data(dataset_dir=None):
    """
    Load DOE parameters, CFD temperature fields and mesh coordinates.

    Parameters
    ----------
    dataset_dir : str | Path | None
        Directory holding `model_setup.json` and the `sim_*.npz` files.
        Defaults to `DEFAULT_DATASET_DIR` (`CHIPCOOLING_DATA_DIR` env var,
        or `<repo_root>/1_Dataset`).

    Returns
    -------
    parameters  : (N, 6) float32 ndarray
    fields      : (N, n_points) float32 ndarray
    coords      : (n_points, 3) float32 ndarray
    param_names : list[str]
    """
    dataset_dir = (
        Path(dataset_dir) if dataset_dir is not None else Path(DEFAULT_DATASET_DIR)
    )

    with open(dataset_dir / "model_setup.json") as f:
        setup = json.load(f)

    doe = setup["doe_configuration"]
    param_names, columns = [], []

    for bc in sorted(doe.keys()):
        for pname in sorted(doe[bc].keys()):
            vals = doe[bc][pname]
            if isinstance(vals, list):
                param_names.append(f"{bc}.{pname}")
                columns.append(vals)

    parameters = np.asarray(columns, dtype=np.float32).T

    sim_files = sorted(dataset_dir.glob("sim_*.npz"), key=_sim_sort_key)
    if len(sim_files) != len(parameters):
        raise RuntimeError(
            f"Found {len(sim_files)} simulation files but "
            f"{len(parameters)} parameter combinations."
        )

    fields, coords = [], None
    for fp in sim_files:
        # allow_pickle=False: the arrays are plain numeric data, and unpickling
        # an untrusted .npz would allow arbitrary code execution.
        with np.load(fp, allow_pickle=False) as d:
            fields.append(np.asarray(d[FIELD_KEY], dtype=np.float32))
            if coords is None:
                coords = np.asarray(d[COORD_KEY], dtype=np.float32)

    return parameters, np.asarray(fields, dtype=np.float32), coords, param_names


# ---------------------------------------------------------------------------
# Grid / mask construction
# ---------------------------------------------------------------------------


def build_domain_mask(coords, y_grid, z_grid):
    """
    Boolean mask (H, W) -- True where at least one CFD node falls in the cell.
    Built from coordinates only; no temperature field is used.
    """
    y = coords[:, 1]
    z = coords[:, 2]

    def edges(grid):
        e = np.empty(len(grid) + 1, dtype=np.float64)
        e[1:-1] = 0.5 * (grid[:-1] + grid[1:])
        e[0] = grid[0] - 0.5 * (grid[1] - grid[0])
        e[-1] = grid[-1] + 0.5 * (grid[-1] - grid[-2])
        return e

    y_edges = edges(y_grid)
    z_edges = edges(z_grid)

    iy = np.searchsorted(y_edges, y, side="right") - 1
    iz = np.searchsorted(z_edges, z, side="right") - 1

    valid = (iy >= 0) & (iy < len(y_grid)) & (iz >= 0) & (iz < len(z_grid))
    mask = np.zeros((len(y_grid), len(z_grid)), dtype=bool)
    mask[iy[valid], iz[valid]] = True
    return mask


def build_grid_cache(coords, fields, cache_path: Path, grid_h=GRID_H, grid_w=GRID_W):
    """
    Rasterise all CFD fields onto a regular (grid_h x grid_w) grid, or load
    a compatible cache from disk.

    Returns
    -------
    grid_fields_nan : (N, H, W) float32  -- NaN outside domain
    grid_fields     : (N, H, W) float32  -- 0.0 outside domain
    mask            : (H, W) bool
    y_grid          : (H,) float64
    z_grid          : (W,) float64
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            d = np.load(cache_path)
            required = {
                "cache_version",
                "grid_fields_nan",
                "grid_fields",
                "mask",
                "y_grid",
                "z_grid",
            }
            if required.issubset(d.files):
                if int(d["cache_version"]) == CACHE_VERSION and d["grid_fields"].shape[
                    1:
                ] == (grid_h, grid_w):
                    logger.info(f"Loading cached FNO grid data from {cache_path}")
                    return (
                        d["grid_fields_nan"],
                        d["grid_fields"],
                        d["mask"].astype(bool),
                        d["y_grid"],
                        d["z_grid"],
                    )
            logger.info("Existing cache is incompatible. Rebuilding.")
        except Exception as exc:
            logger.info(f"Could not read cache ({exc}). Rebuilding.")

    y = coords[:, 1]
    z = coords[:, 2]
    y_grid = np.linspace(y.min(), y.max(), grid_h)
    z_grid = np.linspace(z.min(), z.max(), grid_w)
    Zg, Yg = np.meshgrid(z_grid, y_grid)

    mask = build_domain_mask(coords, y_grid, z_grid)

    n = len(fields)
    grid_fields_nan = np.full((n, grid_h, grid_w), np.nan, dtype=np.float32)

    logger.info(f"Rasterising {n} CFD fields onto {grid_h}x{grid_w} grid...")
    t0 = time.time()

    for i in range(n):
        vals = griddata((z, y), fields[i], (Zg, Yg), method="linear")
        missing = mask & ~np.isfinite(vals)
        if np.any(missing):
            nearest = griddata((z, y), fields[i], (Zg, Yg), method="nearest")
            vals[missing] = nearest[missing]
        vals[~mask] = np.nan
        grid_fields_nan[i] = vals.astype(np.float32)
        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{n} ({time.time()-t0:.0f}s elapsed)")

    grid_fields = np.nan_to_num(grid_fields_nan, nan=0.0).astype(np.float32)

    np.savez_compressed(
        cache_path,
        cache_version=np.array(CACHE_VERSION, dtype=np.int32),
        grid_fields_nan=grid_fields_nan,
        grid_fields=grid_fields,
        mask=mask.astype(np.uint8),
        y_grid=y_grid,
        z_grid=z_grid,
    )
    logger.info(f"Cached FNO grid data to {cache_path}")
    return grid_fields_nan, grid_fields, mask, y_grid, z_grid


# ---------------------------------------------------------------------------
# Input / target construction and normalisation
# ---------------------------------------------------------------------------


def build_input_channels(parameters, mask, H=GRID_H, W=GRID_W):
    """
    Build the (N, 7, H, W) FNO input tensor.
    Channels 0-5 = physical parameters broadcast over the grid.
    Channel 6    = CFD-domain mask (0/1).
    """
    n, n_params = parameters.shape
    if n_params != 6:
        raise ValueError(f"Expected 6 parameters, got {n_params}.")

    X = np.zeros((n, 7, H, W), dtype=np.float32)
    for i in range(6):
        X[:, i] = parameters[:, i, None, None]
    X[:, 6] = mask.astype(np.float32)
    return X


def normalize_inputs(X_train, X_test):
    """
    Z-score normalise the six physical parameter channels (0-5).
    Channel 6 (domain mask) is left exactly as 0/1.

    Returns X_train_n, X_test_n, mean, std  (mean/std shape: (1,6,1,1))
    """
    mean = X_train[:, :6].mean(axis=(0, 2, 3), keepdims=True)
    std = X_train[:, :6].std(axis=(0, 2, 3), keepdims=True) + 1e-8

    X_train_n = X_train.copy()
    X_test_n = X_test.copy()
    X_train_n[:, :6] = (X_train[:, :6] - mean) / std
    X_test_n[:, :6] = (X_test[:, :6] - mean) / std

    return (
        X_train_n.astype(np.float32),
        X_test_n.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def normalize_targets(Y_train, Y_test, mask):
    """
    Normalise temperature using only valid training-domain cells.
    Invalid cells are zeroed after normalisation.

    Returns Y_train_n, Y_test_n, y_mean (scalar), y_std (scalar)
    """
    valid_train = Y_train[:, 0][:, mask]
    y_mean = float(np.mean(valid_train))
    y_std = float(np.std(valid_train)) + 1e-8

    Y_train_n = (Y_train - y_mean) / y_std
    Y_test_n = (Y_test - y_mean) / y_std
    Y_train_n[:, 0, ~mask] = 0.0
    Y_test_n[:, 0, ~mask] = 0.0

    return (
        Y_train_n.astype(np.float32),
        Y_test_n.astype(np.float32),
        np.float32(y_mean),
        np.float32(y_std),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(Y_true, Y_pred, mask):
    """
    R2, RMSE and MAE on the FNO grid (valid cells only).
    Returns a dict with dataset-wide scalars and per-sample arrays.
    """
    true_v = Y_true[:, 0][:, mask]
    pred_v = Y_pred[:, 0][:, mask]

    r2_s, rmse_s, mae_s = [], [], []
    for i in range(len(true_v)):
        r2_s.append(r2_score(true_v[i], pred_v[i]))
        rmse_s.append(np.sqrt(mean_squared_error(true_v[i], pred_v[i])))
        mae_s.append(mean_absolute_error(true_v[i], pred_v[i]))

    tf = true_v.reshape(-1)
    pf = pred_v.reshape(-1)
    return {
        "R2": float(r2_score(tf, pf)),
        "RMSE": float(np.sqrt(mean_squared_error(tf, pf))),
        "MAE": float(mean_absolute_error(tf, pf)),
        "mean_R2": float(np.mean(r2_s)),
        "mean_RMSE": float(np.mean(rmse_s)),
        "mean_MAE": float(np.mean(mae_s)),
        "R2_per_sample": np.asarray(r2_s),
        "RMSE_per_sample": np.asarray(rmse_s),
        "MAE_per_sample": np.asarray(mae_s),
    }


# ---------------------------------------------------------------------------
# Thermal resistance
# ---------------------------------------------------------------------------


def compute_rth(fields, coords, parameters):
    """Per-chip thermal resistance. Retained for compatibility."""
    z = coords[:, 2]
    z_mid = (z.min() + z.max()) / 2
    right, left = z >= z_mid, z < z_mid

    Q1 = parameters[:, PARAM_IDX["Q1"]] * CHIP_AREA
    Q2 = parameters[:, PARAM_IDX["Q2"]] * CHIP_AREA
    T_in1 = parameters[:, PARAM_IDX["T_in1"]]
    T_in2 = parameters[:, PARAM_IDX["T_in2"]]

    n = len(fields)
    tmax1 = np.array([fields[i, right].max() for i in range(n)])
    tmax2 = np.array([fields[i, left].max() for i in range(n)])
    return (tmax1 - T_in1) / Q1, (tmax2 - T_in2) / Q2, tmax1, tmax2
