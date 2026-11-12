# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""JAX loader for the "Transient k-epsilon" lid-driven-cavity dataset (a
transient RANS k-epsilon OpenFOAM study; the full export has 11 lid speeds x
201 snapshots each, steps 0..200, of which this loader's default
`CASES` uses 5), for training `scirex.models.transolver.Transolver`
in its "irregular" point-cloud mode.

Expects a directory layout of::

    <data_dir>/
        01k/Data01k0.<step>.csv
        02k/Data02k0.<step>.csv
        04k/Data04k0.<step>.csv
        14k/Data14k0.<step>.csv
        16k/Data16k0.<step>.csv

(the full export also holds 06k, 08k, 10k, 12k, 18k and 20k; pass `cases`
to select them)

one sub-folder per case (named `<lid_velocity*10:02d>k`, e.g. `04k` ==
lid speed 0.4 m/s -- confirmed by `max(U:0) == lid speed` on every
snapshot), each holding one CSV per write-out step (`<step>` an integer,
NOT zero-padded, occasional gaps from write failures -- see below). Every
CSV has the OpenFOAM `sample`/`foamToVTK` header
`"p","U:0","U:1","U:2","Points:0","Points:1","Points:2"` and covers a
129x129x2 point mesh (0.1m x 0.1m cavity, single cell thick in z via an
`empty` front/back patch -- the z=0 and z=0.01 layers carry byte-identical
field values, confirmed by direct comparison, so only the z=0 layer
(16,641 points) is kept here; this halves memory for free and avoids feeding
the model duplicate points).

Unlike the AirfRANS/Car-Design-ShapeNetCar loaders in this same directory,
every snapshot shares the EXACT SAME 16,641-point mesh (same coordinates, in
the same CSV row order -- OpenFOAM's `sample` dumps points in mesh-vertex
order, which does not change across time or case), so samples here can be
stacked into a genuine dense `(n_samples, 16641, C)` array and trained with
real minibatches, no per-sample loop / point-count subsampling required for
*correctness* (subsampling is still used during training purely as a
compute-speed knob).

Per-point feature layout::

    x : (16641, 4) = [pos_x/0.1, pos_y/0.1, t_norm, U_lid]
        t_norm = step_index / 200 (200 == the longest case's step count, used
        as a shared timescale across cases so t_norm means the same thing
        for every case).

        `t_norm` is a normalized step INDEX, not a physical time. No
        `controlDict` ships with this dataset (each case folder holds only
        its CSVs), so neither the seconds per snapshot nor whether every case
        really used the same `writeInterval` can be read off the data.
        An earlier version of this docstring asserted a shared write interval
        as fact; that was unsourced, and it propagated into the
        physics-informed trainers as a wrong time step (see below). What IS
        measurable: fitting the physical step independently per case against
        the momentum residual gives 0.106-0.124 s, a ~15% spread -- consistent
        with a shared interval near 0.11 s, but evidence for it rather than
        proof of it.

        Anything needing a real time step -- in particular the du/dt in a PDE
        residual -- must use that fitted value (`--dt-phys`, and note it is
        non-dimensionalized per case as `dt_phys * U_lid / DOMAIN_SIZE`),
        never this 1/200.
    y : (16641, 3) = [p, Ux, Uy]  (Uz is identically 0 -- 2D flow -- and
        dropped)

`t_norm` and `U_lid` are broadcast to every point of a snapshot (global
per-sample conditioning, same convention as a "class token" -- there is no
per-point analogue for either).

Requires only `numpy` (CSV parsing via `numpy.genfromtxt`).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CASES: tuple[str, ...] = ("01k", "02k", "04k", "14k", "16k")
N_POINTS = 16_641  # 129 x 129, single z-layer
T_NORM_DIVISOR = 200.0  # longest case's step count -- shared timescale
DOMAIN_SIZE = 0.1  # metres, both x and y

OUTPUT_FIELDS: tuple[str, ...] = ("p", "Ux", "Uy")
INPUT_FIELDS: tuple[str, ...] = ("pos_x", "pos_y", "t_norm", "U_lid")

_FNAME_RE = re.compile(r"Data(\d\d k)0\.(\d+)\.csv".replace(" ", ""))


def lid_velocity(case: str) -> float:
    """`"04k" -> 0.4`, `"14k" -> 1.4` -- confirmed against `max(U:0)`
    of every snapshot in that case folder (the moving-lid boundary).
    """
    return int(case[:2]) / 10.0


def get_case_steps(data_dir: str | Path, case: str) -> list[int]:
    """Sorted list of available step indices for one case folder (skips
    stray non-data files, e.g. editor lock files left behind by someone
    inspecting a CSV -- seen once, for `14k`).
    """
    case_dir = Path(data_dir) / case
    steps = []
    for p in case_dir.iterdir():
        m = _FNAME_RE.fullmatch(p.name)
        if m and m.group(1) == case:
            steps.append(int(m.group(2)))
    return sorted(steps)


def _read_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pos[N,2], p[N], U[N,2]) for the z=0 layer of one CSV.

    Columns are looked up by name (numpy sanitizes the OpenFOAM header's
    `"U:0"`/`"Points:0"` into `U0`/`Points0`) rather than assumed to
    be in a fixed position, since this dataset's CSVs come from two
    different OpenFOAM export tools (`sample`/`foamToVTK`, see module
    docstring) that are not guaranteed to emit columns in the same order.
    """
    data: Any = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    try:
        z = data["Points2"]
        # Tolerance-based, not exact equality: the z=0 layer is text-parsed
        # from OpenFOAM's ASCII export and isn't guaranteed to be a literal
        # "0" (e.g. floating-point noise like 1e-8). 1e-6 is well above
        # realistic parsing noise and well below the 0.01 gap to the other
        # (z=0.01) layer, so it can't merge the two layers together.
        d = data[np.isclose(z, 0.0, rtol=0, atol=1e-6)]
        pos = np.stack([d["Points0"], d["Points1"]], axis=-1)
        p = d["p"]
        U = np.stack([d["U0"], d["U1"]], axis=-1)
    except ValueError as e:
        raise ValueError(
            f"{path}: unexpected CSV header {data.dtype.names!r}, expected "
            "columns for p, U:0, U:1, U:2, Points:0, Points:1, Points:2"
        ) from e
    if d.shape[0] != N_POINTS:
        raise ValueError(f"{path}: expected {N_POINTS} z=0 points, got {d.shape[0]}")
    return pos, p, U


def _read_case(
    data_dir: str | Path, case: str, cache_dir: str | Path | None, use_cache: bool
) -> dict:
    """Reads (or loads from cache) every available snapshot of one case.

    Returns dict with `steps` (n_steps,), `P` (n_steps, 16641),
    `U` (n_steps, 16641, 2), `pos` (16641, 2) -- the last identical
    across every case (verified below), kept per-case only for a cheap
    self-consistency check.
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{case}.npz"
        if use_cache and cache_path.exists():
            with np.load(cache_path) as npz:
                return {k: npz[k] for k in ("steps", "P", "U", "pos")}

    steps = get_case_steps(data_dir, case)
    if not steps:
        raise ValueError(f"No snapshots found for case {case!r} under {data_dir}")

    case_dir = Path(data_dir) / case
    pos_ref = None
    P_list, U_list = [], []
    for step in steps:
        pos, p, U = _read_snapshot(case_dir / f"Data{case}0.{step}.csv")
        if pos_ref is None:
            pos_ref = pos
        P_list.append(p)
        U_list.append(U)

    result = {
        "steps": np.asarray(steps, dtype=np.int32),
        "P": np.stack(P_list, axis=0),
        "U": np.stack(U_list, axis=0),
        "pos": pos_ref,
    }

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **result)

    return result


def load_cavity_kepsilon(
    data_dir: str | Path,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    n_test_frac: float = 0.1,
    seed: int = 0,
    verbose: bool = True,
    cases: tuple[str, ...] | None = None,
) -> dict:
    """Loads the full Transient k-epsilon cavity dataset into two dense
    arrays (train/test), normalized.

    Parameters
    ----------
    data_dir : path
        Root `Transient_k-e` directory (holds `01k/`, `02k/`, ...).
    cache_dir : path, optional
        If given, each case's raw `(steps, P, U, pos)` is cached to
        `<cache_dir>/<case>.npz` after the first CSV read (CSV parsing of
        ~900 33k-row files is the dominant one-time cost; cached loads are
        near-instant).
    n_test_frac : float, default 0.1
        Fraction of ALL (case, step) snapshots held out as test, via a
        global random shuffle (so the split spans all selected lid speeds,
        testing interpolation across time within each case rather than
        extrapolation to an unseen lid speed).
    seed : int, default 0
        Seed for the train/test shuffle.
    cases : tuple[str, ...], optional
        Which case folders to load, e.g. `("01k", "02k", "04k")` for a
        subset. Defaults to the module-level `CASES` (the full 5-case
        set) when not given, so existing callers are unaffected.

    Returns
    -------
    dict with keys:
        `pos` : (16641, 2) shared mesh coordinates, UN-normalized (metres).
        `X_train`, `Y_train`, `X_test`, `Y_test` : dense
            `(n, 16641, 4)` / `(n, 16641, 3)` float32 arrays, normalized.
        `meta_train`, `meta_test` : `(n, 2)` int arrays of
            `(case_index, step)` per sample, for bookkeeping/plotting
            (`cases[case_index]` -- the `cases` key below, NOT
            necessarily the module-level `CASES` -- recovers the case
            string).
        `cases` : the resolved case tuple actually used (either the
            `cases` argument, or module-level `CASES` if omitted).
        `coef_norm` : `(mean_in, std_in, mean_out, std_out)`, computed
            over the train split only and already applied to both splits.
        `model_kwargs` : ``dict(space_dim=4, fun_dim=0, out_dim=3,
            unified_pos=False, geometry="irregular")``.
    """
    cases = tuple(cases) if cases is not None else CASES
    t0 = perf_counter()
    per_case = {}
    for case in cases:
        per_case[case] = _read_case(data_dir, case, cache_dir, use_cache)
        if verbose:
            n = per_case[case]["steps"].shape[0]
            logger.info(f"[cavity_kepsilon] {case}: {n} snapshots loaded")
            sys.stdout.flush()

    pos_ref = per_case[cases[0]]["pos"]
    for case in cases[1:]:
        if not np.array_equal(per_case[case]["pos"], pos_ref):
            raise ValueError(
                f"Mesh mismatch: case {case!r}'s point coordinates differ from "
                f"{cases[0]!r}'s -- samples cannot be stacked into a dense array."
            )

    # Assemble the flat (case, step) sample list and its dense X/Y arrays.
    xs, ys, metas = [], [], []
    pos_norm = (pos_ref / DOMAIN_SIZE).astype(np.float32)  # (16641, 2) in [0, 1]
    for case_idx, case in enumerate(cases):
        c = per_case[case]
        u_lid = lid_velocity(case)
        n_steps = c["steps"].shape[0]
        for i in range(n_steps):
            t_norm = c["steps"][i] / T_NORM_DIVISOR
            cond = np.array([t_norm, u_lid], dtype=np.float32)
            cond_b = np.broadcast_to(cond, (N_POINTS, 2))
            x = np.concatenate([pos_norm, cond_b], axis=1).astype(np.float32)
            y = np.concatenate([c["P"][i][:, None], c["U"][i]], axis=1).astype(
                np.float32
            )
            xs.append(x)
            ys.append(y)
            metas.append((case_idx, int(c["steps"][i])))

    X_all = np.stack(xs, axis=0)  # (n_samples, 16641, 4)
    Y_all = np.stack(ys, axis=0)  # (n_samples, 16641, 3)
    meta_all = np.asarray(metas, dtype=np.int32)  # (n_samples, 2)
    n_samples = X_all.shape[0]

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * n_test_frac)))
    test_idx, train_idx = order[:n_test], order[n_test:]

    X_train_raw, Y_train_raw = X_all[train_idx], Y_all[train_idx]
    X_test_raw, Y_test_raw = X_all[test_idx], Y_all[test_idx]

    mean_in = X_train_raw.reshape(-1, X_train_raw.shape[-1]).mean(axis=0)
    std_in = X_train_raw.reshape(-1, X_train_raw.shape[-1]).std(axis=0)
    mean_out = Y_train_raw.reshape(-1, Y_train_raw.shape[-1]).mean(axis=0)
    std_out = Y_train_raw.reshape(-1, Y_train_raw.shape[-1]).std(axis=0)

    def _norm_x(a: np.ndarray) -> np.ndarray:
        return (a - mean_in) / (std_in + 1e-8)

    def _norm_y(a: np.ndarray) -> np.ndarray:
        return (a - mean_out) / (std_out + 1e-8)

    if verbose:
        elapsed = perf_counter() - t0
        logger.info(
            f"[cavity_kepsilon] n_total={n_samples} n_train={len(train_idx)} "
            f"n_test={len(test_idx)} ({elapsed:.1f}s)"
        )
        sys.stdout.flush()

    return {
        "pos": pos_ref,
        "X_train": _norm_x(X_train_raw).astype(np.float32),
        "Y_train": _norm_y(Y_train_raw).astype(np.float32),
        "X_test": _norm_x(X_test_raw).astype(np.float32),
        "Y_test": _norm_y(Y_test_raw).astype(np.float32),
        "meta_train": meta_all[train_idx],
        "meta_test": meta_all[test_idx],
        "cases": cases,
        "coef_norm": (mean_in, std_in, mean_out, std_out),
        "model_kwargs": dict(
            space_dim=4, fun_dim=0, out_dim=3, unified_pos=False, geometry="irregular"
        ),
    }


def grid_index_maps(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Builds the (row, col) grid index for every flat point in `pos`, so a
    flat `(N,)`/`(n, N)` point-cloud array can be scatter-reshaped into a
    dense `(H, W)`/`(n, H, W)` image via `grid[..., gi, gj] = flat`.

    Kept separate from row order in the source CSVs (never assumed to already
    be raster order) -- built purely from each point's own (x, y) coordinate
    value, so it is correct regardless of how `sample`/`foamToVTK` chose
    to order mesh points in the file.
    """
    xs = np.unique(pos[:, 0])
    ys = np.unique(pos[:, 1])
    H, W = len(xs), len(ys)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    gi = np.array([xi[v] for v in pos[:, 0]], dtype=np.int64)
    gj = np.array([yi[v] for v in pos[:, 1]], dtype=np.int64)
    return gi, gj, H, W


def to_grid_arrays(data: dict) -> dict:
    """Reshapes the flat point-cloud arrays from `load_cavity_kepsilon` into
    dense channel-first images, for the grid-based models (FNO, WNO)
    -- Transolver's "irregular" point-cloud mode uses the
    flat `X_train`/`Y_train` arrays directly and does not need this.

    Returns the same dict as `load_cavity_kepsilon` plus:
        `X_train_grid`, `X_test_grid` : `(n, 4, H, W)` float32
        `Y_train_grid`, `Y_test_grid` : `(n, 3, H, W)` float32
        `H`, `W` : mesh resolution (129, 129 for this dataset)
    """
    gi, gj, H, W = grid_index_maps(data["pos"])

    def _to_grid(flat: np.ndarray) -> np.ndarray:
        # flat: (n, N, C) -> (n, C, H, W)
        n, _, C = flat.shape
        out = np.zeros((n, C, H, W), dtype=np.float32)
        for c in range(C):
            out[:, c, gi, gj] = flat[:, :, c]
        return out

    out = dict(data)
    out["X_train_grid"] = _to_grid(data["X_train"])
    out["Y_train_grid"] = _to_grid(data["Y_train"])
    out["X_test_grid"] = _to_grid(data["X_test"])
    out["Y_test_grid"] = _to_grid(data["Y_test"])
    out["H"], out["W"] = H, W
    return out
