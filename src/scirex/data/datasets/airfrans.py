# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""JAX loader for the AirfRANS 2D airfoil RANS-CFD dataset (SST turbulence
closure, OpenFOAM-generated), mirroring
`src/scirex/data/datasets/shapenet_car.py`'s irregular
point-cloud loader for the Transolver paper's own Car-Design-ShapeNetCar
benchmark -- same per-sample (variable point count), train-only-normalized,
list-of-dicts return shape, so the AirfRANS GINO trainer
(`scripts/benchmarks/airfrans/_gino.py`) can use the same per-sample loop.

Expects a directory layout of::

    <data_dir>/
        <case_id>/
            <case_id>_internal.vtu    # volume mesh: p, U, nut, implicit_distance
            <case_id>_aerofoil.vtp    # airfoil-surface boundary mesh
            <case_id>_freestream.vtp  # far-field boundary mesh -- its solved U
                                       # field is this loader's source for the
                                       # per-case inlet-velocity input feature
        ...

one folder per simulation case, `<case_id>` e.g.
`airFoil2D_SST_63.952_9.03_0.666_3.739_0.0_15.525` (encodes inlet velocity
magnitude/angle and the airfoil's Bezier shape parameters, but with an
ambiguous/variable-length convention across the NACA-4 vs NACA-5 shape
families -- opaque to this loader, used only as a directory/cache key; the
inlet velocity is read from `_freestream.vtp` instead, see below).

Unlike Car-Design-ShapeNetCar (separate surface-pressure and volume-velocity
meshes merged into one array), every AirfRANS `_internal.vtu` point already
carries all four target fields (`U` -- x/y components, `p`, `nut`) plus
the input signed-distance-to-airfoil field (`implicit_distance`), so no
merge step is needed. A per-point `normal` substitute (unit direction from
the nearest airfoil-surface point, matching
`shapenet_car.py`'s `_get_sdf` direction convention) is
computed via a `NearestNeighbors` query against the sample's own
`_aerofoil.vtp` points, and doubles as the `surf` boundary mask
(`dist < 1e-5`, i.e. mesh nodes that coincide exactly with an airfoil
surface point -- OpenFOAM's wall-conforming mesh generation guarantees this).

**Inlet velocity (u_inf_x, u_inf_y)**: the AirfRANS paper's own input feature
vector is R^7 -- spatial coordinates, INLET VELOCITY, the SDF distance, and
the surface normal -- but earlier versions of this loader only built the 5
geometry-only channels, omitting the per-case flow condition (Reynolds
number/angle-of-attack) entirely. Since every point in a case shares the same
freestream condition, and local (position/SDF/direction) features alone
can't distinguish "this airfoil region at low Re/AoA" from "...at high
Re/AoA", omitting it left the model unable to condition on the single
biggest driver of the target fields -- this was found to be the dominant
cause of a large paper-vs-reproduction benchmark gap (particularly for
pressure, which scales with the square of inlet speed and is most sensitive
to angle-of-attack-driven stagnation-point/suction-peak shifts). Fixed by
reading `_freestream.vtp`'s solved `U` field (the far-field boundary's
converged velocity, which is very tightly clustered around the true
prescribed inlet condition -- empirically <0.3% relative spread across its
~1700 boundary points for a sample case) and taking its per-case MEDIAN
(x, y) as a constant, broadcast to every point in that case.

Per-sample feature layout::

    x   : (N_i, 7) = [pos_x, pos_y, implicit_distance, dir_x, dir_y,
                       u_inf_x, u_inf_y]
    y   : (N_i, 4) = [Ux, Uy, p, nut]
    pos : (N_i, 2)
    surf: (N_i,) bool -- True for airfoil-wall mesh nodes

`N_i` (point count) varies slightly case to case (~180k typical), so
samples are returned as a plain Python list of per-sample dicts, NOT stacked
-- same convention as `load_shapenet_car`.

Requires `vtk` (the `benchmark` extra) and `scikit-learn`.
"""

import logging
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from scirex._internal.extras import requires_extra

logger = logging.getLogger(__name__)


def _load_vtu(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    pts = vtk_to_numpy(data.GetPoints().GetData())
    pd = data.GetPointData()
    fields = {
        pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i))
        for i in range(pd.GetNumberOfArrays())
    }
    return pts, fields


def _load_vtp(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    pts = vtk_to_numpy(data.GetPoints().GetData())
    pd = data.GetPointData()
    fields = {
        pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i))
        for i in range(pd.GetNumberOfArrays())
    }
    return pts, fields


def get_volume_triangles(vtu_path: str | Path) -> np.ndarray:
    """Re-triangulate the volume mesh's own quad-cell connectivity (not a
    Delaunay re-fit of the point cloud), for plotting. Every AirfRANS
    `_internal.vtu` cell is a 4-node quad; splitting each along one
    diagonal gives two triangles per cell.

    A Delaunay re-fit of the raw point cloud would happily bridge across the
    thin airfoil surface (points on the upper and lower surface can be close
    in Euclidean distance despite being mesh-topologically distant), smearing
    a spurious blend of upper/lower-surface field values across the airfoil
    interior in any resulting plot -- this uses the mesh's actual
    connectivity instead, so nothing bridges across the surface. Matches
    `Dataset_samples/generate_samples.py`'s own approach.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    conn = vtk_to_numpy(reader.GetOutput().GetCells().GetData())
    quads = conn.reshape(-1, 5)[:, 1:5]
    return np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def get_case_ids(data_dir: str | Path) -> list[str]:
    """Sorted list of case-folder names directly under `data_dir`."""
    data_dir = Path(data_dir)
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir())


# Number of input channels _read_raw_sample produces (pos_x, pos_y, sdf,
# dir_x, dir_y, u_inf_x, u_inf_y). Used to detect and invalidate stale
# `Dataset_cache/*.npz` entries written before the inlet-velocity fix
# (which only had 5 channels) -- see this module's docstring.
N_X_CHANNELS = 7


def _read_freestream_velocity(data_dir: str | Path, case_id: str) -> np.ndarray:
    """Per-case inlet velocity `(u_inf_x, u_inf_y)`: the per-case MEDIAN of
    `_freestream.vtp`'s solved `U` field over its far-field boundary
    points -- see this module's docstring on why the median (tightly
    clustered around the true prescribed inlet condition, robust to the few
    wake-affected boundary points) rather than the raw per-case-id string.
    """
    case_dir = Path(data_dir) / case_id
    fpts, ffields = _load_vtp(case_dir / f"{case_id}_freestream.vtp")
    del fpts
    U_inf = ffields["U"][:, :2].astype(np.float32)
    return np.median(U_inf, axis=0)


def _read_raw_sample(data_dir: str | Path, case_id: str) -> dict:
    """Reads one case's `_internal.vtu`/`_aerofoil.vtp`/`_freestream.vtp`
    triple and builds the `(x, y, pos, surf)` feature arrays described in
    this module's docstring.
    """
    from sklearn.neighbors import NearestNeighbors

    case_dir = Path(data_dir) / case_id
    pts, fields = _load_vtu(case_dir / f"{case_id}_internal.vtu")
    pos = pts[:, :2].astype(np.float32)  # z is a constant span extrusion, drop it

    U = fields["U"][:, :2].astype(np.float32)
    p = fields["p"].astype(np.float32).reshape(-1)
    nut = fields["nut"].astype(np.float32).reshape(-1)
    sdf = fields["implicit_distance"].astype(np.float32).reshape(-1)

    spts, _ = _load_vtp(case_dir / f"{case_id}_aerofoil.vtp")
    surf_pos = spts[:, :2].astype(np.float32)

    nbrs = NearestNeighbors(n_neighbors=1).fit(surf_pos)
    dists, indices = nbrs.kneighbors(pos)
    dists = dists.reshape(-1)
    nearest = surf_pos[indices[:, 0]]
    dirs = (pos - nearest) / (dists[:, None] + 1e-8)
    surf = dists < 1e-5  # wall-conforming mesh: exact coincidence at the surface

    u_inf = _read_freestream_velocity(data_dir, case_id)
    u_inf_bcast = np.broadcast_to(u_inf, (pos.shape[0], 2))

    x = np.concatenate([pos, sdf[:, None], dirs, u_inf_bcast], axis=1).astype(
        np.float32
    )
    y = np.stack([U[:, 0], U[:, 1], p, nut], axis=1).astype(np.float32)

    return {"x": x, "y": y, "pos": pos, "surf": surf}


def _read_sample(
    data_dir: str | Path,
    case_id: str,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict:
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{case_id}.npz"
        if use_cache and cache_path.exists():
            with np.load(cache_path) as npz:
                if npz["x"].shape[1] == N_X_CHANNELS:
                    return {
                        "x": npz["x"],
                        "y": npz["y"],
                        "pos": npz["pos"],
                        "surf": npz["surf"],
                    }
                # Stale cache entry from before the inlet-velocity fix (5
                # channels, not N_X_CHANNELS=7) -- fall through and rebuild.

    sample = _read_raw_sample(data_dir, case_id)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **sample)

    return sample


@requires_extra("vtk", extra="benchmark")
def load_airfrans(
    data_dir: str | Path,
    n_test: int = 100,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    max_cases: int | None = None,
    task: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """Loads the AirfRANS dataset with a fixed train/test split.

    Parameters
    ----------
    data_dir : path
        Root directory holding one sub-folder per simulation case (see this
        module's docstring).
    n_test : int, default 100
        Number of cases held out as the "unseen" test split. Ignored when
        `task` is given.
    seed : int, default 0
        Seed for the train/test shuffle -- deterministic and reproducible
        across training/inference runs (both must pass the same `seed`).
        Ignored when `task` is given.
    cache_dir : path, optional
        If given, each case's computed `(x, y, pos, surf)` arrays are
        cached to `<cache_dir>/<case_id>.npz` after the first VTK read.
    use_cache : bool, default True
        Whether to read from `cache_dir` if a cached `.npz` already
        exists there.
    max_cases : int, optional
        If given, only the first `max_cases` case ids (after sorting, before
        the shuffle) are used in total -- for fast smoke tests. Ignored when
        `task` is given.
    task : str or None, optional
        One of "full", "scarce", "reynolds" or "aoa"; by default None (a random
        split). If given, use the AirfRANS paper's own official train/test case-id
        split for that benchmark task, read from `manifest_path` (the
        `<task>_train`/`<task>_test` keys of `Dataset/manifest.json`,
        e.g. "full" -> 800 train / 200 test) instead of a random split --
        makes results directly comparable to the paper's Table 2/3 baselines.
        Overrides `n_test`, `seed`, and `max_cases`.
    manifest_path : path, optional
        Path to the official split manifest (JSON with
        `{task}_train`/`{task}_test` keys of case-id lists). Only used
        when `task` is given. Defaults to `<data_dir>/manifest.json`.

    Returns
    -------
    dict with keys:
        `train` : list of per-sample dicts (`x`, `y`, `pos`, `surf`,
            `case_id`), normalized, NOT stacked (point count varies).
        `test` : same structure, for the held-out split.
        `coef_norm` : `(mean_in, std_in, mean_out, std_out)` -- per-channel
            population mean/std (`ddof=0`, `eps=1e-8`) over the training
            split's points only, already applied to both splits.
        `model_kwargs` : ``dict(space_dim=7, fun_dim=0, out_dim=4,
            unified_pos=False, geometry="irregular")``, ready to pass into
            `scirex.models.transolver.Transolver`.
    """
    if task is not None:
        import json

        manifest_path = (
            Path(manifest_path) if manifest_path else Path(data_dir) / "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        train_ids = manifest[f"{task}_train"]
        test_ids = manifest[f"{task}_test"]
    else:
        case_ids = get_case_ids(data_dir)
        if max_cases is not None:
            case_ids = case_ids[:max_cases]
        if n_test >= len(case_ids):
            raise ValueError(
                f"n_test={n_test} >= number of available cases ({len(case_ids)})"
            )

        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(case_ids).tolist()
        test_ids = shuffled[:n_test]
        train_ids = shuffled[n_test:]

    def _read_filtered(ids: list[str], split_name: str) -> list[dict]:
        raw = []
        n_total = len(ids)
        t0 = perf_counter()
        logger.info(f"load_airfrans: reading {n_total} {split_name} case(s)...")
        sys.stdout.flush()
        for i, case_id in enumerate(ids):
            sample = _read_sample(data_dir, case_id, cache_dir, use_cache)
            sample["case_id"] = case_id
            raw.append(sample)
            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                elapsed = perf_counter() - t0
                logger.info(
                    f"load_airfrans: {split_name} {i + 1}/{n_total} "
                    f"({elapsed:.1f}s elapsed)"
                )
                sys.stdout.flush()
        return raw

    train_raw = _read_filtered(train_ids, "train")
    test_raw = _read_filtered(test_ids, "test")

    x_all = np.concatenate([s["x"] for s in train_raw], axis=0)
    y_all = np.concatenate([s["y"] for s in train_raw], axis=0)
    mean_in, std_in = x_all.mean(axis=0), x_all.std(axis=0)
    mean_out, std_out = y_all.mean(axis=0), y_all.std(axis=0)

    def _normalize(raw_list: list[dict]) -> list[dict]:
        out = []
        for s in raw_list:
            out.append(
                {
                    "x": (s["x"] - mean_in) / (std_in + 1e-8),
                    "y": (s["y"] - mean_out) / (std_out + 1e-8),
                    "pos": s["pos"],
                    "surf": s["surf"],
                    "case_id": s["case_id"],
                }
            )
        return out

    return {
        "train": _normalize(train_raw),
        "test": _normalize(test_raw),
        "coef_norm": (mean_in, std_in, mean_out, std_out),
        "model_kwargs": dict(
            space_dim=7, fun_dim=0, out_dim=4, unified_pos=False, geometry="irregular"
        ),
    }


# Output channel names, in the order used throughout this module and its
# consumers (`scripts/benchmarks/airfrans/`).
OUTPUT_FIELDS: tuple[str, ...] = ("Ux", "Uy", "p", "nut")
