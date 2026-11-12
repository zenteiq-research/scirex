# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""JAX loader for the Transolver paper's own Car-Design-ShapeNetCar benchmark.

Reads Nobuyuki Umetani's raw `mlcfd_data.zip` (linked from
`Car-Design-ShapeNetCar/README.md` in the reference PyTorch repo), predicts
**velocity + pressure** jointly, and expects a directory layout of::

    <data_dir>/
        param0/
            <sample_id>/
                quadpress_smpl.vtk   # surface mesh: quad cells, point pressure
                hexvelo_smpl.vtk     # volume points: point velocity vectors
            ...
        param1/
            ...
        ...
        param8/

(9 folds, `889` samples total across all folds combined -- matches
`dataset/load_dataset.py::get_samples` in the reference repo). One fold is
held out as the validation split (`fold_id`, default `0`); the rest are
training. Some sample directories in the raw download legitimately contain
only `param1.txt`/`param2.txt` with no VTK files (observed in practice,
not corruption); `load_shapenet_car` skips these silently-per-sample but
prints a one-line count, matching `dataset.py::get_datalist`'s
`continue`-on-missing-file behavior.

Ports `dataset/dataset.py::get_datalist`'s raw-file branch (VTK reading,
SDF, normals, feature assembly, train-only normalization) faithfully, with
one deliberate omission: edge-index / mesh-connectivity construction
(`get_edges`/`get_edge_index`, and the `torch_geometric`/`torch-cluster`
dependency that comes with it) is skipped entirely, because
`models/Transolver.py::Model.forward` never reads `edge_index` -- it only
uses `cfd_data.x` and `cfd_data.pos`. That machinery exists in the
reference repo only for other (GNN-based) model architectures sharing the
same `dataset.py`.

Unlike every other Transolver dataset loader in this repo, samples here are
NOT stacked into one array: each car mesh has a different number of points,
so `load_shapenet_car` returns a plain Python list of per-sample dicts
(`x`, `y`, `pos`, `surf`) instead of one batched tensor. Consumed by
`scripts/benchmarks/shapenetcar/train.py`'s per-sample training loop
(batch_size=1, matching the reference exactly).

Requires `vtk` (the `benchmark` extra) and `scikit-learn`. Both install
cleanly via pip, unlike `open3d` -- no known Python-version blocker as of this
writing.
"""

import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from scirex._internal.extras import requires_extra

logger = logging.getLogger(__name__)


def _load_unstructured_grid_data(file_name: str | Path) -> Any:
    import vtk

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(file_name))
    # Legacy VTK readers only auto-populate GetVectors()/GetScalars() with
    # whichever array the file marks as "active" for that attribute type;
    # ReadAllVectorsOn/ReadAllScalarsOn additionally load every other
    # vector-/scalar-shaped point-data array, which _get_point_vectors and
    # _get_point_scalars below can then fall back to searching by shape.
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    return reader.GetOutput()


def _get_point_vectors(point_data: Any) -> Any:
    """point_data.GetVectors() only returns the reader's designated "active"
    vector array. Some real-world VTK files carry velocity as a
    3-component point-data array without marking it active, leaving
    GetVectors() as None even though the data is present -- fall back to
    scanning all point-data arrays for one with 3 components.
    """
    vecs = point_data.GetVectors()
    if vecs is not None:
        return vecs
    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        if arr is not None and arr.GetNumberOfComponents() == 3:
            return arr
    raise ValueError(
        "No 3-component (vector) point-data array found "
        f"(GetVectors() was None, and no fallback array matched; "
        f"{point_data.GetNumberOfArrays()} array(s) present)."
    )


def _get_point_scalars(point_data: Any) -> Any:
    """Same fallback as _get_point_vectors, for the single-component
    (pressure) array.
    """
    scal = point_data.GetScalars()
    if scal is not None:
        return scal
    for i in range(point_data.GetNumberOfArrays()):
        arr = point_data.GetArray(i)
        if arr is not None and arr.GetNumberOfComponents() == 1:
            return arr
    raise ValueError(
        "No scalar point-data array found "
        f"(GetScalars() was None, and no fallback array matched; "
        f"{point_data.GetNumberOfArrays()} array(s) present)."
    )


def _unstructured_grid_data_to_poly_data(
    unstructured_grid_data: Any,
) -> tuple[Any, Any]:
    import vtk

    filt = vtk.vtkDataSetSurfaceFilter()
    filt.SetInputData(unstructured_grid_data)
    filt.Update()
    return filt.GetOutput(), filt


def _get_sdf(target: np.ndarray, boundary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance + direction from each `target` point to its nearest
    `boundary` point. Ports `dataset.py::get_sdf` exactly.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = boundary[indices[:, 0]]
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def _get_normal(unstructured_grid_data: Any) -> np.ndarray:
    """Per-point surface normals (cell normals interpolated to points).
    Ports `dataset.py::get_normal` exactly, including its
    renormalize-then-retry-on-NaN behavior.
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

    unstructured_grid_data.GetCellData().SetNormals(
        normal_filter.GetOutput().GetCellData().GetNormals()
    )
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()

    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
    if np.isnan(normal).sum() > 0:
        return _get_normal(unstructured_grid_data)  # re-compute, matching reference
    return normal


def get_samples(data_dir: str | Path) -> list[list[str]]:
    """Enumerate the 9 `param{0..8}` folds, each a list of sample IDs
    (relative `"param{i}/<sample_id>"` paths). Ports
    `dataset/load_dataset.py::get_samples`.
    """
    data_dir = Path(data_dir)
    folds = [f"param{i}" for i in range(9)]
    samples = []
    for fold in folds:
        fold_dir = data_dir / fold
        fold_samples = []
        if fold_dir.is_dir():
            for entry in sorted(fold_dir.iterdir()):
                if entry.is_dir():
                    fold_samples.append(f"{fold}/{entry.name}")
        samples.append(fold_samples)
    return samples


def _read_raw_sample(data_dir: str | Path, sample_rel_path: str) -> dict:
    """Reads one `quadpress_smpl.vtk`/`hexvelo_smpl.vtk` pair and builds
    the `(init, target, pos, surf)` feature arrays. Ports
    `dataset.py::get_datalist`'s raw-file branch (lines computing
    `velo`/`press`/`sdf`/`normal`/`init`/`target`/`surf`),
    omitting only the edge-index construction (unused by `Transolver.Model`).
    """
    from vtk.util.numpy_support import vtk_to_numpy

    sample_dir = Path(data_dir) / sample_rel_path
    ug_press = _load_unstructured_grid_data(sample_dir / "quadpress_smpl.vtk")
    ug_velo = _load_unstructured_grid_data(sample_dir / "hexvelo_smpl.vtk")

    velo = vtk_to_numpy(_get_point_vectors(ug_velo.GetPointData()))
    press = vtk_to_numpy(_get_point_scalars(ug_press.GetPointData()))
    points_velo = vtk_to_numpy(ug_velo.GetPoints().GetData())
    points_press = vtk_to_numpy(ug_press.GetPoints().GetData())

    sdf_velo, normal_velo = _get_sdf(points_velo, points_press)
    sdf_press = np.zeros(points_press.shape[0])
    normal_press = _get_normal(ug_press)

    surface = {tuple(p) for p in points_press}
    exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
    velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

    pos_ext = points_velo[exterior_indices]
    pos_surf = points_press
    sdf_ext = sdf_velo[exterior_indices]
    sdf_surf = sdf_press
    normal_ext = normal_velo[exterior_indices]
    normal_surf = normal_press
    velo_ext = velo[exterior_indices]
    velo_surf = np.array(
        [
            velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
            for p in pos_surf
        ]
    )
    press_ext = np.zeros([len(exterior_indices), 1])
    press_surf = press.reshape(-1, 1)

    init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
    init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
    target_ext = np.c_[velo_ext, press_ext]
    target_surf = np.c_[velo_surf, press_surf]

    surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))]).astype(bool)
    pos = np.concatenate([pos_ext, pos_surf]).astype(np.float32)
    init = np.concatenate([init_ext, init_surf]).astype(np.float32)
    target = np.concatenate([target_ext, target_surf]).astype(np.float32)

    return {"x": init, "y": target, "pos": pos, "surf": surf}


def _read_sample(
    data_dir: str | Path,
    sample_rel_path: str,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict | None:
    """Returns None (instead of raising) if the sample has no cache hit AND
    is missing either raw VTK file -- some sample directories in the raw
    download legitimately contain only param1.txt/param2.txt with no VTK
    files, matching dataset.py::get_datalist's continue-on-missing-file
    behavior in the reference implementation.
    """
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{sample_rel_path.replace('/', '_')}.npz"
        if use_cache and cache_path.exists():
            with np.load(cache_path) as npz:
                return {
                    "x": npz["x"],
                    "y": npz["y"],
                    "pos": npz["pos"],
                    "surf": npz["surf"],
                }

    sample_dir = Path(data_dir) / sample_rel_path
    if (
        not (sample_dir / "quadpress_smpl.vtk").exists()
        or not (sample_dir / "hexvelo_smpl.vtk").exists()
    ):
        return None

    sample = _read_raw_sample(data_dir, sample_rel_path)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **sample)

    return sample


@requires_extra("vtk", extra="benchmark")
def load_shapenet_car(
    data_dir: str | Path,
    fold_id: int = 0,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict:
    """Loads the Car-Design-ShapeNetCar dataset for one train/val fold split.

    Parameters
    ----------
    data_dir : path
        Root directory of the raw dataset (`param0/` .. `param8/`, each a
        folder of per-sample subdirectories -- see this module's docstring).
    fold_id : int, default 0
        Which of the 9 folds to hold out as the validation split; the other
        8 folds become the training split. Matches `main.py`'s
        `--fold_id` (default `0`).
    cache_dir : path, optional
        If given, each sample's computed `(x, y, pos, surf)` arrays are
        cached to `<cache_dir>/<fold>_<sample_id>.npz` after the first VTK
        read, and reloaded from there on subsequent calls (`use_cache=True`)
        -- mirrors the reference's `--save_dir`/`--preprocessed` workflow,
        skipping the expensive VTK/SDF/normal computation on reruns.
    use_cache : bool, default True
        Whether to read from `cache_dir` if a cached `.npz` already
        exists there. Set False to force recomputation (e.g. after changing
        the raw data) while still writing fresh caches.

    Returns
    -------
    dict with keys:
        `train` : list of per-sample dicts (`x`: (N_i, 7), `y`: (N_i, 4),
            `pos`: (N_i, 3), `surf`: (N_i,) bool, `sample_rel_path`: str
            e.g. `"param3/1a0bc9ab..."`) -- NOT stacked, since N_i (point
            count) varies per car mesh. `sample_rel_path` is only used by
            drag-coefficient evaluation (`shapenet_car_drag.py`) to
            locate the sample's raw `quadpress_smpl.vtk` for cell/normal
            geometry; ignore it for training.
        `val` : same structure, for the held-out fold.
        `coef_norm` : `(mean_in, std_in, mean_out, std_out)`, each a
            `(7,)` or `(4,)` numpy array -- per-channel population mean/std
            (`ddof=0`, `eps=1e-8`) computed over the training split only,
            already applied to both `train` and `val`'s `x`/`y`
            (matching `dataset.py::get_datalist`'s normalization, which
            fits on train and applies the same stats to val -- no leakage).
        `model_kwargs` : ``dict(space_dim=7, fun_dim=0, out_dim=4,
            unified_pos=False, geometry="irregular")``, ready to pass into
            `scirex.models.transolver.Transolver`.
    """
    samples = get_samples(data_dir)
    if not (0 <= fold_id < len(samples)):
        raise ValueError(f"fold_id must be in [0, {len(samples) - 1}], got {fold_id}")

    train_paths = [p for i, fold in enumerate(samples) if i != fold_id for p in fold]
    val_paths = samples[fold_id]
    if not train_paths or not val_paths:
        raise ValueError(
            f"data_dir={data_dir!r} produced 0 train or 0 val samples -- check "
            "the param0..param8 folder layout described in this module's docstring."
        )

    def _read_filtered(paths: list[str], split_name: str) -> list[dict]:
        # Reading raw (uncached) samples is slow -- VTK parse + a fresh
        # sklearn NearestNeighbors fit/query per sample for the SDF, plus a
        # vtkPolyDataNormals pass -- and this runs for every sample in the
        # split before training's first epoch even starts. Print periodic
        # progress so a several-minute-long silent gap here doesn't look
        # like a hang, especially on the first (uncached) run.
        raw, n_skipped = [], 0
        n_total = len(paths)
        t0 = perf_counter()
        logger.info(f"load_shapenet_car: reading {n_total} {split_name} sample(s)...")
        sys.stdout.flush()
        for i, p in enumerate(paths):
            sample = _read_sample(data_dir, p, cache_dir, use_cache)
            if sample is None:
                n_skipped += 1
            else:
                # Attach the relative path here (not inside _read_sample/the
                # cache) so the .npz cache format stays untouched -- a
                # sample's raw-file location is a property of *where it came
                # from in this call*, not of its cached array contents.
                # Downstream: drag-coefficient evaluation needs this to
                # locate the sample's quadpress_smpl.vtk for cell/normal
                # geometry (see shapenet_car_drag.py).
                sample["sample_rel_path"] = p
                raw.append(sample)
            if (i + 1) % 25 == 0 or (i + 1) == n_total:
                elapsed = perf_counter() - t0
                progress = f"{split_name} {i + 1}/{n_total}"
                logger.info(f"load_shapenet_car: {progress} ({elapsed:.1f}s elapsed)")
                sys.stdout.flush()
        if n_skipped:
            logger.info(
                f"load_shapenet_car: skipped {n_skipped} incomplete {split_name} "
                "sample(s) (missing quadpress_smpl.vtk/hexvelo_smpl.vtk, no cache hit)."
            )
        return raw

    train_raw = _read_filtered(train_paths, "train")
    val_raw = _read_filtered(val_paths, "val")
    if not train_raw or not val_raw:
        raise ValueError(
            f"data_dir={data_dir!r} produced 0 train or 0 val samples after "
            "filtering out incomplete ones -- check the param0..param8 folder "
            "layout described in this module's docstring."
        )

    # Pooled (population, ddof=0) per-channel mean/std over the training
    # split's points only, mirroring get_datalist's two-pass streaming
    # accumulation exactly (mathematically equivalent to concatenating all
    # training points and calling .mean(axis=0)/.std(axis=0) directly).
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
                    "sample_rel_path": s["sample_rel_path"],
                }
            )
        return out

    return {
        "train": _normalize(train_raw),
        "val": _normalize(val_raw),
        "coef_norm": (mean_in, std_in, mean_out, std_out),
        "model_kwargs": dict(
            space_dim=7, fun_dim=0, out_dim=4, unified_pos=False, geometry="irregular"
        ),
    }
