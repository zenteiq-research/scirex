# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data pipeline for the lid-driven cavity benchmark, following the protocol
of Costa Rocha et al. 2023 (Appl. Sci. 13(5), 3165).

This replaces the protocol used by ``src/scirex/data/datasets/
cavity_kepsilon.py`` (5 lid speeds, random snapshot split, direct
point-query regression) with the one the paper actually ran, so that the RMSE
numbers produced here are comparable to its tables.

Paper protocol (Sections 2.1, 2.2, 2.4)::

    train Re : 1000, 2000, 4000, 6000, 14000, 16000, 18000, 20000
    test  Re : 10000                       (a lid speed never seen in training)
    unused   : 8000, 12000                 ("to leave a reasonable gap")
    mesh     : 128 x 128 cells, 0.1 x 0.1 m cavity
    time     : first 10 s only, recorded every 0.1 s  -> steps 0..100
    windows  : sequences of ``n_in`` input frames -> ``n_out`` output frames
               (the paper's headline setting is 3 -> 1)
    fields   : (p, Ux, Uy); Uz is identically zero for this 2D case

Re = U_lid * L / nu with L = 0.1 m and nu = 1e-5 m^2/s, so case ``"10k"``
(U_lid = 1.0 m/s) is exactly the paper's Re = 10,000 test case.

Grid convention. The OpenFOAM export samples MESH VERTICES, giving a 129x129
point lattice for a 128x128 cell mesh. The paper's models consume the 128x128
cell field, so vertices are averaged to cell centres here (the mean of a
cell's four corners). Cropping a row/column instead, as an earlier
script of this repo did, would discard the moving-lid boundary row, which is
the one boundary that drives the entire flow.

Reading the raw CSVs is slow, so each case is converted once to
``<cache_dir>/<case>.npz`` holding the (101, 128, 128, 3) cell grid;
:func:`ensure_prepared` does this on first use.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from scirex.data import UnitGaussianNormalizer

# Paper Section 2.2: the Re sweep, in units of 1000. Folder "NNk" == Re NN000
# and lid speed NN/10 m/s.
# The raw export has 11 case folders (these 9 plus "08k" and "12k"); the
# paper leaves Re 8,000 and 12,000 out, so 9 cases are used: 8 train, 1 test.
TRAIN_CASES: tuple[str, ...] = ("01k", "02k", "04k", "06k", "14k", "16k", "18k", "20k")
TEST_CASE = "10k"
ALL_CASES: tuple[str, ...] = (*TRAIN_CASES, TEST_CASE)

N_VERT = 129  # vertices per side (129 x 129 = 16,641 sampled points)
N_CELL = 128  # cells per side, the paper's stated mesh
N_STEPS = 101  # steps 0..100 == the first 10 s at the 0.1 s write interval
# (each raw case folder holds 201 CSVs, steps 0..200; only the first 101 are read.
# 98 windows per case for 3 in / 1 out: 101 - 3 - 1 + 1)
DT = 0.1  # s, write interval
DOMAIN_SIZE = 0.1  # m
NU = 1e-5  # m^2/s
DX = DOMAIN_SIZE / N_CELL  # cell size, for the mass-imbalance term

FIELDS: tuple[str, ...] = ("p", "Ux", "Uy")

DEFAULT_CACHE_DIR = Path("Dataset_cache/cavity_paper")

_FNAME_RE = re.compile(r"Data(\d\dk)0\.(\d+)\.csv")


def lid_velocity(case: str) -> float:
    """``"04k" -> 0.4`` m/s (verified against ``max(U:0)`` of each snapshot)."""
    return int(case[:2]) / 10.0


def reynolds(case: str) -> float:
    """``"10k" -> 10000.0``."""
    return lid_velocity(case) * DOMAIN_SIZE / NU


def cell_centres(height: int = N_CELL, width: int = N_CELL) -> np.ndarray:
    """Cell-centre coordinates of the grid, as ``(height * width, 2)`` in
    ``[0, 1]^2``, row-major to match the field arrays.

    The field arrays are indexed ``(H, W)`` with ``H`` along y and ``W`` along
    x (see ``_physics.py``), so row ``h * width + w`` of the result is
    ``(x, y)`` of the cell at ``[h, w]``. Coordinates are normalised by
    :data:`DOMAIN_SIZE` rather than given in metres because the GNO search
    radii that consume them are quoted in domain fractions.

    A mesh-free model needs the coordinates the grid models get for free from
    a :class:`~scirex.layers.GridEmbedding2D`, and they are
    a property of the dataset's grid convention, so they are defined here
    alongside it.
    """
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def vertex_to_cell(v: np.ndarray) -> np.ndarray:
    """(..., 129, 129, C) vertex field -> (..., 128, 128, C) cell field."""
    return 0.25 * (
        v[..., :-1, :-1, :]
        + v[..., 1:, :-1, :]
        + v[..., :-1, 1:, :]
        + v[..., 1:, 1:, :]
    )


def vertex_sort_index(pos: np.ndarray) -> np.ndarray:
    """Row-major (y then x) ordering of the 16,641 sampled vertices.

    The CSV rows arrive in OpenFOAM mesh-vertex order, which is not
    guaranteed to be a raster scan, so the lattice is rebuilt explicitly by
    sorting on rounded coordinates (rounding absorbs the ASCII round-trip
    noise in the exported positions).

    Deliberately not ``scirex.data.datasets.cavity_kepsilon.
    grid_index_maps``, which solves the same problem for the other cavity
    suite: that one takes ``np.unique`` of the raw float coordinates, so the
    same round-trip noise would give it more than 129 distinct values and a
    silently wrong mesh size, and it indexes rows by x where this benchmark
    needs rows by y.
    """
    key_x = np.round(pos[:, 0] / DX).astype(np.int64)
    key_y = np.round(pos[:, 1] / DX).astype(np.int64)
    if key_x.min() < 0 or key_x.max() != N_VERT - 1 or key_y.max() != N_VERT - 1:
        raise ValueError(
            f"unexpected vertex lattice: x in [{key_x.min()}, {key_x.max()}], "
            f"y in [{key_y.min()}, {key_y.max()}], expected 0..{N_VERT - 1}"
        )
    order = np.lexsort((key_x, key_y))  # y major, x minor
    if order.shape[0] != N_VERT * N_VERT:
        raise ValueError(f"expected {N_VERT * N_VERT} vertices, got {order.shape[0]}")
    return order


def _read_snapshot_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos[16641, 2], fields[16641, 3]) for the z=0 layer of one CSV.

    Uses ``pandas.read_csv`` (C parser) rather than ``numpy.genfromtxt``:
    these are 33k-row files and the whole prepared dataset is ~900 of them.
    """
    import pandas as pd

    df = pd.read_csv(path, dtype=np.float32)
    df.columns = [c.strip().strip('"').replace(":", "") for c in df.columns]
    missing = {"p", "U0", "U1", "Points0", "Points1", "Points2"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}; got {list(df.columns)}"
        )

    # Keep only the z=0 layer: the mesh is one cell thick in z (an `empty`
    # patch) and the two layers carry byte-identical field values.
    z = df["Points2"].to_numpy()
    keep = np.isclose(z, 0.0, rtol=0, atol=1e-6)
    d = df[keep]
    if d.shape[0] != N_VERT * N_VERT:
        raise ValueError(
            f"{path}: expected {N_VERT * N_VERT} z=0 points, got {d.shape[0]}"
        )

    pos = np.stack([d["Points0"].to_numpy(), d["Points1"].to_numpy()], axis=-1)
    fields = np.stack(
        [d["p"].to_numpy(), d["U0"].to_numpy(), d["U1"].to_numpy()], axis=-1
    )
    return pos.astype(np.float32), fields.astype(np.float32)


def prepare_case(case: str, raw_dir: Path, cache_dir: Path) -> Path:
    """Writes ``<cache_dir>/<case>.npz`` holding ``grid`` (101, 128, 128, 3)."""
    out_path = cache_dir / f"{case}.npz"
    if out_path.exists():
        return out_path

    t0 = perf_counter()
    case_dir = raw_dir / case
    wanted = []
    for p in case_dir.iterdir():
        m = _FNAME_RE.fullmatch(p.name)
        if m and m.group(1) == case and int(m.group(2)) < N_STEPS:
            wanted.append((int(m.group(2)), p))
    wanted.sort()
    if len(wanted) != N_STEPS:
        raise ValueError(
            f"{case}: expected {N_STEPS} snapshots (steps 0..{N_STEPS - 1}) "
            f"under {case_dir}, found {len(wanted)}"
        )

    pos = None
    frames = []
    for _, path in wanted:
        pos_i, f_i = _read_snapshot_csv(path)
        if pos is None:
            pos = pos_i
        frames.append(f_i)
    steps = np.asarray([s for s, _ in wanted], dtype=np.int32)
    fields = np.stack(frames, axis=0)

    order = vertex_sort_index(pos)
    vert = fields[:, order, :].reshape(N_STEPS, N_VERT, N_VERT, len(FIELDS))
    grid = vertex_to_cell(vert).astype(np.float32)  # (101, 128, 128, 3)

    # Sanity: the lid row must carry this case's lid speed.
    lid_row_ux = float(np.max(grid[:, -1, :, 1]))
    expected = lid_velocity(case)
    if not (0.4 * expected <= lid_row_ux <= 1.05 * expected):
        raise ValueError(
            f"{case}: top-row max Ux = {lid_row_ux:.4f} m/s is inconsistent with "
            f"lid speed {expected} m/s; the grid orientation is probably wrong"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, grid=grid, steps=steps, u_lid=np.float32(expected))
    print(
        f"[data] prepared {case}: {grid.shape} from {len(wanted)} CSVs "
        f"in {perf_counter() - t0:.1f}s -> {out_path}"
    )
    sys.stdout.flush()
    return out_path


def ensure_prepared(cases: tuple[str, ...], raw_dir: Path, cache_dir: Path) -> None:
    """Converts any of ``cases`` that is not yet in ``cache_dir``."""
    for case in cases:
        prepare_case(case, raw_dir, cache_dir)


def load_prepared(case: str, cache_dir: Path) -> np.ndarray:
    """(101, 128, 128, 3) physical-unit grid of one case."""
    with np.load(cache_dir / f"{case}.npz") as npz:
        return npz["grid"]


def make_windows(
    grid: np.ndarray, n_in: int, n_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """(T, H, W, C) -> inputs (n, n_in, H, W, C), targets (n, n_out, H, W, C).

    Sample ``i`` maps frames ``[i, i+n_in)`` to ``[i+n_in, i+n_in+n_out)``,
    the paper's "forecast t_{i+3} given {t_i, t_i+1, t_i+2}" for 3:1.
    """
    T = grid.shape[0]
    n = T - n_in - n_out + 1
    if n <= 0:
        raise ValueError(f"need at least {n_in + n_out} frames, got {T}")
    xs = np.stack([grid[i : i + n_in] for i in range(n)], axis=0)
    ys = np.stack([grid[i + n_in : i + n_in + n_out] for i in range(n)], axis=0)
    return xs, ys


def build_train_val(
    cache_dir: Path,
    n_in: int,
    n_out: int,
    val_frac: float = 0.1,
    seed: int = 0,
) -> dict:
    """Normalised train / validation windows from the training cases only.

    The validation split is carved out of the TRAINING cases (the paper uses a
    validation loss for its LR scheduler and early stopping). The Re = 10,000
    test case is never loaded here; only ``infer.py`` touches it.

    Normalisation is per-channel, computed on the training split only, by
    ``scirex``' own :class:`~scirex.data.UnitGaussianNormalizer`.
    Its mean and standard deviation are stored in the checkpoint so inference
    can map predictions back to physical units without reloading training data.
    """
    xs, ys = [], []
    for case in TRAIN_CASES:
        a, b = make_windows(load_prepared(case, cache_dir), n_in, n_out)
        xs.append(a)
        ys.append(b)
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_val = max(1, int(round(val_frac * X.shape[0])))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    normalizer = fit_normalizer(X_train)
    mean, std = channel_stats(normalizer)
    return dict(
        X_train=normalise(X_train, normalizer),
        Y_train=normalise(Y_train, normalizer),
        X_val=normalise(X_val, normalizer),
        Y_val=normalise(Y_val, normalizer),
        normalizer=normalizer,
        mean=mean,
        std=std,
    )


def fit_normalizer(a: np.ndarray) -> UnitGaussianNormalizer:
    """Per-channel statistics of ``a``, reduced over every axis but the last."""
    axis = tuple(range(a.ndim - 1))
    return UnitGaussianNormalizer.create(axis=axis).fit(jnp.asarray(a))


def normalizer_from_stats(mean: np.ndarray, std: np.ndarray) -> UnitGaussianNormalizer:
    """The normaliser a checkpoint's stored mean and standard deviation describe."""
    return UnitGaussianNormalizer(
        mean=jnp.asarray(mean, jnp.float32), std=jnp.asarray(std, jnp.float32)
    )


def channel_stats(
    normalizer: UnitGaussianNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, std)`` as flat per-channel vectors, ready to store as JSON."""
    return (
        np.asarray(normalizer.mean, np.float32).reshape(-1),
        np.asarray(normalizer.std, np.float32).reshape(-1),
    )


def normalise(a: np.ndarray, normalizer: UnitGaussianNormalizer) -> np.ndarray:
    return np.asarray(normalizer.transform(jnp.asarray(a)), np.float32)


def denormalise(a: np.ndarray, normalizer: UnitGaussianNormalizer) -> np.ndarray:
    return np.asarray(normalizer.inverse_transform(jnp.asarray(a)), np.float32)
