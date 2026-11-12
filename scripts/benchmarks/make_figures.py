# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark figures in one style.

Every model page and every comparison page shows the same kinds of figure,
drawn here with one canvas size per kind, one font, fixed colours and plain
labels ("Train", "Test"):

  error_curve.png   training and test error per epoch       1280 x 720
  compute_cost.png  training time per epoch                 1280 x 720
  contour.png       prediction, ground truth, absolute error 1920 x 640
                    (one row per model on a comparison page)

The curves are read from each run's training history; the contours from the
test samples exported by ``make_landing.py export`` (same cache format).

Usage (from the repo root)::

    python scripts/benchmarks/make_figures.py curves --history-root <ARCHIVE>
    python scripts/benchmarks/make_figures.py contours
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from scripts.benchmarks.make_landing import (  # noqa: E402
    DPI,
    ERROR_CMAP,
    FIELD_CMAP,
    FONT,
)

ASSETS = _REPO_ROOT / "doc/assets/benchmarks"
CURVE_SIZE = (8.0, 4.5)  # 1280 x 720 at DPI
TRAIN_COLOR, TEST_COLOR = "#1f77b4", "#ff7f0e"
MODEL_COLORS = {
    "FNO": "#1f77b4",
    "WNO": "#ff7f0e",
    "GINO": "#2ca02c",
    "GIWO": "#9467bd",
    "Transolver": "#d62728",
}
LABELS = {
    "fno": "FNO",
    "wno": "WNO",
    "gino": "GINO",
    "giwo": "GIWO",
    "transolver": "Transolver",
}

# One entry per benchmark: where each model's history is in the archive, how
# its keys are named, and what the curve measures.
#   runs:  model -> list of (history path, run label); the first run is the
#          one on the model page, a second one is drawn dashed
CURVES = {
    "pipe": dict(
        problem="Pipe flow",
        ylabel="Relative L2 error",
        test="Test",
        train_key="train_loss",
        test_key="test_rel_l2",
        time_key="secs",
        runs={
            m: [(f"pipe/{m}/history.json", None)]
            for m in ("fno", "wno", "gino", "transolver")
        },
    ),
    "navier_stokes": dict(
        problem="Navier-Stokes 2D",
        ylabel="Relative L2 error (10-step rollout)",
        test="Test",
        train_key="train_loss",
        test_key="test_rel_l2",
        time_key="secs",
        runs={
            m: [(f"navier_stokes/{m}/history.json", None)]
            for m in ("fno", "wno", "gino", "transolver")
        },
    ),
    "cavity": dict(
        problem="Lid-driven cavity",
        ylabel="Loss (normalised MSE + mass term)",
        test="Validation",
        train_key="train_loss",
        test_key="val_loss",
        time_key="secs",
        runs={
            m: [
                (f"lid_driven_cavity_{m}/{m}_3to1/history.json", "default size"),
                (f"lid_driven_cavity_{m}/best_{m}_3to1/history.json", "searched size"),
            ]
            for m in ("fno", "wno", "gino", "transolver")
        },
    ),
    "chip_cooling": dict(
        problem="Chip cooling",
        ylabel="Relative L2 error",
        test="Test",
        train_key="loss",
        test_key="val_loss",
        time_key="time",
        runs={
            "fno": [("chip_cooling/fno/history.json", None)],
            "wno": [("chip_cooling/wno/history.json", None)],
            "gino": [("chip_cooling/gino/history.json", None)],
            "transolver": [("chip_cooling/transolver_full_norm/history.json", None)],
        },
    ),
    "airfrans": dict(
        problem="AirfRANS",
        ylabel="Volume MSE (normalised fields)",
        test="Validation",
        train_key="train_vol",
        test_key="val_vol",
        time_key="secs",
        runs={
            "wno": [("airfrans/wno/history.json", None)],
            "gino": [("airfrans/gino/loss_curve.csv", None)],
            "transolver": [("airfrans/transolver/history.json", None)],
        },
    ),
    "shapenetcar": dict(
        problem="Car-Design-ShapeNetCar",
        ylabel="Loss (normalised MSE)",
        test="Validation",
        train_key="train_loss",
        test_key="val_loss",
        time_key="secs",
        runs={
            m: [(f"shapenetcar_{m}/history.json", None)]
            for m in ("gino", "giwo", "transolver")
        },
    ),
    "car_cfd": dict(
        problem="Car-CFD",
        ylabel="Relative L2 error of the pressure",
        test="Test",
        train_key="train_err",
        test_key="test_err",
        time_key="secs",
        runs={
            m: [(f"car_cfd_{m}/history.json", None)]
            for m in ("gino", "giwo", "transolver")
        },
    ),
}

# The GINO AirfRANS run wrote a CSV with its own column names.
CSV_KEYS = {"train_vol": "train_vol", "val_vol": "test_vol", "secs": "sec"}


def load_history(path: Path, keys: tuple[str, str, str]) -> dict[str, np.ndarray]:
    """``epoch``, ``train``, ``test`` and ``secs`` arrays from one history file.

    A history is a list of per-epoch records, a dict of per-epoch lists, or a
    CSV; epochs without a test value are dropped from the test curve only.
    """
    train_key, test_key, time_key = keys
    if path.suffix == ".csv":
        with path.open() as f:
            rows = list(csv.DictReader(f))
        col = lambda k: np.array([float(r[CSV_KEYS.get(k, k)] or "nan") for r in rows])  # noqa: E731
        train, test, secs = col(train_key), col(test_key), col(time_key)
    else:
        h = json.loads(path.read_text())
        if isinstance(h, dict):
            train, test, secs = (
                np.asarray(h[k], float) for k in (train_key, test_key, time_key)
            )
        else:
            col = lambda k: np.array(
                [float(r.get(k, np.nan) if r.get(k) is not None else np.nan) for r in h]
            )
            train, test, secs = col(train_key), col(test_key), col(time_key)
    return dict(epoch=np.arange(len(train)), train=train, test=test, secs=secs)


def _style(ax, title: str, ylabel: str, log: bool = True) -> None:
    if log:
        from matplotlib.ticker import LogLocator, ScalarFormatter

        ax.set_yscale("log")
        # plain numbers (0.01, 20) instead of 10^-2, 2x10^1 when the range is short
        lo, hi = ax.get_ylim()
        if hi / lo < 100:
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_formatter(fmt)
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(LogLocator(subs=()))
    ax.set_title(title, pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _line(ax, x, y, **kw) -> None:
    ok = np.isfinite(y) & (y > 0)
    ax.plot(x[ok], y[ok], linewidth=1.6, **kw)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


def render_curves(root: Path, out: Path, only: str | None = None) -> None:
    """Every model's error_curve.png and compute_cost.png, and the comparison ones."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT)
    for bench, spec in CURVES.items():
        if only and bench != only:
            continue
        keys = (spec["train_key"], spec["test_key"], spec["time_key"])
        test_name = spec["test"]
        loaded = {
            m: [(load_history(root / p, keys), lab) for p, lab in runs]
            for m, runs in spec["runs"].items()
        }

        for m, runs in loaded.items():
            name = LABELS[m]
            fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
            for (h, lab), ls in zip(runs, ("-", "--")):
                suffix = f", {lab}" if lab else ""
                _line(
                    ax,
                    h["epoch"],
                    h["train"],
                    color=TRAIN_COLOR,
                    linestyle=ls,
                    label=f"Train{suffix}",
                )
                _line(
                    ax,
                    h["epoch"],
                    h["test"],
                    color=TEST_COLOR,
                    linestyle=ls,
                    label=f"{test_name}{suffix}",
                )
            _style(ax, f"{name}  ·  {spec['problem']}", spec["ylabel"])
            ax.legend(frameon=True, framealpha=0.9, edgecolor="none")
            _save(fig, out / f"{bench}_{m}" / "error_curve.png")

            h = runs[0][0]
            fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
            _line(ax, h["epoch"], h["secs"], color=MODEL_COLORS[name])
            _style(ax, f"{name}  ·  {spec['problem']}", "Training time per epoch (s)")
            ax.annotate(
                "epoch 0 includes compilation",
                xy=(0, h["secs"][0]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color="#555555",
            )
            _save(fig, out / f"{bench}_{m}" / "compute_cost.png")

        # comparison: the test curve and the time per epoch of every model
        for kind, key, ylabel in (
            (
                "error_curve",
                "test",
                f"{test_name}: {spec['ylabel'][0].lower()}{spec['ylabel'][1:]}",
            ),
            ("compute_cost", "secs", "Training time per epoch (s)"),
        ):
            fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
            for m, runs in loaded.items():
                h = runs[0][0]
                _line(
                    ax,
                    h["epoch"],
                    h[key],
                    color=MODEL_COLORS[LABELS[m]],
                    label=LABELS[m],
                )
            _style(ax, spec["problem"], ylabel)
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
            _save(fig, out / f"{bench}_comparison" / f"{kind}.png")
        print(
            f"[figures] {bench}: curves of {', '.join(LABELS[m] for m in loaded)}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Export: every test sample of a released checkpoint, in the landing format
# ---------------------------------------------------------------------------
# Each exporter returns the dict that make_landing.render reads: kind, problem,
# model, quantity, unit, n_test, ids, rel (per stored sample), truth, pred and
# the geometry. ``rel_all`` holds the error of every test sample.


def _rel(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    n = len(truth)
    return np.linalg.norm((pred - truth).reshape(n, -1), axis=1) / np.linalg.norm(
        truth.reshape(n, -1), axis=1
    )


def export_pipe(model: str, ckpt: Path, data_dir: Path) -> dict:
    from scripts.benchmarks.pipe._checkpoint import load_checkpoint
    from scripts.benchmarks.pipe._config import PROTOCOL
    from scripts.benchmarks.pipe._data import load_split
    from scripts.benchmarks.pipe.infer import predict

    data = load_split(data_dir, PROTOCOL)
    ck = load_checkpoint(ckpt)
    h, w = data["grid"]
    pred = predict(ck, data["xy_test"], 8)[..., 0].reshape(-1, h, w)
    truth = data["y_test"][..., 0].reshape(-1, h, w)
    xy = data["xy_test"].reshape(-1, h, w, 2)
    rel = _rel(pred, truth)
    return dict(
        kind="mesh2d",
        problem="Pipe flow",
        model=LABELS[model],
        quantity="velocity u",
        unit="",
        n_test=len(rel),
        ids=np.arange(len(rel)),
        rel=rel,
        truth=truth,
        pred=pred,
        X=xy[..., 0],
        Y=xy[..., 1],
    )


def export_navier_stokes(model: str, ckpt: Path, data_dir: Path) -> dict:
    from scripts.benchmarks.navier_stokes._checkpoint import load_checkpoint
    from scripts.benchmarks.navier_stokes._config import DATA_FILE, PROTOCOL
    from scripts.benchmarks.navier_stokes._data import load_split
    from scripts.benchmarks.navier_stokes.infer import predict

    data = load_split(Path(data_dir) / DATA_FILE, PROTOCOL)
    ck = load_checkpoint(ckpt)
    h, w = data["grid"]
    t_in, t_out = PROTOCOL["t_in"], PROTOCOL["t_out"]
    pred = predict(ck, data["a_test"], t_out, 20)[..., -1].reshape(
        -1, h, w
    )  # last rollout step
    truth = data["u_test"][..., -1].reshape(-1, h, w)
    rel = _rel(pred, truth)
    return dict(
        kind="grid",
        problem="Navier-Stokes 2D",
        model=LABELS[model],
        quantity=f"vorticity at t = {t_in + t_out}",
        unit="",
        n_test=len(rel),
        ids=np.arange(len(rel)),
        rel=rel,
        truth=truth,
        pred=pred,
    )


def export_cavity(model: str, ckpt: Path, data_dir: Path) -> dict:
    from scripts.benchmarks.make_landing import export_cavity as landing

    c = landing(ckpt, data_dir, select=lambda rel: np.arange(len(rel)))
    c["model"] = LABELS[model]
    return c


def export_darcy(model: str, ckpt: Path, data_dir: Path) -> dict:
    import jax
    import jax.numpy as jnp

    from scripts.benchmarks._batch_eval import predict_in_batches
    from scripts.benchmarks._ckpt import load_params
    from scripts.benchmarks.darcy._config import PROTOCOL, RUNS
    from scripts.benchmarks.darcy._data import load_split
    from scripts.benchmarks.darcy._models import prepare

    run = RUNS[model]
    data = load_split(data_dir, PROTOCOL)
    prep = prepare(run.model, run.arch, data)
    params = load_params(prep.init(jax.random.PRNGKey(0)), ckpt)
    n, s = data["test_u"].shape[0], data["test_u"].shape[-1]
    pred = np.asarray(
        predict_in_batches(
            jax.jit(prep.predict),
            params,
            jnp.asarray(prep.test_x),
            run.opt["batch_size"],
        )
    ).reshape(n, s, s)
    truth = data["test_u"].reshape(n, s, s)
    rel = _rel(pred, truth)
    return dict(
        kind="grid",
        problem="Darcy flow",
        model=LABELS[model],
        quantity="pressure",
        unit="",
        n_test=n,
        ids=np.arange(n),
        rel=rel,
        truth=truth,
        pred=pred,
    )


def export_car_cfd(model: str, ckpt: Path, data_dir: Path) -> dict:
    from scripts.benchmarks.make_landing import export_car_cfd as landing

    c = landing(ckpt, data_dir, select=lambda rel: np.arange(len(rel)))
    c["model"] = LABELS[model]
    return c


def _masked_delaunay(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Triangles of scattered points, without the long ones that bridge holes
    (the pin fins of the cold plate)."""
    from matplotlib.tri import Triangulation

    tri = Triangulation(x, y).triangles
    p = np.stack([x, y], -1)[tri]
    edge = np.max(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=-1), axis=1)
    return tri[edge < 3.0 * np.median(edge)].astype(np.int32)


def export_chip_cooling(model: str, ckpt: Path, data_dir: Path) -> dict:
    """Temperature on the 2D slice, on the points each model is scored on:
    in-domain grid cells for FNO and WNO, mesh nodes for GINO and Transolver."""
    from scripts.benchmarks.chip_cooling._checkpoint import (
        load_config,
        load_params_into,
    )
    from scripts.benchmarks.chip_cooling._config import DEFAULT_CACHE_DIR, PROTOCOL
    from scripts.benchmarks.chip_cooling._data import load_raw
    from scripts.benchmarks.chip_cooling._models import get_spec
    from scripts.benchmarks.chip_cooling.infer import field_values, points_mm
    from scripts.benchmarks.chip_cooling.train import load_split

    raw = load_raw(data_dir)
    run, config = load_config(ckpt)
    spec = get_spec(run.model)
    data = load_split(spec.kind, raw, PROTOCOL, DEFAULT_CACHE_DIR)
    stored = config["normalisation"]
    if spec.kind == "mesh" and not np.isclose(
        stored["t_mean"], data["normalisation"]["t_mean"]
    ):
        # a run trained with other statistics (transolver_full_norm): use its own
        p_mean, p_std = (
            np.asarray(stored["p_mean"], np.float32),
            np.asarray(stored["p_std"], np.float32),
        )
        data["params_n"] = ((raw["parameters"] - p_mean) / p_std).astype(np.float32)
        data["normalisation"] = dict(stored, p_mean=p_mean, p_std=p_std)
    prepared = spec.prepare(run.arch, data)
    net = spec.build(run.arch, PROTOCOL["grid"])
    params = load_params_into(
        ckpt, spec.init(net, run.arch, prepared, PROTOCOL["seed"])
    )
    pred_full = spec.predict(net, params, prepared, data)
    ck = dict(spec=spec, data=data)
    truth_full = data["Y_test" if spec.kind == "grid" else "fields_test"]
    pred, truth = field_values(ck, pred_full), field_values(ck, truth_full)
    z, y = points_mm(ck, raw)
    rel = _rel(pred, truth)
    mae = float(np.mean(np.abs(pred - truth)))
    print(f"[figures] chip_cooling/{model}: MAE {mae:.4f} K", flush=True)
    out = dict(
        kind="tri2d",
        problem="Chip cooling",
        model=LABELS[model],
        quantity="temperature",
        unit="K",
        n_test=len(rel),
        ids=np.arange(len(rel)),
        rel=rel,
        truth=truth,
        pred=pred,
        x=z,
        y=y,
        tri=_masked_delaunay(z, y),
    )
    return out


def export_shapenetcar(model: str, ckpt: Path, data_dir: Path) -> dict:
    """Surface pressure on every validation car."""
    import jax.numpy as jnp

    from scirex.data.datasets.shapenet_car_drag import load_surface_mesh
    from scripts.benchmarks.shapenetcar import _gino_geometry
    from scripts.benchmarks.shapenetcar._checkpoint import load_checkpoint
    from scripts.benchmarks.shapenetcar._data import DEFAULT_CACHE_DIR, load_split
    from scripts.benchmarks.shapenetcar.infer import _apply_model

    ck = load_checkpoint(ckpt)
    cache = Path(os.environ.get("SHAPENET_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()
    data = load_split(
        Path(data_dir), ck["config"]["training"]["fold_id"], cache, use_cache=True
    )
    samples = data["val"]
    if ck["run"].model in ("gino", "giwo"):
        a = ck["run"].arch
        _gino_geometry.attach_neighbors(
            samples, a["latent_grid"], a["in_gno_radius"], a["out_gno_radius"]
        )
    _, _, mean_out, std_out = ck["coef_norm"]
    out = dict(
        kind="surface",
        problem="Car-Design-ShapeNetCar",
        model=LABELS[model],
        quantity="surface pressure",
        unit="Pa",
        sample_word="test car",
    )
    rel, kept = [], 0
    for s in samples:
        o = (
            _apply_model(ck["model"], ck["params"], jnp.asarray(s["x"])[None], s)
            * std_out
            + mean_out
        )
        yv = s["y"] * std_out + mean_out
        p_pred, p_true = o[s["surf"], -1], yv[s["surf"], -1]
        vtk = Path(data_dir) / s["sample_rel_path"] / "quadpress_smpl.vtk"
        if not vtk.exists():
            continue
        pts, quads = load_surface_mesh(vtk)
        quads = np.asarray(quads)
        out[f"v{kept}"] = np.asarray(pts, np.float32)
        out[f"t{kept}"] = np.concatenate(
            [quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]
        ).astype(np.int32)
        out[f"truth{kept}"], out[f"pred{kept}"] = (
            p_true.astype(np.float32),
            p_pred.astype(np.float32),
        )
        rel.append(float(np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)))
        kept += 1
    rel = np.array(rel)
    out.update(n_test=kept, ids=np.arange(kept), rel=rel)
    return out


def export_airfrans(
    model: str, ckpt: Path, data_dir: Path, cases: list[str] | None = None
) -> dict:
    """|U| / U_inf around the airfoil for the median case (and ``cases``)."""
    from vtk.util.numpy_support import vtk_to_numpy

    from scripts.benchmarks.airfrans._checkpoint import load_checkpoint
    from scripts.benchmarks.airfrans._config import DEFAULT_CACHE_DIR, PROTOCOL
    from scripts.benchmarks.airfrans._data import load_case, load_split
    from scripts.benchmarks.airfrans._metrics import CaseGeometry
    from scripts.benchmarks.airfrans.infer import WINDOW, evaluate

    ck = load_checkpoint(ckpt)
    split = load_split(Path(data_dir), PROTOCOL["n_val"])
    rows, preds = evaluate(ck, split, Path(data_dir), DEFAULT_CACHE_DIR)
    ids = [r["case"] for r in rows]
    vol = np.array([r["volume"] for r in rows])
    keep = [ids[int(np.argsort(vol)[len(vol) // 2])]] + [
        c for c in (cases or []) if c not in ids[:0]
    ]
    keep = list(dict.fromkeys(keep))
    out = dict(
        kind="tri2d",
        problem="AirfRANS",
        model=LABELS[model],
        quantity="speed |U| / U$_\\infty$",
        unit="",
        n_test=len(ids),
        ids=np.array([ids.index(c) for c in keep]),
        rel=np.zeros(len(keep)),
        rel_all=vol,
        case_names=np.array(keep),
        window=np.array(WINDOW),
    )
    for k, cid in enumerate(keep):
        raw = load_case(Path(data_dir), DEFAULT_CACHE_DIR, cid)
        geo = CaseGeometry(Path(data_dir), cid)
        quads = vtk_to_numpy(geo.internal.GetCells().GetConnectivityArray()).reshape(
            -1, 4
        )
        tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        P = raw["pos"]
        c = P[tri].mean(1)
        sel = (
            (c[:, 0] > WINDOW[0] - 0.1)
            & (c[:, 0] < WINDOW[1] + 0.1)
            & (np.abs(c[:, 1]) < WINDOW[3] + 0.1)
        )
        used = np.unique(tri[sel])
        remap = -np.ones(len(P), np.int64)
        remap[used] = np.arange(len(used))
        t_true = np.linalg.norm(raw["y"][used, :2], axis=1) / geo.u_inf
        t_pred = np.linalg.norm(preds[cid][used, :2], axis=1) / geo.u_inf
        out[f"x{k}"], out[f"y{k}"] = (
            P[used, 0].astype(np.float32),
            P[used, 1].astype(np.float32),
        )
        out[f"tri{k}"] = remap[tri[sel]].astype(np.int32)
        out[f"outline{k}"] = np.asarray(geo.outline(), np.float32)
        out[f"truth{k}"], out[f"pred{k}"] = (
            t_true.astype(np.float32),
            t_pred.astype(np.float32),
        )
        out["rel"][k] = np.linalg.norm(t_pred - t_true) / np.linalg.norm(t_true)
    return out


EXPORT = dict(
    pipe=export_pipe,
    navier_stokes=export_navier_stokes,
    cavity=export_cavity,
    darcy=export_darcy,
    car_cfd=export_car_cfd,
    chip_cooling=export_chip_cooling,
    shapenetcar=export_shapenetcar,
    airfrans=export_airfrans,
)


def export(
    bench: str, model: str, ckpt: Path, data_dir: Path, cache: Path, cases=None
) -> None:
    import jax

    jax.config.update("jax_enable_x64", False)
    c = (
        EXPORT[bench](model, ckpt, data_dir, cases)
        if bench == "airfrans"
        else EXPORT[bench](model, ckpt, data_dir)
    )
    c["rel_all"] = c["rel"] if "rel_all" not in c else c["rel_all"]
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / f"{bench}_{model}.npz", **c)
    print(
        f"[figures] {bench}/{model}: mean test rel. L2 {np.mean(c['rel_all']):.5f} over {len(c['rel_all'])}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Contours: one model's median-error sample; all models on one sample
# ---------------------------------------------------------------------------

# the order of the models in a comparison, most accurate model first where known
ORDER = ("transolver", "fno", "wno", "gino", "giwo")


def _aspect(c: dict) -> float:
    """Width over height of the drawn domain, between 1 and 4."""
    if c["kind"] == "grid":
        h, w = np.asarray(c["truth"]).shape[-2:] if "truth" in c else (1, 1)
        a = w / h
    elif c["kind"] == "tri2d":
        if "window" in c:
            wd = c["window"]
            a = (wd[1] - wd[0]) / (wd[3] - wd[2])
        else:
            x = c["x"] if "x" in c else c["x0"]
            y = c["y"] if "y" in c else c["y0"]
            a = np.ptp(x) / np.ptp(y)
    elif c["kind"] == "surface":
        a = 1.0  # a 3D view fills a square panel
    else:
        a = 1.0
    return float(np.clip(a, 1.0, 4.0))


def median_index(c: dict) -> int:
    return int(np.argsort(c["rel"])[len(c["rel"]) // 2])


def render_model_contour(c: dict, path: Path) -> None:
    from scripts.benchmarks.make_landing import canvas, draw_sample

    fig, axes, caxes = canvas(c)
    draw_sample(fig, axes, caxes, c, median_index(c))
    _save(fig, path)


def render_comparison_contour(cs: dict[str, dict], path: Path, problem: str) -> None:
    """Ground truth and every model's prediction and error on one test sample:
    the median-error sample of the first model in ``ORDER``."""
    import matplotlib.pyplot as plt

    from scripts.benchmarks.make_landing import (
        _sample,
        draw_panel,
        error_max,
        field_range,
        is_elongated,
    )

    plt.rcParams.update(FONT)
    models = [m for m in ORDER if m in cs]
    ref = cs[models[0]]
    sample = int(ref["ids"][median_index(ref)])
    ks = {m: int(np.flatnonzero(cs[m]["ids"] == sample)[0]) for m in models}
    # every model is drawn on the points it is scored on (chip cooling: grid
    # cells for FNO and WNO, mesh nodes for the others), against its own truth
    truth = _sample(ref, ks[models[0]])[0]
    truths = {m: _sample(cs[m], ks[m])[0] for m in models}
    preds = {m: _sample(cs[m], ks[m])[1] for m in models}
    errs = {m: np.abs(preds[m] - truths[m]) for m in models}
    lo, hi = field_range(truth, *preds.values())
    emax = max(error_max(e) for e in errs.values())
    proj = "3d" if ref["kind"] == "surface" else None
    n = len(models)
    if is_elongated(ref):
        # rows: ground truth, then each model's prediction and its error
        rows = 1 + 2 * n
        fig = plt.figure(figsize=(12.0, 0.6 + 1.25 * rows), dpi=DPI, facecolor="white")
        h = 1.0 / (rows + 0.6)
        cells = [("Ground truth", truth, FIELD_CMAP, lo, hi)]
        for m in models:
            cells += [
                (f"{LABELS[m]}: prediction", preds[m], FIELD_CMAP, lo, hi),
                (f"{LABELS[m]}: absolute error", errs[m], ERROR_CMAP, 0.0, emax),
            ]
        for i, (title, v, cmap, a, b) in enumerate(cells):
            ax = fig.add_axes([0.22, 1 - (i + 1.45) * h, 0.62, 0.86 * h])
            cax = fig.add_axes([0.86, 1 - (i + 1.45) * h, 0.008, 0.86 * h])
            draw_panel(
                fig, ax, cax, ref, ks[models[0]], title, v, cmap, a, b, side_label=True
            )
    else:
        # columns: ground truth, then one per model; top row fields, bottom row errors.
        # The height follows the shape of the domain, so wide domains leave no empty bands.
        cols = 1 + n
        aspect = _aspect(ref)
        col_w = 3.4
        panel_w = 0.72 * col_w
        panel_h = panel_w / aspect
        title_h, row_gap = 0.8, 0.55
        H = title_h + 2 * (panel_h + row_gap)
        fig = plt.figure(figsize=(col_w * cols, H), dpi=DPI, facecolor="white")
        w = 1.0 / cols

        def box(j, row):
            y0 = (row_gap * 0.4 + (1 - row) * (panel_h + row_gap)) / H
            return [j * w + 0.008, y0, panel_w / (col_w * cols), panel_h / H]

        for j, (title, v) in enumerate(
            [("Ground truth", truth)] + [(LABELS[m], preds[m]) for m in models]
        ):
            b = box(j, 0)
            ax = fig.add_axes(b, projection=proj)
            cax = fig.add_axes([j * w + w * 0.76, b[1], 0.006, b[3]])
            k = ks[models[0]] if j == 0 else ks[models[j - 1]]
            src = ref if j == 0 else cs[models[j - 1]]
            draw_panel(fig, ax, cax, src, k, title, v, FIELD_CMAP, lo, hi)
            cax.set_visible(j == cols - 1)
        for j, m in enumerate(models, start=1):
            b = box(j, 1)
            ax = fig.add_axes(b, projection=proj)
            cax = fig.add_axes([j * w + w * 0.76, b[1], 0.006, b[3]])
            rel = cs[m]["rel"][ks[m]]
            draw_panel(
                fig,
                ax,
                cax,
                cs[m],
                ks[m],
                f"Error, rel. L2 {100 * rel:.2f}%",
                errs[m],
                ERROR_CMAP,
                0.0,
                emax,
            )
            cax.set_visible(j == cols - 1)
    word = ref.get("sample_word", "test sample")
    fig.suptitle(
        f"{problem}  ·  {ref['quantity']}  ·  {word} {sample + 1} of {ref['n_test']}",
        y=0.985,
    )
    _save(fig, path)


def render_contours(cache: Path, out: Path, bench: str) -> None:
    from scripts.benchmarks.make_landing import load_cache

    cs = {
        p.stem.split("_")[-1]: load_cache(p)
        for p in sorted(cache.glob(f"{bench}_*.npz"))
        if p.stem[len(bench) + 1 :] in LABELS
    }
    for m, c in cs.items():
        render_model_contour(c, out / f"{bench}_{m}" / "contour.png")
    if len(cs) > 1:
        render_comparison_contour(
            cs,
            out / f"{bench}_comparison" / "contour.png",
            next(iter(cs.values()))["problem"],
        )
    print(
        f"[figures] {bench}: contours of {', '.join(LABELS[m] for m in cs)}", flush=True
    )


# ---------------------------------------------------------------------------
# Extras: the physics-only scaling and the AirfRANS accuracy chart, from the
# committed metrics.json files
# ---------------------------------------------------------------------------

FIELDS = (1000, 2000, 5000, 10000, 20000)


def render_scaling(assets: Path, out: Path) -> None:
    """Physics-only Darcy: test error against the number of coefficient fields."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT)
    floor = json.loads((assets / "darcy_q4_s85_spec_floor/metrics.json").read_text())
    floor = 100 * floor["results"]["res84_majority"]["mapped"]["rel_l2_mean"]
    curves = {}
    for m in ("transolver", "fno", "gino", "wno"):
        runs = [
            json.loads(
                (
                    assets / f"darcy_physics_{m}/fields_{n}_seed0/metrics.json"
                ).read_text()
            )
            for n in FIELDS
        ]
        data = json.loads((assets / f"darcy_{m}/metrics.json").read_text())[
            "final_test_rel_l2_mean"
        ]
        curves[m] = (
            [100 * r["final_test_rel_l2_mean"] for r in runs],
            100 * data,
            runs[-1].get("steps_run"),
        )

    def axes_style(ax, title):
        ax.set_xscale("log")
        ax.set_xticks(FIELDS, [f"{n // 1000}k" for n in FIELDS])
        ax.minorticks_off()
        ax.set_xlabel("Coefficient fields seen by the physics loss")
        ax.set_ylabel("Test relative L2 error (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(title, pad=8)
        ax.grid(True, color="#dddddd", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for m, (errs, data, steps) in curves.items():
        name = LABELS[m]
        fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
        ax.plot(
            FIELDS,
            errs,
            "o-",
            color=MODEL_COLORS[name],
            linewidth=1.8,
            label="Physics loss, no labels",
        )
        for n, e in zip(FIELDS, errs):
            ax.annotate(
                f"{e:.2f}%",
                (n, e),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.axhline(
            data,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            label=f"Data loss, 1000 labels ({data:.2f}%)",
        )
        ax.axhline(
            floor,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"Formulation floor ({floor:.2f}%)",
        )
        axes_style(ax, f"{name}  ·  Darcy flow, physics-only")
        ax.set_ylim(0, 1.2 * max(errs + [data]))
        ax.legend(frameon=False)
        _save(fig, out / f"darcy_physics_{m}" / "scaling.png")

    fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
    for m, (errs, _, steps) in curves.items():
        ax.plot(
            FIELDS,
            errs,
            "o-",
            color=MODEL_COLORS[LABELS[m]],
            linewidth=1.8,
            label=LABELS[m],
        )
    ax.axhline(
        0.57,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Transolver paper, data loss (0.57%)",
    )
    ax.axhline(
        floor,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Formulation floor ({floor:.2f}%)",
    )
    axes_style(ax, "Darcy flow, physics-only")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(fig, out / "darcy_physics_comparison" / "scaling.png")
    print("[figures] darcy_physics: scaling", flush=True)


# Transolver paper, Table 3: Transolver and GINO rows
AIRFRANS_PAPER = {
    "Transolver": (0.0037, 0.0142, 0.103, 0.9978),
    "GINO": (0.0297, 0.0482, 0.1821, 0.9958),
}


def render_airfrans_accuracy(assets: Path, out: Path) -> None:
    """The four official AirfRANS metrics, published and measured, as bars."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT)
    measured = {}
    for m in ("transolver", "wno"):
        r = json.loads((assets / f"airfrans_{m}/metrics.json").read_text())["results"]
        measured[LABELS[m]] = (r["volume"], r["surf"], r["relerr_cl"], r["spearman_cl"])
    g = json.loads((assets / "airfrans_gino/metrics.json").read_text())
    measured["GINO"] = (g["volume"], g["surf"], g["C_L_rel_err"], g["rho_L"])
    bars = [
        (f"{k} (paper)", v, MODEL_COLORS[k], 0.35) for k, v in AIRFRANS_PAPER.items()
    ]
    bars += [(k, v, MODEL_COLORS[k], 1.0) for k, v in measured.items()]
    titles = [
        "Volume MSE",
        "Surface MSE",
        "Lift coefficient, relative error",
        "1 − Spearman ρ of the lift",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), layout="constrained")
    for i, (ax, title) in enumerate(zip(axes, titles)):
        vals = [(1 - v[3]) if i == 3 else v[i] for _, v, _, _ in bars]
        x = np.arange(len(bars))
        ax.bar(
            x,
            vals,
            color=[c for _, _, c, _ in bars],
            alpha=1.0,
            edgecolor="#333333",
            linewidth=0.5,
        )
        for xi, (_, _, _, a), v in zip(x, bars, vals):
            ax.patches[xi].set_alpha(a)
            ax.annotate(
                f"{v:.2g}",
                (xi, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.set_yscale("log")
        ax.set_xticks(x, [b[0] for b in bars], rotation=35, ha="right")
        ax.set_title(title, pad=8)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("AirfRANS  ·  200 test cases  ·  lower is better (log scale)")
    _save(fig, out / "airfrans_comparison" / "accuracy.png")
    print("[figures] airfrans: accuracy", flush=True)


# ---------------------------------------------------------------------------
# Summary: for every benchmark, the accuracy and the time per epoch of each
# model as bars, from the same values as the matrix pages
# ---------------------------------------------------------------------------


def render_summary(out: Path) -> None:
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import make_benchmark_matrices as mb

    plt.rcParams.update(FONT)
    for p in mb.PROBLEMS:
        acc, cost = {}, {}
        for m in mb.MODELS:
            if not mb.has_page(p, m):
                continue
            v = mb.value(p, m, p.accuracy) if p.accuracy else None
            if v is not None:
                acc[m] = 100 * v
            t = mb.value(p, m, p.epoch_s) if p.epoch_s else None
            if p.key in ("darcy", "darcy_physics"):
                tt, n = (
                    mb.value(p, m, (("total_time_s",),)),
                    mb.value(p, m, (("epochs",),)),
                )
                t = tt / n if tt is not None and n else None
            if p.key == "shapenetcar":
                tt = mb.value(p, m, (("runs", "@", "train_seconds"),))
                n = mb.value(p, m, (("runs", "@", "n_epochs"),))
                t = tt / n if tt is not None and n else None
            if p.key == "chip_cooling":
                d = mb.load(p, m)
                t = mb.mean_of(mb.dig(d, "time")) if d else None
            if t is not None:
                cost[m] = t
        for name, vals, ylabel, fmt in (
            ("summary_accuracy", acc, "Test error (%)", lambda v: f"{v:.3g} %"),
            (
                "summary_cost",
                cost,
                "Training time per epoch (s)",
                lambda v: f"{v:.3g} s",
            ),
        ):
            fig, ax = plt.subplots(figsize=CURVE_SIZE, layout="constrained")
            ms = list(vals)
            x = np.arange(len(ms))
            ax.bar(
                x,
                [vals[m] for m in ms],
                width=0.6,
                color=[MODEL_COLORS[LABELS[m]] for m in ms],
            )
            for xi, m in zip(x, ms):
                ax.annotate(
                    fmt(vals[m]),
                    (xi, vals[m]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                )
            ax.set_xticks(x, [LABELS[m] for m in ms])
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.15 * max(vals.values()))
            sub = (
                p.metric
                if name == "summary_accuracy"
                else "mean wall-clock time of one training epoch"
            )
            ax.set_title(f"{p.title}\n{sub}", pad=8)
            ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            _save(fig, out / f"{p.key}_comparison" / f"{name}.png")
        print(
            f"[figures] {p.key}: summary bars ({len(acc)} accuracy, {len(cost)} cost)",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="step", required=True)
    c = sub.add_parser(
        "curves",
        help="error_curve.png and compute_cost.png from the training histories",
    )
    c.add_argument(
        "--history-root",
        type=Path,
        required=True,
        help="checkpoint archive holding the histories",
    )
    c.add_argument("--only", choices=sorted(CURVES), help="one benchmark")
    e = sub.add_parser(
        "export", help="run a released checkpoint on the test set (GPU, data)"
    )
    e.add_argument("benchmark", choices=sorted(EXPORT))
    e.add_argument("model", choices=sorted(LABELS))
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--data-dir", type=Path, required=True)
    e.add_argument(
        "--case",
        action="append",
        help="AirfRANS: also keep this case (for the comparison)",
    )
    x = sub.add_parser(
        "extras",
        help="physics-only scaling and AirfRANS accuracy, from committed metrics",
    )
    sm = sub.add_parser(
        "summary", help="accuracy and time-per-epoch bars of every benchmark"
    )
    r = sub.add_parser(
        "contours", help="contour.png of every exported model and the comparison"
    )
    r.add_argument("benchmark", choices=sorted(EXPORT))
    for p in (c, e, r, x, sm):
        p.add_argument("--out", type=Path, default=ASSETS)
        p.add_argument(
            "--cache",
            type=Path,
            default=_REPO_ROOT / "scripts/benchmarks/checkpoints/figures",
        )
    args = ap.parse_args()
    if args.step == "curves":
        render_curves(args.history_root, args.out, args.only)
    elif args.step == "export":
        export(
            args.benchmark,
            args.model,
            args.checkpoint,
            args.data_dir,
            args.cache,
            args.case,
        )
    elif args.step == "summary":
        render_summary(args.out)
    elif args.step == "extras":
        render_scaling(ASSETS, args.out)
        render_airfrans_accuracy(ASSETS, args.out)
    else:
        render_contours(args.cache, args.out, args.benchmark)


if __name__ == "__main__":
    main()
