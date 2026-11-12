# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""AirfRANS benchmark, step 2 of 2: evaluate the saved checkpoints.

Loads every run in ``_config.RUNS`` that has a checkpoint, scores it on the
200 official test cases with the official Transolver metrics (``_metrics.py``)
and writes, for each run, ``contour.png``, ``error_curve.png``,
``compute_cost.png`` and ``metrics.json`` to
``doc/assets/benchmarks/airfrans_<run>/``. A GINO checkpoint
(``train.py --run gino``) is scored the same way by GINO's own evaluation in
``_gino.py``. No training happens here.

Usage (from the repo root)::

    export AIRFRANS_DATA_DIR=/path/to/AirfRANS/Dataset
    python scripts/benchmarks/airfrans/infer.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from scripts.benchmarks._ckpt import resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.airfrans import _gino  # noqa: E402
from scripts.benchmarks.airfrans._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
    sha256,
)
from scripts.benchmarks.airfrans._config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    GINO_ARGS,
    GINO_CONTOUR_CASE,
    MODEL_LABELS,
    PAPER_REFERENCE,
    PROTOCOL,
    RUNS,
)
from scripts.benchmarks.airfrans._data import (  # noqa: E402
    load_case,
    load_split,
    prepare_case,
    to_physical,
)
from scripts.benchmarks.airfrans._metrics import (  # noqa: E402
    CaseGeometry,
    cover_predict,
    field_scores,
    impose_wall,
    summarise,
)

jax.config.update("jax_enable_x64", False)

# Line colour of each model's compute-cost plot.
COST_COLORS = {"transolver": "#9467bd", "wno": "#1f77b4"}
WINDOW = (-0.5, 1.5, -0.45, 0.45)  # contour view around the airfoil (chords)


def evaluate(
    ck: dict, split, data_dir: Path, cache_dir: Path
) -> tuple[list[dict], dict]:
    """Official scores of every test case. Returns the per-case rows and the
    physical prediction of every case (for the contour plot)."""
    spec, model, params, stats = ck["spec"], ck["model"], ck["params"], ck["stats"]
    out_mean, out_std = ck["official_out"]
    apply = jax.jit(lambda p, x, pos, extra: spec.apply(model, p, x, pos, extra))
    rng = np.random.default_rng(0)
    rows, preds = [], {}
    for j, cid in enumerate(split.test):
        raw = load_case(data_dir, cache_dir, cid)
        d = prepare_case(raw, stats)
        extra = spec.extra(ck["run"].arch, raw)
        pred_n, passes = cover_predict(
            lambda idx, d=d, extra=extra: apply(
                params, jnp.asarray(d["x"][idx]), jnp.asarray(d["pos"][idx]), extra
            ),
            len(d["x"]),
            PROTOCOL["subsample"],
            rng,
        )
        pred = impose_wall(to_physical(pred_n, d["scale"], stats), raw["surf"])
        scores = field_scores(pred, raw["y"], raw["surf"], out_mean, out_std)
        geo = CaseGeometry(data_dir, cid)
        cd_t, cl_t = geo.coefficients()
        cd_p, cl_p = geo.coefficients(pred)
        rows.append(
            dict(
                case=cid,
                passes=passes,
                **scores,
                cd_true=float(cd_t),
                cl_true=float(cl_t),
                cd_pred=float(cd_p),
                cl_pred=float(cl_p),
                relerr_cd=float(abs((cd_t - cd_p) / cd_t)),
                relerr_cl=float(abs((cl_t - cl_p) / cl_t)),
            )
        )
        preds[cid] = pred.astype(np.float32)
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(split.test)} cases", flush=True)
    return rows, preds


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_contour(
    rows, preds, data_dir: Path, cache_dir: Path, label: str, path: Path
) -> str:
    """Truth, prediction and absolute error of |U|/U_inf and Cp, and the surface
    pressure, for the test case with the median Volume error. Returns its id."""
    vol = np.array([r["volume"] for r in rows])
    r = rows[int(np.argsort(vol)[len(vol) // 2])]
    cid = r["case"]
    raw = load_case(data_dir, cache_dir, cid)
    geo = CaseGeometry(data_dir, cid)
    from vtk.util.numpy_support import vtk_to_numpy

    quads = vtk_to_numpy(geo.internal.GetCells().GetConnectivityArray()).reshape(-1, 4)
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    P = raw["pos"]
    c = P[tri].mean(1)
    keep = (
        (c[:, 0] > WINDOW[0] - 0.1)
        & (c[:, 0] < WINDOW[1] + 0.1)
        & (np.abs(c[:, 1]) < WINDOW[3] + 0.1)
    )
    T = Triangulation(P[:, 0], P[:, 1], tri[keep])
    u_inf, aoa = geo.u_inf, geo.alpha
    q = 0.5 * u_inf**2
    y, p = raw["y"], preds[cid]
    fields = [
        (
            "|U| / U$_\\infty$",
            np.linalg.norm(y[:, :2], axis=1) / u_inf,
            np.linalg.norm(p[:, :2], axis=1) / u_inf,
            "viridis",
            None,
        ),
        (
            "Cp",
            y[:, 2] / q,
            p[:, 2] / q,
            "RdBu_r",
            TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0),
        ),
    ]
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.32, wspace=0.12)
    loop = geo.outline()
    for i, (name, t, pr, cmap, norm) in enumerate(fields):
        kw = (
            dict(norm=norm)
            if norm is not None
            else dict(vmin=min(t.min(), pr.min()), vmax=max(t.max(), pr.max()))
        )
        for j, (title, v, cm, extra_kw) in enumerate(
            [
                ("Ground truth", t, cmap, kw),
                (f"{label} prediction", pr, cmap, kw),
                ("Absolute error", np.abs(pr - t), "magma", {}),
            ]
        ):
            ax = fig.add_subplot(gs[i, j])
            im = ax.tripcolor(
                T, v, shading="gouraud", cmap=cm, rasterized=True, **extra_kw
            )
            ax.fill(loop[:, 0], loop[:, 1], color="0.25", zorder=3)
            ax.set_xlim(WINDOW[0], WINDOW[1])
            ax.set_ylim(WINDOW[2], WINDOW[3])
            ax.set_aspect("equal")
            ax.set_title(f"{title}: {name}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax = fig.add_subplot(gs[2, :])
    order = geo.wall_order
    xs = vtk_to_numpy(geo.airfoil.GetPoints().GetData())[:, 0]
    ax.plot(xs, -y[order, 2] / q, ".", ms=3, color="0.15", label="Ground truth")
    ax.plot(
        xs, -p[order, 2] / q, ".", ms=2, color="#9467bd", label=f"{label} prediction"
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("-Cp on the airfoil surface")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(
        f"Test case {cid} (median Volume error {r['volume']:.2e}): "
        f"U$_\\infty$ = {u_inf:.1f} m/s, angle of attack {aoa:+.2f}°, "
        f"C$_L$ true {r['cl_true']:.4f} / predicted {r['cl_pred']:.4f}",
        fontsize=12,
    )
    _save(fig, path)
    return cid


def plot_error_curve(history: list[dict], label: str, path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    val = [h for h in history if "val_vol" in h]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs, [h["train_vol"] for h in history], lw=1.2, label="train, volume MSE"
    )
    ax.plot(
        epochs, [h["train_surf"] for h in history], lw=1.2, label="train, surface MSE"
    )
    ax.plot(
        [h["epoch"] for h in val],
        [h["val_vol"] for h in val],
        "o-",
        ms=3,
        lw=1.5,
        label="validation (80 cases), volume MSE",
    )
    ax.plot(
        [h["epoch"] for h in val],
        [h["val_surf"] for h in val],
        "s-",
        ms=3,
        lw=1.5,
        label="validation (80 cases), surface MSE",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised scaled targets, log scale)")
    ax.set_title(f"{label} on AirfRANS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_compute_cost(history: list[dict], label: str, color: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        [h["epoch"] for h in history], [h["secs"] for h in history], color=color, lw=1.5
    )
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(
        LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 1))
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Wall time per epoch (s, log scale)", fontsize=12)
    ax.set_title(f"Per-epoch compute cost ({label} on AirfRANS)", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=11)
    _save(fig, path)


def timing(history: list[dict]) -> dict:
    secs = np.asarray([h["secs"] for h in history])
    return dict(
        first_epoch_s=float(secs[0]),
        steady_epoch_s=float(np.median(secs[1:])) if len(secs) > 1 else float(secs[0]),
        total_h=float(sum(h["secs"] + h.get("val_secs", 0.0) for h in history) / 3600),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    add_standard_args(ap, out_flag="--out-dir", out_default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a an error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)
    require_gpu(args.allow_cpu)
    jax.config.update(
        "jax_default_matmul_precision", PROTOCOL["additions"]["matmul_precision"]
    )
    split = load_split(args.data_dir, PROTOCOL["n_val"])
    done = 0
    for name, run in RUNS.items():
        if only and name != only:
            continue
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint under {run_dir}, skipped")
            continue
        t0 = perf_counter()
        ck = load_checkpoint(run_dir)
        label = MODEL_LABELS[run.model]
        rows, preds = evaluate(ck, split, args.data_dir, args.cache_dir)
        res = summarise(rows)
        out = args.out_dir / f"airfrans_{name}"
        case = plot_contour(
            rows, preds, args.data_dir, args.cache_dir, label, out / "contour.png"
        )
        plot_error_curve(ck["history"], label, out / "error_curve.png")
        plot_compute_cost(
            ck["history"],
            label,
            COST_COLORS.get(run.model, "#1f77b4"),
            out / "compute_cost.png",
        )
        metrics = dict(
            problem="airfrans",
            model=run.model,
            label="Zenteiq",
            dataset=dict(
                source="AirfRANS (Bonnet et al., NeurIPS 2022), manifest 'full' task",
                train="manifest full_train minus its last 80 cases (720)",
                validation="last 80 cases of full_train (logged only)",
                test="manifest full_test (200)",
                nodes="every node of the internal mesh, no crop",
                input="7 official node features + log10 wall distance + log10 Re",
                output="Ux, Uy, p, nut",
            ),
            protocol=PROTOCOL,
            metric=(
                "official Transolver Results_test: Volume / Surf = normalised MSE "
                "on non-wall / wall nodes, mean over 4 channels and 200 cases, "
                "wall Ux, Uy, nut set to 0; C_L, C_D from integrated "
                "pressure and wall shear stress; Spearman rho over the 200 cases"
            ),
            weights=(
                "EMA of the weights after the last epoch "
                "(nothing selected on held-out data)"
            ),
            arch=ck["config"]["run"]["arch"],
            n_params=ck["config"]["n_params"],
            vs_paper_params=ck["config"]["n_params"] / PAPER_REFERENCE["n_params_used"]
            - 1,
            results=res,
            paper_reference=PAPER_REFERENCE,
            vs_paper=dict(
                volume_ratio=res["volume"] / PAPER_REFERENCE["volume"],
                surf_ratio=res["surf"] / PAPER_REFERENCE["surf"],
                relerr_cl_ratio=res["relerr_cl"] / PAPER_REFERENCE["relerr_cl"],
            ),
            contour_case=case,
            timing=timing(ck["history"]),
            checkpoint_sha256=sha256(resolve_weights_path(run_dir)),
            provenance=ck["config"]["provenance"],
        )
        attach(metrics, [run_dir])
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        done += 1
        print(
            f"[{name}] Volume {res['volume']:.5f}  Surf {res['surf']:.5f}  "
            f"C_L error {res['relerr_cl']:.4f}  "
            f"rho_L {res['spearman_cl']:.5f}  ({ck['config']['n_params']:,} params, "
            f"{(perf_counter() - t0) / 60:.1f} min) -> {out}"
        )
    gino_dir = args.ckpt_root / "gino"
    if only:  # one named checkpoint: the GINO evaluation is a separate run
        pass
    elif (gino_dir / "params_final.msgpack").exists():
        _gino.main(
            [
                *GINO_ARGS,
                "--data-dir",
                str(args.data_dir),
                "--cache-dir",
                str(args.cache_dir / "gino"),
                "--output-dir",
                str(args.out_dir / "airfrans_gino"),
                "--eval-only",
                str(gino_dir / "params_final.msgpack"),
                "--history",
                str(gino_dir / "loss_curve.csv"),
                "--contour-case",
                GINO_CONTOUR_CASE,
            ]
        )
        done += 1
    else:
        print(f"[gino] no checkpoint under {gino_dir}, skipped")
    if not done:
        # Checkpoints are not in git, so on a fresh clone there is nothing to
        # evaluate; writing metrics.json would wipe the committed results.
        raise SystemExit(
            f"no checkpoints under {args.ckpt_root}; metrics.json not written"
        )


if __name__ == "__main__":
    main()
