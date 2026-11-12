# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Chip cooling benchmark, step 2 of 2: evaluate the saved checkpoints.

Loads every run in ``_config.RUNS`` that has a checkpoint, evaluates it on
the 400 test samples (in kelvin: over the in-domain grid cells for FNO and
WNO, over the mesh nodes for GINO and Transolver) and writes, for each run,
``contour.png``, ``error_curve.png``, ``compute_cost.png`` and
``metrics.json`` to ``doc/assets/benchmarks/chip_cooling_<run>/``. When all
four runs are there it also writes the comparison figures to
``doc/assets/benchmarks/chip_cooling_comparison/``. No training happens here.

Usage (from the repo root)::

    export CHIPCOOLING_DATA_DIR=/path/to/1_Dataset
    python scripts/benchmarks/chip_cooling/infer.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scirex.data.datasets.chipcooling import evaluate_model  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.chip_cooling._checkpoint import (  # noqa: E402
    check_normalisation,
    is_complete,
    load_config,
    load_history,
    load_params_into,
)
from scripts.benchmarks.chip_cooling._config import (  # noqa: E402
    CONTOUR_SAMPLE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    MODEL_LABELS,
    PROTOCOL,
    RUNS,
)
from scripts.benchmarks.chip_cooling._data import load_raw  # noqa: E402
from scripts.benchmarks.chip_cooling._models import get_spec  # noqa: E402
from scripts.benchmarks.chip_cooling._plots import (  # noqa: E402
    plot_loss_history,
    plot_one,
)
from scripts.benchmarks.chip_cooling.train import load_split  # noqa: E402

jax.config.update("jax_enable_x64", False)

# Line colour of each model in the comparison figures.
COLORS = {
    "fno": "#e8603c",
    "gino": "#7b3fa0",
    "wno": "#00808a",
    "transolver": "#7a7a10",
}


def load_run(run_dir: Path, raw: dict, protocol: dict, cache_dir: Path) -> dict:
    """Rebuilds a run's model, restores its weights and prepares its data."""
    run, config = load_config(run_dir)
    spec = get_spec(run.model)
    data = load_split(spec.kind, raw, protocol, cache_dir)
    check_normalisation(config["normalisation"], data["normalisation"], run.name)
    prepared = spec.prepare(run.arch, data)
    model = spec.build(run.arch, protocol["grid"])
    template = spec.init(model, run.arch, prepared, protocol["seed"])
    return dict(
        run=run,
        spec=spec,
        model=model,
        params=load_params_into(run_dir, template),
        data=data,
        prepared=prepared,
        history=load_history(run_dir),
        config=config,
        protocol=protocol,
    )


def field_values(ck: dict, field: np.ndarray) -> np.ndarray:
    """The values a model is scored on: in-domain grid cells or mesh nodes,
    ``(n, n_points)``."""
    if ck["spec"].kind == "grid":
        return field[:, 0][:, ck["data"]["mask"]]
    return field


def points_mm(ck: dict, raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """(z, y) in mm of the points :func:`field_values` returns."""
    if ck["spec"].kind == "grid":
        d = ck["data"]
        Zg, Yg = np.meshgrid(d["z_grid"] * 1000, d["y_grid"] * 1000)
        return Zg[d["mask"]], Yg[d["mask"]]
    return raw["coords"][:, 2] * 1000, raw["coords"][:, 1] * 1000


def test_metrics(ck: dict, pred: np.ndarray) -> dict:
    """R², RMSE, MAE (pooled and per-sample means), max AE and the mean
    relative L2, in kelvin; plus the per-sample arrays."""
    data = ck["data"]
    if ck["spec"].kind == "grid":
        metrics = evaluate_model(data["Y_test"], pred, data["mask"])
        true_v, pred_v = field_values(ck, data["Y_test"]), field_values(ck, pred)
    else:
        true_v, pred_v = data["fields_test"], pred
        r2_s = [r2_score(t, p) for t, p in zip(true_v, pred_v, strict=True)]
        rmse_s = [np.sqrt(mean_squared_error(t, p)) for t, p in zip(true_v, pred_v)]
        mae_s = [mean_absolute_error(t, p) for t, p in zip(true_v, pred_v)]
        tf, pf = true_v.reshape(-1), pred_v.reshape(-1)
        metrics = dict(
            R2=float(r2_score(tf, pf)),
            RMSE=float(np.sqrt(mean_squared_error(tf, pf))),
            MAE=float(mean_absolute_error(tf, pf)),
            mean_R2=float(np.mean(r2_s)),
            mean_RMSE=float(np.mean(rmse_s)),
            mean_MAE=float(np.mean(mae_s)),
            R2_per_sample=np.asarray(r2_s),
            RMSE_per_sample=np.asarray(rmse_s),
            MAE_per_sample=np.asarray(mae_s),
        )
    metrics["MaxAE"] = float(np.max(np.abs(pred_v - true_v)))
    metrics["mean_RelL2"] = float(
        np.mean(
            np.linalg.norm(pred_v - true_v, axis=1) / np.linalg.norm(true_v, axis=1)
        )
    )
    return metrics


def inference_ms(ck: dict, k: int) -> float:
    """Wall time of one test sample's prediction, after compilation."""
    args = (ck["model"], ck["params"], ck["prepared"], ck["data"], [k])
    ck["spec"].predict(*args)
    t0 = time.perf_counter()
    ck["spec"].predict(*args)
    return (time.perf_counter() - t0) * 1e3


def write_assets(
    ck: dict,
    raw: dict,
    pred: np.ndarray,
    out: Path,
    test: dict,
    run_dir: Path,
    sample: int = CONTOUR_SAMPLE,
) -> None:
    """contour.png (test sample ``sample``), error_curve.png, compute_cost.png
    and metrics.json (the training history, the test metrics and the provenance)."""
    label = MODEL_LABELS[ck["run"].model]
    out.mkdir(parents=True, exist_ok=True)
    k = sample
    g = len(raw["parameters"]) - ck["protocol"]["ntest"] + k
    z_mm, y_mm = points_mm(ck, raw)
    truth = ck["data"]["Y_test" if ck["spec"].kind == "grid" else "fields_test"]
    true_k = field_values(ck, truth)[k]
    pred_k = field_values(ck, pred)[k]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plot_one(
            z_mm,
            y_mm,
            pred_k,
            true_k,
            k,
            inference_ms(ck, k),
            raw["parameters"][g],
            raw["names"],
            td,
            title_prefix=label,
        )
        shutil.copy(td / f"comparison_sample_{k:04d}.png", out / "contour.png")
        plot_loss_history(ck["history"], td, title_prefix=f"{label} (ZenteiQ)")
        shutil.copy(td / "loss_curves.png", out / "error_curve.png")
        shutil.copy(td / "training_time.png", out / "compute_cost.png")
    scalars = {
        k: float(v) for k, v in test.items() if np.ndim(v) == 0
    }  # drop the per-sample arrays
    metrics = {**ck["history"], "test": scalars, "n_params": ck["config"]["n_params"]}
    attach(metrics, [run_dir])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def write_comparison(out_dir: Path) -> None:
    """The four models' error curves and per-epoch cost on shared axes, and
    their contour figures side by side."""
    C = out_dir / "chip_cooling_comparison"
    C.mkdir(parents=True, exist_ok=True)
    order = ["fno", "gino", "wno", "transolver"]
    H = {
        k: json.loads((out_dir / f"chip_cooling_{k}/metrics.json").read_text())
        for k in order
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in order:
        h = H[k]
        tr = h.get("train_err", h["loss"])
        te = h.get("test_l2", h["val_loss"])
        ep = np.arange(1, len(te) + 1)
        ax.semilogy(
            ep, te, color=COLORS[k], lw=1.8, label=f"{MODEL_LABELS[k]} validation"
        )
        ax.semilogy(
            ep, tr, color=COLORS[k], lw=1.4, ls=":", label=f"{MODEL_LABELS[k]} train"
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked relative L2 (log scale)")
    ax.set_title("Chip cooling: training and validation error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(C / "error_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in order:
        t = H[k]["time"]
        ax.semilogy(
            np.arange(1, len(t) + 1), t, color=COLORS[k], lw=1.4, label=MODEL_LABELS[k]
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per epoch (log scale)")
    ax.set_title(
        "Chip cooling: training cost per epoch (epoch 1 includes JIT compilation)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C / "compute_cost.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    for ax, k in zip(axs.ravel(), order, strict=True):
        ax.imshow(plt.imread(out_dir / f"chip_cooling_{k}/contour.png"))
        ax.set_axis_off()
        ax.set_title(MODEL_LABELS[k], fontsize=22, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C / "contour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)

    require_gpu(args.allow_cpu)
    raw = load_raw(args.data_dir)
    done = []
    for name in [only] if only else RUNS:
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint under {run_dir}, skipped")
            continue
        ck = load_run(run_dir, raw, PROTOCOL, args.cache_dir)
        pred = ck["spec"].predict(ck["model"], ck["params"], ck["prepared"], ck["data"])
        m = test_metrics(ck, pred)
        print(
            f"[{name}] R2 {m['R2']:.6f}  RMSE {m['RMSE']:.3f} K  MAE {m['MAE']:.3f} K"
            f"  MaxAE {m['MaxAE']:.3f} K  mean RelL2 {m['mean_RelL2']:.2e}"
            f"  ({ck['config']['n_params']:,} params)"
        )
        write_assets(ck, raw, pred, args.out_dir / f"chip_cooling_{name}", m, run_dir)
        done.append(name)

    if not done:
        # Checkpoints are not in git, so on a fresh clone there is nothing to
        # evaluate; writing anything would wipe the committed results.
        raise SystemExit(f"no checkpoints under {args.ckpt_root}; nothing written")
    if len(done) == len(RUNS):
        write_comparison(args.out_dir)
    print(f"[infer] wrote {args.out_dir}/chip_cooling_<run>/")


if __name__ == "__main__":
    main()
