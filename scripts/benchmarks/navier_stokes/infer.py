# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Navier-Stokes benchmark, step 2 of 2: evaluate the saved checkpoints.

Loads every run in ``_config.RUNS`` that has a checkpoint, rolls it out
autoregressively over the 200 test samples and writes, for each run,
``doc/assets/benchmarks/navier_stokes_<run>/``:

* ``contour.png``: ground truth, prediction and absolute error at four
  rollout steps, for the test sample with the median error;
* ``rollout.png``: the relative L2 at each of the 10 steps;
* ``error_curve.png`` and ``compute_cost.png``: the training history;
* ``metrics.json``: the run, its test relative L2 and its timing.

When all four runs are present, ``navier_stokes_comparison/`` gets the four
models side by side. No training happens here.

Usage (from the repo root)::

    export NS_DATA_DIR=/path/to/dir   # NavierStokes_V1e-5_N1200_T20.mat
    python scripts/benchmarks/navier_stokes/infer.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
# Deterministic GPU kernels, so repeated evaluations give the same numbers.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
).strip()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.benchmarks._ckpt import resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import (  # noqa: E402
    plot_compute_cost,
    plot_error_curves,
    save_metrics_json,
)
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.navier_stokes._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
    sha256,
)
from scripts.benchmarks.navier_stokes._config import (  # noqa: E402
    DATA_FILE,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    MODEL_LABELS,
    PAPER_REL_L2,
    PROTOCOL,
    RUNS,
    TRANSOLVER_PARAMS,
)
from scripts.benchmarks.navier_stokes._data import (  # noqa: E402
    copy_last_frame,
    load_split,
    rollout,
)

jax.config.update("jax_enable_x64", False)
# Full float32 matmuls (no TF32), so the metric does not depend on the GPU's
# default precision.
jax.config.update("jax_default_matmul_precision", "highest")

#: Rollout steps shown in ``contour.png`` (0-based). Frame k of ``u`` is at
#: time t = k + 1 (the file's ``t``), so step s is at t = t_in + s + 1.
CONTOUR_STEPS = (0, 3, 6, 9)
#: Line colour of each model in the comparison plots.
COLORS = {
    "transolver": "#2a78d6",
    "gino": "#eda100",
    "fno": "#eb6834",
    "wno": "#1baf7a",
}


def predict(ck: dict, a: np.ndarray, t_out: int, batch_size: int) -> np.ndarray:
    """Autoregressive rollout of ``a`` ``(n, H * W, t_in)`` as float32
    ``(n, H * W, t_out)``."""
    spec, model, grid = ck["spec"], ck["model"], ck["grid"]
    run = jax.jit(
        lambda p, fx: rollout(lambda h: spec.apply(model, p, h, grid), fx, t_out)
    )
    out = [
        np.asarray(run(ck["params"], jnp.asarray(a[i : i + batch_size])), np.float32)
        for i in range(0, len(a), batch_size)
    ]
    return np.concatenate(out)


def per_sample_rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``||pred - truth|| / ||truth||`` per sample, over all predicted frames:
    the benchmark's metric."""
    n = truth.shape[0]
    diff = (pred - truth).reshape(n, -1)
    return np.linalg.norm(diff, axis=1) / np.linalg.norm(truth.reshape(n, -1), axis=1)


def per_step_rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mean over samples of the relative L2 of each predicted frame."""
    return (np.linalg.norm(pred - truth, axis=1) / np.linalg.norm(truth, axis=1)).mean(
        axis=0
    )


def timing(history: list[dict]) -> dict:
    secs = np.asarray([h["secs"] for h in history])
    return dict(
        first_epoch_s=float(secs[0]),
        steady_epoch_s=float(np.median(secs[1:])) if len(secs) > 1 else float(secs[0]),
        total_h=float(secs.sum() / 3600),
    )


def _save(fig, path: Path, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_contour(truth, pred, rel, grid, t_in, label, path: Path) -> int:
    """Ground truth, prediction and absolute error at :data:`CONTOUR_STEPS`,
    for the test sample with the median error. Returns that sample's index."""
    k = int(np.argsort(rel)[len(rel) // 2])
    t = truth[k].reshape(*grid, -1)
    p = pred[k].reshape(*grid, -1)
    vmax = float(np.abs(t).max())
    emax = float(np.abs(p - t)[..., list(CONTOUR_STEPS)].max())
    n = len(CONTOUR_STEPS)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n + 1.2, 9.6), layout="constrained")
    rows = [
        ("Ground truth", t, "RdBu_r", -vmax, vmax),
        (f"{label}", p, "RdBu_r", -vmax, vmax),
        ("Abs. error", np.abs(p - t), "magma", 0.0, emax),
    ]
    for r, (name, field, cmap, lo, hi) in enumerate(rows):
        for c, s in enumerate(CONTOUR_STEPS):
            ax = axes[r, c]
            im = ax.imshow(
                field[..., s].T,
                origin="lower",
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                interpolation="bicubic",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {t_in + s + 1}", fontsize=16, fontweight="bold")
            if c == 0:
                ax.set_ylabel(name, fontsize=16, fontweight="bold")
        cbar = fig.colorbar(im, ax=axes[r, :], fraction=0.02)
        cbar.set_label(
            "|error|" if r == 2 else "vorticity", fontsize=14, fontweight="bold"
        )
        cbar.ax.tick_params(labelsize=12)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
    fig.suptitle(
        f"Test sample {k} (median error, relative L2 {rel[k]:.4f})",
        fontsize=17,
        fontweight="bold",
    )
    _save(fig, path, dpi=90)
    return k


def plot_rollout(curves: dict, t_in: int, path: Path) -> None:
    """Relative L2 per rollout step for each model in ``curves``."""
    n = len(next(iter(curves.values())))
    steps = np.arange(t_in + 1, t_in + 1 + n)  # time of each frame
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, err in curves.items():
        ax.plot(
            steps,
            err,
            marker="o",
            lw=2,
            color=COLORS.get(name),
            label=MODEL_LABELS[name],
        )
    ax.set_xticks(steps)
    ax.set_xlabel("Predicted time t (each frame fed back as input)")
    ax.set_ylabel("Relative L2 (mean over test samples)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Error along the 10-step rollout")
    fig.tight_layout()
    _save(fig, path)


def write_comparison(out_dir: Path, models: list[str], curves: dict) -> None:
    """``navier_stokes_comparison/``: each model's figures side by side, and
    the four rollout curves on one plot."""
    layouts = {
        "contour.png": (2, 2),
        "error_curve.png": (2, 2),
        "compute_cost.png": (2, 2),
    }
    for name, (n_rows, n_cols) in layouts.items():
        images = [plt.imread(out_dir / f"navier_stokes_{m}" / name) for m in models]
        h, w = images[0].shape[:2]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * w / 150, n_rows * (h / 150 + 0.4)),
            squeeze=False,
        )
        for ax, image, model in zip(axes.ravel(), images, models, strict=True):
            ax.imshow(image)
            ax.set_title(MODEL_LABELS[model], fontsize=18, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        _save(fig, out_dir / "navier_stokes_comparison" / name, dpi=110)
    plot_rollout(
        curves,
        PROTOCOL["t_in"],
        out_dir / "navier_stokes_comparison" / "rollout.png",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    add_standard_args(ap, out_flag="--out-dir", out_default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)

    require_gpu(args.allow_cpu)
    data = load_split(args.data_dir / DATA_FILE, PROTOCOL)
    a_test, truth = data["a_test"], data["u_test"]
    t_in, t_out = PROTOCOL["t_in"], PROTOCOL["t_out"]
    baseline_pred = copy_last_frame(a_test, t_out)
    baseline = dict(
        test_rel_l2=float(per_sample_rel_l2(baseline_pred, truth).mean()),
        per_step_rel_l2=per_step_rel_l2(baseline_pred, truth).tolist(),
    )

    runs, curves = {}, {}
    for name, run in RUNS.items():
        if only and name != only:
            continue
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint under {run_dir}, skipped")
            continue
        ck = load_checkpoint(run_dir)
        if ck["run"].optimizer != run.optimizer:
            print(
                f"[{name}] warning: checkpoint trained with {ck['run'].optimizer!r},"
                f" but the run now uses {run.optimizer!r}; retrain it"
            )
        label = MODEL_LABELS[run.model]
        out = args.out_dir / f"navier_stokes_{name}"
        pred = predict(ck, a_test, t_out, args.batch_size)
        rel = per_sample_rel_l2(pred, truth)
        steps = per_step_rel_l2(pred, truth)
        curves[name] = steps
        sample = plot_contour(
            truth, pred, rel, ck["grid"], t_in, label, out / "contour.png"
        )
        plot_rollout(
            {name: steps},
            t_in,
            out / "rollout.png",
        )
        history = ck["history"]
        epochs = [h["epoch"] for h in history]
        plot_error_curves(
            epochs,
            [h["train_loss"] for h in history],
            [h["test_rel_l2"] for h in history],
            out / "error_curve.png",
            title=f"{label} on Navier-Stokes",
            train_label="rel. L2",
        )
        plot_compute_cost(
            epochs,
            [h["secs"] for h in history],
            out / "compute_cost.png",
            title=f"Per-epoch compute cost ({label} on Navier-Stokes)",
        )
        runs[name] = dict(
            model=run.model,
            arch=ck["config"]["run"]["arch"],
            cycle_momentum=ck["config"]["run"].get("cycle_momentum", False),
            optimizer=ck["run"].optimizer,
            n_params=ck["config"]["n_params"],
            vs_transolver_params=ck["config"]["n_params"] / TRANSOLVER_PARAMS - 1,
            test_rel_l2=dict(
                mean=float(rel.mean()),
                median=float(np.median(rel)),
                p90=float(np.percentile(rel, 90)),
                max=float(rel.max()),
            ),
            per_step_rel_l2=steps.tolist(),
            final_epoch_test_rel_l2_logged=history[-1]["test_rel_l2"],
            contour_sample=sample,
            timing=timing(history),
            checkpoint_sha256=sha256(resolve_weights_path(run_dir)),
            provenance=ck["config"]["provenance"],
        )
        print(
            f"[{name}] test rel. L2 mean {rel.mean():.5f}  median {np.median(rel):.5f}"
            f"  ({ck['config']['n_params']:,} params)"
        )

    if not runs:
        # Checkpoints are not in git, so on a fresh clone there is nothing to
        # evaluate; writing metrics.json would wipe the committed results.
        raise SystemExit(
            f"no checkpoints under {args.ckpt_root}; metrics.json not written"
        )
    common = dict(
        label="Zenteiq",
        dataset=dict(
            file=DATA_FILE,
            train="first 1000 samples",
            test="last 200 samples",
            grid="64x64",
            input=f"vorticity at t = 1-{t_in}",
            output=f"vorticity at t = {t_in + 1}-{t_in + t_out}, one step at a time",
        ),
        protocol=PROTOCOL,
        metric="relative L2 per test sample over the 10 frames of the "
        "autoregressive rollout (test_l2_full of exp_ns.py)",
        weights="last epoch (no validation split, so no epoch selection)",
        paper_reference=dict(
            model="Transolver",
            test_rel_l2=PAPER_REL_L2,
            source="Wu et al. 2024 (ICML), Transolver",
        ),
        copy_last_frame_baseline=baseline,
    )
    for name, run in runs.items():
        save_metrics_json(
            attach(
                {"problem": "navier_stokes", **common, **run},
                [checkpoint_dir(args, name)],
            ),
            args.out_dir / f"navier_stokes_{name}" / "metrics.json",
        )
    if set(runs) == set(RUNS):
        write_comparison(args.out_dir, list(RUNS), curves)
        print(f"[infer] wrote {args.out_dir / 'navier_stokes_comparison'}")
    print(
        f"[infer] wrote metrics.json and figures to {args.out_dir}/navier_stokes_<run>/"
    )


if __name__ == "__main__":
    main()
