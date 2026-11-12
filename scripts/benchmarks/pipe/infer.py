# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Pipe flow benchmark, step 2 of 2: evaluate the saved checkpoints.

Loads every run and seed in ``_config.RUNS`` that has a checkpoint,
evaluates it on the 200 test samples and writes, for each run,
``metrics.json`` (every seed, and the mean and standard deviation over seeds)
and ``contour.png``, ``error_curve.png`` and ``compute_cost.png`` (seed 0) to
``doc/assets/benchmarks/pipe_<run>/``. When all four runs are present,
``pipe_comparison/`` gets the four models side by side. No training happens
here.

Usage (from the repo root)::

    export PIPE_DATA_DIR=/path/to/pipe
    python scripts/benchmarks/pipe/infer.py
    python scripts/benchmarks/pipe/infer.py --run transolver fno
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402

from scripts.benchmarks._ckpt import resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import add_standard_args  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import plot_error_curves  # noqa: E402
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.pipe._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
    sha256,
)
from scripts.benchmarks.pipe._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    MODEL_LABELS,
    PROTOCOL,
    RUNS,
    SHARED_RECIPE_RUNS,
    TRANSOLVER_PARAMS,
    protocol_for,
)
from scripts.benchmarks.pipe._data import load_split  # noqa: E402

jax.config.update("jax_enable_x64", False)

# Line colour of each model in the compute-cost plots.
COST_COLORS = {
    "transolver": "#808000",
    "wno": "#1f77b4",
    "fno": "#2ca02c",
    "gino": "#d62728",
}
#: The seed whose figures are drawn: fixed, not chosen by its test error.
FIGURE_SEED = 0


def predict(ck: dict, x_flat: np.ndarray, batch_size: int) -> np.ndarray:
    """Physical-unit predictions as float32 ``(n, H * W, 1)``.

    A model may compute in a lower precision (WNO's default compute dtype is
    bfloat16). Its output is cast to float32 before it is decoded: the
    normaliser decodes in its input's dtype, so decoding a bfloat16 output
    would round the physical-unit prediction a second time.
    """
    spec, grid, model, params = ck["spec"], ck["grid"], ck["model"], ck["params"]
    decode = ck["y_normalizer"].decode
    apply = jax.jit(
        lambda p, inputs: decode(spec.apply(model, p, inputs).astype(jnp.float32))
    )
    x = ck["x_normalizer"].encode(x_flat).astype(np.float32)
    prepared = spec.prepare(ck["run"].arch, x, grid)
    bs = spec.batch_size or batch_size
    out = [
        np.asarray(
            apply(params, spec.take(prepared, np.arange(i, min(i + bs, len(x))))),
            dtype=np.float32,
        )
        for i in range(0, len(x), bs)
    ]
    return np.concatenate(out)


def per_sample_rel_l2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``||pred - truth|| / ||truth||`` per sample, the benchmark's metric."""
    n = truth.shape[0]
    diff = (pred - truth).reshape(n, -1)
    return np.linalg.norm(diff, axis=1) / np.linalg.norm(truth.reshape(n, -1), axis=1)


def timing(history: list[dict]) -> dict:
    secs = np.asarray([h["secs"] for h in history])
    return dict(
        first_epoch_s=float(secs[0]),
        steady_epoch_s=float(np.median(secs[1:])) if len(secs) > 1 else float(secs[0]),
        total_min=float(secs.sum() / 60),
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_contour(xy, truth, pred, rel, grid, label, path: Path) -> int:
    """Truth, prediction and absolute error on the physical mesh, for the
    test sample with the median error. Returns that sample's index."""
    k = int(np.argsort(rel)[len(rel) // 2])
    X = xy[k, :, 0].reshape(grid)
    Y = xy[k, :, 1].reshape(grid)
    t = truth[k, :, 0].reshape(grid)
    p = pred[k, :, 0].reshape(grid)
    vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    panels = [
        ("Ground truth", t, "viridis", dict(vmin=vmin, vmax=vmax)),
        (f"{label} prediction", p, "viridis", dict(vmin=vmin, vmax=vmax)),
        ("Absolute error", np.abs(p - t), "magma", {}),
    ]
    for ax, (name, field, cmap, kw) in zip(axes, panels, strict=True):
        im = ax.pcolormesh(X, Y, field, shading="gouraud", cmap=cmap, **kw)
        ax.set_title(name)
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("x")
    fig.suptitle(
        f"Test sample {k} (median error, relative L2 {rel[k]:.2e})", fontsize=12
    )
    fig.tight_layout()
    _save(fig, path)
    return k


def plot_compute_cost(history: list[dict], label: str, color: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        [h["epoch"] for h in history],
        [h["secs"] for h in history],
        color=color,
        lw=1.5,
    )
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(
        LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 1))
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Wall time per epoch (s, log scale)", fontsize=12)
    ax.set_title(f"Per-epoch compute cost ({label} on Pipe)", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, path)


def _same_canvas(images: list[np.ndarray]) -> list[np.ndarray]:
    """Centres every image on a white canvas of the largest one's size, so
    that the side-by-side panels share one scale."""
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    out = []
    for image in images:
        canvas = np.ones((h, w, image.shape[2]), image.dtype)
        top, left = (h - image.shape[0]) // 2, (w - image.shape[1]) // 2
        canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
        out.append(canvas)
    return out


def write_comparison(out_dir: Path, names: list[str], histories: dict) -> None:
    """``pipe_comparison/``: every run's seed-0 compute cost on one plot, and
    its error curves and contour plots side by side."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        history = histories[name]
        ax.plot(
            [h["epoch"] for h in history],
            [h["secs"] for h in history],
            color=COST_COLORS.get(name),
            lw=1.5,
            label=MODEL_LABELS[RUNS[name].model],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per epoch (log scale)")
    ax.set_title("Pipe: training cost per epoch, seed 0 (epoch 0 includes JIT)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "pipe_comparison" / "compute_cost.png")
    for figure in ("error_curve.png", "contour.png"):
        images = _same_canvas(
            [plt.imread(out_dir / f"pipe_{n}" / figure) for n in names]
        )
        h, w = images[0].shape[:2]
        fig, axes = plt.subplots(2, 2, figsize=(2 * w / 150, 2 * (h / 150 + 0.4)))
        for ax, image, name in zip(axes.ravel(), images, names, strict=True):
            ax.imshow(image)
            ax.set_title(MODEL_LABELS[RUNS[name].model], fontsize=18, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        _save(fig, out_dir / "pipe_comparison" / figure)


def save_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    add_standard_args(
        ap, out_flag="--out-dir", out_default=DEFAULT_OUT_DIR, run_flag=False
    )
    ap.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size"])
    ap.add_argument(
        "--run",
        nargs="+",
        default=list(RUNS),
        metavar="NAME",
        help="run(s) to evaluate (default: every run with a checkpoint); with "
        "--checkpoint, the run that directory holds",
    )
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only_dir, fig_seed = None, FIGURE_SEED
    if (
        args.checkpoint is not None
    ):  # one seed's directory: <run>/seed<N>, or the run directory
        only_dir = args.checkpoint.expanduser()
        if not only_dir.is_dir():
            ap.error(f"--checkpoint {only_dir} is not a directory")
        seed_dir = only_dir.name.startswith("seed") and only_dir.name[4:].isdigit()
        guess = only_dir.parent.name if seed_dir else only_dir.name
        args.run = [guess] if args.run == list(RUNS) else args.run
        only_seed = int(only_dir.name[4:]) if seed_dir else PROTOCOL["seeds"][0]
        fig_seed = only_seed
        if len(args.run) != 1:
            ap.error("--checkpoint evaluates one run; give a single --run")
    unknown = [n for n in args.run if n not in RUNS]
    if unknown:
        ap.error(f"unknown run(s) {unknown}; choose from {list(RUNS)}")

    require_gpu(args.allow_cpu)
    data = load_split(args.data_dir, PROTOCOL)
    xy_test = data["xy_test"]
    truth = data["y_test"]

    runs, histories = {}, {}
    for name in args.run:
        run = RUNS[name]
        label = MODEL_LABELS[run.model]
        out = args.out_dir / f"pipe_{name}"
        seeds = {}
        for seed in [only_seed] if only_dir else PROTOCOL["seeds"]:
            run_dir = only_dir or args.ckpt_root / name / f"seed{seed}"
            if not is_complete(run_dir):
                print(f"[{name} seed {seed}] no checkpoint under {run_dir}, skipped")
                continue
            ck = load_checkpoint(run_dir)
            pred = predict(ck, xy_test, args.batch_size)
            rel = per_sample_rel_l2(pred, truth)
            history = ck["history"]
            if seed == fig_seed:
                sample = plot_contour(
                    xy_test, truth, pred, rel, ck["grid"], label, out / "contour.png"
                )
                histories[name] = history
                plot_compute_cost(
                    history,
                    label,
                    COST_COLORS.get(run.model, "#1f77b4"),
                    out / "compute_cost.png",
                )
                plot_error_curves(
                    [h["epoch"] for h in history],
                    [h["train_loss"] for h in history],
                    [h["test_rel_l2"] for h in history],
                    out / "error_curve.png",
                    title=f"{label} on Pipe (seed {seed})",
                    train_label="rel. L2",
                )
            n_params = ck["config"]["n_params"]
            arch = ck["config"]["run"]["arch"]
            seeds[str(seed)] = dict(
                test_rel_l2=dict(
                    mean=float(rel.mean()),
                    median=float(np.median(rel)),
                    p90=float(np.percentile(rel, 90)),
                    max=float(rel.max()),
                ),
                final_epoch_test_rel_l2_logged=history[-1]["test_rel_l2"],
                timing=timing(history),
                checkpoint_sha256=sha256(resolve_weights_path(run_dir)),
                provenance=ck["config"]["provenance"],
            )
            print(
                f"[{name} seed {seed}] test rel. L2 mean {rel.mean():.5f}  "
                f"median {np.median(rel):.5f}  ({n_params:,} params)"
            )
        if not seeds:
            continue
        if str(fig_seed) not in seeds:
            # Seed 0 draws the figures; without it the run is left out
            # rather than shown with another seed's figures.
            print(f"[{name}] seed {fig_seed} is missing, run skipped")
            continue
        means = np.asarray([s["test_rel_l2"]["mean"] for s in seeds.values()])
        runs[name] = dict(
            model=run.model,
            arch=arch,
            n_params=n_params,
            vs_transolver_params=n_params / TRANSOLVER_PARAMS - 1,
            test_rel_l2_over_seeds=dict(
                n_seeds=len(means),
                mean=float(means.mean()),
                # Sample standard deviation across seeds.
                std=float(means.std(ddof=1)) if len(means) > 1 else None,
                min=float(means.min()),
                max=float(means.max()),
            ),
            figure_seed=fig_seed,
            contour_sample=sample,
            seeds=seeds,
        )
        print(
            f"[{name}] {len(means)} seed(s): test rel. L2 {means.mean():.5f}"
            + (f" +/- {means.std(ddof=1):.5f}" if len(means) > 1 else "")
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
            train="first 1000 samples",
            test="last 200 samples",
            grid="129x129",
            input="(x, y) mesh coordinates",
            output="channel 0 of Pipe_Q.npy",
        ),
        protocol=PROTOCOL,
        metric="relative L2 per test sample, in physical units",
        weights="last epoch (no validation split, so no epoch selection)",
    )
    for name, run in runs.items():
        save_metrics_json(
            attach(
                {
                    "problem": "pipe",
                    "model": run["model"],
                    **common,
                    "protocol": protocol_for(RUNS[name]),
                    **run,
                },
                [only_dir]
                if only_dir
                else [
                    args.ckpt_root / name / f"seed{s}"
                    for s in PROTOCOL["seeds"]
                    if is_complete(args.ckpt_root / name / f"seed{s}")
                ],
            ),
            args.out_dir / f"pipe_{name}" / "metrics.json",
        )
    if set(SHARED_RECIPE_RUNS) <= set(runs):
        write_comparison(args.out_dir, list(SHARED_RECIPE_RUNS), histories)
    print(f"[infer] wrote metrics.json and figures to {args.out_dir}/pipe_<run>/")


if __name__ == "__main__":
    main()
