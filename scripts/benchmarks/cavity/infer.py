# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Lid-driven cavity benchmark, step 2 of 2: evaluate saved checkpoints.

Loads every checkpoint written by ``train.py``, scores it on the held-out
Re = 10,000 case with the metric of Costa Rocha et al. 2023 (RMSE in m/s
pooled over p, Ux and Uy), and writes the figures and ``metrics.json`` used
by the ``doc/benchmarks/cavity/results/comparative.md``. No training happens here.

For each run it reports:

* the single-step error (``n_in`` frames in, ``n_out`` frames out), the
  paper's Tables 1, 2 and 4;
* the error 10 steps (1 s) ahead, reached by feeding each prediction back in
  as input, the paper's Tables 5 and 6;
* the same two numbers for simply copying the last input frame, as a
  baseline every model has to beat.

Usage (from the repo root)::

    python scripts/benchmarks/cavity/infer.py

Outputs go under ``doc/assets/benchmarks/``, one folder per model:

* ``cavity_<model>/`` holds that model's ``metrics.json``, ``contour.png``,
  ``rollout.png``, ``error_curve.png`` and ``compute_cost.png``;
* ``cavity_comparison/`` holds the figures that compare the four models, for
  ``doc/benchmarks/cavity/results/comparative.md``: ``contour.png``,
  ``accuracy.png``, ``rollout.png``, ``error_curve.png`` and
  ``compute_cost.png``.
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import save_metrics_json  # noqa: E402
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.cavity import _reference as ref  # noqa: E402
from scripts.benchmarks.cavity._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
    sha256,
)
from scripts.benchmarks.cavity._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    MODEL_LABELS,
    MODELS,
    PROTOCOL,
    RUNS,
)
from scripts.benchmarks.cavity._data import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DOMAIN_SIZE,
    DT,
    TEST_CASE,
    TRAIN_CASES,
    denormalise,
    ensure_prepared,
    load_prepared,
    make_windows,
    normalise,
    reynolds,
)
from scripts.benchmarks.cavity._physics import rmse_metrics  # noqa: E402

DEFAULT_OUT_ROOT = _REPO_ROOT / "doc/assets/benchmarks"
COMPARISON_DIR = "cavity_comparison"  # the four-model comparison figures
HORIZON = 10  # 1 s at the 0.1 s write interval: one lid pass at 1 m/s
MODEL_COLORS = {
    "fno": "#eb6834",
    "gino": "#7b3fa0",
    "wno": "#00838f",
    "transolver": "#827717",
}
BASELINE_COLOR = "#555555"


# ─────────────────────────────── evaluation ────────────────────────────────
def single_step(ck: dict, grid: np.ndarray, batch_size: int) -> dict:
    """Error of one forward pass over every test window."""
    run = ck["run"]
    X, Y = make_windows(grid, run.n_in, run.n_out)
    apply = jax.jit(ck["model"].apply)
    normalizer = ck["normalizer"]
    pred = predict_in_batches(apply, ck["params"], normalise(X, normalizer), batch_size)
    return rmse_metrics(denormalise(pred, normalizer), Y)


def rollout(ck: dict, grid: np.ndarray, horizon: int, batch_size: int) -> dict:
    """Error ``horizon`` steps ahead, feeding predictions back in as input.

    A model with ``n_out > 1`` produces ``n_out`` frames per pass; the
    window then slides forward by those frames, as the paper "un-rolls" it.
    Every start time with ``horizon`` frames of ground truth after its input
    window is used.
    """
    run = ck["run"]
    normalizer = ck["normalizer"]
    starts = np.arange(grid.shape[0] - run.n_in - horizon + 1)
    apply = jax.jit(ck["model"].apply)

    def unroll(params, window):
        produced = []
        while len(produced) < horizon:
            pred = apply(params, window)  # (B, n_out, H, W, C)
            produced.extend(pred[:, k] for k in range(run.n_out))
            window = jnp.concatenate([window, pred], axis=1)[:, -run.n_in :]
        return jnp.stack(produced[:horizon], axis=1)

    windows = np.stack([grid[s : s + run.n_in] for s in starts])
    pred = predict_in_batches(
        unroll, ck["params"], normalise(windows, normalizer), batch_size
    )
    truth = np.stack([grid[s + run.n_in : s + run.n_in + horizon] for s in starts])
    return _rollout_errors(denormalise(pred, normalizer), truth)


def persistence(grid: np.ndarray, n_in: int, horizon: int) -> dict:
    """Copying the last input frame forward, the no-model baseline."""
    X, Y = make_windows(grid, n_in, 1)
    n_seq = grid.shape[0] - n_in - horizon + 1
    last = np.stack([grid[s + n_in - 1] for s in range(n_seq)])
    pred = np.repeat(last[:, None], horizon, axis=1)
    truth = np.stack([grid[s + n_in : s + n_in + horizon] for s in range(n_seq)])
    step = rmse_metrics(X[:, -1:], Y)
    return dict(
        n_in=n_in,
        single_step=dict(rmse=step["rmse"], rmse_vel=step["rmse_vel"]),
        rollout=_rollout_errors(pred, truth),
    )


def _rollout_errors(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth  # (n_seq, horizon, H, W, C)
    return dict(
        horizon=int(err.shape[1]),
        n_sequences=int(err.shape[0]),
        rmse=float(np.sqrt(np.mean(err**2))),
        rmse_vel=float(np.sqrt(np.mean(err[..., 1:3] ** 2))),
        per_step_rmse=[
            float(np.sqrt(np.mean(err[:, k] ** 2))) for k in range(err.shape[1])
        ],
    )


def timing(history: list[dict]) -> dict:
    secs = [h["secs"] for h in history]
    return dict(
        first_epoch_s=float(secs[0]) if secs else None,
        mean_epoch_s=float(np.mean(secs[1:])) if len(secs) > 1 else None,
        total_min=float(np.sum(secs) / 60.0),
    )


def paper_numbers(run) -> dict:
    """The paper's numbers for the same setting, where it reports one.

    Empty for a model the paper never ran; see ``_reference.NOT_IN_PAPER``.
    """
    out = {}
    if run.n_in == 3 and run.n_out == 1:
        if run.model in ref.TABLE1_50_EPOCHS:
            out["table1_50_epochs"] = ref.TABLE1_50_EPOCHS[run.model]
        if run.model in ref.SINGLE_STEP_TARGET:
            out["table2_300_epochs"] = ref.SINGLE_STEP_TARGET[run.model]
    if run.model == "fno" and run.n_out == 1 and run.n_in in ref.TABLE4_FNO_N_TO_1:
        out["table4_single_step"] = ref.TABLE4_FNO_N_TO_1[run.n_in]
    if run.model == "fno" and run.n_out == 1 and run.n_in in ref.TABLE5_FNO_N_TO_10:
        out["table5_10_steps"] = ref.TABLE5_FNO_N_TO_10[run.n_in]
    if run.model == "fno" and run.n_in == 3 and run.n_out in ref.TABLE6_FNO_3_TO_N:
        out["table6_10_steps"] = ref.TABLE6_FNO_3_TO_N[run.n_out]
    return out


# ──────────────────────────────── figures ──────────────────────────────────
def _speed(frame: np.ndarray) -> np.ndarray:
    return np.hypot(frame[..., 1], frame[..., 2])


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_contour(
    grid, cks, step: int, out: Path, labels: list[str] | None = None
) -> None:
    """Velocity magnitude at one test frame: truth, the copy-last-frame
    baseline and each model (top), and each one's velocity error (bottom)."""
    truth = grid[step]
    panels = [("Copy last frame", grid[step - 1])]
    for k, ck in enumerate(cks):
        run = ck["run"]
        x = normalise(grid[step - run.n_in : step][None], ck["normalizer"])
        pred = ck["model"].apply(ck["params"], jnp.asarray(x))
        pred = denormalise(np.asarray(pred), ck["normalizer"])[0, 0]
        panels.append((labels[k] if labels else MODEL_LABELS[run.model], pred))

    errs = [
        np.hypot(p[..., 1] - truth[..., 1], p[..., 2] - truth[..., 2])
        for _, p in panels
    ]
    vmax, emax = float(_speed(truth).max()), float(max(e.max() for e in errs))
    extent = (0, DOMAIN_SIZE, 0, DOMAIN_SIZE)
    n = len(panels) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.6))
    im = axes[0, 0].imshow(
        _speed(truth), origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="viridis"
    )
    axes[0, 0].set_title("Ground truth")
    axes[1, 0].axis("off")
    for j, ((name, pred), err) in enumerate(zip(panels, errs), start=1):
        im = axes[0, j].imshow(
            _speed(pred),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=vmax,
            cmap="viridis",
        )
        axes[0, j].set_title(name)
        ie = axes[1, j].imshow(
            err, origin="lower", extent=extent, vmin=0, vmax=emax, cmap="magma"
        )
        axes[1, j].set_title(f"{name}: error\nRMS {np.sqrt(np.mean(err**2)):.2e} m/s")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, label="|u| (m/s)")
    fig.colorbar(ie, ax=axes[1, 1:].tolist(), fraction=0.02, label="|u error| (m/s)")
    fig.suptitle(
        f"Re {reynolds(TEST_CASE):,.0f} (unseen), t = {step * DT:.1f} s, "
        "one step ahead from the previous frames"
    )
    _save(fig, out)


def plot_accuracy(runs: dict, base: dict, out: Path) -> None:
    """Single-step RMSE, 3 frames in and 1 out: paper vs ours, per model."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(MODELS))
    w = 0.27
    bars = [
        (
            "Paper (Table 2)",
            [ref.SINGLE_STEP_TARGET.get(m) for m in MODELS],
            "#bbbbbb",
        ),
        ("Ours, default size", [_rmse(runs, f"{m}_3to1") for m in MODELS], "#8fb3de"),
        (
            "Ours, searched size",
            [_rmse(runs, f"best_{m}_3to1") for m in MODELS],
            "#2a78d6",
        ),
    ]
    for k, (label, vals, color) in enumerate(bars):
        pos = x + (k - 1) * w
        ax.bar(pos, [v or 0 for v in vals], w, label=label, color=color)
        for xi, v in zip(pos, vals):
            if v:
                ax.text(xi, v * 1.06, f"{v:.1e}", ha="center", fontsize=7)
    ax.axhline(
        base["single_step"]["rmse"],
        color=BASELINE_COLOR,
        ls="--",
        label=f"Copy last frame ({base['single_step']['rmse']:.2e})",
    )
    for xi, m in zip(x, MODELS):
        if m in ref.NOT_IN_PAPER:
            ax.text(
                xi - w,
                base["single_step"]["rmse"] * 0.85,
                "not in\nthe paper",
                ha="center",
                va="top",
                fontsize=7,
                color="#777777",
            )
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.set_yscale("log")
    ax.set_ylabel("RMSE on Re 10,000 (m/s, log scale)")
    ax.set_title("One step ahead, 3 frames in, 1 out (lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    _save(fig, out)


def plot_rollout(runs: dict, base: dict, out: Path) -> None:
    """Ten steps ahead: per-step RMSE (left), pooled RMSE vs paper (right)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))
    steps = np.arange(1, HORIZON + 1)
    # The FNO entries are the paper's time-window study. Every other
    # architecture contributes both of its 3:1 sizes, taken from MODELS so a
    # new one appears here without editing this function. Both sizes are
    # shown because which one rolls out best is not the same question as
    # which one is most accurate one step ahead.
    styles = {
        "fno_1to1": ("FNO 1 in, 1 out", "#f4a582", "-"),
        "fno_3to1": ("FNO 3 in, 1 out", "#eb6834", "-"),
        "fno_3to3": ("FNO 3 in, 3 out", "#b2182b", "-"),
        **{
            name: (f"{MODEL_LABELS[m]} 3 in, 1 out, {size}", MODEL_COLORS[m], ls)
            for m in MODELS
            if m != "fno"
            for name, ls, size in (
                (f"{m}_3to1", ":", "default size"),
                (f"best_{m}_3to1", "--", "searched size"),
            )
        },
    }
    for name, (label, color, ls) in styles.items():
        if name in runs:
            a1.plot(
                steps,
                runs[name]["rollout"]["per_step_rmse"],
                ls,
                color=color,
                label=label,
                lw=2,
            )
    a1.plot(
        steps,
        base["rollout"]["per_step_rmse"],
        ":",
        color=BASELINE_COLOR,
        label="Copy last frame",
        lw=2,
    )
    a1.set_xlabel("Steps ahead (0.1 s each)")
    a1.set_ylabel("RMSE (m/s)")
    a1.set_title("Error at each step of a 10-step rollout")
    a1.grid(True, alpha=0.3)
    # Under the axes rather than over them: this legend holds two entries
    # per architecture, and covered the curves it labels once there were six.
    a1.legend(
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        borderaxespad=0,
    )

    rows = [
        ("1 in, 1 out", "fno_1to1", ref.TABLE5_FNO_N_TO_10[1]),
        ("3 in, 1 out", "fno_3to1", ref.TABLE5_FNO_N_TO_10[3]),
        ("3 in, 3 out", "fno_3to3", ref.TABLE6_FNO_3_TO_N[3]),
    ]
    x = np.arange(len(rows))
    ours = [runs[r]["rollout"]["rmse"] if r in runs else 0 for _, r, _ in rows]
    paper = [p for _, _, p in rows]
    a2.bar(x - 0.2, paper, 0.4, label="Paper (Tables 5, 6)", color="#bbbbbb")
    a2.bar(x + 0.2, ours, 0.4, label="Ours", color=MODEL_COLORS["fno"])
    for xi, v in zip(np.r_[x - 0.2, x + 0.2], paper + ours):
        if v:
            a2.text(xi, v * 1.02, f"{v:.1e}", ha="center", fontsize=7)
    a2.axhline(
        base["rollout"]["rmse"],
        color=BASELINE_COLOR,
        ls="--",
        label=f"Copy last frame ({base['rollout']['rmse']:.2e})",
    )
    a2.set_xticks(x, [f"FNO {r}" for r, _, _ in rows])
    a2.set_ylabel("RMSE over the 10 steps (m/s)")
    a2.set_title("FNO 10 steps ahead, paper vs ours")
    a2.grid(True, axis="y", alpha=0.3)
    a2.legend(fontsize=8)
    _save(fig, out)


def _three_to_one_runs(cks_by_name: dict):
    for m in MODELS:
        for name, ls, size in (
            (f"{m}_3to1", "-", "default"),
            (f"best_{m}_3to1", "--", "searched"),
        ):
            if name in cks_by_name:
                yield name, m, ls, size


def plot_error_curve(cks_by_name: dict, out: Path) -> None:
    """Train and validation loss per epoch for the 3-in, 1-out runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, size in zip(axes, ("default", "searched")):
        for name, m, _, s in _three_to_one_runs(cks_by_name):
            if s != size:
                continue
            h = cks_by_name[name]["history"]
            ep = [r["epoch"] for r in h]
            ax.plot(
                ep,
                [r["val_loss"] for r in h],
                "-",
                color=MODEL_COLORS[m],
                lw=2,
                label=f"{MODEL_LABELS[m]} validation",
            )
            ax.plot(
                ep,
                [r["train_loss"] for r in h],
                ":",
                color=MODEL_COLORS[m],
                lw=1.5,
                label=f"{MODEL_LABELS[m]} train",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{size.capitalize()} size, 3 frames in, 1 out")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Loss: MSE (normalised) + 0.1 x mass term, log scale")
    _save(fig, out)


def plot_compute_cost(cks_by_name: dict, out: Path) -> None:
    """Seconds per training epoch for the 3-in, 1-out runs."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for name, m, ls, size in _three_to_one_runs(cks_by_name):
        h = cks_by_name[name]["history"]
        ax.plot(
            [r["epoch"] for r in h],
            [r["secs"] for r in h],
            ls,
            color=MODEL_COLORS[m],
            lw=1.5,
            label=f"{MODEL_LABELS[m]}, {size} size",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per epoch (log scale)")
    ax.set_title("Training cost per epoch (epoch 1 includes JIT compilation)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_model_error_curve(cks_by_name: dict, m: str, out: Path) -> None:
    """Train and validation loss per epoch of one model's two 3-in, 1-out sizes."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, mm, ls, size in _three_to_one_runs(cks_by_name):
        if mm != m:
            continue
        h = cks_by_name[name]["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(
            ep,
            [r["val_loss"] for r in h],
            ls,
            color=MODEL_COLORS[m],
            lw=2,
            label=f"{size.capitalize()} size, validation",
        )
        ax.plot(
            ep,
            [r["train_loss"] for r in h],
            ls,
            color=MODEL_COLORS[m],
            lw=1,
            alpha=0.5,
            label=f"{size.capitalize()} size, train",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss: MSE (normalised) + 0.1 x mass term, log scale")
    ax.set_title(f"{MODEL_LABELS[m]}, 3 frames in, 1 out")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_model_compute_cost(cks_by_name: dict, m: str, out: Path) -> None:
    """Seconds per training epoch of one model's two 3-in, 1-out sizes."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, mm, ls, size in _three_to_one_runs(cks_by_name):
        if mm != m:
            continue
        h = cks_by_name[name]["history"]
        ax.plot(
            [r["epoch"] for r in h],
            [r["secs"] for r in h],
            ls,
            color=MODEL_COLORS[m],
            lw=1.5,
            label=f"{size.capitalize()} size",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per epoch (log scale)")
    ax.set_title(
        f"{MODEL_LABELS[m]}: training cost per epoch "
        "(epoch 1 includes JIT compilation)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out)


def plot_model_rollout(runs: dict, base: dict, m: str, out: Path) -> None:
    """Per-step RMSE of a 10-step rollout for one model's two sizes, with the
    copy-last-frame baseline and, for every model but the FNO, the FNO's
    matched-size run for reference."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    steps = np.arange(1, HORIZON + 1)
    curves = [
        (f"{m}_3to1", f"{MODEL_LABELS[m]}, default size", MODEL_COLORS[m], "-"),
        (f"best_{m}_3to1", f"{MODEL_LABELS[m]}, searched size", MODEL_COLORS[m], "--"),
    ]
    if m != "fno":
        curves.append(
            ("fno_3to1", "FNO, default size (reference)", MODEL_COLORS["fno"], ":")
        )
    for name, label, color, ls in curves:
        if name in runs:
            ax.plot(
                steps,
                runs[name]["rollout"]["per_step_rmse"],
                ls,
                color=color,
                label=label,
                lw=2,
            )
    ax.plot(
        steps,
        base["rollout"]["per_step_rmse"],
        ":",
        color=BASELINE_COLOR,
        label="Copy last frame",
        lw=2,
    )
    ax.set_xlabel("Steps ahead (0.1 s each)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title(f"{MODEL_LABELS[m]}: error at each step of a 10-step rollout")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out)


def _rmse(runs: dict, name: str):
    return runs[name]["single_step"]["rmse"] if name in runs else None


def model_dir(out_root: Path, model: str) -> Path:
    """The asset folder of one model's page: ``<out_root>/cavity_<model>``."""
    return out_root / f"cavity_{model}"


def split_metrics(metrics: dict) -> dict[str, dict]:
    """One ``metrics.json`` per model: the shared setup and baseline, plus
    only that model's runs and size search."""
    return {
        m: {
            **{k: v for k, v in metrics.items() if k not in ("runs", "search")},
            "model": m,
            "runs": {k: v for k, v in metrics["runs"].items() if v["model"] == m},
            "search": {m: metrics["search"][m]} if m in metrics["search"] else {},
        }
        for m in MODELS
    }


def draw_figures(
    grid: np.ndarray,
    cks_by_name: dict,
    runs: dict,
    base: dict,
    out_root: Path,
    contour_step: int,
) -> None:
    """Every figure of the benchmark pages, under ``out_root``.

    The comparison page shows the four FNO-matched sizes side by side; each
    model's own page shows that model's default and searched sizes.
    ``runs`` and ``base`` are the evaluation results of :func:`main`, so the
    figures can be redrawn from saved metrics without evaluating.
    """
    comparison_dir = out_root / COMPARISON_DIR
    comparison_dir.mkdir(parents=True, exist_ok=True)
    matched = [cks_by_name[f"{m}_3to1"] for m in MODELS if f"{m}_3to1" in cks_by_name]
    if matched:
        plot_contour(grid, matched, contour_step, comparison_dir / "contour.png")
    for m in MODELS:
        out_dir = model_dir(out_root, m)
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [n for n in (f"{m}_3to1", f"best_{m}_3to1") if n in cks_by_name]
        if names:
            plot_contour(
                grid,
                [cks_by_name[n] for n in names],
                contour_step,
                out_dir / "contour.png",
                labels=[
                    f"{MODEL_LABELS[m]}, "
                    f"{'searched' if n.startswith('best_') else 'default'} size"
                    for n in names
                ],
            )
        plot_model_rollout(runs, base, m, out_dir / "rollout.png")
        plot_model_error_curve(cks_by_name, m, out_dir / "error_curve.png")
        plot_model_compute_cost(cks_by_name, m, out_dir / "compute_cost.png")
    plot_accuracy(runs, base, comparison_dir / "accuracy.png")
    plot_rollout(runs, base, comparison_dir / "rollout.png")
    plot_error_curve(cks_by_name, comparison_dir / "error_curve.png")
    plot_compute_cost(cks_by_name, comparison_dir / "compute_cost.png")


# ────────────────────────────────── main ───────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    add_standard_args(ap, out_flag="--out-root", out_default=DEFAULT_OUT_ROOT)
    ap.add_argument(
        "--contour-step",
        type=int,
        default=10,
        help="test frame shown in the contour figures (10 = t of 1.0 s)",
    )
    ap.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size"])
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)

    require_gpu(args.allow_cpu)
    ensure_prepared((TEST_CASE,), args.data_dir, args.cache_dir)
    grid = load_prepared(TEST_CASE, args.cache_dir)
    base = persistence(grid, n_in=3, horizon=HORIZON)
    print(
        f"[baseline] copy last frame: 1 step {base['single_step']['rmse']:.4e}, "
        f"10 steps {base['rollout']['rmse']:.4e} m/s"
    )

    # Named runs.
    cks_by_name, runs = {}, {}
    for name in [only] if only else RUNS:
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint under {run_dir}, skipped")
            continue
        ck = load_checkpoint(run_dir)
        cks_by_name[name] = ck
        tr = ck["config"]["training"]
        runs[name] = dict(
            model=ck["run"].model,
            n_in=ck["run"].n_in,
            n_out=ck["run"].n_out,
            arch=ck["config"]["run"]["arch"],
            n_params=ck["config"]["n_params"],
            epochs_run=tr["epochs_run"],
            best_epoch=tr["best_epoch"],
            best_val_loss=tr["best_val_loss"],
            single_step=single_step(ck, grid, args.batch_size),
            rollout=rollout(ck, grid, HORIZON, args.batch_size),
            paper=paper_numbers(ck["run"]),
            timing=timing(ck["history"]),
            checkpoint_sha256=sha256(resolve_weights_path(run_dir)),
        )
        r = runs[name]
        print(
            f"[{name}] 1 step {r['single_step']['rmse']:.4e}  "
            f"10 steps {r['rollout']['rmse']:.4e} m/s  ({r['n_params']:,} params)"
        )

    # Architecture search: ranked on validation loss, as the paper does.
    search = {}
    for m in MODELS:
        rows = []
        for run_dir in sorted((args.ckpt_root / "search").glob(f"search_{m}_*")):
            if not is_complete(run_dir):
                continue
            ck = load_checkpoint(run_dir)
            rows.append(
                dict(
                    name=ck["run"].name,
                    arch=ck["config"]["run"]["arch"],
                    n_params=ck["config"]["n_params"],
                    best_val_loss=ck["config"]["training"]["best_val_loss"],
                    rmse=single_step(ck, grid, args.batch_size)["rmse"],
                )
            )
        if rows:
            rows.sort(key=lambda r: r["best_val_loss"])
            target = ref.SINGLE_STEP_TARGET.get(m)
            search[m] = dict(
                paper_table1_50_epochs=ref.TABLE1_50_EPOCHS.get(m),
                paper_table2_300_epochs=target,
                n_beating_paper_table2=(
                    None if target is None else sum(r["rmse"] <= target for r in rows)
                ),
                configs=rows,
            )
            print(f"[search {m}] {len(rows)} configs, best by val: {rows[0]['name']}")

    metrics = dict(
        problem="cavity",
        label="Zenteiq",
        reference=dict(
            citation=ref.CITATION,
            doi=ref.DOI,
        ),
        dataset=dict(
            train_cases=list(TRAIN_CASES),
            test_case=TEST_CASE,
            grid="128x128 cells",
            frames="steps 0 to 100 (first 10 s, every 0.1 s)",
            fields=["p", "Ux", "Uy"],
        ),
        protocol=PROTOCOL,
        metric="RMSE in m/s pooled over p, Ux, Uy on the test case",
        rollout_horizon=HORIZON,
        baseline_copy_last_frame=base,
        runs=runs,
        search=search,
    )
    for m, per_model in split_metrics(metrics).items():
        mine = [checkpoint_dir(args, n) for n, r in runs.items() if r["model"] == m]
        save_metrics_json(
            attach(per_model, mine), model_dir(args.out_root, m) / "metrics.json"
        )

    draw_figures(grid, cks_by_name, runs, base, args.out_root, args.contour_step)
    print(f"[infer] wrote metrics.json and figures to {args.out_root}/cavity_<model>")


if __name__ == "__main__":
    main()
