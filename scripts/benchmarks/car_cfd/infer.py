# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-CFD benchmark, step 2 of 2: evaluate saved checkpoints.

Loads every checkpoint written by ``train.py`` for a run named in
``_config.RUNS`` (plus the headline seeds), re-scores it on the 111
held-out cars with the GINO paper's de-normalised relative L2, and writes
the figures and ``metrics.json`` used by the
``doc/benchmarks/car_cfd/results/comparative.md``. No training happens here.

Usage (from the repo root)::

    python scripts/benchmarks/car_cfd/infer.py

There is no ``--run``: all three headline checkpoints (``paper``,
``transolver_val`` and ``giwo_val``; train them with ``train.py --run paper
transolver_val giwo_val``) must exist, or it stops with an error. Other
complete checkpoints, such as ``transolver_val_seed1``, are added to the
per-model figures; ``<run>_val<N>`` screens are ignored.

Outputs go under ``doc/assets/benchmarks/``, one folder per model:

* ``car_cfd_<model>/`` holds that model's ``metrics.json`` (every run of the
  model, its headline run and, for Transolver, the seed spread),
  ``contour.png`` (the headline run's surface pressure on the first test
  car), ``error_curve.png`` and ``compute_cost.png``;
* ``car_cfd_comparison/`` holds the figures that compare the three headline
  runs, for ``doc/benchmarks/car_cfd/results/comparative.md``: ``contour.png``,
  ``accuracy.png``, ``error_curve.png`` and ``compute_cost.png``.

``truth.vtu``/``pred.vtu`` of each headline run's first test car go to
``scripts/benchmarks/vtk_outputs/car_cfd/<model>/`` for ParaView
(gitignored; skipped with ``--no-vtk`` or without the ``vtk`` package).
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from scirex.losses import LpLoss  # noqa: E402
from scripts.benchmarks._ckpt import load_params, resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import save_metrics_json  # noqa: E402
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.car_cfd import _reference as ref  # noqa: E402
from scripts.benchmarks.car_cfd._checkpoint import (  # noqa: E402
    count_params,
    is_complete,
    read_checkpoint,
)
from scripts.benchmarks.car_cfd._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_VTK_ROOT,
    HEADLINE,
    MODEL_LABELS,
    MODELS,
    PROTOCOL,
    RUNS,
    SEEDS,
    build_config,
)
from scripts.benchmarks.car_cfd._data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    load_data_module,
    load_test_mesh,
)
from scripts.benchmarks.car_cfd._models import (  # noqa: E402
    build,
    predict_denormalised,
)
from scripts.benchmarks.car_cfd._vtk import export_vtk_comparison  # noqa: E402
from scripts.benchmarks.car_cfd.train import build_trainer  # noqa: E402

COMPARISON_DIR = "car_cfd_comparison"  # the three-model comparison figures
MODEL_COLORS = {"gino": "#7b3fa0", "transolver": "#827717", "giwo": "#00838f"}
PAPER_COLOR = "#bbbbbb"
TEST_METRIC = "de-normalised relative L2 of the surface pressure, 111 test cars"


# ─────────────────────────────── evaluation ────────────────────────────────
def load_run(run_dir: Path, data_root: Path, download: bool = True) -> dict:
    """One checkpoint's model and data processor with its trained weights,
    plus its run, history, saved config and test loader."""
    ck = read_checkpoint(run_dir)
    run, saved = ck["run"], ck["config"]
    training = saved["training"]
    config = build_config(run, training["n_train"], training["n_test"])

    # Same n_train and n_test as training: the pressure normaliser (and the
    # coordinate bounding box) is fitted on all n_train + n_test cars loaded,
    # so it is the same fit only when both counts match.
    data_module = load_data_module(
        data_root,
        run.sdf_query_resolution,
        training["n_train"],
        training["n_test"],
        download=download,
    )
    # Neighbour-search logs go to a scratch directory: the checkpoint is read-only.
    log_dir = Path(tempfile.mkdtemp(prefix="car_cfd_neighbor_logs_"))
    model, processor = build(run, config, data_module, log_dir)

    # GINO's parameter shapes come from the data (latent grid, channels), so
    # the template is initialised on a real test batch, then overwritten.
    test_loader = data_module.test_loader(batch_size=1, shuffle=False)
    model.init(processor.preprocess(next(iter(test_loader))))
    model.params = load_params(model.params, resolve_weights_path(run_dir))
    return dict(
        ck,
        build_config=config,
        model=model,
        processor=processor,
        test_loader=test_loader,
    )


def evaluate_run(run_dir: Path, data_root: Path, download: bool = True) -> dict:
    """Re-scores one checkpoint on the test cars.

    Returns its metrics (curves, final/best/re-scored test error, config)
    and its de-normalised ``(truth, pred)`` on the first test car.
    """
    loaded = load_run(run_dir, data_root, download)
    run, history, saved = loaded["run"], loaded["history"], loaded["config"]
    training = saved["training"]
    config = loaded["build_config"]
    model, processor = loaded["model"], loaded["processor"]
    test_loader = loaded["test_loader"]
    sample0 = next(iter(test_loader))

    trainer = build_trainer(model, processor, n_epochs=0)
    rescored = float(
        trainer.evaluate({"l2": LpLoss(d=2, p=2)}, test_loader, log_prefix="test")[
            "test_l2"
        ]
    )
    truth, pred = predict_denormalised(model, processor, sample0)
    final = history[-1]["test_err"]
    best = min(history, key=lambda h: h["test_err"])
    print(
        f"[{run_dir.name}] re-scored test L2 {rescored:.4%} "
        f"(history's final epoch: {final:.4%})",
        flush=True,
    )
    secs = [h["secs"] for h in history]
    metrics = {
        "architecture": run.architecture,
        "epochs": [h["epoch"] for h in history],
        "train_err": [h["train_err"] for h in history],
        "test_err": [h["test_err"] for h in history],
        "epoch_time_s": secs,
        "lr": [h["lr"] for h in history],
        "final_train_err": history[-1]["train_err"],
        "final_test_err": final,
        "rescored_test_err": rescored,
        "best_test_err": best["test_err"],
        "best_epoch": best["epoch"],
        "mean_epoch_time_s": sum(secs) / len(secs),
        "train_seconds": training["train_seconds"],
        "n_params": count_params(model.params),
        "seed": training.get("seed", PROTOCOL["seed"]),
        "config": {
            **saved["run"],
            "n_train": training["n_train"],
            "n_test": training["n_test"],
            **_effective_arch(run, config),
        },
        "provenance": saved["provenance"],
    }
    return dict(metrics=metrics, truth=truth, pred=pred)


def _effective_arch(run, config) -> dict:
    """The model settings actually applied, where the run's overrides alone
    would mislead (GINO reads ``out_gno_radius``, not ``gno_radius``)."""
    if run.architecture not in ("gino", "giwo"):
        return {}
    arch = {
        "out_gno_radius": config.model.get("out_gno_radius", 0.033),
        "use_in_gno": config.model.use_in_gno,
        "hidden_channels": config.model.fno_hidden_channels,
    }
    if run.architecture == "giwo":
        arch.update({k: v for k, v in run.model.items() if k.startswith("wno_")})
    return arch


def seed_spread(results: dict, run: str) -> dict | None:
    """Mean and sample standard deviation of ``run``'s final test error over
    seed 0 and the seeds in :data:`SEEDS`, or ``None`` without extra seeds."""
    names = [run] + [f"{run}_seed{s}" for s in SEEDS.get(run, ())]
    names = [n for n in names if n in results]
    if len(names) < 2:
        return None
    errs = [results[n]["metrics"]["final_test_err"] for n in names]
    return dict(
        runs=names,
        seeds=[results[n]["metrics"]["seed"] for n in names],
        final_test_err=errs,
        mean=float(np.mean(errs)),
        std=float(np.std(errs, ddof=1)),
    )


def paper_numbers() -> dict:
    return dict(citation=ref.CITATION, doi=ref.DOI, table2=ref.TABLE2)


# ──────────────────────────────── figures ──────────────────────────────────
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_surface(ax, pts, triangles, values, cmap, norm) -> None:
    """One car: every triangle coloured by the mean of its 3 vertex values,
    from the elevated three-quarter view of ``shapenetcar``'s figures."""
    faces = values[triangles].mean(axis=1)
    ax.add_collection3d(
        Poly3DCollection(
            pts[triangles],
            facecolors=plt.get_cmap(cmap)(norm(faces)),
            edgecolor="none",
        )
    )
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
    ax.set_box_aspect(np.ptp(pts, axis=0))
    ax.view_init(elev=20.0, azim=-60.0)
    ax.set_axis_off()


def plot_contour(
    vertices: np.ndarray,
    triangles: np.ndarray,
    truth: np.ndarray,
    preds: list[tuple[str, np.ndarray]],
    out: Path,
) -> None:
    """Surface pressure on one test car: ground truth and each prediction
    (top), each prediction's absolute error (bottom), one colour scale per
    row."""
    # Mesh axes are (width, height, length); plot (width, length, height) so
    # the car stands upright, as plot_car_surface_pressure does.
    pts = vertices[:, [0, 2, 1]]
    errs = [np.abs(p - truth) for _, p in preds]
    lo, hi = float(truth.min()), float(truth.max())
    p_norm = Normalize(vmin=lo, vmax=hi)
    e_norm = Normalize(vmin=0.0, vmax=float(max(e.max() for e in errs)))

    n = len(preds) + 1
    fig = plt.figure(figsize=(4.6 * n, 7.6))
    ax = fig.add_subplot(2, n, 1, projection="3d")
    _draw_surface(ax, pts, triangles, truth, "jet", p_norm)
    ax.set_title("Ground truth")
    p_axes, e_axes = [ax], []
    for j, ((name, pred), err) in enumerate(zip(preds, errs), start=2):
        ax = fig.add_subplot(2, n, j, projection="3d")
        _draw_surface(ax, pts, triangles, pred, "jet", p_norm)
        rel = np.linalg.norm(pred - truth) / np.linalg.norm(truth)
        ax.set_title(f"{name}\nrel. L2 {rel:.2%}")
        p_axes.append(ax)
        ax = fig.add_subplot(2, n, n + j, projection="3d")
        _draw_surface(ax, pts, triangles, err, "magma", e_norm)
        ax.set_title(f"{name}: abs. error")
        e_axes.append(ax)
    for norm, cmap, axes, label in (
        (p_norm, "jet", p_axes, "Pressure"),
        (e_norm, "magma", e_axes, "|Pressure error|"),
    ):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=axes, fraction=0.02, shrink=0.7, label=label)
    fig.suptitle("Surface pressure, first test car (de-normalised)")
    _save(fig, out)


def plot_accuracy(results: dict, spreads: dict, out: Path) -> None:
    """Final test error of each model's headline run, with the GINO paper's
    Table 2 decoder row; the seed spread as an error bar where there is one."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = ["GINO, paper\n(Table 2)"] + [MODEL_LABELS[m] for m in MODELS]
    vals = [ref.TABLE2["gino_decoder"]["test"] * 100]
    errs, colors = [0.0], [PAPER_COLOR]
    for m in MODELS:
        spread = spreads.get(m)
        if spread is not None:
            vals.append(spread["mean"] * 100)
            errs.append(spread["std"] * 100)
        else:
            vals.append(results[HEADLINE[m]]["metrics"]["final_test_err"] * 100)
            errs.append(0.0)
        colors.append(MODEL_COLORS[m])
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.6, color=colors)
    spread = [k for k, e in enumerate(errs) if e]
    ax.errorbar(
        x[spread],
        [vals[k] for k in spread],
        yerr=[errs[k] for k in spread],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    for xi, v, e in zip(x, vals, errs):
        text = f"{v:.2f}%" + (f"\n± {e:.2f}" if e else "")
        ax.text(xi, v + e + 0.2, text, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test relative L2 (%)")
    ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.25)
    ax.set_title("Car-CFD surface pressure, ~75M parameters (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def _curve(ax, metrics: dict, color, label: str, lw: float = 2.0) -> None:
    ep = np.asarray(metrics["epochs"]) + 1
    ax.plot(ep, metrics["test_err"], "-", color=color, lw=lw, label=f"{label} test")
    ax.plot(
        ep, metrics["train_err"], ":", color=color, lw=lw * 0.75, label=f"{label} train"
    )


def plot_error_curve(runs: list[tuple[str, dict, str]], out: Path, title: str) -> None:
    """Test (solid) and training (dotted) relative L2 per epoch."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for label, metrics, color in runs:
        _curve(ax, metrics, color, label)
    ax.axhline(
        ref.TABLE2["gino_decoder"]["test"],
        color=PAPER_COLOR,
        ls="--",
        label="GINO paper, test (Table 2)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2 (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, out)


def plot_compute_cost(runs: list[tuple[str, dict, str]], out: Path, title: str) -> None:
    """Seconds per training epoch (500 cars, one at a time)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for label, metrics, color in runs:
        ax.plot(
            np.asarray(metrics["epochs"]) + 1,
            metrics["epoch_time_s"],
            color=color,
            lw=1.5,
            label=f"{label} (mean {metrics['mean_epoch_time_s']:.0f} s)",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per epoch")
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out)


def _model_runs(results: dict, m: str) -> list[tuple[str, dict, str]]:
    """Every evaluated run of model ``m``, headline first in the model's
    colour, the rest in shades of grey."""
    names = [n for n in results if results[n]["metrics"]["architecture"] == m]
    names.sort(key=lambda n: (n != HEADLINE[m], n))
    greys = plt.get_cmap("Greys")(np.linspace(0.75, 0.4, max(len(names) - 1, 1)))
    return [
        (n, results[n]["metrics"], MODEL_COLORS[m] if k == 0 else greys[k - 1])
        for k, n in enumerate(names)
    ]


def model_dir(out_root: Path, model: str) -> Path:
    """The asset folder of one model's page: ``<out_root>/car_cfd_<model>``."""
    return out_root / f"car_cfd_{model}"


# ────────────────────────────────── main ───────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    add_standard_args(
        ap,
        out_flag="--out-root",
        out_default=DEFAULT_OUT_ROOT,
        data_flag="--data-root",
        data_default=DEFAULT_DATA_ROOT,
    )
    ap.add_argument("--vtk-root", type=Path, default=DEFAULT_VTK_ROOT)
    ap.add_argument("--no-vtk", action="store_true", help="skip the .vtu export")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(
        ap,
        args,
        [*RUNS, *(f"{r}_seed{s}" for r, seeds in SEEDS.items() for s in seeds)],
    )

    require_gpu(args.allow_cpu)

    if only:  # one checkpoint: score it, no figures or cross-model comparison
        res = evaluate_run(
            checkpoint_dir(args, only), args.data_root, not args.no_download
        )
        save_metrics_json(
            attach(
                {
                    "problem": "car_cfd",
                    "run": only,
                    "test_metric": TEST_METRIC,
                    "metrics": res["metrics"],
                    "paper": paper_numbers(),
                },
                [checkpoint_dir(args, only)],
            ),
            args.out_root / only / "metrics.json",
        )
        print(f"[{only}] {res['metrics']['final_test_err']:.4%} test error")
        return

    # Every named run with a complete checkpoint, plus the headline seeds.
    names = [n for n in RUNS if is_complete(args.ckpt_root / n)]
    names += [
        f"{run}_seed{s}"
        for run, seeds in SEEDS.items()
        for s in seeds
        if is_complete(args.ckpt_root / f"{run}_seed{s}")
    ]
    missing = [h for h in HEADLINE.values() if h not in names]
    if missing:
        ap.error(f"no complete checkpoint for headline run(s) {missing}")
    results = {
        n: evaluate_run(args.ckpt_root / n, args.data_root, not args.no_download)
        for n in names
    }

    vertices, triangles = load_test_mesh(args.data_root, 0)
    truth = results[HEADLINE[MODELS[0]]]["truth"]
    spreads = {}
    for m in MODELS:
        head = HEADLINE[m]
        out_dir = model_dir(args.out_root, m)
        runs = _model_runs(results, m)
        spreads[m] = seed_spread(results, head)
        plot_contour(
            vertices,
            triangles,
            truth,
            [(MODEL_LABELS[m], results[head]["pred"])],
            out_dir / "contour.png",
        )
        plot_error_curve(
            runs,
            out_dir / "error_curve.png",
            f"{MODEL_LABELS[m]} on Car-CFD: training/test relative L2 per epoch",
        )
        plot_compute_cost(
            runs,
            out_dir / "compute_cost.png",
            f"{MODEL_LABELS[m]} on Car-CFD: training cost per epoch",
        )
        save_metrics_json(
            attach(
                {
                    "problem": "car_cfd",
                    "model": m,
                    "label": MODEL_LABELS[m],
                    "test_metric": TEST_METRIC,
                    "headline_run": head,
                    "headline": results[head]["metrics"]["final_test_err"],
                    "seed_spread": spreads[m],
                    "runs": {name: results[name]["metrics"] for name, _, _ in runs},
                    "paper": paper_numbers(),
                },
                [args.ckpt_root / name for name, _, _ in runs],
            ),
            out_dir / "metrics.json",
        )
        print(f"[{m}] wrote {out_dir}", flush=True)
        if args.no_vtk or importlib.util.find_spec("vtk") is None:
            continue
        export_vtk_comparison(
            vertices,
            triangles,
            results[head]["truth"],
            results[head]["pred"],
            args.vtk_root / m,
        )

    comparison_dir = args.out_root / COMPARISON_DIR
    headline = [
        (MODEL_LABELS[m], results[HEADLINE[m]]["metrics"], MODEL_COLORS[m])
        for m in MODELS
    ]
    plot_contour(
        vertices,
        triangles,
        truth,
        [(MODEL_LABELS[m], results[HEADLINE[m]]["pred"]) for m in MODELS],
        comparison_dir / "contour.png",
    )
    plot_accuracy(results, spreads, comparison_dir / "accuracy.png")
    plot_error_curve(
        headline,
        comparison_dir / "error_curve.png",
        "Car-CFD, headline runs: training/test relative L2 per epoch",
    )
    plot_compute_cost(
        headline,
        comparison_dir / "compute_cost.png",
        "Car-CFD, headline runs: training cost per epoch",
    )
    print(f"[comparison] wrote {comparison_dir}", flush=True)


if __name__ == "__main__":
    main()
