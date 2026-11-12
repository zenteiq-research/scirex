# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-Design-ShapeNetCar benchmark, step 2 of 2: evaluate saved checkpoints.

Loads every checkpoint written by ``train.py``, re-scores it on its held-out
fold with the Transolver paper's own metrics -- the split MSE (training loss,
for the error-curve plot), the paper's Table 3-style relative L2 ("Volume"
velocity / "Surf" pressure), and the paper's post-hoc drag-coefficient
evaluation (mean relative Cd error "c_d", Spearman rank correlation "rho_d")
-- and writes the figures and ``metrics.json`` used by
``doc/benchmarks/shapenetcar/results/comparative.md``. No
training happens here.

The drag-coefficient metric and the surface-pressure contour need the raw
``quadpress_smpl.vtk`` per sample (cell geometry); both are skipped
per-sample when it's absent (e.g. a cache-only checkpoint), falling back to a
2D point-cloud projection for the contour -- see ``_data.py``'s module
docstring and ``src/scirex/data/datasets/shapenet_car_drag.py``.

Usage (from the repo root)::

    python scripts/benchmarks/shapenetcar/infer.py \\
        --data-dir ~/data/mlcfd_data/training_data

Outputs go to ``doc/assets/benchmarks/shapenetcar_<model>/`` -- one folder
per model (``shapenetcar_transolver``, ``shapenetcar_gino``,
``shapenetcar_giwo``), each with its own ``metrics.json``, ``contour.png``,
``error_curve.png`` and ``compute_cost.png`` for that model's headline run
(:data:`HEADLINE_RUN`); any other run of the same model (e.g. the cheaper
``"gino"``/``"giwo"`` radius probes) gets its own name-prefixed figures in
the same folder, matching the ``lid_driven_cavity``/``darcy`` convention of
one asset folder per ``<problem>_<model>``. A fourth folder,
``shapenetcar_comparison/``, holds the three headline runs' ``accuracy.png``
and ``compute_cost.png`` side by side. Legacy ``.vtk`` files (ParaView) still
go to ``scripts/benchmarks/vtk_outputs/shapenetcar/<run>/`` (gitignored).
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scirex.data.datasets.shapenet_car_drag import (  # noqa: E402
    compute_drag_coefficient,
    load_surface_mesh,
    mean_relative_drag_error,
    spearman_drag_correlation,
)
from scripts.benchmarks._ckpt import resolve_weights_path  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import (  # noqa: E402
    plot_car_surface_pressure,
    plot_compute_cost,
    plot_error_curves,
    plot_point_cloud_scalar_field,
    save_metrics_json,
)
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.shapenetcar import _gino_geometry  # noqa: E402
from scripts.benchmarks.shapenetcar import _reference as ref  # noqa: E402
from scripts.benchmarks.shapenetcar._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
    sha256,
)
from scripts.benchmarks.shapenetcar._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    MODEL_LABELS,
    RUNS,
)
from scripts.benchmarks.shapenetcar._data import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    load_split,
)
from scripts.benchmarks.shapenetcar._physics import rel_l2_metrics  # noqa: E402

# See train.py's comment on this: `scirex` forces jax_enable_x64=True as a
# tensorly import-time side effect; restore float32 here explicitly too, so
# this script doesn't silently depend on train.py's import ordering.
jax.config.update("jax_enable_x64", False)

DEFAULT_OUT_ROOT = _REPO_ROOT / "doc/assets/benchmarks"
DEFAULT_VTK_ROOT = _REPO_ROOT / "scripts/benchmarks/vtk_outputs/shapenetcar"
COMPARISON_DIR = "shapenetcar_comparison"  # the three-model comparison figures
MODEL_COLORS = {"transolver": "#827717", "gino": "#2a78d6", "giwo": "#b2182b"}

#: Which registered run represents each model on its own page and in the
#: comparison figures -- the one actually trained for the paper's full
#: 200-epoch protocol (see ``_config.py``'s comments on why the other,
#: cheaper-radius runs of the same model were not: they were 20/21-epoch
#: cost-matching probes, not headline results).
HEADLINE_RUN: dict[str, str] = {
    "transolver": "transolver",
    "gino": "gino_wide_radius",
    "giwo": "giwo_transolver_cost_matched",
}


# ── VTK export (legacy .vtk, not XML .vtu -- see this package's __init__.py
# for why ParaView renders of these are the images doc/benchmarks/
# transolver.md links, not an XML .vtu) ────────────────────────────────────


def _build_quad_grid(points: np.ndarray, quads: np.ndarray, pressure: np.ndarray):
    import vtk
    from vtk.util import numpy_support as vnp

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64)))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    for quad in quads:
        cell = vtk.vtkQuad()
        for j, pid in enumerate(quad):
            cell.GetPointIds().SetId(j, int(pid))
        grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    pressure_arr = vnp.numpy_to_vtk(np.ascontiguousarray(pressure, dtype=np.float64))
    pressure_arr.SetName("pressure")
    grid.GetPointData().AddArray(pressure_arr)
    grid.GetPointData().SetActiveScalars("pressure")
    return grid


def export_vtk_comparison_legacy(
    points: np.ndarray,
    quads: np.ndarray,
    truth_pressure: np.ndarray,
    pred_pressure: np.ndarray,
    out_dir: Path,
) -> None:
    """Writes ``truth.vtk``/``pred.vtk`` (legacy, non-XML VTK) into
    ``out_dir`` so the two can be opened side by side in ParaView."""
    import vtk

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, field in (("truth", truth_pressure), ("pred", pred_pressure)):
        grid = _build_quad_grid(points, quads, field)
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(str(out_dir / f"{name}.vtk"))
        writer.SetInputData(grid)
        writer.SetFileTypeToBinary()
        writer.Write()


# ──────────────────────────────── evaluation ───────────────────────────────


def _apply_model(model, params, x: jnp.ndarray, sample: dict) -> np.ndarray:
    """``model.apply`` with GINO's ``neighbors_in``/``neighbors_out`` forwarded
    only when ``sample`` carries them (attached by :func:`_gino_geometry.
    attach_neighbors` for a ``"gino"`` run) -- Transolver's own ``__call__``
    has no such parameters and raises on an unknown kwarg, so this mirrors
    ``_physics.py``'s ``make_loss``'s own conditional-``extra`` pattern rather
    than passing ``neighbors_in=None`` unconditionally."""
    extra = {}
    if "neighbors_in" in sample:
        extra = dict(
            neighbors_in=sample["neighbors_in"], neighbors_out=sample["neighbors_out"]
        )
    return np.asarray(model.apply(params, x=x, fx=None, **extra)[0])


def evaluate(ck: dict, val_samples: list[dict], data_dir: Path) -> dict:
    """Split MSE, Table 3-style relative L2, and (when raw VTK is available)
    the drag-coefficient metrics, over ``val_samples``."""
    model, params = ck["model"], ck["params"]
    mean_in, std_in, mean_out, std_out = ck["coef_norm"]
    press_weight = ck["config"]["training"]["press_weight"]

    if ck["run"].model in ("gino", "giwo"):
        arch = ck["run"].arch
        _gino_geometry.attach_neighbors(
            val_samples,
            arch["latent_grid"],
            arch["in_gno_radius"],
            arch["out_gno_radius"],
        )

    press_losses, velo_losses = [], []
    volume_rel_l2s, surface_rel_l2s = [], []
    gt_coefs, pred_coefs, n_missing_vtk = [], [], 0

    for sample in val_samples:
        x = jnp.asarray(sample["x"])[None, :, :]
        out = _apply_model(model, params, x, sample)
        y, surf = sample["y"], sample["surf"]

        surf_f = surf.astype(out.dtype)
        n_surf = max(1.0, surf_f.sum())
        press_mse = float(np.sum(surf_f * (out[:, -1] - y[:, -1]) ** 2) / n_surf)
        press_losses.append(press_mse)
        velo_losses.append(float(np.mean((out[:, :-1] - y[:, :-1]) ** 2)))

        out_denorm = out * std_out + mean_out
        y_denorm = y * std_out + mean_out
        l2 = rel_l2_metrics(out_denorm, y_denorm, surf)
        volume_rel_l2s.append(l2["volume"])
        surface_rel_l2s.append(l2["surface"])

        vtk_path = data_dir / sample["sample_rel_path"] / "quadpress_smpl.vtk"
        if not vtk_path.exists():
            n_missing_vtk += 1
            continue
        pred_coefs.append(
            compute_drag_coefficient(
                vtk_path, out_denorm[surf, -1], out_denorm[surf, :-1]
            )
        )
        gt_coefs.append(
            compute_drag_coefficient(vtk_path, y_denorm[surf, -1], y_denorm[surf, :-1])
        )

    val_press, val_velo = float(np.mean(press_losses)), float(np.mean(velo_losses))
    c_d = rho_d = None
    if gt_coefs:
        c_d = mean_relative_drag_error(gt_coefs, pred_coefs)
        if len(gt_coefs) > 1:
            rho_d = spearman_drag_correlation(gt_coefs, pred_coefs)
            if np.isnan(rho_d):
                rho_d = None
    if n_missing_vtk:
        print(
            f"[{ck['run'].name}:infer] drag eval skipped {n_missing_vtk} val "
            "sample(s) without a raw quadpress_smpl.vtk on disk"
        )

    return dict(
        val_loss=val_velo + press_weight * val_press,
        val_press_mse=val_press,
        val_velo_mse=val_velo,
        rel_l2_volume=float(np.mean(volume_rel_l2s)),
        rel_l2_surface=float(np.mean(surface_rel_l2s)),
        c_d_mean_relative_drag_error=c_d,
        rho_d_spearman_drag_correlation=rho_d,
        n_drag_eval_samples=len(gt_coefs),
    )


def paper_numbers(model: str) -> dict:
    """The paper's Table 3 numbers, absent for a model it never ran
    (``_reference.NOT_IN_PAPER``)."""
    return {} if model in ref.NOT_IN_PAPER else dict(ref.TABLE3)


# ──────────────────────────── per-model asset layout ───────────────────────


def model_dir(out_root: Path, model: str) -> Path:
    """The asset folder of one model's page: ``<out_root>/shapenetcar_<model>``,
    matching ``lid_driven_cavity``/``darcy``'s own ``model_dir``."""
    return out_root / f"shapenetcar_{model}"


def split_metrics(metrics: dict) -> dict[str, dict]:
    """One ``metrics.json`` per model: the shared header, plus only that
    model's own runs (e.g. ``shapenetcar_gino``'s holds both ``"gino"`` and
    ``"gino_wide_radius"``)."""
    return {
        m: {
            **{k: v for k, v in metrics.items() if k != "runs"},
            "runs": {k: v for k, v in metrics["runs"].items() if v["model"] == m},
        }
        for m in HEADLINE_RUN
    }


def plot_comparison_accuracy(runs: dict, out: Path) -> None:
    """The paper's Table 3 metrics, headline run per model, side by side."""
    specs = [
        ("rel_l2_volume", "Volume rel. L2 (velocity)", ref.TABLE3["volume_rel_l2"]),
        ("rel_l2_surface", "Surface rel. L2 (pressure)", ref.TABLE3["surface_rel_l2"]),
        ("c_d_mean_relative_drag_error", "C_D mean rel. error", ref.TABLE3["c_d"]),
        (
            "rho_d_spearman_drag_correlation",
            "rho_D Spearman corr. (higher better)",
            ref.TABLE3["rho_d"],
        ),
    ]
    models = [m for m in HEADLINE_RUN if HEADLINE_RUN[m] in runs]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.2 * len(specs), 4.2))
    for ax, (key, title, paper_val) in zip(axes, specs):
        vals = [runs[HEADLINE_RUN[m]].get(key) for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        ax.bar([MODEL_LABELS[m] for m in models], [v or 0 for v in vals], color=colors)
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(i, v * 1.03, f"{v:.3f}", ha="center", fontsize=8)
        ax.axhline(paper_val, color="#777777", ls="--", label=f"Paper {paper_val:.3f}")
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Car-Design-ShapeNetCar: Table 3 metrics, headline run per model")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_compute_cost(runs: dict, out: Path) -> None:
    """Seconds per training epoch, headline run per model, one line each."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, name in HEADLINE_RUN.items():
        if name not in runs:
            continue
        history = runs[name]["history"]
        ax.plot(
            [h["epoch"] for h in history],
            [h["secs"] for h in history],
            label=MODEL_LABELS[m],
            color=MODEL_COLORS[m],
            lw=2,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Compute cost (s / epoch, log scale)")
    ax.set_title("Car-Design-ShapeNetCar: compute cost per epoch, by model")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────── main ───────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    add_standard_args(ap, out_flag="--out-root", out_default=DEFAULT_OUT_ROOT)
    ap.add_argument("--vtk-root", type=Path, default=DEFAULT_VTK_ROOT)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)

    require_gpu(args.allow_cpu)
    if only and args.vtk_root == DEFAULT_VTK_ROOT:  # keep a one-off run out of the repo
        args.vtk_root = args.out_root / "vtk"
    data_dir = args.data_dir.expanduser()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else None

    runs: dict[str, dict] = {}
    for name in [only] if only else RUNS:
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint under {run_dir}, skipped")
            continue
        ck = load_checkpoint(run_dir)
        fold_id = ck["config"]["training"]["fold_id"]
        data = load_split(data_dir, fold_id, cache_dir, use_cache=True)
        val_samples = data["val"]

        metrics = evaluate(ck, val_samples, data_dir)
        tr = ck["config"]["training"]
        runs[name] = dict(
            model=ck["run"].model,
            arch=ck["config"]["run"]["arch"],
            n_params=ck["config"]["n_params"],
            n_epochs=tr["n_epochs"],
            n_train=tr["n_train"],
            n_val=tr["n_val"],
            fold_id=fold_id,
            train_seconds=tr["train_seconds"],
            history=ck["history"],
            **metrics,
            paper=paper_numbers(ck["run"].model),
            checkpoint_sha256=sha256(resolve_weights_path(run_dir)),
        )
        print(
            f"[{name}] val_loss={metrics['val_loss']:.6f}  "
            f"rel_l2_volume={metrics['rel_l2_volume']:.6f}  "
            f"rel_l2_surface={metrics['rel_l2_surface']:.6f}  "
            f"({ck['config']['n_params']:,} params)"
        )

        # ── Plots + VTK export for this run ──────────────────────────────────
        # Each run's figures go to its model's own asset folder
        # (doc/assets/benchmarks/shapenetcar_<model>/); the headline run
        # (HEADLINE_RUN) keeps unprefixed filenames since it's the one that
        # model's page links by default, every other run of the same model
        # (e.g. the cheap "gino" radius probe alongside "gino_wide_radius")
        # gets its own name-prefixed files in that same folder -- otherwise
        # every run after the first would clobber the previous one's
        # contour.png/error_curve.png/compute_cost.png.
        out_dir = model_dir(args.out_root, ck["run"].model)
        prefix = "" if name == HEADLINE_RUN.get(ck["run"].model) else f"{name}_"
        label = MODEL_LABELS.get(ck["run"].model, ck["run"].model)
        history = ck["history"]
        epochs = [h["epoch"] for h in history]
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_error_curves(
            epochs,
            [h["train_loss"] for h in history],
            [h["val_loss"] for h in history],
            out_dir / f"{prefix}error_curve.png",
            title=f"{label} on Car-Design-ShapeNetCar: training/test error (Zenteiq)",
            train_label="split MSE (velo + press_weight*press)",
        )
        plot_compute_cost(
            epochs,
            [h["secs"] for h in history],
            out_dir / f"{prefix}compute_cost.png",
            title=f"{label} on Car-Design-ShapeNetCar: compute cost per epoch (Zenteiq)",
        )

        sample0 = val_samples[0]
        x0 = jnp.asarray(sample0["x"])[None, :, :]
        out0 = _apply_model(ck["model"], ck["params"], x0, sample0)
        _, _, mean_out, std_out = ck["coef_norm"]
        contour_title = f"{label} on Car-Design-ShapeNetCar: surface pressure (Zenteiq)"
        vtk_path0 = data_dir / sample0["sample_rel_path"] / "quadpress_smpl.vtk"
        if vtk_path0.exists():
            surf0 = sample0["surf"]
            out0_denorm = out0 * std_out + mean_out
            y0_denorm = sample0["y"] * std_out + mean_out
            plot_car_surface_pressure(
                vtk_path0,
                out0_denorm[surf0, -1],
                y0_denorm[surf0, -1],
                out_dir / f"{prefix}contour.png",
                title=contour_title,
            )
            points0, quads0 = load_surface_mesh(vtk_path0)
            export_vtk_comparison_legacy(
                points0,
                quads0,
                y0_denorm[surf0, -1],
                out0_denorm[surf0, -1],
                args.vtk_root / name,
            )
            print(f"[{name}] VTK files: {args.vtk_root / name} (open in ParaView)")
        else:
            plot_point_cloud_scalar_field(
                sample0["pos"],
                out0[:, -1],
                sample0["y"][:, -1],
                out_dir / f"{prefix}contour.png",
                title=contour_title,
            )

    metrics = dict(
        problem="shapenetcar",
        label="Zenteiq",
        reference=dict(citation=ref.CITATION, doi=ref.DOI),
        metric="split MSE during training; Table 3-style relative L2 and drag "
        "coefficient for paper comparison",
        runs=runs,
    )
    for m, per_model in split_metrics(metrics).items():
        mine = [checkpoint_dir(args, n) for n, r in runs.items() if r["model"] == m]
        save_metrics_json(
            attach(per_model, mine), model_dir(args.out_root, m) / "metrics.json"
        )

    comparison_dir = args.out_root / COMPARISON_DIR
    plot_comparison_accuracy(runs, comparison_dir / "accuracy.png")
    plot_comparison_compute_cost(runs, comparison_dir / "compute_cost.png")

    print(
        f"[infer] wrote metrics.json and figures to {args.out_root}/shapenetcar_<model>"
    )


if __name__ == "__main__":
    main()
