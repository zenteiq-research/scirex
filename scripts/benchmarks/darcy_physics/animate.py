# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Darcy animation: the best model's prediction across the test
set.

Loads the final weights of the reported physics-only Transolver (10k
coefficient fields, the lowest error of the four models, see
``doc/benchmarks/darcy_physics/results/comparative.md``), rebuilds its label-free
inference exactly as ``train_transolver.py`` does, and predicts the 200 test
fields. Before drawing anything it checks that the mean test error equals the
one in the run's committed ``metrics.json``, so the animation shows the
published model.

Each frame is one test field, ``--n-cases`` of them picked evenly from the
lowest to the highest error: coefficient, prediction, released solution and
absolute error, on one colour scale per panel for the whole animation.

Written to ``animations/darcy_physics/evolution.gif`` (a diagnostic, like the
other ``animations/`` folders, not part of the published doc assets). The
animated cases are also saved to ``--cache`` (under the gitignored
``scripts/benchmarks/checkpoints/``), so ``--from-cache`` re-renders the GIF
without the data, the model or a GPU.

Usage (from the repo root)::

    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth{1,2}.mat
    python scripts/benchmarks/darcy_physics/animate.py
    python scripts/benchmarks/darcy_physics/animate.py --from-cache
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
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    build_grid_aligned_q4_mesh,
)
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import load_params  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.darcy._data import (  # noqa: E402
    GlobalNormalizer,
    load_split,
    rel_l2,
)
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    HEADLINE_FIELDS,
    PROTOCOL,
    run_dir,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    FE_PRECISION,
    cell_centre_interpolation,
    sample_extra_coefficients,
    sin_boundary_factor,
    split_coefficient,
    torsion_output_scale,
)
from scripts.benchmarks.darcy_physics.train_transolver import (  # noqa: E402
    build_model,
    make_predict_nodes,
)

jax.config.update("jax_enable_x64", False)

BEST_MODEL = "transolver"
DEFAULT_OUT = _REPO_ROOT / "animations/darcy_physics/evolution.gif"
DEFAULT_CACHE = _REPO_ROOT / "scripts/benchmarks/checkpoints/darcy_physics/frames.npz"


def predict_test_set(
    metrics: dict, checkpoint: Path, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(test_a, pred, truth)``, each ``(n_test, s, s)``: the run's
    predictions at the released sample points, as ``train_transolver.py``
    scores them."""
    rate, cell_mode = PROTOCOL["subsample_rate"], metrics["cell_mode"]
    split = load_split(
        data_dir,
        dict(
            PROTOCOL,
            n_train=metrics["n_train_released"],
            n_test=metrics["n_test"],
            subsample_rate=1,
        ),
    )
    # Only the training coefficients are needed: they fix the input
    # normaliser (fitted on everything the run trained on, released + sampled).
    train_a, _ = split_coefficient(split["train_a"], rate, cell_mode)
    a_lo, a_hi = float(split["train_a"].min()), float(split["train_a"].max())
    if metrics["extra_coefficients"] > 0:
        extra_a, _ = sample_extra_coefficients(
            metrics["extra_coefficients"], split["train_a"].shape[1], rate, cell_mode
        )
        train_a = np.concatenate([train_a.astype(np.float32), extra_a])
    test_a, _ = split_coefficient(split["test_a"], rate, cell_mode)
    test_u = split["test_u"][:, ::rate, ::rate]
    n_side = test_a.shape[1]
    n_points = n_side * n_side

    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, a_lo, a_hi)
    boundary_factor = jnp.asarray(sin_boundary_factor(xy), dtype=jnp.float32)
    to_points = jnp.asarray(cell_centre_interpolation(n_side, rate), jnp.float32)

    normalizer = GlobalNormalizer(train_a[..., None])
    test_x = normalizer.encode(test_a[..., None]).reshape(-1, n_points, 1)

    model = build_model(
        n_side,
        n_hidden=metrics["n_hidden"],
        n_layers=metrics["n_layers"],
        n_head=metrics["n_head"],
        slice_num=metrics["slice_num"],
        ref=metrics["ref"],
    )
    template = model.init(
        jax.random.PRNGKey(0),
        x=jnp.zeros((1, n_points, 2), dtype=jnp.float32),
        fx=jnp.asarray(test_x[:1]),
    )
    params = load_params(template, checkpoint)
    predict_nodes = make_predict_nodes(model, n_points, scale, offset, boundary_factor)

    @jax.jit
    def predict_points(p, x):
        grid = predict_nodes(p, x).reshape(-1, n_side, n_side)
        return jnp.einsum(
            "ij,bjk,lk->bil", to_points, grid, to_points, precision="highest"
        )

    pred = predict_in_batches(
        predict_points, params, jnp.asarray(test_x), metrics["batch_size"]
    )
    return test_a, np.asarray(pred), test_u


def pick_cases(errors: np.ndarray, n: int) -> list[int]:
    """``n`` test cases evenly spread from the lowest to the highest error."""
    order = np.argsort(errors)
    ranks = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
    return [int(order[r]) for r in ranks]


def _sci_formatter() -> ScalarFormatter:
    """Colour-bar ticks as short mantissas with one ``x 10^n`` label."""
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    return formatter


def save_gif(
    case_ids: np.ndarray,
    test_a: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    errors: np.ndarray,
    out: Path,
    fps: float,
) -> None:
    """One frame per case; every array holds only the animated cases, in
    frame order (``(n, s, s)`` fields, ``(n,)`` ids and errors)."""
    err = np.abs(pred - truth)
    u_levels = np.linspace(
        min(pred.min(), truth.min()), max(pred.max(), truth.max()), 41
    )
    e_levels = np.linspace(0.0, float(err.max()), 41)
    a_levels = np.linspace(float(test_a.min()), float(test_a.max()), 3)
    # The same panel names and colour-bar labels as the data-driven Darcy GIF.
    panels = (
        ("Coefficient", test_a, "cividis", a_levels, "a(x)", "%g"),
        ("Prediction", pred, "viridis", u_levels, "pressure u", "%g"),
        ("Ground truth", truth, "viridis", u_levels, "pressure u", "%g"),
        ("Abs. error", err, "magma", e_levels, "|u error|", _sci_formatter()),
    )

    with plt.rc_context({"font.size": 15, "axes.titlesize": 18}):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for ax, (_, field, cmap, levels, label, fmt) in zip(axes, panels):
            im = ax.contourf(field[0], levels=levels, cmap=cmap)
            ticks = levels if len(levels) <= 3 else MaxNLocator(5)
            fig.colorbar(
                im, ax=ax, fraction=0.046, pad=0.04, label=label, ticks=ticks,
                format=fmt,
            )  # fmt: skip

        def draw(k: int) -> list:
            for ax, (name, field, cmap, levels, _, _) in zip(axes, panels):
                ax.clear()
                ax.contourf(field[k], levels=levels, cmap=cmap)
                ax.set_title(name)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
            fig.suptitle(
                "Darcy Flow, Model: Transolver (physics-only, no labels), "
                f"Test case: {case_ids[k]}, rel. L2: {100 * errors[k]:.2f}%",
                fontsize=18,
            )
            return []

        draw(0)
        fig.tight_layout()
        anim = FuncAnimation(fig, draw, frames=len(case_ids), blit=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
    print(f"[animate] wrote {len(case_ids)} frames to {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Final weights of the run; defaults to the path in its metrics.json.",
    )
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help="Where the animated cases are saved (not committed).",
    )
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-render from --cache only (no data, model or GPU), e.g. after a "
        "cosmetic change.",
    )
    args = ap.parse_args()

    if not args.from_cache:
        require_gpu()
        metrics_path = run_dir(BEST_MODEL, HEADLINE_FIELDS[BEST_MODEL]) / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        checkpoint = args.checkpoint or _REPO_ROOT / metrics["checkpoint_final"]
        test_a, pred, truth = predict_test_set(metrics, checkpoint, args.data_dir)

        errors = np.array([rel_l2(p, t) for p, t in zip(pred, truth)])
        reported = metrics["final_test_rel_l2_mean"]
        print(
            f"[animate] mean test rel. L2 {errors.mean():.5f} "
            f"(metrics.json: {reported:.5f})",
            flush=True,
        )
        if abs(errors.mean() - reported) > 1e-4:
            raise SystemExit(
                f"checkpoint {checkpoint} does not reproduce {metrics_path}: "
                f"{errors.mean():.5f} vs {reported:.5f}"
            )
        ids = np.array(pick_cases(errors, args.n_cases))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.cache,
            case_ids=ids,
            test_a=test_a[ids],
            pred=pred[ids],
            truth=truth[ids],
            errors=errors[ids],
        )
        print(f"[animate] cached the animated cases in {args.cache}", flush=True)

    c = np.load(args.cache)
    save_gif(
        c["case_ids"],
        c["test_a"],
        c["pred"],
        c["truth"],
        c["errors"],
        args.out,
        args.fps,
    )


if __name__ == "__main__":
    main()
