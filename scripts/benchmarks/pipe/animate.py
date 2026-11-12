# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Pipe training-progress GIF: prediction, ground truth and absolute error
on one fixed test sample, one frame per captured epoch.

Renders the ``frames.npz`` a training run saves with ``--animate``
(``train.py``) or ``--anim-frames`` (``_transolver.py``). The published GIF
is of the best model, Transolver (the current one is from an earlier
``_transolver.py`` run); to redraw it from the benchmark's run::

    python scripts/benchmarks/pipe/train.py --run transolver --seed 0 \\
        --animate --data-dir <path-to-Dataset>
    python scripts/benchmarks/pipe/animate.py \\
        --frames scripts/benchmarks/checkpoints/pipe/transolver/seed0/frames.npz

Written to ``animations/pipe/evolution.gif`` (a training diagnostic, like
``animations/cavity/``, not part of the published doc assets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "animations/pipe/evolution.gif"


def save_pipe_evolution_gif(
    frames: dict, out_path: Path, fps: int = 4, hold_last: int = 8
) -> Path:
    """Writes the GIF. ``hold_last`` repeats the final frame so the converged
    state stays on screen before the loop restarts."""
    grid = tuple(int(g) for g in frames["grid"])
    X = frames["xy"][:, 0].reshape(grid)
    Y = frames["xy"][:, 1].reshape(grid)
    truth = frames["truth_frame"].reshape(grid)
    preds = frames["pred_frames"].reshape(-1, *grid)
    errs = np.abs(preds - truth)
    epochs = [int(e) for e in frames["epochs"]]

    vmin, vmax = float(truth.min()), float(truth.max())
    # Log scale, so the error stays visible as it falls by orders of
    # magnitude; capped at the 99.5th percentile so a few early outliers
    # don't set the whole range.
    emax = float(np.percentile(errs, 99.5))
    enorm = LogNorm(vmin=emax * 1e-3, vmax=emax)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), layout="constrained")

    def panel(ax, field, title, label, **kw):
        mesh = ax.pcolormesh(X, Y, field, shading="gouraud", **kw)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02, label=label)
        return mesh

    pred_mesh = panel(
        axes[0], preds[0], "Prediction", "u", cmap="viridis", vmin=vmin, vmax=vmax
    )
    panel(axes[1], truth, "Ground truth", "u", cmap="viridis", vmin=vmin, vmax=vmax)
    err_mesh = panel(
        axes[2],
        np.maximum(errs[0], enorm.vmin),
        "Abs. error",
        "|u error| (log)",
        cmap="magma",
        norm=enorm,
    )
    suptitle = fig.suptitle("")

    def update(i: int):
        i = min(i, len(epochs) - 1)
        pred_mesh.set_array(preds[i].ravel())
        err_mesh.set_array(np.maximum(errs[i], enorm.vmin).ravel())
        suptitle.set_text(f"Pipe flow -- epoch {epochs[i] + 1}")
        return []

    update(0)
    anim = FuncAnimation(fig, update, frames=len(epochs) + hold_last, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    print(
        f"[animate] wrote {len(epochs)} frames (epochs {epochs[0] + 1}-"
        f"{epochs[-1] + 1}) to {out_path}"
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="frames.npz saved by a training run",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    save_pipe_evolution_gif(dict(np.load(args.frames)), args.out, fps=args.fps)


if __name__ == "__main__":
    main()
