# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Navier-Stokes rollout GIF: ground truth, prediction and absolute error on
one test sample, one frame per predicted time step.

Each prediction is fed back as the model's next input, so the GIF shows how
the error grows along the 10-step rollout. The published GIF is of the best
model, Transolver, on the test sample with its median error::

    python scripts/benchmarks/navier_stokes/infer.py     # writes metrics.json
    python scripts/benchmarks/navier_stokes/animate.py --run transolver

Written to ``animations/navier_stokes/evolution.gif`` (like
``animations/chip_cooling/``, it shows the trained model on test data, not
the training progress).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.navier_stokes._checkpoint import load_checkpoint  # noqa: E402
from scripts.benchmarks.navier_stokes._config import (  # noqa: E402
    DATA_FILE,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    MODEL_LABELS,
    PROTOCOL,
    RUNS,
)
from scripts.benchmarks.navier_stokes._data import load_split  # noqa: E402
from scripts.benchmarks.navier_stokes.infer import predict  # noqa: E402

DEFAULT_OUT = _REPO_ROOT / "animations/navier_stokes/evolution.gif"


def save_rollout_gif(
    truth: np.ndarray,
    pred: np.ndarray,
    grid: tuple[int, int],
    t_in: int,
    title: str,
    out_path: Path,
    fps: int = 2,
    hold_last: int = 3,
) -> Path:
    """Writes the GIF. ``truth`` and ``pred`` are ``(H * W, t_out)``;
    ``hold_last`` repeats the final frame before the loop restarts."""
    t = truth.reshape(*grid, -1)
    p = pred.reshape(*grid, -1)
    err = np.abs(p - t)
    n = t.shape[-1]
    vmax = float(np.abs(t).max())
    emax = float(err.max())
    rel = np.linalg.norm((p - t).reshape(-1, n), axis=0) / np.linalg.norm(
        t.reshape(-1, n), axis=0
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    panels = [
        ("Ground truth", t, "RdBu_r", -vmax, vmax, "vorticity"),
        ("Prediction", p, "RdBu_r", -vmax, vmax, "vorticity"),
        ("Abs. error", err, "magma", 0.0, emax, "|error|"),
    ]
    images = []
    for ax, (name, field, cmap, lo, hi, label) in zip(axes, panels, strict=True):
        im = ax.imshow(
            field[..., 0].T,
            origin="lower",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="bicubic",
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label=label)
        images.append((im, field))
    suptitle = fig.suptitle("")

    def update(i: int):
        i = min(i, n - 1)
        for im, field in images:
            im.set_data(field[..., i].T)
        suptitle.set_text(
            f"{title} -- t = {t_in + i + 1} (step {i + 1}/{n}), "
            f"relative L2 {rel[i]:.4f}"
        )
        return []

    update(0)
    anim = FuncAnimation(fig, update, frames=n + hold_last, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"[animate] wrote {n} frames (t = {t_in + 1}-{t_in + n}) to {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", default="transolver", choices=list(RUNS))
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="test sample; default: the run's median-error sample from its "
        "metrics.json",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--metrics-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()

    sample = args.sample
    if sample is None:
        metrics = args.metrics_dir / f"navier_stokes_{args.run}" / "metrics.json"
        sample = json.loads(metrics.read_text())["contour_sample"]

    require_gpu(args.allow_cpu)
    data = load_split(args.data_dir / DATA_FILE, PROTOCOL)
    ck = load_checkpoint(args.ckpt_root / args.run)
    a = data["a_test"][sample : sample + 1]
    pred = predict(ck, a, PROTOCOL["t_out"], batch_size=1)[0]
    save_rollout_gif(
        data["u_test"][sample],
        pred,
        ck["grid"],
        PROTOCOL["t_in"],
        f"Navier-Stokes, {MODEL_LABELS[RUNS[args.run].model]}, test sample {sample}",
        args.out,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
