# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-CFD animation: the best model's surface pressure across the test cars.

Loads the reported Transolver checkpoint (the best Car-CFD run, see
``doc/benchmarks/car_cfd/results/comparative.md``), scores all 111 test cars, and
shows ``--n-cars`` of them from the lowest to the highest relative L2. Each
frame is one car: prediction, ground truth and absolute error of the
de-normalised surface pressure, on one colour scale per panel for the whole
animation.

Written to ``animations/car_cfd/evolution.gif`` (a diagnostic, like the
other ``animations/`` folders, not part of the published doc assets).

Usage (from the repo root)::

    python scripts/benchmarks/car_cfd/animate.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.car_cfd._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    HEADLINE,
)
from scripts.benchmarks.car_cfd._data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    load_test_mesh,
)
from scripts.benchmarks.car_cfd._models import predict_denormalised  # noqa: E402
from scripts.benchmarks.car_cfd.infer import _draw_surface, load_run  # noqa: E402

BEST_MODEL = "transolver"
DEFAULT_OUT = _REPO_ROOT / "animations/car_cfd/evolution.gif"


def predict_cars(loaded: dict) -> list[dict]:
    """De-normalised ``truth``/``pred`` and relative L2 of every test car,
    in test-set order."""
    cars = []
    for i, sample in enumerate(loaded["test_loader"]):
        truth, pred = predict_denormalised(loaded["model"], loaded["processor"], sample)
        rel = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
        cars.append(dict(index=i, truth=truth, pred=pred, rel=rel))
    return cars


def pick_cars(cars: list[dict], n: int) -> list[dict]:
    """``n`` cars evenly spread over the error ranking, lowest error first."""
    ranked = sorted(cars, key=lambda c: c["rel"])
    idx = np.unique(np.round(np.linspace(0, len(ranked) - 1, n)).astype(int))
    return [ranked[i] for i in idx]


def save_gif(frames: list[dict], out: Path, fps: float, label: str) -> None:
    # One colour scale per panel type for the whole animation, clipped at the
    # 0.5th/99.5th percentiles so a few stagnation-point vertices do not wash
    # out every frame. Colour bars mark the clipping.
    values = np.concatenate([f[k] for f in frames for k in ("truth", "pred")])
    lo, hi = np.percentile(values, [0.5, 99.5])
    emax = np.percentile(
        np.concatenate([np.abs(f["pred"] - f["truth"]) for f in frames]), 99.5
    )
    p_norm, e_norm = Normalize(lo, hi), Normalize(0.0, emax)

    fig = plt.figure(figsize=(15, 4.6))
    axes = [fig.add_subplot(1, 3, k + 1, projection="3d") for k in range(3)]
    for norm, cmap, ax, name in (
        (p_norm, "jet", axes[:2], "Pressure"),
        (e_norm, "magma", axes[2:], "|Pressure error|"),
    ):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, fraction=0.025, shrink=0.7, extend="both", label=name)
    title = fig.suptitle("")

    def draw(i: int):
        f = frames[i]
        pts = f["vertices"][:, [0, 2, 1]]  # (width, length, height): upright
        panels = [
            ("Prediction", f["pred"], "jet", p_norm),
            ("Ground truth", f["truth"], "jet", p_norm),
            ("Abs. error", np.abs(f["pred"] - f["truth"]), "magma", e_norm),
        ]
        for ax, (name, v, cmap, norm) in zip(axes, panels, strict=True):
            ax.clear()
            _draw_surface(ax, pts, f["triangles"], v, cmap, norm)
            ax.set_title(name)
        title.set_text(
            f"Car-CFD, Model: {label}, test car {f['index'] + 1} of 111, "
            f"relative L2 {f['rel']:.2%} ({i + 1}/{len(frames)}, "
            "lowest to highest error)"
        )
        return axes

    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--n-cars", type=int, default=12)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()

    require_gpu(args.allow_cpu)
    run_dir = args.ckpt_root / HEADLINE[BEST_MODEL]
    loaded = load_run(run_dir, args.data_root, not args.no_download)
    cars = predict_cars(loaded)
    print(
        f"[{run_dir.name}] mean test relative L2 "
        f"{np.mean([c['rel'] for c in cars]):.4%} over {len(cars)} cars",
        flush=True,
    )
    frames = pick_cars(cars, args.n_cars)
    for f in frames:
        f["vertices"], f["triangles"] = load_test_mesh(args.data_root, f["index"])
    save_gif(frames, args.out, args.fps, "Transolver")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
