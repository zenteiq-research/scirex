# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Chip-cooling animation: the best model's prediction across the test set.

Loads the FNO checkpoint (the best chip-cooling model, see
``doc/benchmarks/chip_cooling/results/comparative.md``), predicts the test set and
shows ``--n-samples`` test samples spread evenly over the total chip heat
flux. Each frame shows one sample on the in-domain cells of the 128x256 grid:
prediction, ground truth and absolute error of the temperature, on one colour
scale per panel for the whole animation.

Written to ``animations/chip_cooling/evolution.gif`` (a diagnostic, like the
other ``animations/`` folders, not part of the published doc assets).

Usage (from the repo root)::

    export CHIPCOOLING_DATA_DIR=/path/to/1_Dataset
    python scripts/benchmarks/chip_cooling/animate.py
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

from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.chip_cooling._config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
)
from scripts.benchmarks.chip_cooling._data import load_raw  # noqa: E402
from scripts.benchmarks.chip_cooling.infer import load_run  # noqa: E402

BEST_RUN = "fno"

DEFAULT_OUT = _REPO_ROOT / "animations/chip_cooling/evolution.gif"
FLUX = ("chip1.thermal.heat_flux", "chip2.thermal.heat_flux")
T_IN = ("inlet1.thermal.total_temperature", "inlet2.thermal.total_temperature")


def pick_samples(total_flux: np.ndarray, n: int) -> list[int]:
    """``n`` test samples evenly spread over the total heat flux, in its order."""
    order = np.argsort(total_flux)
    idx = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
    return [int(order[i]) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()

    require_gpu(args.allow_cpu)
    raw = load_raw(args.data_dir)
    ck = load_run(args.ckpt_root / BEST_RUN, raw, PROTOCOL, args.cache_dir)
    data = ck["data"]
    pred = ck["spec"].predict(ck["model"], ck["params"], ck["prepared"], data)[:, 0]
    truth = data["Y_test"][:, 0]
    mask = data["mask"]
    y_mm = data["y_grid"] * 1000
    z_mm = data["z_grid"] * 1000
    parameters, names = raw["parameters"], raw["names"]
    params = parameters[-PROTOCOL["ntest"] :]
    col = {name: k for k, name in enumerate(names)}
    flux = params[:, [col[f] for f in FLUX]]
    t_in = params[:, [col[t] for t in T_IN]]
    samples = pick_samples(flux.sum(1), args.n_samples)

    Zg, Yg = np.meshgrid(z_mm, y_mm)
    zs, ys = Zg[mask], Yg[mask]
    t_lo = float(min(truth[:, mask].min(), pred[:, mask].min()))
    t_hi = float(max(truth[:, mask].max(), pred[:, mask].max()))
    # One error scale for the whole animation, at the 99.5th percentile of the
    # shown samples; the colour bar marks the clipping.
    emax = float(np.percentile(np.abs(pred - truth)[samples][:, mask], 99.5))

    # Three equal panels; each colour bar gets its own narrow column, with a
    # spacer so the first bar's labels clear the error panel.
    fig = plt.figure(figsize=(16, 2.9))
    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 0.025, 0.2, 1, 0.025],
        wspace=0.08,
        left=0.01,
        right=0.95,
        top=0.78,
        bottom=0.04,
    )
    axes = [fig.add_subplot(gs[0, k]) for k in (0, 1, 4)]
    # The in-domain grid cells as points, as in the static FNO figure, sized to
    # close the gaps between them at this figure size.
    kw = dict(marker="s", s=11, linewidths=0)
    first = samples[0]
    scs = [
        axes[0].scatter(
            zs, ys, c=pred[first][mask], cmap="jet", vmin=t_lo, vmax=t_hi, **kw
        ),
        axes[1].scatter(
            zs, ys, c=truth[first][mask], cmap="jet", vmin=t_lo, vmax=t_hi, **kw
        ),
        axes[2].scatter(
            zs,
            ys,
            c=np.abs(pred[first] - truth[first])[mask],
            cmap="magma",
            vmin=0,
            vmax=emax,
            **kw,
        ),
    ]
    for ax, name in zip(
        axes, ("Prediction", "Ground truth", "Abs. error"), strict=True
    ):
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(z_mm[0], z_mm[-1])
        ax.set_ylim(y_mm[0], y_mm[-1])
    fig.colorbar(scs[0], cax=fig.add_subplot(gs[0, 2])).set_label("T (K)")
    fig.colorbar(scs[2], cax=fig.add_subplot(gs[0, 5]), extend="max").set_label(
        "|error| (K)"
    )
    title = fig.suptitle("")

    def draw(i: int):
        s = samples[i]
        scs[0].set_array(pred[s][mask])
        scs[1].set_array(truth[s][mask])
        scs[2].set_array(np.abs(pred[s] - truth[s])[mask])
        title.set_text(
            f"Chip cooling -- test sample {s} ({i + 1}/{len(samples)}): chip heat "
            f"flux {flux[s, 0] / 1e6:.2f} / {flux[s, 1] / 1e6:.2f} MW/m², inlet "
            f"{t_in[s, 0]:.0f} / {t_in[s, 1]:.0f} K"
        )
        return [*scs, title]

    draw(0)
    anim = FuncAnimation(fig, draw, frames=len(samples), blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=90)
    plt.close(fig)
    print(f"[animate] wrote {len(samples)} frames to {args.out}")


if __name__ == "__main__":
    main()
