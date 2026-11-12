# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""AirfRANS animation: the best model's prediction across the test cases.

Loads the reported Transolver checkpoint (the best AirfRANS run, see
``doc/benchmarks/airfrans/results/comparative.md``) and predicts ``--n-cases`` test
cases spread evenly over the angle of attack, with the official inference
(random 32,000-node subsets covering every node). Each frame shows one case
around the airfoil: prediction, ground truth and absolute error of the
velocity magnitude ``|U| / U_inf``, on one colour scale per panel for the
whole animation.

Written to ``animations/airfrans/evolution.gif`` (a diagnostic, like the
other ``animations/`` folders, not part of the published doc assets).

Usage (from the repo root)::

    export AIRFRANS_DATA_DIR=/path/to/AirfRANS/Dataset
    python scripts/benchmarks/airfrans/animate.py
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.airfrans._checkpoint import load_checkpoint  # noqa: E402
from scripts.benchmarks.airfrans._config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
)
from scripts.benchmarks.airfrans._data import (  # noqa: E402
    load_case,
    load_split,
    prepare_case,
    to_physical,
)
from scripts.benchmarks.airfrans._metrics import (  # noqa: E402
    CaseGeometry,
    cover_predict,
    impose_wall,
)

jax.config.update("jax_enable_x64", False)

BEST_RUN = "transolver"
WINDOW = (-0.5, 1.5, -0.45, 0.45)  # view around the airfoil (chords), as infer.py
DEFAULT_OUT = _REPO_ROOT / "animations/airfrans/evolution.gif"


def pick_cases(test: list[str], n: int) -> list[str]:
    """``n`` test cases evenly spread over the angle of attack, in its order."""
    by_alpha = sorted(test, key=lambda cid: float(cid.split("_")[3]))
    idx = np.unique(np.round(np.linspace(0, len(by_alpha) - 1, n)).astype(int))
    return [by_alpha[i] for i in idx]


def triangulation(geo: CaseGeometry, pos: np.ndarray) -> Triangulation:
    """The case's quad mesh split into triangles, kept around the window."""
    from vtk.util.numpy_support import vtk_to_numpy

    quads = vtk_to_numpy(geo.internal.GetCells().GetConnectivityArray()).reshape(-1, 4)
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    c = pos[tri].mean(1)
    keep = (
        (c[:, 0] > WINDOW[0] - 0.1)
        & (c[:, 0] < WINDOW[1] + 0.1)
        & (np.abs(c[:, 1]) < WINDOW[3] + 0.1)
    )
    return Triangulation(pos[:, 0], pos[:, 1], tri[keep])


def predict_cases(ck: dict, cases: list[str], data_dir: Path, cache_dir: Path):
    spec, model, params, stats = ck["spec"], ck["model"], ck["params"], ck["stats"]
    apply = jax.jit(lambda p, x, pos, extra: spec.apply(model, p, x, pos, extra))
    rng = np.random.default_rng(0)
    frames = []
    for j, cid in enumerate(cases):
        raw = load_case(data_dir, cache_dir, cid)
        d = prepare_case(raw, stats)
        extra = spec.extra(ck["run"].arch, raw)
        pred_n, _ = cover_predict(
            lambda idx, d=d, extra=extra: apply(
                params, jnp.asarray(d["x"][idx]), jnp.asarray(d["pos"][idx]), extra
            ),
            len(d["x"]),
            PROTOCOL["subsample"],
            rng,
        )
        pred = impose_wall(to_physical(pred_n, d["scale"], stats), raw["surf"])
        geo = CaseGeometry(data_dir, cid)
        u_inf = geo.u_inf
        frames.append(
            dict(
                tri=triangulation(geo, raw["pos"]),
                outline=geo.outline(),
                truth=np.linalg.norm(raw["y"][:, :2], axis=1) / u_inf,
                pred=np.linalg.norm(pred[:, :2], axis=1) / u_inf,
                u_inf=u_inf,
                alpha=geo.alpha,
            )
        )
        print(f"  {j + 1}/{len(cases)} {cid}", flush=True)
    return frames


def in_window(frame: dict, values: np.ndarray) -> np.ndarray:
    x, y = frame["tri"].x, frame["tri"].y
    inside = (x > WINDOW[0]) & (x < WINDOW[1]) & (y > WINDOW[2]) & (y < WINDOW[3])
    return values[inside]


def save_gif(frames: list[dict], out: Path, fps: float) -> None:
    # One colour scale per panel type for the whole animation, at the 99.5th
    # percentile inside the view: a few nodes at the leading-edge suction peak
    # would otherwise wash out every frame. Colour bars mark the clipping.
    vmax = float(
        np.percentile(
            np.concatenate(
                [in_window(f, f[k]) for f in frames for k in ("truth", "pred")]
            ),
            99.5,
        )
    )
    emax = float(
        np.percentile(
            np.concatenate(
                [in_window(f, np.abs(f["pred"] - f["truth"])) for f in frames]
            ),
            99.5,
        )
    )
    # Three equal panels; each colour bar gets its own narrow column, with a
    # spacer so the first bar's labels clear the error panel.
    fig = plt.figure(figsize=(15.5, 3.2))
    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[1, 1, 0.035, 0.22, 1, 0.035],
        wspace=0.08,
        left=0.01,
        right=0.94,
        top=0.8,
        bottom=0.04,
    )
    axes = [fig.add_subplot(gs[0, k]) for k in (0, 1, 4)]
    sm_u = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, vmax))
    sm_e = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, emax))
    fig.colorbar(sm_u, cax=fig.add_subplot(gs[0, 2]), extend="max").set_label(
        "|U| / U$_\\infty$"
    )
    fig.colorbar(sm_e, cax=fig.add_subplot(gs[0, 5]), extend="max").set_label(
        "|error| / U$_\\infty$"
    )
    title = fig.suptitle("")

    def draw(i: int):
        f = frames[i]
        panels = [
            ("Prediction", f["pred"], "viridis", vmax),
            ("Ground truth", f["truth"], "viridis", vmax),
            ("Abs. error", np.abs(f["pred"] - f["truth"]), "magma", emax),
        ]
        for ax, (name, v, cmap, hi) in zip(axes, panels, strict=True):
            ax.clear()
            ax.tripcolor(
                f["tri"],
                v,
                shading="gouraud",
                cmap=cmap,
                vmin=0,
                vmax=hi,
                rasterized=True,
            )
            ax.fill(f["outline"][:, 0], f["outline"][:, 1], color="0.25", zorder=3)
            ax.set_xlim(WINDOW[0], WINDOW[1])
            ax.set_ylim(WINDOW[2], WINDOW[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name)
        title.set_text(
            f"AirfRANS -- test case {i + 1}/{len(frames)}: angle of attack "
            f"{f['alpha']:+.1f}°, U$_\\infty$ = {f['u_inf']:.1f} m/s"
        )
        return []

    draw(0)
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"[animate] wrote {len(frames)} frames to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is an error",
    )
    args = ap.parse_args()
    require_gpu(args.allow_cpu)
    jax.config.update(
        "jax_default_matmul_precision", PROTOCOL["additions"]["matmul_precision"]
    )
    ck = load_checkpoint(args.ckpt_root / BEST_RUN)
    cases = pick_cases(load_split(args.data_dir, PROTOCOL["n_val"]).test, args.n_cases)
    frames = predict_cases(ck, cases, args.data_dir, args.cache_dir)
    save_gif(frames, args.out, args.fps)


if __name__ == "__main__":
    main()
