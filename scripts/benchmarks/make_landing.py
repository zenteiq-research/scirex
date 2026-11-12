# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The home-page animations: one style for every benchmark.

Each clip shows the released model of one benchmark on 8 test samples, from
the lowest to the highest error: prediction, ground truth and absolute error,
on one canvas size, one font and one colour scheme. Two steps:

  export  run the trained model on the test set (GPU, data and checkpoint)
          and save the 8 samples to ``<cache>/<benchmark>.npz``
  render  draw ``doc/assets/landing/<benchmark>.mp4`` and a poster
          ``<benchmark>.png`` from the cache (CPU only, needs ffmpeg)

Usage (from the repo root)::

    python scripts/benchmarks/make_landing.py export pipe \\
        --checkpoint <CKPT_DIR> --data-dir <DATA_DIR>
    python scripts/benchmarks/make_landing.py render pipe
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

BENCHMARKS = ("darcy_physics", "pipe", "cavity", "car_cfd")
N_SAMPLES = 8
DEFAULT_CACHE = _REPO_ROOT / "scripts/benchmarks/checkpoints/landing"
DEFAULT_OUT = _REPO_ROOT / "doc/assets/landing"

# One look for every clip
WIDTH_IN, HEIGHT_IN, DPI = 12.0, 4.0, 160  # 1920 x 640 pixels
SECONDS_PER_SAMPLE = 1.5
FPS = 30  # a standard rate: browsers stall on clips of under one frame per second
FIELD_CMAP, ERROR_CMAP = "viridis", "magma"
FONT = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 15,
}


def pick(errors: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """``n`` sample indices evenly spread from the lowest to the highest error."""
    order = np.argsort(errors)
    return order[np.linspace(0, len(order) - 1, n).round().astype(int)]


# ---------------------------------------------------------------------------
# Export: the released model on the test set
# ---------------------------------------------------------------------------


def export_pipe(ckpt: Path, data_dir: Path) -> dict:
    from scripts.benchmarks.pipe._checkpoint import load_checkpoint
    from scripts.benchmarks.pipe._config import PROTOCOL
    from scripts.benchmarks.pipe._data import load_split
    from scripts.benchmarks.pipe.infer import per_sample_rel_l2, predict

    data = load_split(data_dir, PROTOCOL)
    ck = load_checkpoint(ckpt)
    pred = predict(ck, data["xy_test"], 8)
    truth = data["y_test"]
    rel = per_sample_rel_l2(pred, truth)
    ids = pick(rel)
    h, w = data["grid"]
    xy = data["xy_test"][ids].reshape(-1, h, w, 2)
    return dict(
        kind="mesh2d",
        problem="Pipe flow",
        model=ck["run"].label if hasattr(ck["run"], "label") else "Transolver",
        quantity="velocity u",
        unit="",
        n_test=len(rel),
        ids=ids,
        rel=rel[ids],
        truth=truth[ids, :, 0].reshape(-1, h, w),
        pred=pred[ids, :, 0].reshape(-1, h, w),
        X=xy[..., 0],
        Y=xy[..., 1],
    )


def export_cavity(ckpt: Path, data_dir: Path, select=None) -> dict:
    import jax

    from scripts.benchmarks._batch_eval import predict_in_batches
    from scripts.benchmarks.cavity._checkpoint import load_checkpoint
    from scripts.benchmarks.cavity._data import (
        DEFAULT_CACHE_DIR,
        TEST_CASE,
        denormalise,
        ensure_prepared,
        load_prepared,
        make_windows,
        normalise,
    )

    # data_dir holds either the prepared <case>.npz files or the raw cases
    if not (Path(data_dir) / f"{TEST_CASE}.npz").exists():
        ensure_prepared((TEST_CASE,), data_dir, DEFAULT_CACHE_DIR)
        data_dir = DEFAULT_CACHE_DIR
    grid = load_prepared(TEST_CASE, data_dir)
    ck = load_checkpoint(ckpt)
    run = ck["run"]
    X, Y = make_windows(grid, run.n_in, run.n_out)
    pred = predict_in_batches(
        jax.jit(ck["model"].apply), ck["params"], normalise(X, ck["normalizer"]), 8
    )
    pred = np.asarray(denormalise(pred, ck["normalizer"]))
    speed = lambda f: np.hypot(f[..., 1], f[..., 2])  # noqa: E731
    p, t = (
        speed(pred[:, 0] if pred.ndim == 5 else pred),
        speed(Y[:, 0] if Y.ndim == 5 else Y),
    )
    rel = np.linalg.norm((p - t).reshape(len(t), -1), axis=1) / np.linalg.norm(
        t.reshape(len(t), -1), axis=1
    )
    # time order reads better than error order for a flow that evolves
    ids = (
        np.linspace(len(t) // 10, len(t) - 1, N_SAMPLES).round().astype(int)
        if select is None
        else np.asarray(select(rel))
    )
    return dict(
        kind="grid",
        problem="Lid-driven cavity",
        model="FNO",
        quantity="speed |u|",
        unit="m/s",
        n_test=len(t),
        ids=ids,
        rel=rel[ids],
        rel_all=rel,
        truth=t[ids],
        pred=p[ids],
        sample_word="time step",
    )


def export_darcy_physics(ckpt: Path, data_dir: Path) -> dict:
    import json

    from scripts.benchmarks.darcy_physics.animate import (
        BEST_MODEL,
        HEADLINE_FIELDS,
        predict_test_set,
        run_dir,
    )

    metrics = json.loads(
        (run_dir(BEST_MODEL, HEADLINE_FIELDS[BEST_MODEL]) / "metrics.json").read_text()
    )
    _, pred, truth = predict_test_set(metrics, ckpt, data_dir)
    pred, truth = np.asarray(pred), np.asarray(truth)
    rel = np.linalg.norm(
        (pred - truth).reshape(len(truth), -1), axis=1
    ) / np.linalg.norm(truth.reshape(len(truth), -1), axis=1)
    ids = pick(rel)
    return dict(
        kind="grid",
        problem="Darcy flow, no labels",
        model="Transolver",
        quantity="pressure",
        unit="",
        n_test=len(rel),
        ids=ids,
        rel=rel[ids],
        truth=truth[ids],
        pred=pred[ids],
    )


def export_car_cfd(ckpt: Path, data_dir: Path, select=None) -> dict:
    from scripts.benchmarks.car_cfd._data import load_test_mesh
    from scripts.benchmarks.car_cfd.animate import predict_cars
    from scripts.benchmarks.car_cfd.infer import load_run

    cars = predict_cars(load_run(ckpt, data_dir, False))
    rel = np.array([c["rel"] for c in cars])
    ids = pick(rel) if select is None else np.asarray(select(rel))
    out = dict(
        kind="surface",
        problem="Car aerodynamics",
        model="Transolver",
        quantity="surface pressure",
        unit="Pa",
        n_test=len(rel),
        ids=ids,
        rel=rel[ids],
        rel_all=rel,
        sample_word="test car",
    )
    for k, i in enumerate(ids):
        v, t = load_test_mesh(data_dir, int(i))
        out[f"v{k}"], out[f"t{k}"] = v, t
        out[f"truth{k}"], out[f"pred{k}"] = (
            np.ravel(cars[i]["truth"]),
            np.ravel(cars[i]["pred"]),
        )
    return out


EXPORT = dict(
    pipe=export_pipe,
    cavity=export_cavity,
    darcy_physics=export_darcy_physics,
    car_cfd=export_car_cfd,
)


# ---------------------------------------------------------------------------
# Render: one canvas, one font, one colour scheme
# ---------------------------------------------------------------------------


def _sample(c: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth and prediction of stored sample ``k``; meshes that differ per
    sample are stored under numbered keys (``truth0``, ``v0``, ...)."""
    if f"truth{k}" in c:
        return c[f"truth{k}"], c[f"pred{k}"]
    return c["truth"][k], c["pred"][k]


def _draw(ax, c: dict, k: int, values: np.ndarray, cmap: str, vmin: float, vmax: float):
    kind = c["kind"]
    if kind == "grid":
        return ax.imshow(
            values,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
    if kind == "mesh2d":
        return ax.pcolormesh(
            c["X"][k],
            c["Y"][k],
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="gouraud",
        )
    if kind == "tri2d":
        from matplotlib.tri import Triangulation

        key = lambda n: f"{n}{k}" if f"{n}{k}" in c else n  # noqa: E731
        tri = Triangulation(c[key("x")], c[key("y")], c[key("tri")])
        m = ax.tripcolor(
            tri,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="gouraud",
            rasterized=True,
        )
        if key("outline") in c:  # a solid body inside the domain (the airfoil)
            o = c[key("outline")]
            ax.fill(o[:, 0], o[:, 1], color="0.3", zorder=3)
        if "window" in c:
            w = c["window"]
            ax.set_xlim(w[0], w[1])
            ax.set_ylim(w[2], w[3])
        return m
    from matplotlib.colors import Normalize

    v, t = c[f"v{k}"], c[f"t{k}"]
    surf = ax.plot_trisurf(
        v[:, 0],
        v[:, 2],
        v[:, 1],
        triangles=t,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    norm = Normalize(vmin, vmax)
    surf.set_facecolor(
        __import__("matplotlib").colormaps[cmap](norm(values[t].mean(axis=1)))
    )
    ax.view_init(elev=22, azim=-125)
    ax.set_box_aspect(np.ptp(v[:, [0, 2, 1]], axis=0), zoom=1.45)
    ax.set_axis_off()
    import matplotlib.cm as cm

    return cm.ScalarMappable(norm=norm, cmap=cmap)


def _exponent(x: float) -> int:
    """Power of ten to factor out of a colour bar, 0 when the numbers read well as they are."""
    if x == 0 or 1e-2 <= x < 1e4:
        return 0
    return int(np.floor(np.log10(x)))


def is_elongated(c: dict) -> bool:
    """A domain much wider than tall (the pipe), drawn as stacked rows."""
    if c["kind"] == "mesh2d":
        return bool(np.ptp(c["X"]) > 2.5 * np.ptp(c["Y"]))
    return False


def draw_panel(
    fig,
    ax,
    cax,
    c: dict,
    k: int,
    title: str,
    values,
    cmap: str,
    vmin: float,
    vmax: float,
    side_label: bool = False,
) -> None:
    """One field on one panel with its colour bar, in the common style."""
    e = _exponent(max(abs(vmin), abs(vmax)))
    s = 10.0**-e
    m = _draw(ax, c, k, values * s, cmap, vmin * s, vmax * s)
    if side_label and e:
        # rows stacked one above the other: a scale on top of a colour bar would sit
        # on the lowest tick of the bar above, so the scale goes in the row label
        title = f"{title}\n(\u00d710$^{{{e}}}$)"
    if side_label:
        ax.text(
            -0.01,
            0.5,
            title,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=FONT["axes.titlesize"],
        )
    else:
        ax.set_title(title, pad=6)
    if c["kind"] != "surface":
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # the colour bar spans exactly the drawn panel
        ax.apply_aspect()
        pos = ax.get_position()
        cax.set_position([pos.x1 + 0.012, pos.y0, 0.008, pos.height])
    cb = fig.colorbar(m, cax=cax)
    cb.ax.tick_params(labelsize=10)
    # the factored-out power of ten sits on top of the colour bar
    if e and not side_label:
        cb.ax.set_title(f"×10$^{{{e}}}$", fontsize=10, pad=4, loc="left")


def field_range(*fields) -> tuple[float, float]:
    """Colour range of the fields: the 0.1-99.9 percentiles, so a few extreme
    values (a leading edge, a corner) do not wash out the rest."""
    v = np.concatenate([np.ravel(f) for f in fields])
    return float(np.percentile(v, 0.1)), float(np.percentile(v, 99.9))


def error_max(err: np.ndarray) -> float:
    """Top of the error colour scale: ignores the largest 0.5% (a lid corner, a sharp edge)."""
    return float(np.percentile(err, 99.5)) or float(err.max()) or 1.0


def title_of(c: dict, k: int) -> str:
    unit = f" ({c['unit']})" if c.get("unit") else ""
    word = c.get("sample_word", "test sample")
    return (
        f"{c['problem']}  ·  {c['model']}  ·  {c['quantity']}{unit}  ·  "
        f"{word} {int(c['ids'][k]) + 1} of {c['n_test']}  ·  relative L2 {100 * c['rel'][k]:.2f}%"
    )


def canvas(c: dict):
    """The 1920 x 640 figure with its three panels and colour bars."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(FONT)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, facecolor="white")
    proj = "3d" if c["kind"] == "surface" else None
    if is_elongated(c):
        # three rows instead of three columns
        axes = [fig.add_axes([0.12, 0.615 - 0.295 * i, 0.74, 0.27]) for i in range(3)]
        caxes = [fig.add_axes([0.88, 0.635 - 0.295 * i, 0.008, 0.19]) for i in range(3)]
    else:
        axes = [
            fig.add_axes([0.02 + 0.33 * i, 0.06, 0.25, 0.76], projection=proj)
            for i in range(3)
        ]
        caxes = [fig.add_axes([0.28 + 0.33 * i, 0.12, 0.008, 0.64]) for i in range(3)]
    return fig, axes, caxes


def draw_sample(
    fig, axes, caxes, c: dict, k: int, lo: float | None = None, hi: float | None = None
) -> None:
    """Prediction, ground truth and absolute error of sample ``k``; the field
    scale is ``lo``-``hi`` (default: this sample's range)."""
    truth, pred = _sample(c, k)
    err = np.abs(pred - truth)
    if lo is None or hi is None:
        lo, hi = field_range(truth, pred)
    for ax, cax in zip(axes, caxes):
        ax.clear()
        cax.clear()
    panels = [
        ("Prediction", pred, FIELD_CMAP, lo, hi),
        ("Ground truth", truth, FIELD_CMAP, lo, hi),
        ("Absolute error", err, ERROR_CMAP, 0.0, error_max(err)),
    ]
    for (title, values, cmap, vmin, vmax), ax, cax in zip(panels, axes, caxes):
        draw_panel(
            fig,
            ax,
            cax,
            c,
            k,
            title,
            values,
            cmap,
            vmin,
            vmax,
            side_label=is_elongated(c),
        )
    fig.suptitle(title_of(c, k), y=0.965)


def load_cache(path: Path) -> dict:
    c = dict(np.load(path, allow_pickle=False))
    return {k: (v.item() if v.ndim == 0 else v) for k, v in c.items()}


def render(name: str, cache: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    c = load_cache(cache / f"{name}.npz")
    n = len(c["ids"])
    lo = min(min(_sample(c, k)[0].min(), _sample(c, k)[1].min()) for k in range(n))
    hi = max(max(_sample(c, k)[0].max(), _sample(c, k)[1].max()) for k in range(n))
    fig, axes, caxes = canvas(c)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(out_dir / f"{name}.mp4"), dpi=DPI):
        for k in range(n):
            draw_sample(fig, axes, caxes, c, k, lo, hi)
            for _ in range(round(SECONDS_PER_SAMPLE * FPS)):  # hold the sample
                writer.grab_frame(facecolor="white")
    draw_sample(fig, axes, caxes, c, n // 2, lo, hi)
    fig.savefig(out_dir / f"{name}.png", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"[landing] {out_dir / name}.mp4 and .png", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=("export", "render"))
    ap.add_argument("benchmark", choices=BENCHMARKS + ("all",))
    ap.add_argument(
        "--checkpoint", type=Path, help="released checkpoint directory (export)"
    )
    ap.add_argument("--data-dir", type=Path, help="dataset folder (export)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    names = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    for name in names:
        if args.step == "export":
            import jax

            jax.config.update("jax_enable_x64", False)
            c = EXPORT[name](args.checkpoint, args.data_dir)
            args.cache.mkdir(parents=True, exist_ok=True)
            np.savez(args.cache / f"{name}.npz", **c)
            print(
                f"[landing] {name}: mean rel. L2 of the shown samples {np.mean(c['rel']):.4f}",
                flush=True,
            )
        else:
            render(name, args.cache, args.out)


if __name__ == "__main__":
    main()
