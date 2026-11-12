# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Training-progress GIF: prediction (varies per epoch), ground truth
(constant -- the same validation sample throughout) and absolute error
(varies per epoch) for one fixed held-out sample.

Used by ``train.py`` scripts to show how a run's prediction converges
during training, as opposed to ``_plot_utils.py``'s static before/after
figures. Written to ``animations/<benchmark>/<run>/`` at the repo root (not
``doc/assets/``: these are a training-diagnostic byproduct, not part of the
published docs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _default_field(frame: np.ndarray) -> np.ndarray:
    """Velocity magnitude from a ``[p, u, v]`` channel-last frame."""
    return np.hypot(frame[..., 1], frame[..., 2])


def log_spaced_epochs(n_epochs: int, n_frames: int) -> list[int]:
    """Zero-based epochs at which to capture a frame: roughly ``n_frames``
    of them, log-spaced so the fast early learning gets most of the frames,
    always ending at the last epoch. Duplicates at the low end are dropped,
    so fewer than ``n_frames`` may come back.
    """
    epochs = np.round(np.geomspace(1, n_epochs, n_frames)).astype(int) - 1
    return [int(e) for e in np.unique(epochs)]


def save_prediction_evolution_gif(
    pred_frames: list[np.ndarray],
    truth_frame: np.ndarray,
    epochs: list[int],
    out_path: Path,
    title: str = "",
    fps: int = 12,
    field_fn=None,
    value_label: str = "|u| (m/s)",
    error_label: str = "|u error| (m/s)",
    title_template: str | None = None,
) -> Path:
    """Writes an animated GIF with three panels: prediction, ground truth,
    absolute error -- all sharing one colour scale per panel type so the
    convergence is visually comparable across frames.

    Parameters
    ----------
    pred_frames : one array per captured epoch, shape ``(H, W, C)``.
    truth_frame : the same fixed sample's ground truth, shape ``(H, W, C)``
        -- constant across the animation.
    epochs : the epoch number each entry of ``pred_frames`` was captured at
        (same length as ``pred_frames``), shown in the title per frame.
    field_fn : reduces a ``(H, W, C)`` frame to the 2D scalar field to plot.
        Defaults to velocity magnitude (``[p, u, v]`` channels, as the
        lid-driven-cavity benchmark uses) -- pass an identity function for
        an already-scalar field.
    value_label, error_label : colourbar labels of the prediction and error
        panels. Default to the lid-driven cavity's velocity magnitude.
    title_template : per-frame title with an ``{epoch}`` field, e.g.
        ``"Darcy Flow, Model: Transolver, Epoch: {epoch}"``. When omitted the
        title is ``f"{title} -- epoch {epoch}"``.
    """
    import matplotlib.pyplot as plt

    if field_fn is None:
        field_fn = _default_field

    truth2d = field_fn(truth_frame)
    pred2d = [field_fn(p) for p in pred_frames]
    err2d = [np.abs(p - truth2d) for p in pred2d]

    vmax = float(max(truth2d.max(), max(p.max() for p in pred2d)))
    emax = float(max(e.max() for e in err2d)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    im_pred = axes[0].imshow(
        pred2d[0], origin="lower", vmin=0, vmax=vmax, cmap="viridis"
    )
    axes[0].set_title("Prediction")
    axes[1].imshow(truth2d, origin="lower", vmin=0, vmax=vmax, cmap="viridis")
    axes[1].set_title("Ground truth")
    im_err = axes[2].imshow(err2d[0], origin="lower", vmin=0, vmax=emax, cmap="magma")
    axes[2].set_title("Abs. error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_pred, ax=axes[0], fraction=0.046, label=value_label)
    fig.colorbar(im_err, ax=axes[2], fraction=0.046, label=error_label)
    if title_template is None:
        escaped = title.replace("{", "{{").replace("}", "}}")
        title_template = f"{escaped} -- epoch {{epoch}}"
    suptitle = fig.suptitle(title_template.format(epoch=epochs[0]))
    fig.tight_layout()

    def update(i):
        im_pred.set_data(pred2d[i])
        im_err.set_data(err2d[i])
        suptitle.set_text(title_template.format(epoch=epochs[i]))
        return im_pred, im_err, suptitle

    anim = FuncAnimation(fig, update, frames=len(pred2d), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(
        f"[animate] wrote {len(pred2d)} frames "
        f"(epochs {epochs[0]}-{epochs[-1]}) to {out_path}"
    )
    return out_path


def save_point_cloud_evolution_gif(
    pos: np.ndarray,
    pred_frames: list[np.ndarray],
    truth: np.ndarray,
    epochs: list[int],
    out_path: Path,
    title: str = "",
    fps: int = 2,
    cmap: str = "viridis",
    levels: int = 50,
) -> Path:
    """Point-cloud analogue of :func:`save_prediction_evolution_gif`: a 3-panel
    GIF (prediction / ground truth / abs error) for a scalar field with no
    grid structure (e.g. ShapeNetCar surface pressure), via a 2D
    ``tricontourf`` projection of ``pos``'s first two columns -- same
    rendering convention as ``_plot_utils.plot_point_cloud_scalar_field``.

    Unlike ``save_prediction_evolution_gif``'s ``imshow`` (which can update
    in place), a ``tricontourf`` has no persistent array to update, so each
    frame clears and redraws its three axes from scratch.
    """
    import matplotlib.pyplot as plt

    pos = np.asarray(pos)
    truth = np.asarray(truth).ravel()
    pred_frames = [np.asarray(p).ravel() for p in pred_frames]
    err_frames = [np.abs(p - truth) for p in pred_frames]

    vmin = float(min(truth.min(), min(p.min() for p in pred_frames)))
    vmax = float(max(truth.max(), max(p.max() for p in pred_frames)))
    emax = float(max(e.max() for e in err_frames)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    mappables: dict[int, object] = {}

    def draw(i: int) -> None:
        for ax in axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
        specs = [
            ("Prediction", pred_frames[i], cmap, vmin, vmax),
            ("Ground truth", truth, cmap, vmin, vmax),
            ("Abs. error", err_frames[i], "magma", 0.0, emax),
        ]
        for idx, (ax, (name, values, cm, vlo, vhi)) in enumerate(zip(axes, specs)):
            try:
                im = ax.tricontourf(
                    pos[:, 0],
                    pos[:, 1],
                    values,
                    levels=levels,
                    cmap=cm,
                    vmin=vlo,
                    vmax=vhi,
                )
            except (RuntimeError, ValueError):
                im = ax.scatter(
                    pos[:, 0], pos[:, 1], c=values, cmap=cm, s=4.0, vmin=vlo, vmax=vhi
                )
            ax.set_title(name)
            mappables[idx] = im
        fig.suptitle(f"{title} -- epoch {epochs[i]}")

    draw(0)
    fig.colorbar(mappables[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(mappables[2], ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()

    def update(i):
        draw(i)
        return []

    anim = FuncAnimation(fig, update, frames=len(pred_frames), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(
        f"[animate] wrote {len(pred_frames)} frames "
        f"(epochs {epochs[0]}-{epochs[-1]}) to {out_path}"
    )
    return out_path


def save_point_cloud_evolution_gif_3d(
    pos: np.ndarray,
    pred_frames: list[np.ndarray],
    truth: np.ndarray,
    epochs: list[int],
    out_path: Path,
    title: str = "",
    fps: int = 2,
    cmap: str = "jet",
    elev: float = 20.0,
    azim: float = -60.0,
    point_size: float = 3.0,
) -> Path:
    """3D isometric variant of :func:`save_point_cloud_evolution_gif`: scatters
    the surface points in 3D (``Axes3D``) instead of flattening them to a 2D
    ``tricontourf`` projection -- the flat version reads as a front view
    rather than a recognizable car silhouette.

    Axis remap and camera angle match ``_plot_utils.plot_car_surface_
    pressure``'s own three-quarter view: dataset columns are (width, height,
    length); remapped to plot-space (width, length, height) so height
    renders vertical-screen, matching the paper's own renders -- see that
    function's docstring for why the remap is needed.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)

    pos = np.asarray(pos)[:, [0, 2, 1]]
    truth = np.asarray(truth).ravel()
    pred_frames = [np.asarray(p).ravel() for p in pred_frames]
    err_frames = [np.abs(p - truth) for p in pred_frames]

    vmin = float(min(truth.min(), min(p.min() for p in pred_frames)))
    vmax = float(max(truth.max(), max(p.max() for p in pred_frames)))
    emax = float(max(e.max() for e in err_frames)) or 1.0
    box_aspect = tuple(np.ptp(pos[:, k]) for k in range(3))

    fig = plt.figure(figsize=(15.5, 4.8))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    mappables: dict[int, object] = {}

    def draw(i: int) -> None:
        specs = [
            ("Prediction", pred_frames[i], cmap, vmin, vmax),
            ("Ground truth", truth, cmap, vmin, vmax),
            ("Abs. error", err_frames[i], "magma", 0.0, emax),
        ]
        for idx, (ax, (name, values, cm, vlo, vhi)) in enumerate(zip(axes, specs)):
            ax.clear()
            im = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                c=values,
                cmap=cm,
                vmin=vlo,
                vmax=vhi,
                s=point_size,
            )
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect(box_aspect)
            ax.view_init(elev=elev, azim=azim)
            mappables[idx] = im
        fig.suptitle(f"{title} -- epoch {epochs[i]}")

    draw(0)
    fig.colorbar(mappables[0], ax=axes[0], fraction=0.046, pad=0.08)
    fig.colorbar(mappables[2], ax=axes[2], fraction=0.046, pad=0.08)
    fig.tight_layout()

    def update(i):
        draw(i)
        return []

    anim = FuncAnimation(fig, update, frames=len(pred_frames), blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(
        f"[animate] wrote {len(pred_frames)} frames "
        f"(epochs {epochs[0]}-{epochs[-1]}) to {out_path}"
    )
    return out_path


def rebuild_point_cloud_from_cache(
    cache_path: Path, out_path: Path, three_d: bool = True, **kwargs
) -> Path:
    """Re-renders a point-cloud GIF from the ``frames.npz`` a training run
    cached next to it (same cache ``train.py`` writes, holding ``pos`` too)
    -- for a cosmetic-only change (view angle, title, colours) that doesn't
    need a full retrain. ``three_d`` picks
    :func:`save_point_cloud_evolution_gif_3d` (default) over the flat 2D
    :func:`save_point_cloud_evolution_gif`.

    ``kwargs`` are forwarded to whichever is picked (``title``, ``fps``,
    ``cmap``, and for the 3D version ``elev``/``azim``/``point_size``).
    """
    data = np.load(cache_path)
    fn = (
        save_point_cloud_evolution_gif_3d if three_d else save_point_cloud_evolution_gif
    )
    return fn(
        data["pos"],
        list(data["pred_frames"]),
        data["truth_frame"],
        list(data["epochs"]),
        out_path,
        **kwargs,
    )


def rebuild_from_cache(cache_path: Path, out_path: Path, **kwargs) -> Path:
    """Re-renders the GIF from the ``frames.npz`` a training run cached next
    to it (see ``train.py``'s ``anim_out`` handling) -- for a cosmetic-only
    change (title, fps, colours) that doesn't need a full retrain.

    ``kwargs`` are forwarded to :func:`save_prediction_evolution_gif`
    (``title``, ``fps``, ``field_fn``, ``value_label``, ``error_label``,
    ``title_template``).
    """
    data = np.load(cache_path)
    return save_prediction_evolution_gif(
        list(data["pred_frames"]),
        data["truth_frame"],
        list(data["epochs"]),
        out_path,
        **kwargs,
    )
