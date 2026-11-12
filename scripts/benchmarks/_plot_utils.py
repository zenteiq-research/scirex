# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared plotting helpers for the benchmark gallery.

These produce the static PNGs embedded in the documentation:

* ``plot_contours`` - an input / prediction / ground-truth / abs-error
  panel for a single 2D test sample.
* ``plot_error_curves`` - train-vs-test error over epochs, optionally
  overlaid with the neuraloperator reference curve for comparison.

The figures are deterministic given the captured arrays, so the docs build
never has to execute training (see scripts/benchmarks/run_*.py).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed when run on a GPU box / CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _to_2d(arr) -> np.ndarray:
    """Squeeze a (C, H, W) / (1, C, H, W) array down to a single 2D field."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    while a.ndim > 2:
        a = a[0]
    return a


def plot_contours(
    inp,
    pred,
    truth,
    out_path: str | Path,
    title: str = "",
    cmap: str = "viridis",
) -> Path:
    """Save a 4-panel contour figure: input, prediction, truth, abs-error.

    Parameters
    ----------
    inp, pred, truth : array-like
        2D fields (extra leading channel/batch dims are squeezed away).
    out_path : str | Path
        destination PNG path (parent dirs are created).
    title : str
        figure suptitle (e.g. the model + problem name).
    """
    inp2d, pred2d, truth2d = _to_2d(inp), _to_2d(pred), _to_2d(truth)
    err2d = np.abs(pred2d - truth2d)

    fields = [
        ("Input", inp2d, cmap),
        ("Prediction", pred2d, cmap),
        ("Ground truth", truth2d, cmap),
        ("Abs. error", err2d, "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, field, cm) in zip(axes, fields):
        im = ax.contourf(field, levels=50, cmap=cm)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_prediction_truth(
    pred,
    truth,
    out_path: str | Path,
    title: str = "",
    cmap: str = "viridis",
) -> Path:
    """Save a 3-panel contour figure: prediction, truth, abs-error.

    Same as ``plot_contours`` but without the "input" panel, for cases where
    the saved run artifacts only kept the ground truth and prediction (e.g.
    the WNO Burgers/Navier-Stokes scripts' ``evolution.pkl``, which does not
    retain the input field).
    """
    pred2d, truth2d = _to_2d(pred), _to_2d(truth)
    err2d = np.abs(pred2d - truth2d)

    fields = [
        ("Prediction", pred2d, cmap),
        ("Ground truth", truth2d, cmap),
        ("Abs. error", err2d, "magma"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (name, field, cm) in zip(axes, fields):
        im = ax.contourf(field, levels=50, cmap=cm)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multi_instance_grid(
    kappas,
    preds,
    truths,
    out_path: str | Path,
    row_labels: list[str] | None = None,
    title: str = "",
    cmap: str = "viridis",
) -> Path:
    """Save a ``(K, 4)`` grid of contour panels -- one row per problem
    instance, columns kappa / prediction / ground truth / abs-error -- for
    a zero-shot operator evaluation across ``K`` different held-out
    instances (e.g. unseen coefficient-field draws). The kappa column
    makes each row's actual instance (not just its (A, f) label) visible
    at a glance, alongside how well the operator did on it.

    Same per-panel convention as :func:`plot_prediction_truth`, stacked
    into one figure instead of ``K`` separate ones, so instance-to-instance
    generalization is visible at a glance rather than across ``K`` files.

    Each column (kappa / prediction+ground-truth / abs-error) shares one
    color normalization across all ``K`` rows -- prediction and ground
    truth also share each other's scale, not just their own row's -- so
    color alone is comparable row-to-row and between prediction and truth,
    instead of every panel silently rescaling to its own min/max.
    """
    kappas2d = [_to_2d(kp) for kp in kappas]
    preds2d = [_to_2d(p) for p in preds]
    truths2d = [_to_2d(t) for t in truths]
    errs2d = [np.abs(p - t) for p, t in zip(preds2d, truths2d)]
    k = len(preds2d)

    kappa_vmin, kappa_vmax = (
        min(f.min() for f in kappas2d),
        max(f.max() for f in kappas2d),
    )
    pt_vmin = min(f.min() for f in preds2d + truths2d)
    pt_vmax = max(f.max() for f in preds2d + truths2d)
    err_vmin, err_vmax = 0.0, max(f.max() for f in errs2d)

    kappa_norm = Normalize(vmin=kappa_vmin, vmax=kappa_vmax)
    pred_truth_norm = Normalize(vmin=pt_vmin, vmax=pt_vmax)
    err_norm = Normalize(vmin=err_vmin, vmax=err_vmax)

    # contourf's own `levels` (an int) picks level *values* from each call's
    # own data range regardless of `norm` -- passing a `norm` alone only
    # changes how those values map to color, not the range itself. Explicit
    # level arrays spanning each column-group's shared min/max are what
    # actually makes the contour levels (and so the colorbar) agree across
    # rows, which a `norm` by itself does not.
    kappa_levels = np.linspace(kappa_vmin, kappa_vmax, 51)
    pred_truth_levels = np.linspace(pt_vmin, pt_vmax, 51)
    err_levels = np.linspace(err_vmin, err_vmax, 51)

    fig, axes = plt.subplots(k, 4, figsize=(14.0, 3.2 * k))
    if k == 1:
        axes = axes[None, :]
    col_titles = ["kappa", "Prediction", "Ground truth", "Abs. error"]
    for row in range(k):
        fields = [kappas2d[row], preds2d[row], truths2d[row], errs2d[row]]
        cmaps = ["cividis", cmap, cmap, "magma"]
        norms = [kappa_norm, pred_truth_norm, pred_truth_norm, err_norm]
        levels_per_col = [
            kappa_levels,
            pred_truth_levels,
            pred_truth_levels,
            err_levels,
        ]
        for col, (field, cm, norm, levels) in enumerate(
            zip(fields, cmaps, norms, levels_per_col)
        ):
            ax = axes[row, col]
            im = ax.contourf(field, levels=levels, cmap=cm, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(col_titles[col])
        if row_labels is not None:
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9)
    fig.tight_layout()
    if title:
        # tight_layout() has no notion of the suptitle, so it happily
        # leaves zero headroom for it -- reserve a fixed ~0.5in strip at
        # the top (as a *fraction* of the figure height, which grows with
        # k) so the title never overlaps row 0's column headers.
        fig.subplots_adjust(top=min(0.96, 1.0 - 0.5 / (3.2 * k)))
        fig.suptitle(title, fontsize=14)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_1d_prediction_truth(
    truth,
    pred,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Save a 1D line plot: ground truth vs. prediction (WNO Burgers)."""
    x_grid = np.arange(np.asarray(truth).shape[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, np.squeeze(truth), label="Ground truth", lw=2)
    ax.plot(x_grid, np.squeeze(pred), "--", label="WNO prediction", lw=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_point_cloud_scalar_field(
    pos,
    pred,
    truth,
    out_path: str | Path,
    title: str = "",
    cmap: str = "viridis",
    levels: int = 50,
    point_size: float = 4.0,
) -> Path:
    """Save a 3-panel filled-contour figure (prediction / ground truth /
    abs-error) for a scalar field defined on an irregular point cloud, e.g.
    Transolver's Car-Design-ShapeNetCar surface pressure.

    Unlike ``plot_contours`` (regular grid, ``contourf``), points here have
    no grid structure, so this Delaunay-triangulates the point cloud's first
    two coordinates and draws a filled ``tricontourf`` over that mesh
    (``tricontourf`` gives the same continuous filled-contour look every
    other benchmark's ``contour.png`` has, rather than sparse individual
    dots). Falls back to a plain coloured
    scatter if triangulation fails (e.g. too few or degenerate/collinear
    points, which can happen on tiny smoke-test fixtures).

    Parameters
    ----------
    pos : array-like, shape (N, >=2)
        Point coordinates; only the first two columns are used for the 2D
        projection/triangulation.
    pred, truth : array-like, shape (N,)
        Scalar field values (e.g. pressure, stress) at each point.
    out_path : str | Path
        destination PNG path (parent dirs are created).
    title : str
        figure suptitle.
    """
    pos = np.asarray(pos)
    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    err = np.abs(pred - truth)

    fields = [
        ("Prediction", pred, cmap),
        ("Ground truth", truth, cmap),
        ("Abs. error", err, "magma"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, values, cm) in zip(axes, fields):
        try:
            im = ax.tricontourf(pos[:, 0], pos[:, 1], values, levels=levels, cmap=cm)
        except (RuntimeError, ValueError):
            im = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap=cm, s=point_size)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_car_surface_pressure(
    vtk_path: str | Path,
    pred,
    truth,
    out_path: str | Path,
    title: str = "",
    cmap: str = "jet",
    elev: float = 20.0,
    azim: float = -60.0,
) -> Path:
    """Save a 3-panel 3D car-surface contour figure (prediction / ground
    truth / abs-error), matching the rendering style of the Transolver
    paper's own Car-Design-ShapeNetCar pressure figures.

    Unlike ``plot_point_cloud_scalar_field`` (2D Delaunay triangulation of a
    coordinate projection -- which distorts the true car geometry and can
    degrade into a scatter of thin slivers where the projection folds the
    surface over itself), this reads the actual CFD surface mesh -- quad-cell
    connectivity straight from the sample's ``quadpress_smpl.vtk`` via
    ``scirex.data.datasets.shapenet_car_drag.load_surface_mesh`` --
    and renders it as a flat-shaded 3D solid (each quad colored by the mean
    of its 4 vertex values).

    Parameters
    ----------
    vtk_path : str | Path
        the sample's ``quadpress_smpl.vtk``.
    pred, truth : array-like, shape (N_points,)
        per-point scalar field (e.g. denormalized surface pressure), in the
        same point order as ``vtk_path`` -- see
        ``shapenet_car_drag.py``'s module docstring for the exact
        point-order requirement (denormalize first; slice with the sample's
        own ``surf`` mask, no extra reordering needed).
    elev, azim : float
        ``Axes3D.view_init`` camera angles, applied to the *plot*-space axes
        described below (not the raw dataset axes) -- the defaults reproduce
        the paper's own three-quarter elevated view (roof + one side + the
        nose visible).
    """
    from scirex.data.datasets.shapenet_car_drag import load_surface_mesh

    points, quads = load_surface_mesh(vtk_path)
    # Dataset axis convention (see shapenet_car_drag.py's module
    # docstring: frontal area is the convex hull of points[:, :2], drag is
    # projected onto +z): column 0 = width, column 1 = height, column 2 =
    # length/flow. matplotlib's Poly3DCollection/view_init always render the
    # *third* plotted coordinate as vertical-screen, so plotting (x, y, z)
    # directly puts the car's length axis vertical -- rendering it as a tall
    # capsule rather than a car (this was the actual bug: contour.png showed
    # a vertical capsule instead of a recognizable car silhouette). Remap to
    # plot-space (width, length, height) so height ends up vertical-screen,
    # matching the paper's own three-quarter renders (see Figure 9 /
    # case-study figures in the paper and reference repo).
    plot_points = points[:, [0, 2, 1]]
    verts = plot_points[quads]  # (n_cells, 4, 3)

    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    err = np.abs(pred - truth)

    fields = [
        ("Prediction", pred, cmap),
        ("Ground truth", truth, cmap),
        ("Abs. error", err, "magma"),
    ]
    span = np.ptp(plot_points, axis=0)
    span = np.where(span > 0, span, 1.0)  # guard degenerate axes

    fig = plt.figure(figsize=(16, 5.5))
    for i, (name, values, cm) in enumerate(fields):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        face_vals = values[quads].mean(axis=1)
        norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
        facecolors = plt.get_cmap(cm)(norm(face_vals))
        ax.add_collection3d(
            Poly3DCollection(verts, facecolors=facecolors, edgecolor="none")
        )
        ax.set_xlim(plot_points[:, 0].min(), plot_points[:, 0].max())
        ax.set_ylim(plot_points[:, 1].min(), plot_points[:, 1].max())
        ax.set_zlim(plot_points[:, 2].min(), plot_points[:, 2].max())
        ax.set_box_aspect(span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name)
        ax.set_axis_off()
        mappable = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def count_params_from_checkpoint(ckpt_path: str | Path) -> int:
    """Count parameters directly from a saved checkpoint's raw bytes.

    Uses ``flax.serialization.msgpack_restore``, which needs no template
    pytree -- so callers don't have to rebuild the model just to count
    parameters.
    """
    from flax import serialization

    restored = serialization.msgpack_restore(Path(ckpt_path).read_bytes())
    leaves = _tree_leaves(restored)
    return sum(int(np.asarray(leaf).size) for leaf in leaves)


def _tree_leaves(obj):
    """Flatten a nested dict/list of arrays without depending on jax."""
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _tree_leaves(value)
    elif isinstance(obj, list | tuple):
        for value in obj:
            yield from _tree_leaves(value)
    else:
        yield obj


# Legend labels are fixed by convention: our JAX implementation is "Zenteiq"
# and the neuraloperator baseline is "Reference".
LABEL_OURS = "Zenteiq"
LABEL_REF = "Reference"


def plot_error_curves(
    epochs,
    train_err,
    test_err,
    out_path: str | Path,
    title: str = "",
    ref_test_err=None,
    train_label: str = "H1 loss",
) -> Path:
    """Save a train-vs-test error curve, optionally overlaying the reference.

    Parameters
    ----------
    epochs : sequence[int]
    train_err, test_err : sequence[float]
        per-epoch Zenteiq training and test errors.
    ref_test_err : sequence[float] | None
        per-epoch Reference (neuraloperator) test error for comparison.
    train_label : str
        name of the training loss shown in the legend (FNO trains with H1
        loss; WNO trains with plain relative-L2, so its callers pass
        "rel. L2" here instead).

    Notes
    -----
    The axis is log-scale, which cannot represent zero or negative values
    -- a problem for a physics/energy-functional training loss, which is
    typically negative for most of training (it's a variational energy
    being minimized past zero, not a bounded error metric), silently
    dropping nearly every point and leaving only the first step or two
    visible. Plotting ``abs(train_err)`` shows the loss's magnitude/
    convergence trend instead, which is what's actually informative here;
    this is a no-op for callers whose training loss is already
    non-negative (H1 loss, relative-L2).
    """
    train_err_plot = np.abs(np.asarray(train_err, dtype=float))
    train_label_plot = (
        f"|{train_label}|" if np.any(np.asarray(train_err) < 0) else train_label
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        epochs, train_err_plot, label=f"{LABEL_OURS} train ({train_label_plot})", lw=2
    )
    ax.plot(epochs, test_err, label=f"{LABEL_OURS} test (rel. L2)", lw=2)
    if ref_test_err is not None:
        ax.plot(
            epochs[: len(ref_test_err)],
            ref_test_err,
            "--",
            label=f"{LABEL_REF} test (rel. L2)",
            lw=2,
            color="crimson",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_l2_error_only(
    epochs,
    test_err,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Save a single rel-L2-error-vs-step curve, in percent on a log axis.

    A dedicated companion to :func:`plot_error_curves` for benchmarks that
    want the test error highlighted on its own (e.g. an L2-only headline
    figure) without the training-loss curve sharing the axis -- the two
    are often on very different scales (a PDE-residual loss vs. a rel-L2
    error), which can make the error curve hard to read at a glance
    alongside it.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [e * 100 for e in test_err], color="#256abf", lw=2, marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("rel-L2 error (%, log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_compute_cost(
    epochs,
    our_times,
    out_path: str | Path,
    title: str = "",
    ref_times=None,
) -> Path:
    """Save a per-epoch compute-cost (seconds/epoch) plot, Zenteiq vs Reference.

    Parameters
    ----------
    epochs : sequence[int]
    our_times : sequence[float]
        Zenteiq per-epoch wall-clock training time in seconds.
    ref_times : sequence[float] | None
        Reference (neuraloperator) per-epoch training time in seconds.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, our_times, label=LABEL_OURS, lw=2)
    if ref_times is not None:
        ax.plot(
            epochs[: len(ref_times)],
            ref_times,
            "--",
            label=LABEL_REF,
            lw=2,
            color="crimson",
        )
    # Log scale so the one-time JAX/XLA compile at epoch 0 does not swamp the
    # steady-state per-epoch cost.
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Compute cost (s / epoch, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_metrics_json(metrics: dict, out_path: str | Path) -> Path:
    """Persist the captured run metrics so docs/tables can be regenerated."""
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline keeps the pre-commit end-of-file-fixer hook happy.
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return out_path
