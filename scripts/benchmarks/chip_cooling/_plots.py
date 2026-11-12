# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The figures of the chip cooling benchmark's doc pages.

* :func:`plot_one`: prediction, truth and absolute error of one test sample,
  with its parameters and metrics (``contour.png``);
* :func:`plot_loss_history`: the training-history curves, of which
  ``loss_curves.png`` is ``error_curve.png`` and ``training_time.png`` is
  ``compute_cost.png``.

All field plots use ``ax.scatter``, so the same function draws the mesh
nodes (GINO, Transolver) and the in-domain grid cells (FNO, WNO).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    }
)
# Human-readable labels shown in the side-panel of single-prediction plots.
DISPLAY = {
    "chip1.thermal.heat_flux": ("Chip 1 Heat Flux", "W/m²", ".0f"),
    "chip2.thermal.heat_flux": ("Chip 2 Heat Flux", "W/m²", ".0f"),
    "inlet1.momentum.mass_flow_rate": ("Inlet 1 Mass Flow", "kg/s", ".6f"),
    "inlet1.thermal.total_temperature": ("Inlet 1 Temperature", "K", ".1f"),
    "inlet2.momentum.mass_flow_rate": ("Inlet 2 Mass Flow", "kg/s", ".6f"),
    "inlet2.thermal.total_temperature": ("Inlet 2 Temperature", "K", ".1f"),
}


# ---------------------------------------------------------------------------
# Single-prediction comparison plot
# ---------------------------------------------------------------------------


def plot_one(
    z_pts,
    y_pts,
    y_pred,
    y_true,
    idx,
    t_inf_ms,
    x_vals,
    p_names,
    out_dir,
    title_prefix="",
):
    """
    Three-panel scatter plot: Prediction / Truth / Absolute Error,
    with a metrics side panel.

    Parameters
    ----------
    z_pts, y_pts : 1-D arrays of point positions in mm (valid cells only).
    y_pred, y_true : 1-D arrays of temperature values (valid cells only).
    idx          : test-sample index (int), used in filename and title.
    t_inf_ms     : inference time in milliseconds.
    x_vals       : raw parameter values for this sample (length-6 array).
    p_names      : list of parameter name strings (length 6).
    out_dir      : pathlib.Path - where to save the PNG.
    title_prefix : string prepended to the figure suptitle, e.g. "FNO" or
                   "FNO (ZenteiQ)".
    """
    FS = 16
    abs_err = np.abs(y_pred - y_true)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    max_ae = abs_err.max()

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], wspace=0.05, hspace=0.15)
    ax_p = fig.add_subplot(gs[0, 0])
    ax_t = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_text = fig.add_subplot(gs[:, 1])

    panels = [
        (ax_p, y_pred, "Prediction", "jet", "T (K)", dict(vmin=vmin, vmax=vmax)),
        (ax_t, y_true, "Truth", "jet", "T (K)", dict(vmin=vmin, vmax=vmax)),
        (ax_e, abs_err, "Absolute Error", "hot", "|Error| (K)", {}),
    ]

    for ax, vals, title, cmap, label, kwargs in panels:
        sc = ax.scatter(
            z_pts, y_pts, c=vals, cmap=cmap, marker="o", s=4, linewidths=0, **kwargs
        )
        ax.set_xlabel("Z (mm)", fontsize=FS)
        ax.set_ylabel("Y (mm)", fontsize=FS)
        ax.set_title(title, fontsize=FS)
        ax.tick_params(labelsize=FS - 4)
        ax.set_aspect("equal")
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(label, fontsize=FS - 2)
        cb.ax.tick_params(labelsize=FS - 4)

    # Side text panel
    ax_text.axis("off")
    lines = [f"Test Sample {idx}", "", "Input Parameters", "-" * 28]
    for name, val in zip(p_names, x_vals):
        if name in DISPLAY:
            lbl, unit, fmt = DISPLAY[name]
            lines += [f"{lbl}:", f"  {val:{fmt}} {unit}"]
        else:
            lines.append(f"{name}: {val:.6g}")
    lines += [
        "",
        "Metrics",
        "-" * 28,
        f"R²:        {r2:.6f}",
        f"MAE:       {mae:.3f} K",
        f"Max AE:    {max_ae:.3f} K",
        f"Inference: {t_inf_ms:.2f} ms",
    ]
    ax_text.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax_text.transAxes,
        fontsize=FS - 4,
        verticalalignment="top",
        fontfamily="monospace",
    )

    prefix = f"{title_prefix} - " if title_prefix else ""
    fig.suptitle(f"{prefix}Prediction vs CFD - Test Sample {idx}", fontsize=FS, y=0.995)
    plt.tight_layout()

    out = out_dir / f"comparison_sample_{idx:04d}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  Saved {out.name}: R²={r2:.6f}, MAE={mae:.3f} K, Inference={t_inf_ms:.2f} ms"
    )


# ---------------------------------------------------------------------------
# Training history plots  (loss, MAE, time, LR)
# ---------------------------------------------------------------------------


def plot_loss_history(history, plots_dir, title_prefix="FNO"):
    """
    Save four training-history plots from a history dict that contains at
    minimum the keys: loss, val_loss, mae, val_mae, time, lr.
    train_err / test_l2 are used when present (FNO history), otherwise
    loss / val_loss are used as fallback.

    Parameters
    ----------
    history      : dict  - as written by save_artifacts()
    plots_dir    : pathlib.Path
    title_prefix : str   - e.g. "FNO", "FNO (ZenteiQ)"
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(len(history["loss"]))

    train_l2 = history.get("train_err", history["loss"])
    test_l2 = history.get("test_l2", history.get("val_loss", []))

    # Relative L2
    fig, ax = plt.subplots()
    ax.semilogy(epochs, train_l2, label="Train Relative L2", linewidth=1.5)
    if test_l2:
        ax.semilogy(epochs, test_l2, label="Test Relative L2", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2  (log scale)")
    ax.set_ylim(top=1.0)
    ax.set_title(f"{title_prefix} - Relative L2 Error vs Epoch")
    ax.legend()
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=300)
    plt.close(fig)

    # MAE
    fig, ax = plt.subplots()
    ax.semilogy(epochs, history["mae"], label="Train MAE", linewidth=1.5)
    ax.semilogy(epochs, history["val_mae"], label="Test MAE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE - normalised units  (log scale)")
    ax.set_title(f"{title_prefix} - MAE vs Epoch")
    ax.legend()
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    fig.tight_layout()
    fig.savefig(plots_dir / "mae_curves.png", dpi=300)
    plt.close(fig)

    # Time per epoch
    fig, ax = plt.subplots()
    ax.plot(epochs[2:], history["time"][2:], label="Time per Epoch", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Wall-clock time per epoch  (s)")
    ax.set_title(f"{title_prefix} - Training Time per Epoch")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_time.png", dpi=300)
    plt.close(fig)

    # Learning rate
    fig, ax = plt.subplots()
    ax.semilogy(epochs, history["lr"], label="Learning Rate", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate  (log scale)")
    ax.set_title(f"{title_prefix} - Learning Rate Schedule")
    ax.legend()
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    fig.tight_layout()
    fig.savefig(plots_dir / "learning_rate.png", dpi=300)
    plt.close(fig)

    print(f"Saved training history plots to {plots_dir}")
