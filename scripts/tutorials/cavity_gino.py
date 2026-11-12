# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Forecasting a turbulent lid-driven cavity with GINO (Cavity)
============================================================

**Problem.** A 0.1 m square cavity full of fluid at rest is set in motion by
its lid, which slides at a fixed speed. The flow is transient and turbulent
(RANS k-epsilon, solved in OpenFOAM; Costa Rocha et al. 2023). Given the
pressure and velocity on the 128x128 cell grid at three consecutive times,
0.1 s apart, predict the next frame. The model is trained on eight lid speeds
(Re 1,000 to 20,000) and tested on one it never saw, Re 10,000. The model is
the Geometry-Informed Neural Operator (GINO, Li et al. 2023), which is not in
the paper: it reads the 16,384 cells as a point cloud, integrates them onto a
coarse latent grid where an FNO runs, and integrates back to the cells.

**What you will learn.**

1. How to turn the OpenFOAM CSV export into the benchmark's 128x128 cell
   grids, and those into windows of 3 frames in and 1 frame out.
2. How to split by Reynolds number: train and validate on eight lid speeds,
   test on a ninth.
3. How to build ``scirex.models.GINO`` on a fixed point cloud, with the
   neighbour lists of its two graph integrals computed once with
   ``scirex.layers.precompute_neighbors``.
4. How to add a soft mass-conservation penalty to the loss with
   ``scirex.losses.FiniteDiff``, and how to train with the paper's
   plateau schedule, early stopping and best-epoch selection.
5. How to evaluate one step ahead and 10 steps ahead (feeding predictions
   back in), against the copy-the-last-frame baseline.

**Caveat.** The ``--smoke`` mode trains a tiny GINO on a few windows for a few
epochs. It checks that the pipeline works and gives poor predictions. It is
not a benchmark number. For the benchmark protocol (all 706 training windows,
up to 300 epochs, 1,058,861 parameters) use
``scripts/benchmarks/cavity/train.py``; see the "Reproduce at full scale"
section of the tutorial page.

Usage, from the repository root, with the benchmark's cache (nine <case>.npz)::

    python scripts/tutorials/cavity_gino.py --cache-dir ~/data/cavity_paper --smoke

or with the raw CSV export, converted to the cache on the first run::

    python scripts/tutorials/cavity_gino.py \\
        --data-dir ~/data/Transient_k-e --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.GINO``,
# ``scirex.layers.precompute_neighbors``, ``scirex.data.UnitGaussianNormalizer``,
# ``scirex.losses.FiniteDiff`` and ``scirex.utils.count_model_params``. The
# loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
from time import perf_counter

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax

import scirex
from scirex.data import UnitGaussianNormalizer
from scirex.layers import precompute_neighbors
from scirex.losses import FiniteDiff
from scirex.models import GINO
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# The paper's split (Costa Rocha et al. 2023, Sec. 2.2): eight lid speeds to
# train on, Re 10,000 to test on, Re 8,000 and 12,000 left out. A case "NNk"
# has Re NN,000 and a lid speed of NN / 10 m/s.
TRAIN_CASES = ("01k", "02k", "04k", "06k", "14k", "16k", "18k", "20k")
TEST_CASE = "10k"
N_STEPS = 101  # frames 0..100: the first 10 s, one frame every 0.1 s
DT = 0.1  # s
DOMAIN_SIZE = 0.1  # m
N_CELL = 128  # cells per side; the export samples the 129 x 129 corners
DX = DOMAIN_SIZE / N_CELL
FIELDS = ("p", "Ux", "Uy")
HORIZON = 10  # rollout length: 1 s

# Training settings of the benchmark, the same values as PROTOCOL in
# scripts/benchmarks/cavity/_config.py (the paper's Sections 2.2 to 2.4).
PROTOCOL = dict(
    n_in=3,
    n_out=1,
    learning_rate=1e-3,
    lr_factor=0.5,  # halve the learning rate ...
    lr_patience=10,  # ... after 10 epochs without a lower validation loss
    early_stop_patience=50,
    epochs=300,
    batch_size=8,
    mc_weight=0.1,  # weight of the mass-conservation term
    val_frac=0.1,
)

# Model sizes. "full" is the benchmark's gino_3to1 (1,058,861 parameters, 0.2 %
# above the FNO's 1,056,739). "smoke" is a tiny GINO that trains on a CPU.
SIZES = dict(
    full=dict(modes=16, width=34, n_layers=3, latent_grid=32),
    smoke=dict(modes=4, width=8, n_layers=1, latent_grid=16),
)


def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")


# %%
# From the CSV export to cell grids
# ---------------------------------
# Each case folder holds one CSV per frame, ``Data<case>0.<step>.csv``, with
# the columns ``"p","U:0","U:1","U:2","Points:0","Points:1","Points:2"`` at
# the 129 x 129 x 2 mesh corners. The mesh is one cell thick, so only the
# ``z = 0`` layer is kept. The corners are sorted into rows of constant y,
# and each cell gets the mean of its four corners: the 128 x 128 cell field
# the paper's models use. Averaging keeps the lid row, which drives the whole
# flow; cropping a row of corners would lose it. Each case is converted once
# and cached as ``<cache_dir>/<case>.npz``, the same file the benchmark caches.


def read_snapshot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of the 129 x 129 corners on ``z = 0`` and their ``(p, Ux, Uy)``."""
    with open(path) as f:
        header = [h.strip().strip('"') for h in f.readline().split(",")]
    cols = [
        header.index(c) for c in ("p", "U:0", "U:1", "Points:0", "Points:1", "Points:2")
    ]
    table = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, dtype=np.float32)
    table = table[np.isclose(table[:, 5], 0.0, rtol=0, atol=1e-6)]
    return table[:, 3:5], table[:, 0:3]


def corners_to_cells(pos: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """Corner values ``(N, 3)`` -> cell values ``(128, 128, 3)``, rows along y."""
    ix = np.round(pos[:, 0] / DX).astype(np.int64)
    iy = np.round(pos[:, 1] / DX).astype(np.int64)
    corners = np.zeros((N_CELL + 1, N_CELL + 1, fields.shape[1]), np.float32)
    corners[iy, ix] = fields
    return 0.25 * (
        corners[:-1, :-1] + corners[1:, :-1] + corners[:-1, 1:] + corners[1:, 1:]
    )


def prepare_case(raw_dir: Path, case: str, cache_dir: Path) -> Path:
    """Write ``<cache_dir>/<case>.npz`` with ``grid`` ``(101, 128, 128, 3)`` once."""
    out = Path(cache_dir) / f"{case}.npz"
    if out.exists():
        return out
    frames = []
    for step in range(N_STEPS):
        pos, fields = read_snapshot(Path(raw_dir) / case / f"Data{case}0.{step}.csv")
        frames.append(corners_to_cells(pos, fields))
    out.parent.mkdir(parents=True, exist_ok=True)
    u_lid = np.float32(int(case[:2]) / 10.0)
    np.savez_compressed(
        out,
        grid=np.stack(frames),
        steps=np.arange(N_STEPS, dtype=np.int32),
        u_lid=u_lid,
    )
    return out


def load_case(cache_dir: Path, case: str) -> np.ndarray:
    """The ``(101, 128, 128, 3)`` frames of one case, in physical units."""
    with np.load(Path(cache_dir) / f"{case}.npz") as npz:
        return npz["grid"]


# %%
# Windows and the split
# ---------------------
# A sample maps frames ``[i, i + n_in)`` to ``[i + n_in, i + n_in + n_out)``:
# 98 windows per case for 3 in and 1 out. The windows of the eight training
# cases are pooled and 10 % of them, at random, become the validation set,
# which only drives the learning-rate schedule, early stopping and the choice
# of the saved epoch. The test case is never windowed into this pool. The
# normaliser (one mean and standard deviation per field) is fitted on the
# training windows only.


def make_windows(grid: np.ndarray, n_in: int, n_out: int):
    """``(T, H, W, C)`` -> inputs ``(n, n_in, ...)`` and targets ``(n, n_out, ...)``."""
    n = grid.shape[0] - n_in - n_out + 1
    xs = np.stack([grid[i : i + n_in] for i in range(n)])
    ys = np.stack([grid[i + n_in : i + n_in + n_out] for i in range(n)])
    return xs, ys


def build_split(cache_dir: Path, protocol: dict, seed: int, n_train=None, n_val=None):
    """Normalised train and validation windows of the training cases."""
    pairs = [make_windows(load_case(cache_dir, c), protocol["n_in"], protocol["n_out"])
             for c in TRAIN_CASES]  # fmt: skip
    x = np.concatenate([p[0] for p in pairs])
    y = np.concatenate([p[1] for p in pairs])
    perm = np.random.default_rng(seed).permutation(len(x))
    n_val_all = max(1, round(protocol["val_frac"] * len(x)))
    val_idx, train_idx = perm[:n_val_all][:n_val], perm[n_val_all:][:n_train]
    norm = UnitGaussianNormalizer.create(axis=(0, 1, 2, 3)).fit(
        jnp.asarray(x[train_idx])
    )

    def enc(a):
        return np.asarray(norm.transform(jnp.asarray(a)), np.float32)

    print(
        f"windows: {len(train_idx)} train / {len(val_idx)} val "
        f"(of {len(x) - n_val_all} / {n_val_all}) from {len(TRAIN_CASES)} cases"
    )
    return dict(
        x_train=enc(x[train_idx]),
        y_train=enc(y[train_idx]),
        x_val=enc(x[val_idx]),
        y_val=enc(y[val_idx]),
        normalizer=norm,
    )


# %%
# Build the model
# ---------------
# GINO maps between point clouds. Its input points are the 16,384 cell centres
# (in [0, 1]^2), its latent points a ``latent_grid`` x ``latent_grid`` square on
# which an FNO runs, and its output points the cell centres again. A graph
# integral (GNO) carries the field from every cell to the latent points within
# 1.5 latent spacings, and a second one from the latent points back to every
# cell within 1 spacing, which reaches every cell. The mesh never changes, so
# ``geometry`` runs both neighbour searches once and keeps them. The window is
# folded into 9 values per point, as the FNO folds it into channels; GINO
# embeds the coordinates itself. ``fno_split_complex_params=True`` stores the
# latent FNO's complex weights as real pairs, so plain Adam is the right
# optimiser.

_GEOMETRY: dict = {}


def cell_centres(height: int, width: int) -> np.ndarray:
    """``(x, y)`` of every cell centre in ``[0, 1]^2``, row-major, ``(H * W, 2)``."""
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def geometry(height: int, width: int, latent_grid: int) -> dict:
    """Points, latent grid, radii and both neighbour lists, computed once."""
    key = (height, width, latent_grid)
    if key not in _GEOMETRY:
        points = cell_centres(height, width)
        axis = np.linspace(0.0, 1.0, latent_grid, dtype=np.float32)
        lat_x, lat_y = np.meshgrid(axis, axis, indexing="xy")
        latent = np.stack([lat_x.ravel(), lat_y.ravel()], axis=-1)
        spacing = 1.0 / (latent_grid - 1)
        in_radius, out_radius = 1.5 * spacing, 1.0 * spacing
        _GEOMETRY[key] = dict(
            points=jnp.asarray(points)[None],
            latent=jnp.asarray(latent).reshape(1, latent_grid, latent_grid, 2),
            in_radius=in_radius,
            out_radius=out_radius,
            neighbors_in=precompute_neighbors(points, latent, in_radius),
            neighbors_out=precompute_neighbors(latent, points, out_radius),
        )
    return _GEOMETRY[key]


class WindowGINO(nn.Module):
    """``(B, n_in, H, W, 3)`` -> ``(B, n_out, H, W, 3)`` with GINO on the cells."""

    n_out: int
    modes: int = 16
    width: int = 32
    n_layers: int = 1
    latent_grid: int = 32
    gno_channels: int = 32
    gno_mlp_hidden: tuple = (64, 64)

    @nn.compact
    def __call__(self, x):
        batch, steps, n_rows, n_cols, channels = x.shape
        geo = geometry(n_rows, n_cols, self.latent_grid)
        f = x.transpose(0, 2, 3, 1, 4).reshape(batch, n_rows * n_cols, steps * channels)
        y = GINO(
            in_channels=steps * channels,
            out_channels=self.n_out * channels,
            gno_coord_dim=2,
            gno_use_open3d=False,
            in_gno_radius=geo["in_radius"],
            out_gno_radius=geo["out_radius"],
            in_gno_transform_type="nonlinear_kernelonly",
            in_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            out_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            fno_in_channels=self.gno_channels,
            fno_n_modes=(self.modes, self.modes),
            fno_hidden_channels=self.width,
            fno_n_layers=self.n_layers,
            fno_split_complex_params=True,
        )(
            input_geom=geo["points"],
            latent_queries=geo["latent"],
            output_queries=geo["points"],
            x=f,
            neighbors_in=geo["neighbors_in"],
            neighbors_out=geo["neighbors_out"],
        )  # (B, H * W, n_out * C)
        y = y.reshape(batch, n_rows, n_cols, self.n_out, channels)
        return y.transpose(0, 3, 1, 2, 4)


def build_model(size: dict, n_out: int, sample: np.ndarray, seed: int):
    """Build the GINO, initialise it on one window and print the parameter count."""
    model = WindowGINO(n_out=n_out, **size)
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(sample)[None])
    print(f"GINO {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The paper's loss: the mean squared error on the normalised fields, plus 0.1
# times a soft mass-conservation term, the mean absolute divergence of the
# predicted velocity in physical units, made dimensionless with L = 0.1 m and
# U = 1 m/s. ``FiniteDiff`` takes the derivatives: central differences
# inside, one-sided ones at the walls (``periodic=False``). The grid axes are
# (y, x), so Ux is differentiated along the last axis and Uy along the one
# before.

_DIFF = FiniteDiff()


def divergence(vel):
    """``d Ux / dx + d Uy / dy`` of ``(..., H, W, 2)`` velocities in m/s."""
    dudx = _DIFF.derivative(vel[..., 0], axis=-1, spacing=DX, periodic=False)
    dvdy = _DIFF.derivative(vel[..., 1], axis=-2, spacing=DX, periodic=False)
    return dudx + dvdy


def make_loss(model, normalizer, mc_weight: float):
    """Return ``loss_fn(params, x, y) -> (total, (mse, mass_conservation))``."""

    def loss_fn(params, x, y):
        pred = model.apply(params, x)
        mse = jnp.mean((pred - y) ** 2)
        vel = normalizer.inverse_transform(pred)[..., 1:3]
        mc = jnp.mean(jnp.abs(divergence(vel))) * DOMAIN_SIZE / 1.0
        return mse + mc_weight * mc, (mse, mc)

    return loss_fn


# %%
# Train
# -----
# Adam at 1e-3 on batches of 8 windows in a new random order every epoch.
# After each epoch the validation windows are scored. The learning rate is
# halved after 10 epochs without a lower validation loss (``inject_hyperparams``
# makes it a value that can be changed between steps), training stops after
# 50 such epochs, and the weights of the epoch with the lowest validation loss
# are kept. JAX compiles the step on its first call, so the first epoch is
# slower than the others.


def train(model, params, data, epochs, seed, protocol=PROTOCOL):
    """Train; return the best-validation parameters and the history."""
    tx = optax.inject_hyperparams(optax.adam)(learning_rate=protocol["learning_rate"])
    opt_state = tx.init(params)
    loss_fn = make_loss(model, data["normalizer"], protocol["mc_weight"])

    @jax.jit
    def train_step(params, opt_state, x, y):
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    val_loss = jax.jit(lambda p, x, y: loss_fn(p, x, y)[0])
    bs, lr = protocol["batch_size"], protocol["learning_rate"]
    x_tr, y_tr = data["x_train"], data["y_train"]
    rng = np.random.default_rng(seed)
    best = dict(loss=np.inf, epoch=0, params=params)
    since_best = since_drop = 0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = perf_counter()
        order = rng.permutation(len(x_tr))
        losses = []
        for i in range(0, len(x_tr), bs):
            idx = order[i : i + bs]
            params, opt_state, loss = train_step(
                params, opt_state, jnp.asarray(x_tr[idx]), jnp.asarray(y_tr[idx])
            )
            losses.append(float(loss))
        val = np.mean([
            float(val_loss(params, jnp.asarray(data["x_val"][i : i + bs]),
                           jnp.asarray(data["y_val"][i : i + bs])))
            for i in range(0, len(data["x_val"]), bs)
        ])  # fmt: skip
        if val < best["loss"] - 1e-9:
            best = dict(loss=val, epoch=epoch, params=params)
            since_best = since_drop = 0
        else:
            since_best += 1
            since_drop += 1
        if since_drop >= protocol["lr_patience"]:  # ReduceLROnPlateau
            lr *= protocol["lr_factor"]
            opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, jnp.float32)
            since_drop = 0
        secs = perf_counter() - t0
        history.append(
            dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=float(val),
                 lr=lr, secs=secs)
        )  # fmt: skip
        print(
            f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4e} "
            f"val_loss={val:.4e} lr={lr:.1e} epoch_time={secs:.1f}s"
        )
        if since_best >= protocol["early_stop_patience"]:
            print(f"early stop at epoch {epoch}")
            break
    print(f"best validation loss {best['loss']:.4e} at epoch {best['epoch']}")
    return best["params"], history


# %%
# Evaluate
# --------
# On the unseen Re 10,000 case, in physical units, with the paper's metric:
# the RMSE pooled over p, Ux and Uy. One step ahead, every window of the test
# case is predicted from true frames. Ten steps ahead, each prediction is fed
# back in as the newest input frame, from every start that has 10 true frames
# after its window. The baseline copies the last input frame forward: a model
# that cannot beat it has learnt nothing useful about the dynamics.


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square of ``a - b`` over every element."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate(model, params, grid, normalizer, n_in: int, horizon: int = HORIZON):
    """One-step and rollout RMSE on one case, for the model and the baseline."""
    apply = jax.jit(model.apply)

    def predict(window):  # physical (B, n_in, H, W, C) -> physical (B, 1, H, W, C)
        out = apply(params, normalizer.transform(jnp.asarray(window)))
        return np.asarray(normalizer.inverse_transform(out))[:, :1]

    x, y = make_windows(grid, n_in, 1)
    one_step = np.concatenate([predict(x[i : i + 8]) for i in range(0, len(x), 8)])
    starts = np.arange(grid.shape[0] - n_in - horizon + 1)
    window = np.stack([grid[s : s + n_in] for s in starts])
    produced = []
    for _ in range(horizon):
        nxt = np.concatenate(
            [predict(window[i : i + 8]) for i in range(0, len(window), 8)]
        )
        produced.append(nxt[:, 0])
        window = np.concatenate([window[:, 1:], nxt], axis=1)
    truth = np.stack([grid[s + n_in : s + n_in + horizon] for s in starts], axis=1)
    last = np.stack([grid[s + n_in - 1] for s in starts])
    return dict(
        rmse=rmse(one_step, y),
        per_field={f: rmse(one_step[..., k], y[..., k]) for k, f in enumerate(FIELDS)},
        mean_abs_div=float(
            np.mean(np.abs(divergence(jnp.asarray(one_step[..., 1:3]))))
        ),
        baseline_rmse=rmse(x[:, -1:], y),
        rollout_rmse=rmse(np.stack(produced), truth),
        rollout_per_step=[rmse(p, t) for p, t in zip(produced, truth)],
        baseline_rollout_per_step=[rmse(last, t) for t in truth],
        baseline_rollout_rmse=rmse(np.broadcast_to(last, truth.shape), truth),
        one_step=one_step,
    )


# %%
# Plot
# ----
# The standard figure, on the test case at one time: the input (the speed |u|
# of the last input frame), the true and the predicted speed on one colour
# scale, and the absolute error of the speed. A second figure shows the
# rollout error per step against the baseline, and a third the loss per epoch.


def plot_prediction(last_input, truth, pred, path: Path, title: str) -> None:
    """Save input, true speed, predicted speed and absolute error."""

    def speed(f):
        return np.hypot(f[..., 1], f[..., 2])

    vmax = float(speed(truth).max())
    panels = [
        ("Input: |u| of the last input frame", speed(last_input), "viridis", (0, vmax)),
        ("Truth: |u|", speed(truth), "viridis", (0, vmax)),
        ("Prediction: |u|", speed(pred), "viridis", (0, vmax)),
        ("Absolute error of |u|", np.abs(speed(pred) - speed(truth)), "magma", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    extent = (0, DOMAIN_SIZE, 0, DOMAIN_SIZE)
    for ax, (name, values, cmap, lim) in zip(axes, panels):
        vmin, vmax_ = lim if lim else (None, None)
        im = ax.imshow(
            values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax_
        )
        ax.set_title(name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="m/s")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_rollout(per_step: list[float], baseline: list[float], path: Path) -> None:
    """Save the rollout RMSE per step, model and copy-last-frame baseline."""
    steps = np.arange(1, len(per_step) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps * DT, per_step, marker="o", label="model, fed its own predictions")
    ax.plot(steps * DT, baseline, marker="s", ls="--", label="copy last frame")
    ax.set_xlabel("time ahead (s)")
    ax.set_ylabel("RMSE over p, Ux, Uy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and validation loss per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE + 0.1 x mass conservation)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# The data come from ``--cache-dir`` (or ``$CAVITY_CACHE_DIR``): a folder with
# the nine ``<case>.npz`` files, which the benchmark's ``train.py`` and
# ``infer.py`` read with the same flag. Cases missing from it are converted from
# the CSV export in ``--data-dir`` first.
# ``--smoke`` takes a few training windows, uses the tiny GINO and trains for a
# few epochs; the test case is still evaluated in full. Without it the script
# trains the benchmark-size GINO on every training window for up to 300 epochs.
# That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("CAVITY_DATA_DIR"),
        help="Transient_k-e folder, one sub-folder per case (or $CAVITY_DATA_DIR)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("CAVITY_CACHE_DIR", "Dataset_cache/cavity_paper")),
        help="folder of the nine <case>.npz files (or $CAVITY_CACHE_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few windows, tiny GINO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/cavity_gino_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=64, help="windows in smoke mode")
    ap.add_argument("--n-val", type=int, default=16, help="windows in smoke mode")
    ap.add_argument("--plot-step", type=int, default=10, help="test frame to plot")
    args = ap.parse_args(argv)
    args.cache_dir = args.cache_dir.expanduser()
    cases = (*TRAIN_CASES, TEST_CASE)
    if args.data_dir is None and not all(
        (args.cache_dir / f"{c}.npz").exists() for c in cases
    ):
        ap.error("pass --cache-dir with the nine <case>.npz files, or --data-dir")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PROTOCOL
    epochs = args.epochs or (3 if args.smoke else p["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    for case in cases:
        prepare_case(args.data_dir, case, args.cache_dir)
    n_train, n_val = (args.n_train, args.n_val) if args.smoke else (None, None)
    data = build_split(args.cache_dir, p, args.seed, n_train, n_val)
    test = load_case(args.cache_dir, TEST_CASE)
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params = build_model(size, p["n_out"], data["x_train"][0], args.seed)

    params, history = train(model, params, data, epochs, args.seed)
    m = evaluate(model, params, test, data["normalizer"], p["n_in"])
    print(
        f"Re 10,000, one step: RMSE {m['rmse']:.3e} "
        f"(copy last frame {m['baseline_rmse']:.3e})"
    )
    print(
        "  per field: " + ", ".join(f"{k} {v:.2e}" for k, v in m["per_field"].items())
    )
    print(
        f"Re 10,000, {HORIZON} steps: RMSE {m['rollout_rmse']:.3e} "
        f"(copy last frame {m['baseline_rollout_rmse']:.3e})"
    )

    k = args.plot_step
    plot_prediction(
        test[k - 1],
        test[k],
        m["one_step"][k - p["n_in"], 0],
        out_dir / "prediction.png",
        f"Cavity, GINO: Re 10,000 (unseen), t = {k * DT:.1f} s, one step ahead",
    )
    plot_rollout(
        m["rollout_per_step"], m["baseline_rollout_per_step"], out_dir / "rollout.png"
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in m.items() if k != "one_step"}


if __name__ == "__main__":
    main()
