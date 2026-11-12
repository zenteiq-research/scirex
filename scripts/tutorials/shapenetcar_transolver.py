# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting flow around a car with Transolver (Car-Design-ShapeNetCar)
=====================================================================

**Problem.** Given the 3D point cloud of a car (position, signed distance to
the car surface and surface normal at every point), predict the steady
velocity at every point and the pressure on the car surface. Every car is a
different mesh, so the model has to work on irregular point clouds, not on a
fixed grid. The flow is steady, incompressible external flow at
72 km/h (Umetani & Bickel 2018); the model is Transolver (Wu et al. 2024),
which replaces attention over all pairs of points by attention over a small
set of learned "slices".

**What you will learn.**

1. How to load Car-Design-ShapeNetCar with ``scirex.data.load_shapenet_car``.
2. How to build a small data directory of a few cars with symlinks, so that
   the whole tutorial runs on a CPU in minutes.
3. How to build a ``scirex.models.Transolver`` and count its parameters.
4. How to write the training loop in plain JAX and optax: one gradient step
   per car, a velocity-plus-pressure loss, a one-cycle learning-rate schedule.
5. How to evaluate with the relative L2 error on de-normalised fields, and
   how to plot truth, prediction and error on the car surface.

**Caveat.** The ``--smoke`` mode trains a tiny model on a handful of cars for
two epochs. It checks that the pipeline works and gives qualitatively poor
predictions. It is not a benchmark number. For the benchmark protocol (fold 0,
789 training cars, 200 epochs, paper-size model) use
``scripts/benchmarks/shapenetcar/train.py``; see the "Reproduce at full scale"
section of the tutorial page.

Usage, from the repository root::

    # needs the `benchmark` extra (vtk): pip install -e ".[benchmark]"
    python scripts/tutorials/shapenetcar_transolver.py \\
        --data-dir ~/data/mlcfd_data/training_data --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.data.load_shapenet_car``,
# ``scirex.models.Transolver`` and ``scirex.utils.count_model_params``. The
# loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
from time import perf_counter

import flax
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax

import scirex
from scirex.data import load_shapenet_car
from scirex.models import Transolver
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# Training settings of the Transolver paper's protocol for this benchmark,
# the same values as PROTOCOL in scripts/benchmarks/shapenetcar/_config.py.
PROTOCOL = dict(
    n_epochs=200,
    learning_rate=1e-3,
    press_weight=0.5,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1000.0,
)

# Model sizes. "full" is the paper's size (3,860,804 parameters). "smoke" is a
# tiny model that trains on a CPU.
SIZES = dict(
    full=dict(n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32),
    smoke=dict(n_hidden=32, n_layers=2, n_head=4, mlp_ratio=2, slice_num=8),
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
# Get a small data directory
# --------------------------
# ``load_shapenet_car`` reads ``<data_dir>/param0 .. param8``; each ``paramK``
# folder holds one folder per car with ``quadpress_smpl.vtk`` (surface mesh and
# pressure) and ``hexvelo_smpl.vtk`` (volume points and velocity). Fold 0 is
# the validation split (``param0``); all other folds are the training split.
# The loader only lists the folders that exist, so a directory with a small
# ``param0`` and a small ``param1`` is a valid small dataset. ``build_subset``
# makes one out of symlinks, so no data is copied.


def build_subset(src_dir: Path, dst_dir: Path, n_train: int, n_val: int) -> dict:
    """Symlink ``n_val`` cars into ``dst_dir/param0``, ``n_train`` into ``param1``."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    picked = {}
    for fold, count in (("param0", n_val), ("param1", n_train)):
        out = dst_dir / fold
        out.mkdir(parents=True, exist_ok=True)
        cars = [
            d
            for d in sorted((src_dir / fold).iterdir())
            if (d / "quadpress_smpl.vtk").exists() and (d / "hexvelo_smpl.vtk").exists()
        ]
        if len(cars) < count:
            raise ValueError(f"{src_dir / fold} has only {len(cars)} complete cars")
        picked[fold] = [d.name for d in cars[:count]]
        for old in out.iterdir():  # drop links left by an earlier, different subset
            if old.is_symlink() and old.name not in picked[fold]:
                old.unlink()
        for d in cars[:count]:
            link = out / d.name
            if not link.is_symlink() and not link.exists():
                link.symlink_to(d.resolve(), target_is_directory=True)
    return picked


# %%
# Load the data
# -------------
# The loader returns a dict with ``train`` and ``val`` (lists of cars,
# not stacked into one array, because the loader does not assume that all
# cars have the same number of points), ``coef_norm``
# (the per-channel mean and standard deviation of the training cars, already
# applied to the inputs and targets) and ``model_kwargs`` (the Transolver
# settings fixed by the data). Each car is a dict with ``x`` of shape (N, 7)
# = (x, y, z, signed distance, normal_x, normal_y, normal_z), ``y`` of shape
# (N, 4) = (velocity_x, velocity_y, velocity_z, pressure), ``pos`` (N, 3) and
# ``surf`` (N,), a boolean mask that is True on the car surface. The first read
# of a car parses the VTK files and computes the signed distance and the
# normals; the result is cached in ``cache_dir`` and reused by later runs.


def load_data(data_dir: Path, cache_dir: Path | None, fold_id: int = 0) -> dict:
    """Load one train/validation split and print what was read."""
    data = load_shapenet_car(data_dir, fold_id=fold_id, cache_dir=cache_dir)
    n_points = [len(s["x"]) for s in data["train"] + data["val"]]
    print(
        f"cars: {len(data['train'])} train / {len(data['val'])} val | "
        f"points per car: min {min(n_points)}, max {max(n_points)}"
    )
    print(f"model_kwargs = {data['model_kwargs']}")
    return data


# %%
# Build the model
# ---------------
# ``Transolver`` is a Flax module, so the parameters are a separate pytree:
# ``init`` creates them and ``apply`` uses them. The input width is inferred
# from the array that we pass to ``init``.


def build_model(model_kwargs: dict, size: dict, sample: dict, seed: int):
    """Build the Transolver, initialise it on one car and print the parameter count."""
    model = Transolver(**model_kwargs, **size)
    x0 = jnp.asarray(sample["x"])[None, :, :]
    params = model.init(jax.random.PRNGKey(seed), x=x0, fx=None)
    print(f"Transolver {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss is computed on standardised fields, per car:
# the mean squared error of the velocity at all points, plus ``press_weight``
# times the mean squared error of the pressure at the surface points only
# (the pressure is only defined on the car surface).


def make_loss(model, press_weight: float):
    """Return ``loss_fn(params, x, y, surf) -> (total, (loss_press, loss_velo))``."""

    def loss_fn(params, x, y, surf):
        out = model.apply(params, x=x, fx=None)[0]
        surf_f = surf.astype(out.dtype)
        n_surf = jnp.maximum(1.0, surf_f.sum())
        loss_press = jnp.sum(surf_f * (out[:, -1] - y[:, -1]) ** 2) / n_surf
        loss_velo = jnp.mean((out[:, :-1] - y[:, :-1]) ** 2)
        total = loss_velo + press_weight * loss_press
        return total, (loss_press, loss_velo)

    return loss_fn


# %%
# Train
# -----
# As in the reference implementation, one gradient step is taken per car
# (batch size 1), in a new random order every epoch. Adam follows a one-cycle
# schedule: the learning rate rises to its peak over the first 30 % of the
# steps and then decays. JAX compiles the step on its first call (and again
# for any new number of points), so the first epoch is slower than the others.


def train(model, params, train_samples, val_samples, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    total_steps = (len(train_samples) + 1) * epochs
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    tx = optax.adam(learning_rate=schedule)
    opt_state = tx.init(params)
    press_weight = protocol["press_weight"]
    loss_fn = make_loss(model, press_weight)

    @jax.jit
    def train_step(params, opt_state, x, y, surf):
        grads, aux = jax.grad(loss_fn, has_aux=True)(params, x, y, surf)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    @jax.jit
    def eval_step(params, x, y, surf):
        return loss_fn(params, x, y, surf)[1]

    def as_arrays(s):
        return jnp.asarray(s["x"])[None], jnp.asarray(s["y"]), jnp.asarray(s["surf"])

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        press, velo = [], []
        for i in rng.permutation(len(train_samples)):
            params, opt_state, (p, v) = train_step(
                params, opt_state, *as_arrays(train_samples[i])
            )
            press.append(p)
            velo.append(v)
        train_press = float(jnp.mean(jnp.stack(press)))
        train_velo = float(jnp.mean(jnp.stack(velo)))
        secs = perf_counter() - t0

        val = [eval_step(params, *as_arrays(s)) for s in val_samples]
        val_press = float(jnp.mean(jnp.stack([p for p, _ in val])))
        val_velo = float(jnp.mean(jnp.stack([v for _, v in val])))
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_velo + press_weight * train_press,
                val_loss=val_velo + press_weight * val_press,
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_loss={history[-1]['train_loss']:.5f} "
            f"val_loss={history[-1]['val_loss']:.5f} epoch_time={secs:.1f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# The loss lives on standardised fields and mixes two quantities with
# different units, so it does not say how good the prediction is. The reported
# error is the relative L2 error on de-normalised (physical) fields, averaged
# over the validation cars: the velocity over all points ("volume") and the
# pressure over the surface points ("surface").


def rel_l2(pred: np.ndarray, truth: np.ndarray, eps: float = 1e-8) -> float:
    """Relative L2 error ``||pred - truth|| / (||truth|| + eps)``."""
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + eps))


def evaluate(model, params, samples, coef_norm) -> dict:
    """Mean relative L2 error of the velocity (all points) and surface pressure."""
    _, _, mean_out, std_out = coef_norm
    predict = jax.jit(lambda p, x: model.apply(p, x=x, fx=None)[0])
    volume, surface, predictions = [], [], []
    for s in samples:
        out = np.asarray(predict(params, jnp.asarray(s["x"])[None]))
        out_denorm = out * std_out + mean_out
        y_denorm = s["y"] * std_out + mean_out
        surf = s["surf"]
        volume.append(rel_l2(out_denorm[:, :-1], y_denorm[:, :-1]))
        surface.append(rel_l2(out_denorm[surf, -1], y_denorm[surf, -1]))
        predictions.append(out_denorm)
    return dict(
        rel_l2_volume=float(np.mean(volume)),
        rel_l2_surface=float(np.mean(surface)),
        predictions=predictions,
    )


# %%
# Plot
# ----
# The standard figure of the benchmark: the input (here the surface normal
# component along y), the true and predicted surface pressure on one
# validation car, and the absolute error. Truth and prediction share one colour
# scale. We also plot the training and validation loss per epoch.


def plot_surface_pressure(sample, pred_denorm, coef_norm, path: Path) -> None:
    """Save input, true pressure, predicted pressure and absolute error."""
    mean_in, std_in, mean_out, std_out = coef_norm
    surf = sample["surf"]
    pos = sample["pos"][surf]
    normal_y = (sample["x"][surf, 5] * std_in[5] + mean_in[5]).astype(float)
    truth = (sample["y"][surf, -1] * std_out[-1] + mean_out[-1]).astype(float)
    pred = pred_denorm[surf, -1]
    panels = [
        ("Input: surface normal y", normal_y, "coolwarm", None),
        ("Truth: pressure", truth, "jet", (truth.min(), truth.max())),
        ("Prediction: pressure", pred, "jet", (truth.min(), truth.max())),
        ("Absolute error", np.abs(pred - truth), "magma", None),
    ]
    fig = plt.figure(figsize=(18, 4.5))
    for k, (title, values, cmap, lim) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, k, projection="3d")
        vmin, vmax = lim if lim else (None, None)
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 2],
            pos[:, 1],
            c=values,
            s=3,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect(np.ptp(pos[:, [0, 2, 1]], axis=0))
        fig.colorbar(sc, ax=ax, shrink=0.6)
    fig.suptitle("Car-Design-ShapeNetCar, one validation car")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and validation loss per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (velocity MSE + 0.5 x pressure MSE)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` symlinks a few cars, uses the tiny model and trains for two
# epochs. Without it the script trains the paper-size model on the whole fold
# of ``--data-dir`` for 200 epochs. That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("SHAPENETCAR_DATA"),
        help="folder that contains param0..param8 (or set $SHAPENETCAR_DATA)",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="few cars, tiny model, 2 epochs"
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/shapenetcar_tutorial")
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="default: <out-dir>/cache"
    )
    ap.add_argument("--n-train", type=int, default=4, help="cars in the smoke subset")
    ap.add_argument("--n-val", type=int, default=2, help="cars in the smoke subset")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $SHAPENETCAR_DATA")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or out_dir / "cache").expanduser()
    data_dir = Path(args.data_dir).expanduser()

    if args.smoke:
        data_dir = out_dir / "subset"
        picked = build_subset(
            Path(args.data_dir).expanduser(), data_dir, args.n_train, args.n_val
        )
        print(f"subset in {data_dir}: {picked}")
    epochs = args.epochs or (2 if args.smoke else PROTOCOL["n_epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data = load_data(data_dir, cache_dir)
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params = build_model(data["model_kwargs"], size, data["train"][0], args.seed)

    params, history = train(
        model, params, data["train"], data["val"], epochs, args.seed
    )
    metrics = evaluate(model, params, data["val"], data["coef_norm"])
    print(f"rel_l2_volume (velocity)  = {metrics['rel_l2_volume']:.4f}")
    print(f"rel_l2_surface (pressure) = {metrics['rel_l2_surface']:.4f}")

    plot_surface_pressure(
        data["val"][0],
        metrics["predictions"][0],
        data["coef_norm"],
        out_dir / "surface_pressure.png",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k != "predictions"}


if __name__ == "__main__":
    main()
