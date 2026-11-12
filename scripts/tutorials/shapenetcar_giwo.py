# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting flow around a car with GIWO (Car-Design-ShapeNetCar)
===============================================================

**Problem.** Given the 3D point cloud of a car (position, signed distance to
the car surface and surface normal at every point), predict the steady
velocity at every point and the pressure on the car surface. Every car is a
different point cloud. The flow is steady, incompressible external flow at
72 km/h (Umetani & Bickel 2018); the model is GIWO, a "Geo-WNO" of this
repository and not a published architecture: GINO's outer shell (a graph
neural operator carries the point cloud onto a fixed regular grid, a second
one carries the result back onto the car's points) with a Wavelet Neural
Operator (WNO) instead of GINO's FNO on the grid.

**What you will learn.**

1. How to load Car-Design-ShapeNetCar with ``scirex.data.load_shapenet_car``
   and build a small data directory of a few cars.
2. How GIWO sees a point cloud: a fixed latent grid, and two radius-based
   neighbour lists per car, built once with ``scirex.layers``.
3. Why every car's neighbour lists are padded to one length, so that JAX
   compiles the training step only once.
4. How to assemble GIWO from the public blocks ``scirex.layers.GNOBlock``,
   ``scirex.models.WNO`` and ``scirex.layers.ChannelMLP``, and count its
   parameters.
5. How to train it with the benchmark's loss, optimiser and schedule, and how
   to evaluate and plot the surface pressure.

**Caveat.** The ``--smoke`` mode trains a tiny GIWO on a handful of cars for
two epochs. It checks that the pipeline works and gives qualitatively poor
predictions. It is not a benchmark number. For the benchmark protocol (fold 0,
789 training cars, 200 epochs, 3,752,638 parameters) use
``scripts/benchmarks/shapenetcar/train.py --run giwo_transolver_cost_matched``; see the
"Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    # needs the `benchmark` extra (vtk): pip install -e ".[benchmark]"
    python scripts/tutorials/shapenetcar_giwo.py \\
        --data-dir ~/data/mlcfd_data/training_data --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.data.load_shapenet_car``,
# ``scirex.layers.GNOBlock``, ``scirex.layers.ChannelMLP``,
# ``scirex.models.WNO``, ``scirex.layers.kdtree_neighbor_search``,
# ``scirex.layers.neighbors_to_segments`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

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
from scirex.data import load_shapenet_car
from scirex.layers import (
    ChannelMLP,
    GNOBlock,
    kdtree_neighbor_search,
    neighbors_to_segments,
)
from scirex.models import WNO
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# Training settings of the Transolver paper's protocol for this benchmark,
# the same values as PROTOCOL in scripts/benchmarks/shapenetcar/_config.py.
# Every model on this benchmark is trained with them.
PROTOCOL = dict(
    n_epochs=200,
    learning_rate=1e-3,
    press_weight=0.5,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1000.0,
)

# Model sizes. "full" is the benchmark's headline GIWO run,
# "giwo_transolver_cost_matched" (3,752,638 parameters, 2.8 % below
# Transolver's 3,860,804, with neighbour radii raised so that an epoch costs
# about as much as Transolver's). "smoke" is a tiny GIWO with short neighbour
# radii that trains on a CPU.
SIZES = dict(
    full=dict(
        latent_grid=10,
        width=15,
        n_layers=4,
        level=1,
        wavelet="db4",
        gno_mlp_hidden=(64, 64),
        in_gno_radius=0.3,
        out_gno_radius=1.65,
    ),
    smoke=dict(
        latent_grid=8,
        width=8,
        n_layers=2,
        level=1,
        wavelet="db4",
        gno_mlp_hidden=(32, 32),
        in_gno_radius=0.22,
        out_gno_radius=0.6,
    ),
)

# The box the fixed latent grid spans, in the normalised coordinates that
# load_shapenet_car returns: the extent of every fold-0 car (train and
# validation) plus a 10 % margin, as in scripts/benchmarks/shapenetcar.
DOMAIN_MIN = np.array([-2.0, -2.5, -2.5], dtype=np.float32)
DOMAIN_MAX = np.array([2.0, 6.2, 2.7], dtype=np.float32)


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
# The loader returns a dict with ``train`` and ``val`` (lists of cars),
# ``coef_norm`` (the per-channel mean and standard deviation of the training
# cars, already applied to the inputs and targets) and ``model_kwargs``. Each
# car is a dict with ``x`` of shape (N, 7) = (x, y, z, signed distance,
# normal_x, normal_y, normal_z), ``y`` of shape (N, 4) = (velocity_x,
# velocity_y, velocity_z, pressure), ``pos`` (N, 3) and ``surf`` (N,), a
# boolean mask that is True on the car surface. The first read of a car parses
# the VTK files; the result is cached in ``cache_dir``.


def load_data(data_dir: Path, cache_dir: Path | None, fold_id: int = 0) -> dict:
    """Load one train/validation split and print what was read."""
    data = load_shapenet_car(data_dir, fold_id=fold_id, cache_dir=cache_dir)
    n_points = [len(s["x"]) for s in data["train"] + data["val"]]
    print(
        f"cars: {len(data['train'])} train / {len(data['val'])} val | "
        f"points per car: min {min(n_points)}, max {max(n_points)}"
    )
    return data


# %%
# Latent grid and neighbour lists
# -------------------------------
# GIWO, like GINO, works through a fixed regular grid of ``latent_grid**3``
# points over ``DOMAIN_MIN .. DOMAIN_MAX``, the same for every car. Two
# neighbour lists connect a car to it. ``neighbors_in``: for every grid
# point, the car points within ``in_gno_radius`` (the input operator averages
# over them).
# ``neighbors_out``: for every car point, the grid points within
# ``out_gno_radius`` (the output operator sums over them). A KD-tree search
# needs concrete numbers, so it runs once per car before training, not inside
# the jitted step. A grid point far from a given car has no neighbours and
# contributes nothing for that car, which is expected.
#
# Every car finds a different number of neighbours. JAX compiles the training
# step once per input shape, so the lists of all cars are padded to the
# longest one, with every padding edge sent to an extra segment that the
# reduction drops. Without padding, every car would trigger a new compile.


def build_latent_grid(latent_grid: int) -> np.ndarray:
    """The ``(g, g, g, 3)`` regular grid over the fixed domain, axes (x, y, z)."""
    axes = [
        np.linspace(lo, hi, latent_grid, dtype=np.float32)
        for lo, hi in zip(DOMAIN_MIN, DOMAIN_MAX, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def pad_neighbors(neighbors: dict, n_segments: int, max_edges: int) -> dict:
    """Pad the edge arrays to ``max_edges``; padding goes to segment ``n_segments``."""
    index = np.asarray(neighbors["neighbors_index"])
    segment_ids = np.asarray(neighbors["segment_ids"])
    n_pad = max_edges - index.shape[0]
    return {
        "neighbors_index": np.concatenate([index, np.zeros(n_pad, dtype=index.dtype)]),
        "segment_ids": np.concatenate(
            [segment_ids, np.full(n_pad, n_segments, dtype=segment_ids.dtype)]
        ),
        "counts": neighbors["counts"],
    }


def attach_neighbors(samples: list[dict], size: dict) -> dict:
    """Add padded ``neighbors_in``/``neighbors_out`` to every car, in place."""
    latent = build_latent_grid(size["latent_grid"]).reshape(-1, 3)
    raw = []
    for s in samples:
        pos = np.asarray(s["x"][:, :3], dtype=np.float32)
        n_in = neighbors_to_segments(
            kdtree_neighbor_search(
                data=pos, queries=latent, radius=size["in_gno_radius"]
            )
        )
        n_out = neighbors_to_segments(
            kdtree_neighbor_search(
                data=latent, queries=pos, radius=size["out_gno_radius"]
            )
        )
        raw.append((n_in, n_out, pos.shape[0]))
    max_in = max(int(n_in["neighbors_index"].shape[0]) for n_in, _, _ in raw)
    max_out = max(int(n_out["neighbors_index"].shape[0]) for _, n_out, _ in raw)
    for s, (n_in, n_out, n_car) in zip(samples, raw, strict=True):
        s["neighbors_in"] = pad_neighbors(n_in, latent.shape[0], max_in)
        s["neighbors_out"] = pad_neighbors(n_out, n_car, max_out)
    print(f"edges per car (padded): in {max_in:,} | out {max_out:,}")
    return dict(max_in=max_in, max_out=max_out)


# %%
# Build the model
# ---------------
# GIWO lives in ``scirex.experimental``, which is not public API, so this
# script assembles it from public blocks; the submodule names (``gno_in``,
# ``wno``, ``gno_out``, ``projection``) are the same, so the weights of the
# benchmark's GIWO load into it unchanged. The input graph operator averages
# the car's 4 features (signed distance and normal) over the neighbours of
# every grid point; its positional embedding of the coordinates is built in.
# ``WNO`` works on the ``(g, g, g)`` grid in float32 with its own lifting and
# projection, and adds the grid coordinates as channels (``use_grid=True``).
# The output graph operator sums the grid features over the neighbours of
# every car point, and a pointwise MLP maps them to the 4 output fields. The
# wavelet weights are real, so plain Adam is correct.


class GIWO(nn.Module):
    """GNO-in -> WNO on the latent grid -> GNO-out -> pointwise projection."""

    in_channels: int
    out_channels: int
    latent_grid: int
    width: int
    n_layers: int
    level: int
    wavelet: str
    gno_mlp_hidden: tuple[int, ...]
    in_gno_radius: float
    out_gno_radius: float

    def setup(self) -> None:
        gno = dict(
            coord_dim=3,
            pos_embedding_type="nerf",
            pos_embedding_channels=32,
            pos_embedding_max_positions=10000,
            channel_mlp_layers=list(self.gno_mlp_hidden),
            channel_mlp_non_linearity=nn.gelu,
            use_torch_scatter_reduce=False,
            use_open3d_neighbor_search=True,
        )
        self.gno_in = GNOBlock(
            in_channels=self.in_channels,
            out_channels=self.width,
            radius=self.in_gno_radius,
            reduction="mean",
            weighting_fn=None,
            transform_type="nonlinear_kernelonly",
            **gno,
        )
        self.wno = WNO(
            in_channels=self.width,
            out_channels=self.width,
            hidden_channels=self.width,
            n_layers=self.n_layers,
            level=self.level,
            size=(self.latent_grid,) * 3,
            wavelet=self.wavelet,
            mode="symmetric",
            use_grid=True,
            compute_dtype=jnp.float32,
        )
        self.gno_out = GNOBlock(
            in_channels=self.width,
            out_channels=self.width,
            radius=self.out_gno_radius,
            reduction="sum",
            weighting_fn=None,
            transform_type="linear",
            **gno,
        )
        self.projection = ChannelMLP(
            in_channels=self.width,
            out_channels=self.out_channels,
            hidden_channels=4 * self.width,
            n_layers=2,
            n_dim=1,
        )

    def __call__(self, pos, latent, features, neighbors_in, neighbors_out):
        pos, latent = pos[0], latent[0]
        grid = latent.reshape(-1, 3)
        h = self.gno_in(y=pos, x=grid, f_y=features, neighbors=neighbors_in)
        h = self.wno(h.reshape(1, *latent.shape[:-1], -1))
        h = h.reshape(1, -1, self.width)
        out = self.gno_out(y=grid, x=pos, f_y=h, neighbors=neighbors_out)
        out = jnp.transpose(out, (0, 2, 1))
        return jnp.transpose(self.projection(out), (0, 2, 1))


class CarGIWO(nn.Module):
    """GIWO on one car: ``x`` (1, N, 7) -> (1, N, 4)."""

    latent_grid: int
    width: int
    n_layers: int
    level: int
    wavelet: str
    gno_mlp_hidden: tuple[int, ...]
    in_gno_radius: float
    out_gno_radius: float
    out_dim: int = 4

    @nn.compact
    def __call__(self, x, neighbors_in, neighbors_out):
        pos, features = x[:, :, :3], x[:, :, 3:]
        latent = jnp.asarray(build_latent_grid(self.latent_grid))[None]
        return GIWO(
            in_channels=features.shape[-1],
            out_channels=self.out_dim,
            latent_grid=self.latent_grid,
            width=self.width,
            n_layers=self.n_layers,
            level=self.level,
            wavelet=self.wavelet,
            gno_mlp_hidden=self.gno_mlp_hidden,
            in_gno_radius=self.in_gno_radius,
            out_gno_radius=self.out_gno_radius,
        )(pos, latent, features, neighbors_in, neighbors_out)


def build_model(size: dict, sample: dict, seed: int):
    """Build the GIWO, initialise it on one car and print the parameter count."""
    model = CarGIWO(**size)
    params = model.init(
        jax.random.PRNGKey(seed),
        jnp.asarray(sample["x"])[None],
        sample["neighbors_in"],
        sample["neighbors_out"],
    )
    print(f"GIWO {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss is computed on standardised fields, per car: the mean squared error
# of the velocity at all points, plus ``press_weight`` times the mean squared
# error of the pressure at the surface points only (the pressure is only
# defined on the car surface). It is the same loss for every model on this
# benchmark.


def make_loss(model, press_weight: float):
    """Return ``loss_fn(params, x, y, surf, nbr_in, nbr_out) -> (total, parts)``."""

    def loss_fn(params, x, y, surf, neighbors_in, neighbors_out):
        out = model.apply(params, x, neighbors_in, neighbors_out)[0]
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
# steps and then decays. Because the neighbour lists are padded, every car has
# the same input shapes and JAX compiles the step once; the first epoch is
# still slower than the others because of that compile.


def as_arrays(s: dict) -> tuple:
    """One car's model inputs and targets as JAX arrays."""
    return (
        jnp.asarray(s["x"])[None],
        jnp.asarray(s["y"]),
        jnp.asarray(s["surf"]),
        jax.tree_util.tree_map(jnp.asarray, s["neighbors_in"]),
        jax.tree_util.tree_map(jnp.asarray, s["neighbors_out"]),
    )


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
    def train_step(params, opt_state, x, y, surf, nbr_in, nbr_out):
        grads, aux = jax.grad(loss_fn, has_aux=True)(
            params, x, y, surf, nbr_in, nbr_out
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    @jax.jit
    def eval_step(params, x, y, surf, nbr_in, nbr_out):
        return loss_fn(params, x, y, surf, nbr_in, nbr_out)[1]

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
    predict = jax.jit(lambda p, x, ni, no: model.apply(p, x, ni, no)[0])
    volume, surface, predictions = [], [], []
    for s in samples:
        x, _, _, nbr_in, nbr_out = as_arrays(s)
        out = np.asarray(predict(params, x, nbr_in, nbr_out))
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
    fig.suptitle("Car-Design-ShapeNetCar, GIWO, one validation car")
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
# ``--smoke`` symlinks a few cars, uses the tiny GIWO and trains for two
# epochs. Without it the script trains the benchmark-size GIWO on the whole
# fold of ``--data-dir`` for 200 epochs. That mode is meant for a GPU.


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
        "--out-dir", type=Path, default=Path("outputs/shapenetcar_giwo_tutorial")
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
    attach_neighbors(data["train"] + data["val"], size)
    print(f"data loading and neighbour search: {perf_counter() - t0:.1f}s")
    model, params = build_model(size, data["train"][0], args.seed)

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
