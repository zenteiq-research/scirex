# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting Darcy flow with GINO
===============================

**Problem.** Given the permeability ``a(x)`` of a two-phase porous medium on
the unit square (piecewise constant, 3 or 12), predict the steady pressure
``u(x)`` of ``-div(a grad u) = 1`` with ``u = 0`` on the boundary. The data
are Li et al.'s released ``piececonst_r421_N1024`` files, strided from
421x421 to 85x85. The model is the Geometry-Informed Neural Operator (GINO,
Li et al. 2023): a graph neural operator carries the input from a point
cloud to a latent grid, an FNO works on that grid, and a second graph
neural operator carries the result back to the query points.

**What you will learn.**

1. How to read the Darcy ``.mat`` files and build the benchmark's 1000 / 200
   split at 85x85, with per-pixel normalisation.
2. How to treat a grid as a point cloud for ``scirex.models.GINO``: the point
   coordinates, the latent grid and the fixed-radius neighbour lists, built
   once with ``scirex.layers``.
3. How to build GINO from the public ``GINODarcy2d`` preset and count its
   parameters, complex weights included.
4. How to train with the complex-aware ``scirex.training.adamw_cx``, a
   one-cycle learning rate, gradient clipping and Transolver's
   ``relative L2 + 0.1 x gradient`` loss.
5. How to evaluate with the per-sample relative L2 error in physical units,
   and how to plot permeability, truth, prediction and error.

**Caveat.** The ``--smoke`` mode trains a small GINO on a few samples for a
few epochs. It checks that the pipeline works and gives poor predictions. It
is not a benchmark number. For the benchmark protocol (1000 training
samples, 2000 epochs, 2,801,340 parameters) use
``scripts/benchmarks/darcy/train.py``; see the "Reproduce at full scale"
section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/darcy_gino.py --data-dir ~/data/darcy --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.GINO``, the
# ``scirex.config.GINODarcy2d`` preset, ``scirex.layers.kdtree_neighbor_search``
# and ``neighbors_to_segments``, ``scirex.training.adamw_cx`` and
# ``scirex.utils.count_model_params``. The data are read with SciPy and the
# loop is plain JAX and optax.

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
from scipy.io import loadmat

import scirex
from scirex.config import GINODarcy2d
from scirex.layers import kdtree_neighbor_search, neighbors_to_segments
from scirex.models import GINO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# Data split of the Darcy benchmark, the same values as PROTOCOL in
# scripts/benchmarks/darcy/_config.py.
PROTOCOL = dict(
    train_file="piececonst_r421_N1024_smooth1.mat",
    test_file="piececonst_r421_N1024_smooth2.mat",
    n_train=1000,
    n_test=200,
    subsample_rate=5,  # 421 -> 85, as in Li et al.'s own code
)

# GINO's training recipe on this benchmark, RUNS["gino"].opt in the same file:
# AdamW (adamw_cx) with weight decay 1e-4, cosine one-cycle lr with peak 1e-3,
# gradient clipping 1.0, batch 16, 2000 epochs, rel. L2 + 0.1 x gradient loss.
RECIPE = dict(
    learning_rate=1e-3,
    weight_decay=1e-4,
    adam_eps=1e-8,
    grad_clip=1.0,
    grad_loss_weight=0.1,
    batch_size=16,
    epochs=2000,
)

# Model sizes: the FNO inside GINO, and the hidden layers of the two GNO
# channel MLPs. "full" is the benchmark's GINO (2,801,340 parameters, matched
# to Transolver's 2,826,945), with the preset's GNO MLPs. "smoke" is a small
# GINO that trains on a CPU.
SIZES = dict(
    full=dict(hidden_channels=31, modes=16, n_layers=5),
    smoke=dict(
        hidden_channels=16,
        modes=12,
        n_layers=2,
        in_gno_channel_mlp_hidden_layers=[32, 32],
        out_gno_channel_mlp_hidden_layers=[32, 32],
    ),
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
# Load the data
# -------------
# Each ``.mat`` file holds 1024 samples on the 421x421 grid: ``coeff``, the
# permeability (3 or 12), and ``sol``, the pressure. The benchmark trains on
# the first 1000 samples of ``smooth1`` and tests on the first 200 of
# ``smooth2``, and keeps every 5th grid point in both directions (85x85), as
# Li et al.'s own code does.


def load_darcy(
    data_dir: Path, n_train: int, n_test: int, subsample_rate: int = 5
) -> dict:
    """Read the split: ``a`` and ``u`` as ``(n, s, s)`` float32 arrays."""
    out = {}
    for name, file, n in (
        ("train", PROTOCOL["train_file"], n_train),
        ("test", PROTOCOL["test_file"], n_test),
    ):
        path = Path(data_dir) / file
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        raw = loadmat(path.as_posix(), variable_names=("coeff", "sol"))
        if raw["coeff"].shape[0] < n:
            raise ValueError(f"{path} has only {raw['coeff'].shape[0]} samples")
        r = subsample_rate
        out[f"a_{name}"] = raw["coeff"][:n, ::r, ::r].astype(np.float32)
        out[f"u_{name}"] = raw["sol"][:n, ::r, ::r].astype(np.float32)
    print(
        f"samples: {n_train} train / {n_test} test | grid {out['a_train'].shape[1:]} | "
        f"permeability values {np.unique(out['a_train'])} | "
        f"pressure range [{out['u_train'].min():.4f}, {out['u_train'].max():.4f}]"
    )
    return out


# %%
# Normalise
# ---------
# As in the benchmark, the permeability and the pressure each get a per-pixel
# unit Gaussian: one mean and one standard deviation per grid point, over the
# training samples, with ``eps = 1e-5``. GINO takes the fields flattened to
# ``(n, 85 * 85)``, so the normalisers are fitted on flattened arrays. The
# network predicts a normalised pressure, which is decoded before the loss.


class PixelNormalizer:
    """``(v - mean) / (std + eps)`` with a mean and a std per grid point."""

    def __init__(self, train: np.ndarray, eps: float = 1e-5) -> None:
        self.mean = train.mean(axis=0, keepdims=True)
        self.std = train.std(axis=0, keepdims=True)
        self.eps = eps

    def encode(self, v):
        return (v - self.mean) / (self.std + self.eps)

    def decode(self, v):
        return v * (self.std + self.eps) + self.mean


# %%
# Geometry
# --------
# GINO works on point clouds. Here every sample has the same points, the
# 85x85 grid over [0, 1]^2, flattened in the same order as the fields, and the
# same latent grid for the FNO, so the neighbour lists are built once. Each
# GNO layer sums over the points within a radius of its query point; the
# benchmark scales the preset's radii 0.16 and 0.175 (tuned at resolution 16)
# to the grid spacing, ``0.16 * 16 / (s - 1)`` and ``0.175 * 16 / (s - 1)``.
# ``kdtree_neighbor_search`` finds the neighbours and ``neighbors_to_segments``
# turns them into the form the GNO layers take.


def gno_radii(n_side: int) -> tuple[float, float]:
    """The input and output GNO radii for an ``n_side x n_side`` grid."""
    return 0.16 * 16.0 / (n_side - 1), 0.175 * 16.0 / (n_side - 1)


def build_geometry(n_side: int) -> dict:
    """Points, latent grid and neighbour lists of the ``n_side x n_side`` grid."""
    xs = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (s * s, 2)
    in_radius, out_radius = gno_radii(n_side)
    neighbors = {
        name: neighbors_to_segments(kdtree_neighbor_search(xy, xy, radius))
        for name, radius in (("neighbors_in", in_radius), ("neighbors_out", out_radius))
    }
    edges = int(neighbors["neighbors_in"]["neighbors_index"].shape[0])
    print(
        f"geometry: {len(xy)} points | radii {in_radius:.4f}, {out_radius:.4f} | "
        f"input-graph edges {edges:,} ({edges / len(xy):.1f} per point)"
    )
    return dict(
        input_geom=jnp.asarray(xy)[None],
        latent_queries=jnp.asarray(xy.reshape(n_side, n_side, 2))[None],
        output_queries=jnp.asarray(xy)[None],
        **neighbors,
    )


# %%
# Build the model
# ---------------
# ``GINODarcy2d`` in ``scirex.config`` is the GINO preset for this problem:
# one input and one output channel, 2D coordinates, the GNO channel MLPs and
# the FNO settings. We take its values, set the input channel count, the
# radii and the FNO size of this run, and pass them to ``GINO``. The input is
# the normalised permeability at every point, ``(batch, s * s, 1)``.
# ``count_model_params`` counts a complex weight as two.


def build_model(size: dict, geometry: dict, n_side: int, seed: int):
    """Build GINO from the preset, initialise it and print the parameter count."""
    config = dict(GINODarcy2d().to_dict())
    config.pop("model_arch")
    config["in_channels"] = config.pop("data_channels")
    config["in_gno_radius"], config["out_gno_radius"] = gno_radii(n_side)
    config["fno_hidden_channels"] = size["hidden_channels"]
    config["fno_n_layers"] = size["n_layers"]
    config["fno_n_modes"] = [size["modes"], size["modes"]]
    for key in (
        "in_gno_channel_mlp_hidden_layers",
        "out_gno_channel_mlp_hidden_layers",
    ):
        config[key] = size.get(key, config[key])
    model = GINO(**config)
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    x0 = jnp.zeros((1, n_side * n_side, 1), dtype=jnp.float32)
    params = model.init(init_key, x=x0, **geometry)
    print(f"GINO {size} | parameters: {count_model_params(params):,}")
    return model, params, key


# %%
# Loss
# ----
# The loss is Transolver's Darcy loss: the mean relative L2 error of the
# decoded pressure, plus 0.1 times the relative L2 errors of its x- and
# y-derivatives. The derivatives are central differences with zero padding;
# the outer ring of the prediction is set to zero first, because ``u = 0``
# on the boundary. The metric is the plain per-sample relative L2.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all grid points."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.maximum(jnp.linalg.norm(true.reshape(b, -1), axis=1), 1e-12)


def central_diff(grid, dx: float):
    """Zero-padded central differences ``(d/dx, d/dy)`` of ``(batch, s, s)`` fields."""
    padded = jnp.pad(grid, ((0, 0), (1, 1), (1, 1)))
    grad_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * dx)
    grad_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * dx)
    return grad_x, grad_y


def gradient_loss(pred, true, dx: float):
    """Relative L2 of the x- and y-derivatives, with the prediction's border zeroed."""
    pred = pred.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0)
    pred = pred.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    (px, py), (tx, ty) = central_diff(pred, dx), central_diff(true, dx)
    return jnp.mean(rel_l2_per_sample(px, tx)) + jnp.mean(rel_l2_per_sample(py, ty))


def make_loss(model, geometry, u_norm: PixelNormalizer, n_side: int, weight: float):
    """Return ``loss_fn(params, a, u) -> rel. L2 + weight x gradient loss``."""
    dx = 1.0 / n_side

    def loss_fn(params, a, u):
        out = model.apply(params, x=a[..., None], **geometry)[..., 0]
        pred = u_norm.decode(out)  # (batch, s * s)
        loss = jnp.mean(rel_l2_per_sample(pred, u))
        shape = (-1, n_side, n_side)
        return loss + weight * gradient_loss(pred.reshape(shape), u.reshape(shape), dx)

    return loss_fn


# %%
# Optimiser
# ---------
# GINO's FNO has complex spectral weights, and ``jax.grad`` returns the
# conjugate of the steepest-ascent direction for them, so plain
# ``optax.adamw`` would move their imaginary parts the wrong way. ``adamw_cx``
# conjugates complex gradients first and keeps Adam's second moment real;
# real weights are updated exactly as by AdamW. The learning rate follows a
# cosine one-cycle schedule over the whole run, behind gradient clipping.
# ``inject_hyperparams`` would otherwise give the hyperparameters the
# parameters' highest dtype, complex64, so ``hyperparam_dtype`` keeps them
# float32.


def make_optimizer(recipe: dict, total_steps: int):
    """Gradient clipping, then ``adamw_cx`` on a cosine one-cycle learning rate."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps, peak_value=recipe["learning_rate"]
    )
    adam = optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule,
        weight_decay=recipe["weight_decay"],
        eps=recipe["adam_eps"],
    )
    return optax.chain(optax.clip_by_global_norm(recipe["grad_clip"]), adam)


# %%
# Train
# -----
# Batches of 16 samples in a new random order every epoch, one optimiser step
# per batch; the last incomplete batch is dropped, as in the benchmark. Every
# ``eval_every`` epochs the test samples are scored, to draw the error curve;
# no epoch is selected by it, and the weights after the last epoch are the
# result. JAX compiles the step on its first call, so the first epoch is
# slower than the others.


def train(
    model, params, key, geometry, data, u_norm, epochs, eval_every=10, recipe=RECIPE
):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    a_train, u_train = jnp.asarray(data["a_train"]), jnp.asarray(data["u_train"])
    bs = recipe["batch_size"]
    n_train = a_train.shape[0]
    n_side = round(np.sqrt(a_train.shape[1]))
    steps_per_epoch = max(n_train // bs, 1)
    tx = make_optimizer(recipe, steps_per_epoch * epochs)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, geometry, u_norm, n_side, recipe["grad_loss_weight"])

    @jax.jit
    def train_step(params, opt_state, a, u):
        loss, grads = jax.value_and_grad(loss_fn)(params, a, u)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n_train)
        total = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * bs : (b + 1) * bs]
            params, opt_state, loss = train_step(
                params, opt_state, a_train[idx], u_train[idx]
            )
            total += float(loss)
        secs = perf_counter() - t0
        if epoch % eval_every == 0 or epoch == epochs - 1:
            test = evaluate(model, params, geometry, data, u_norm)["test_rel_l2"]
            history.append(
                dict(epoch=epoch, train_loss=total / steps_per_epoch, test_rel_l2=test)
            )
            print(
                f"epoch {epoch}: train_loss={total / steps_per_epoch:.4f} "
                f"test_rel_l2={test:.4f} epoch_time={secs:.1f}s"
            )
    return params, history


# %%
# Evaluate
# --------
# The relative L2 error of every test sample, in physical units. The mean is
# the benchmark's number; the median, the 90th percentile and the worst
# sample show how the error is spread.


def evaluate(model, params, geometry, data, u_norm, batch_size: int = 16) -> dict:
    """Per-sample relative L2 on the test samples, and the predictions."""
    predict = jax.jit(
        lambda p, a: u_norm.decode(model.apply(p, x=a[..., None], **geometry)[..., 0])
    )
    a_test, u_test = data["a_test"], data["u_test"]
    preds = np.concatenate(
        [
            np.asarray(predict(params, jnp.asarray(a_test[i : i + batch_size])))
            for i in range(0, len(a_test), batch_size)
        ]
    )
    errors = np.asarray(rel_l2_per_sample(jnp.asarray(preds), jnp.asarray(u_test)))
    return dict(
        test_rel_l2=float(errors.mean()),
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        worst=float(errors.max()),
        errors=errors,
        predictions=preds,
    )


# %%
# Plot
# ----
# The standard figure of the benchmark for one test sample: the input
# permeability, the true and the predicted pressure on one colour scale, and
# the absolute error. We also plot the loss and the test error per epoch.


def plot_prediction(a, truth, pred, path: Path, title: str) -> None:
    """Save permeability, true pressure, predicted pressure and absolute error."""
    lim = (float(truth.min()), float(truth.max()))
    panels = [
        ("Input: permeability a", a, "viridis", None),
        ("Truth: pressure u", truth, "jet", lim),
        ("Prediction: pressure u", pred, "jet", lim),
        ("Absolute error", np.abs(pred - truth), "magma", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    for ax, (name, values, cmap, vlim) in zip(axes, panels):
        vmin, vmax = vlim if vlim else (None, None)
        im = ax.imshow(
            values, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=(0, 1, 0, 1)
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training loss and the test relative L2 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax.plot(epochs, [h["test_rel_l2"] for h in history], marker="o", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / relative L2 (mean over samples)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few samples, uses the small GINO, trains for a few
# epochs and scores the test samples after every epoch. Without it the script
# trains the benchmark-size GINO on the full split for 2000 epochs and scores
# every 10 epochs, as ``train.py`` does. That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("DARCY_DATA_DIR"),
        help="folder with piececonst_r421_N1024_smooth{1,2}.mat (or $DARCY_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few samples, small GINO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/darcy_gino_tutorial"))
    ap.add_argument("--n-train", type=int, default=96, help="samples in smoke mode")
    ap.add_argument("--n-test", type=int, default=20, help="samples in smoke mode")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $DARCY_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PROTOCOL
    if args.smoke:
        n_train, n_test, eval_every = args.n_train, args.n_test, 1
    else:
        n_train, n_test, eval_every = p["n_train"], p["n_test"], 10
    epochs = args.epochs or (5 if args.smoke else RECIPE["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    raw = load_darcy(
        Path(args.data_dir).expanduser(), n_train, n_test, p["subsample_rate"]
    )
    n_side = raw["a_train"].shape[1]
    flat = {k: v.reshape(len(v), -1) for k, v in raw.items()}
    a_norm, u_norm = PixelNormalizer(flat["a_train"]), PixelNormalizer(flat["u_train"])
    data = dict(
        a_train=a_norm.encode(flat["a_train"]),
        a_test=a_norm.encode(flat["a_test"]),
        u_train=flat["u_train"],
        u_test=flat["u_test"],
    )
    geometry = build_geometry(n_side)
    print(f"data loading and geometry: {perf_counter() - t0:.1f}s")
    model, params, key = build_model(size, geometry, n_side, args.seed)

    params, history = train(
        model, params, key, geometry, data, u_norm, epochs, eval_every
    )
    metrics = evaluate(model, params, geometry, data, u_norm)
    print(
        f"test relative L2: mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} | p90 {metrics['p90']:.4f} | "
        f"worst {metrics['worst']:.4f}"
    )

    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        raw["a_test"][k],
        raw["u_test"][k],
        metrics["predictions"][k].reshape(n_side, n_side),
        out_dir / "prediction.png",
        f"Darcy, GINO: test sample with the median error ({metrics['errors'][k]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
