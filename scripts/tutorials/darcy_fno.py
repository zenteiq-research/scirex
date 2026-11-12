# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting Darcy flow with FNO
==============================

**Problem.** Given the permeability ``a(x)`` of a two-phase porous medium on
the unit square (piecewise constant, 3 or 12), predict the steady pressure
``u(x)`` of ``-div(a grad u) = 1`` with ``u = 0`` on the boundary. The data
are Li et al.'s released ``piececonst_r421_N1024`` files, strided from
421x421 to 85x85. The model is the Fourier Neural Operator (FNO, Li et al.
2021), which mixes information globally with a truncated Fourier transform in
every layer.

**What you will learn.**

1. How to read the Darcy ``.mat`` files and build the benchmark's 1000 / 200
   split at 85x85, with per-pixel normalisation.
2. How to build ``scirex.models.FNO`` for a scalar field on a grid and count
   its parameters, complex weights included.
3. How to train it with the complex-aware AdamW ``scirex.training.adamw_cx``
   on a step-decay learning rate, as in the FNO paper.
4. How to evaluate with the per-sample relative L2 error in physical units,
   and how to plot permeability, truth, prediction and error.

**Caveat.** The ``--smoke`` mode trains a tiny FNO on a few samples for a few
epochs. It checks that the pipeline works and gives poor predictions. It is
not a benchmark number. For the benchmark protocol (1000 training samples,
500 epochs, 2,836,281 parameters) use ``scripts/benchmarks/darcy/train.py``;
see the "Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/darcy_fno.py --data-dir ~/data/darcy --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.FNO``,
# ``scirex.training.adamw_cx`` and ``scirex.utils.count_model_params``. The
# data are read with SciPy and the loop is plain JAX and optax.

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
from scirex.models import FNO
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

# FNO's training recipe on this benchmark, RUNS["fno"].opt in the same file:
# Adam(W) with no weight decay, lr 1e-3 halved every 100 epochs, batch 20.
RECIPE = dict(
    learning_rate=1e-3,
    weight_decay=0.0,
    adam_eps=1e-8,
    decay_epochs=100,
    decay_rate=0.5,
    batch_size=20,
    epochs=500,
)

# Model sizes. "full" is the benchmark's FNO (2,836,281 parameters, matched to
# Transolver's 2,826,945). "smoke" is a tiny FNO that trains on a CPU.
SIZES = dict(
    full=dict(modes=20, hidden_channels=40, n_layers=4),
    smoke=dict(modes=12, hidden_channels=16, n_layers=2),
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
# As in Li et al.'s code, the permeability and the pressure each get a
# per-pixel unit Gaussian: one mean and one standard deviation per grid point,
# over the training samples, with ``eps = 1e-5``. The inputs are encoded once.
# The network predicts a normalised pressure, which is decoded before the
# loss, so the loss and the metric are in physical units.


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
# Build the model
# ---------------
# ``FNO`` is channels-first, so a batch is ``(batch, 1, 85, 85)``: one input
# channel, the permeability. By default FNO appends the (x, y) coordinates of
# the grid as two more channels. ``n_modes=(20, 20)`` keeps 20 Fourier modes
# along each axis in total (10 positive and 10 negative frequencies on the
# first axis). ``count_model_params`` counts a complex weight as two.


def build_model(size: dict, sample: np.ndarray, seed: int):
    """Build the FNO, initialise it on one sample and print the parameter count."""
    model = FNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=size["hidden_channels"],
        n_modes=(size["modes"], size["modes"]),
        n_layers=size["n_layers"],
    )
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    params = model.init(init_key, jnp.asarray(sample)[None])
    print(f"FNO {size} | parameters: {count_model_params(params):,}")
    return model, params, key


# %%
# Loss
# ----
# The loss of a batch is the mean over its samples of the relative L2 error on
# the decoded (physical) pressure. The metric is the same per-sample relative
# L2, averaged over the test samples.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all grid points."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.maximum(jnp.linalg.norm(true.reshape(b, -1), axis=1), 1e-12)


def make_loss(model, u_norm: PixelNormalizer):
    """Return ``loss_fn(params, a, u) -> mean relative L2`` in physical units."""

    def loss_fn(params, a, u):
        pred = u_norm.decode(model.apply(params, a))  # (batch, 1, s, s)
        return jnp.mean(rel_l2_per_sample(pred, u))

    return loss_fn


# %%
# Optimiser
# ---------
# FNO's spectral weights are complex, and ``jax.grad`` returns the conjugate
# of the steepest-ascent direction for them, so plain ``optax.adamw`` would
# move their imaginary parts the wrong way. ``adamw_cx`` conjugates complex
# gradients first and keeps Adam's second moment real; real weights are
# updated exactly as by AdamW. The learning rate is halved every 100 epochs.
# ``inject_hyperparams`` would otherwise give the hyperparameters the
# parameters' highest dtype, complex64, so ``hyperparam_dtype`` keeps them
# float32.


def make_optimizer(recipe: dict, steps_per_epoch: int):
    """``adamw_cx`` on a step-decay learning rate."""
    schedule = optax.exponential_decay(
        init_value=recipe["learning_rate"],
        transition_steps=steps_per_epoch * recipe["decay_epochs"],
        decay_rate=recipe["decay_rate"],
        staircase=True,
    )
    return optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule,
        weight_decay=recipe["weight_decay"],
        eps=recipe["adam_eps"],
    )


# %%
# Train
# -----
# Batches of 20 samples in a new random order every epoch, one optimiser step
# per batch; the last incomplete batch is dropped, as in the benchmark. Every
# ``eval_every`` epochs the test samples are scored, to draw the error curve;
# no epoch is selected by it, and the weights after the last epoch are the
# result. JAX compiles the step on its first call, so the first epoch is
# slower than the others.


def train(model, params, key, data, u_norm, epochs, eval_every=10, recipe=RECIPE):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    a_train, u_train = jnp.asarray(data["a_train"]), jnp.asarray(data["u_train"])
    bs = recipe["batch_size"]
    n_train = a_train.shape[0]
    steps_per_epoch = max(n_train // bs, 1)
    tx = make_optimizer(recipe, steps_per_epoch)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, u_norm)

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
            test = evaluate(model, params, data, u_norm)["test_rel_l2"]
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


def evaluate(model, params, data, u_norm, batch_size: int = 20) -> dict:
    """Per-sample relative L2 on the test samples, and the predictions."""
    predict = jax.jit(lambda p, a: u_norm.decode(model.apply(p, a)))
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
        predictions=preds[:, 0],
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
    ax.set_ylabel("relative L2 (mean over samples)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few samples, uses the tiny FNO, trains for a few epochs
# and scores the test samples after every epoch. Without it the script trains
# the benchmark-size FNO on the full split for 500 epochs and scores every 10
# epochs, as ``train.py`` does. That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("DARCY_DATA_DIR"),
        help="folder with piececonst_r421_N1024_smooth{1,2}.mat (or $DARCY_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few samples, tiny FNO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/darcy_fno_tutorial"))
    ap.add_argument("--n-train", type=int, default=100, help="samples in smoke mode")
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
    a_norm = PixelNormalizer(raw["a_train"][:, None])
    u_norm = PixelNormalizer(raw["u_train"][:, None])
    data = dict(
        a_train=a_norm.encode(raw["a_train"][:, None]),
        a_test=a_norm.encode(raw["a_test"][:, None]),
        u_train=raw["u_train"][:, None],
        u_test=raw["u_test"][:, None],
    )
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params, key = build_model(size, data["a_train"][0], args.seed)

    params, history = train(model, params, key, data, u_norm, epochs, eval_every)
    metrics = evaluate(model, params, data, u_norm)
    print(
        f"test relative L2: mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} | p90 {metrics['p90']:.4f} | "
        f"worst {metrics['worst']:.4f}"
    )

    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        raw["a_test"][k],
        raw["u_test"][k],
        metrics["predictions"][k],
        out_dir / "prediction.png",
        f"Darcy, FNO: test sample with the median error ({metrics['errors'][k]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
