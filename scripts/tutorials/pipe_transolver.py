# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting flow through a deformed pipe with Transolver (Pipe)
==============================================================

**Problem.** Given the coordinates of the 129x129 structured mesh of a pipe
whose centreline is bent differently in every sample, predict the steady
horizontal velocity at every mesh node. The flow is steady, incompressible
Navier-Stokes flow with a parabolic inlet profile (Geo-FNO, Li et al. 2023);
the model is Transolver (Wu et al. 2024), which replaces attention over all
pairs of nodes by attention over a small set of learned "slices".

**What you will learn.**

1. How to read the three Pipe ``.npy`` files and build the 1000 / 200 split
   of the benchmark, with unit-Gaussian normalisation.
2. How to feed a structured mesh to ``scirex.models.Transolver``: the nodes
   as a flat sequence of coordinates, with ``geometry="structured_2d"`` so
   that the slices are computed with convolutions on the mesh.
3. How to match the PyTorch reference's numerics (exact GELU, LayerNorm eps,
   convolution and linear-layer initialisation) and count the parameters.
4. How to train it with the benchmark's recipe: AdamW (all weights are real),
   gradient clipping and a one-cycle learning rate.
5. How to evaluate with the per-sample relative L2 error in physical units,
   and how to plot truth, prediction and error on the physical mesh.

**Caveat.** The ``--smoke`` mode trains a tiny Transolver on a few samples
for a few epochs. It checks that the pipeline works and gives poor
predictions. It is not a benchmark number. For the benchmark protocol (1000
training samples, 500 epochs, 3,073,985 parameters, 3 seeds) use
``scripts/benchmarks/pipe/train.py``; see the "Reproduce at full scale"
section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/pipe_transolver.py --data-dir ~/data/pipe --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.Transolver``,
# ``scirex.training.optim.scale_by_adam_cx`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import functools
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
from scirex.models import Transolver
from scirex.training.optim import scale_by_adam_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# Training settings of the Pipe benchmark, the same values as PROTOCOL in
# scripts/benchmarks/pipe/_config.py (Transolver's recipe, used by every model).
PROTOCOL = dict(
    ntrain=1000,
    ntest=200,
    n_total=1200,
    epochs=500,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-5,
    adam_eps=1e-8,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    max_grad_norm=0.1,
)

# Model sizes. "full" is the paper's launch-script Transolver
# (Transolver_Pipe.sh, 3,073,985 parameters), the size every other
# model on the benchmark is matched to. "smoke" is a tiny Transolver that
# trains on a CPU.
SIZES = dict(
    full=dict(
        n_hidden=128,
        n_layers=8,
        n_head=8,
        mlp_ratio=2,
        slice_num=64,
        ref=8,
        dropout_rate=0.0,
    ),
    smoke=dict(
        n_hidden=32,
        n_layers=2,
        n_head=4,
        mlp_ratio=1,
        slice_num=16,
        ref=8,
        dropout_rate=0.0,
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
# ``Pipe_X.npy`` and ``Pipe_Y.npy`` hold the x and y coordinates of every mesh
# node, ``(2310, 129, 129)``; ``Pipe_Q.npy`` holds three solution channels,
# ``(2310, 3, 129, 129)``. The target is channel 0, the horizontal velocity.
# The benchmark uses the first 1200 samples: the first ``ntrain`` train and
# the last ``ntest`` of the 1200 test. The files are memory-mapped, so only
# the samples that are used are read from disk. ``stride`` keeps every
# ``stride``-th node along both mesh axes: 1, all 129 x 129 nodes, as in the
# benchmark, or 2, 65 x 65 nodes, for faster smoke runs.


def load_pipe(
    data_dir: Path, ntrain: int, ntest: int, n_total: int = 1200, stride: int = 1
) -> dict:
    """Read the split. Coordinates ``(n, H, W, 2)``, velocity ``(n, H, W)``."""
    data_dir = Path(data_dir)
    xs = np.load(data_dir / "Pipe_X.npy", mmap_mode="r")
    ys = np.load(data_dir / "Pipe_Y.npy", mmap_mode="r")
    q = np.load(data_dir / "Pipe_Q.npy", mmap_mode="r")
    if q.shape[0] < n_total or ntrain + ntest > n_total:
        raise ValueError(f"need {ntrain} + {ntest} <= {n_total} <= {q.shape[0]}")
    split = dict(train=np.arange(ntrain), test=np.arange(n_total - ntest, n_total))
    out = {}
    for name, idx in split.items():
        coords = np.stack(
            [xs[idx][:, ::stride, ::stride], ys[idx][:, ::stride, ::stride]], axis=-1
        )
        out[f"x_{name}"] = coords.astype(np.float32)
        out[f"y_{name}"] = np.asarray(q[idx, 0][:, ::stride, ::stride], np.float32)
    print(
        f"samples: {ntrain} train / {ntest} test | mesh {out['x_train'].shape[1:3]} | "
        f"velocity range [{out['y_train'].min():.3f}, {out['y_train'].max():.3f}]"
    )
    return out


# %%
# Normalise
# ---------
# As in the benchmark, the inputs and the targets each get one unit Gaussian,
# fitted on the training samples over the sample and node axes. The inputs
# are encoded once. The targets stay in physical units: the network's output
# is decoded before the loss, so the loss and the metric are in physical units.


class UnitGaussian:
    """``(v - mean) / (std + 1e-8)`` with one mean and std per channel."""

    def __init__(self, train: np.ndarray, axes: tuple[int, ...]) -> None:
        self.mean = train.mean(axis=axes, keepdims=True)[0]
        self.std = train.std(axis=axes, keepdims=True)[0] + 1e-8

    def encode(self, v):
        return (v - self.mean) / self.std

    def decode(self, v):
        return v * self.std + self.mean


# %%
# Build the model
# ---------------
# ``Transolver`` takes the nodes as a sequence, ``(batch, 129 * 129, 2)``.
# Every block projects the nodes onto ``slice_num`` learned slices (soft
# assignments of the nodes to "physical states"), runs attention between the
# slices and sends the result back to every node, so the cost grows linearly
# with the number of nodes. ``geometry="structured_2d"`` with ``H`` and ``W``
# tells it that the sequence is a 129 x 129 mesh, and the projections onto
# the slices are then convolutions on the mesh. ``fun_dim=0`` and
# ``unified_pos=False``: the input is the coordinates only.
#
# The other arguments match the paper's PyTorch code, as the benchmark does:
# exact (erf) GELU instead of Flax's tanh approximation, LayerNorm eps 1e-5
# instead of Flax's 1e-6 and PyTorch's default initialisation of the slice
# convolutions (``conv_init="torch"``). ``match_reference_init`` then applies
# the reference's ``initialize_weights``: every linear kernel drawn from
# N(0, 0.02^2), except the slice projection of every block, which keeps its
# orthogonal initialisation.


def build_model(size: dict, sample: np.ndarray, grid, seed: int):
    """Build the Transolver, initialise it as the reference does, count parameters."""
    model = Transolver(
        space_dim=2,
        fun_dim=0,
        out_dim=1,
        unified_pos=False,
        geometry="structured_2d",
        H=grid[0],
        W=grid[1],
        act=functools.partial(jax.nn.gelu, approximate=False),
        layer_norm_epsilon=1e-5,
        conv_init="torch",
        **size,
    )
    params = model.init(jax.random.PRNGKey(seed), x=jnp.asarray(sample)[None], fx=None)
    params = match_reference_init(params, seed, size["n_layers"])
    print(f"Transolver {size} | parameters: {count_model_params(params):,}")
    return model, params


def match_reference_init(params, seed: int, n_layers: int):
    """Redraw every 2D linear kernel from N(0, 0.02^2), except orthogonal ones.

    The slice projection of each block (one per layer) is initialised
    orthogonally, in the reference as here; it is recognised by its Gram
    matrix. Convolution kernels (4D) are left as they are.
    """
    key = jax.random.PRNGKey(seed)
    count, kept = [0], [0]

    def redraw(path, leaf):
        if getattr(path[-1], "key", None) != "kernel" or leaf.ndim != 2:
            return leaf
        r, c = leaf.shape
        gram = leaf.T @ leaf if r >= c else leaf @ leaf.T
        if min(r, c) > 1 and jnp.allclose(gram, jnp.eye(min(r, c)), atol=1e-2):
            kept[0] += 1
            return leaf
        count[0] += 1
        return 0.02 * jax.random.normal(
            jax.random.fold_in(key, count[0]), leaf.shape, leaf.dtype
        )

    params = jax.tree_util.tree_map_with_path(redraw, params)
    if kept[0] != n_layers:
        raise RuntimeError(f"expected {n_layers} orthogonal kernels, found {kept[0]}")
    return params


# %%
# Loss
# ----
# The loss of one batch is the relative L2 error of every sample, summed over
# the batch, on the decoded (physical) velocity. The metric is the same
# per-sample relative L2, averaged over the test samples.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all nodes."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.linalg.norm(true.reshape(b, -1), axis=1)


def make_loss(model, y_norm: UnitGaussian):
    """Return ``loss_fn(params, x, y) -> summed relative L2`` in physical units."""

    def loss_fn(params, x, y):
        out = model.apply(params, x=x, fx=None)[..., 0]  # (batch, H * W)
        return jnp.sum(rel_l2_per_sample(y_norm.decode(out), y))

    return loss_fn


# %%
# Optimiser
# ---------
# The benchmark trains every model with the complex-aware AdamW of
# ``scirex.training.optim``, because FNO and GINO have complex spectral
# weights. Transolver's weights are all real, and for real weights
# ``scale_by_adam_cx`` is exactly Adam's scaling, so this is AdamW: the
# gradient norm is clipped to 0.1, then Adam, decoupled weight decay and a
# cosine one-cycle learning rate, the paper's own settings.


def make_optimizer(protocol: dict, total_steps: int):
    """AdamW on a cosine one-cycle schedule, after gradient clipping."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(protocol["max_grad_norm"]),
        scale_by_adam_cx(eps=protocol["adam_eps"]),
        optax.add_decayed_weights(protocol["weight_decay"]),
        optax.scale_by_learning_rate(schedule),
    )
    return tx, schedule


# %%
# Train
# -----
# Batches of 8 samples in a new random order every epoch, one optimiser step
# per batch. After every epoch the test samples are scored, to draw the error
# curve; no epoch is selected by it, and the weights after the last epoch
# are the result. JAX compiles the step on its first call, so the first
# epoch is slower than the others.


def train(model, params, data, y_norm, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    bs = protocol["batch_size"]
    ntrain = len(x_train)
    tx, _ = make_optimizer(protocol, -(-ntrain // bs) * epochs)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, y_norm)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        order = rng.permutation(ntrain)
        train_total = 0.0
        for i in range(0, ntrain, bs):
            idx = order[i : i + bs]
            params, opt_state, loss = train_step(
                params, opt_state, jnp.asarray(x_train[idx]), jnp.asarray(y_train[idx])
            )
            train_total += float(loss)
        secs = perf_counter() - t0
        test_total = sum(
            float(
                eval_step(params, jnp.asarray(x_test[i : i + bs]), y_test[i : i + bs])
            )
            for i in range(0, len(x_test), bs)
        )
        history.append(
            dict(
                epoch=epoch,
                train_rel_l2=train_total / ntrain,
                test_rel_l2=test_total / len(x_test),
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_rel_l2={history[-1]['train_rel_l2']:.4f} "
            f"test_rel_l2={history[-1]['test_rel_l2']:.4f} epoch_time={secs:.1f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# The relative L2 error of every test sample, in physical units. The mean is
# the benchmark's number; the median, the 90th percentile and the worst
# sample show how the error is spread, because a few hard geometries
# dominate the mean.


def evaluate(model, params, data, y_norm, batch_size: int = 8) -> dict:
    """Per-sample relative L2 on the test samples, and the predictions."""
    predict = jax.jit(lambda p, x: y_norm.decode(model.apply(p, x=x, fx=None)[..., 0]))
    x_test, y_test = data["x_test"], data["y_test"]
    preds = np.concatenate(
        [
            np.asarray(predict(params, jnp.asarray(x_test[i : i + batch_size])))
            for i in range(0, len(x_test), batch_size)
        ]
    )
    errors = np.asarray(rel_l2_per_sample(jnp.asarray(preds), jnp.asarray(y_test)))
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
# The standard figure of the benchmark, on the physical mesh of one test
# sample: the input (the y coordinate of every node, which shows the bend of
# the pipe), the true and the predicted horizontal velocity on one colour
# scale, and the absolute error. We also plot the error per epoch.


def plot_prediction(coords, truth, pred, path: Path, title: str) -> None:
    """Save input, true velocity, predicted velocity and absolute error."""
    xx, yy = coords[..., 0], coords[..., 1]
    lim = (float(truth.min()), float(truth.max()))
    panels = [
        ("Input: y coordinate of the mesh", yy, "coolwarm", None),
        ("Truth: horizontal velocity", truth, "jet", lim),
        ("Prediction: horizontal velocity", pred, "jet", lim),
        ("Absolute error", np.abs(pred - truth), "magma", None),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), constrained_layout=True)
    for ax, (name, values, cmap, vlim) in zip(axes, panels):
        vmin, vmax = vlim if vlim else (None, None)
        mesh = ax.pcolormesh(
            xx, yy, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud"
        )
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and test relative L2 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_rel_l2"] for h in history], marker="o", label="train")
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
# ``--smoke`` takes a few samples, uses the tiny Transolver and trains for a
# few epochs. Without it the script trains the paper-size Transolver on the
# full split for 500 epochs. That mode is meant for a GPU: the benchmark took
# about 2 h per seed on one.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("PIPE_DATA_DIR"),
        help="folder with Pipe_X.npy, Pipe_Y.npy, Pipe_Q.npy (or set $PIPE_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few samples, tiny Transolver")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/pipe_transolver_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=32, help="samples in smoke mode")
    ap.add_argument("--n-test", type=int, default=8, help="samples in smoke mode")
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="keep every stride-th mesh node (default: 1)",
    )
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $PIPE_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PROTOCOL
    ntrain, ntest = (
        (args.n_train, args.n_test) if args.smoke else (p["ntrain"], p["ntest"])
    )
    epochs = args.epochs or (5 if args.smoke else p["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    stride = args.stride or 1
    raw = load_pipe(
        Path(args.data_dir).expanduser(), ntrain, ntest, p["n_total"], stride
    )
    x_norm = UnitGaussian(raw["x_train"], axes=(0, 1, 2))
    y_norm = UnitGaussian(raw["y_train"], axes=(0, 1, 2))
    grid = raw["y_train"].shape[1:]
    n_nodes = grid[0] * grid[1]
    data = dict(
        x_train=x_norm.encode(raw["x_train"]).reshape(-1, n_nodes, 2),
        x_test=x_norm.encode(raw["x_test"]).reshape(-1, n_nodes, 2),
        y_train=raw["y_train"].reshape(-1, n_nodes),
        y_test=raw["y_test"].reshape(-1, n_nodes),
    )
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params = build_model(size, data["x_train"][0], grid, args.seed)

    params, history = train(model, params, data, y_norm, epochs, args.seed)
    metrics = evaluate(model, params, data, y_norm)
    print(
        f"test relative L2: mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} | p90 {metrics['p90']:.4f} | "
        f"worst {metrics['worst']:.4f}"
    )

    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        raw["x_test"][k],
        raw["y_test"][k],
        metrics["predictions"][k].reshape(grid),
        out_dir / "prediction.png",
        f"Pipe, Transolver: test sample, median error ({metrics['errors'][k]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
