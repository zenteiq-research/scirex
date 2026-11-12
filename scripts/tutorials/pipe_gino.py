# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting flow through a deformed pipe with GINO (Pipe)
========================================================

**Problem.** Given the coordinates of the 129x129 structured mesh of a pipe
whose centreline is bent differently in every sample, predict the steady
horizontal velocity at every mesh node. The flow is steady, incompressible
Navier-Stokes flow with a parabolic inlet profile (Geo-FNO, Li et al. 2023);
the model is the Geometry-Informed Neural Operator (GINO, Li et al. 2023),
which moves the data from the mesh nodes to a latent grid with a graph
neural operator, runs an FNO there and moves the result back to the nodes.

**What you will learn.**

1. How to read the three Pipe ``.npy`` files and build the 1000 / 200 split
   of the benchmark, with unit-Gaussian normalisation.
2. How to build the radius graph of every sample with the public
   ``scirex.layers.build_neighbor_graph`` and ``edge_index_to_neighbors``,
   padded to one size so that the training step compiles once.
3. How to feed ``scirex.models.GINO`` one sample at a time, with the sample's
   own mesh as the FNO's latent grid, and count its parameters.
4. How to keep the benchmark's batch of 8 with a model that takes one sample:
   sum the gradients of 8 samples, then take one optimiser step.
5. How to evaluate with the per-sample relative L2 error in physical units,
   and how to plot truth, prediction and error on the physical mesh.

**Caveat.** The ``--smoke`` mode trains a tiny GINO on a few samples for a
few epochs. It checks that the pipeline works and gives poor predictions. It
is not a benchmark number. For the benchmark protocol (1000 training
samples, 500 epochs, 3,001,956 parameters, 3 seeds) use
``scripts/benchmarks/pipe/train.py``; see the "Reproduce at full scale"
section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/pipe_gino.py --data-dir ~/data/pipe --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.GINO``,
# ``scirex.layers.build_neighbor_graph`` and ``edge_index_to_neighbors``,
# ``scirex.training.optim.scale_by_adam_cx`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

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
from scirex.layers import build_neighbor_graph, edge_index_to_neighbors
from scirex.models import GINO
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

# Model sizes. "full" is the benchmark's GINO (3,001,956 parameters, matched
# to Transolver's 3,073,985); every other setting is GINO's default. "smoke"
# is a tiny GINO that trains on a CPU: smaller FNO and GNO MLPs. The smoke run
# also keeps only every second mesh node (``--stride 2``), so its radius is
# doubled to keep about as many neighbours per node.
SIZES = dict(
    full=dict(
        in_gno_radius=0.05,
        out_gno_radius=0.05,
        fno_n_modes=(16, 16),
        fno_hidden_channels=48,
        fno_n_layers=4,
        fno_lifting_channel_ratio=4,
    ),
    smoke=dict(
        in_gno_radius=0.1,
        out_gno_radius=0.1,
        fno_n_modes=(8, 8),
        fno_hidden_channels=12,
        fno_n_layers=2,
        fno_lifting_channel_ratio=2,
        in_gno_channel_mlp_hidden_layers=(16, 16),
        out_gno_channel_mlp_hidden_layers=(32, 16),
    ),
)

# Neighbour graphs keep at most the 32 nearest neighbours of every node within
# the radius, as in the benchmark.
MAX_NUM_NEIGHBORS = 32


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
# Build the neighbour graphs
# --------------------------
# Every sample has its own mesh, so every sample needs its own graph. One
# radius graph from the nodes to themselves serves both GNO layers (the
# encoder gathers node data onto the latent grid, the decoder gathers the
# latent result back onto the nodes), because the latent grid is the
# sample's own mesh. ``build_neighbor_graph`` returns the edge list, at most
# ``MAX_NUM_NEIGHBORS`` nearest neighbours per node within the radius. All
# graphs are then padded to the length of the largest one, so that the
# training step compiles once; padding edges point one past the last node and
# are ignored. The GNO layers process every edge, padding included, so the
# cost of a step grows with the number of edges. The radius applies to the
# normalised coordinates, so the number of edges depends on the normaliser,
# and so on the training samples it was fitted on.
# ``edge_index_to_neighbors`` turns an edge list into the arrays the GNO
# layers use.


def build_graphs(coords: np.ndarray, radius: float) -> list[dict]:
    """One radius graph per sample of ``(n, N, 2)`` coordinates, all of one length."""
    edges = [
        build_neighbor_graph(
            pos, pos, radius, "radius", max_num_neighbors=MAX_NUM_NEIGHBORS
        )
        for pos in coords
    ]
    max_edges = max(e.shape[1] for e in edges)
    graphs = []
    for pos, e in zip(coords, edges):
        padded = np.full((2, max_edges), len(pos), dtype=np.int32)
        padded[:, : e.shape[1]] = e
        nbrs = edge_index_to_neighbors(jnp.asarray(padded), num_query=len(pos))
        graphs.append({k: np.asarray(v) for k, v in nbrs.items()})
    return graphs


# %%
# Build the model
# ---------------
# ``GINO`` takes one sample at a time. Its inputs are the node coordinates
# (``input_geom`` and ``output_queries``, ``(1, N, 2)``), the latent grid of
# the FNO processor (``latent_queries``, here the sample's own mesh as
# ``(1, H, W, 2)``), a node feature (``x``, a constant 1: the geometry
# enters only through the coordinates) and the two graphs. ``gno_coord_dim=2``
# because the coordinates are 2D. The FNO processor's spectral weights are
# complex; ``count_model_params`` counts each as two real numbers.


def build_model(size: dict, pos: np.ndarray, graph: dict, grid, seed: int):
    """Build the GINO, initialise it on one sample and print the parameter count."""
    model = GINO(in_channels=1, out_channels=1, gno_coord_dim=2, **size)
    params = model.init(jax.random.PRNGKey(seed), **model_inputs(pos, graph, grid))
    print(f"GINO {size} | parameters: {count_model_params(params):,}")
    return model, params


def model_inputs(pos, graph: dict, grid) -> dict:
    """GINO's keyword arguments for one sample with ``N`` nodes at ``pos``."""
    pos = jnp.asarray(pos)[None]  # (1, N, 2)
    nbrs = {k: jnp.asarray(v) for k, v in graph.items()}
    return dict(
        input_geom=pos,
        latent_queries=pos.reshape(1, *grid, 2),
        output_queries=pos,
        x=jnp.ones((1, pos.shape[1], 1), dtype=jnp.float32),
        neighbors_in=nbrs,
        neighbors_out=nbrs,
    )


# %%
# Loss
# ----
# The relative L2 error of one sample on the decoded (physical) velocity. The
# loss of a batch is its sum over the batch; the metric is its mean over the
# test samples.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all nodes."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.linalg.norm(true.reshape(b, -1), axis=1)


def make_loss(model, y_norm: UnitGaussian, grid):
    """Return ``loss_fn(params, inputs, y) -> relative L2`` of one sample."""

    def loss_fn(params, inputs, y):
        out = model.apply(params, **inputs)[0, :, 0].reshape(grid)  # (H, W)
        return jnp.sum(rel_l2_per_sample(y_norm.decode(out)[None], y[None]))

    return loss_fn


# %%
# Optimiser for complex weights
# -----------------------------
# The FNO processor inside GINO has complex spectral weights. For a real
# loss ``L`` and a complex weight ``w``, ``jax.grad`` returns ``dL/dw``, the
# conjugate of the steepest-ascent direction that optax transforms assume.
# With plain
# ``optax.adamw`` the real part of every spectral weight descends but the
# imaginary part ascends. ``scale_by_adam_cx`` conjugates complex gradients
# first and keeps Adam's second moment real (``|g|^2``); real weights are
# updated exactly as by AdamW. Weight decay and the one-cycle learning rate
# follow, after clipping the gradient norm, as in the benchmark.
# ``optimizer="adamw"`` swaps in optax's own Adam scaling, to see the problem.


def make_optimizer(protocol: dict, total_steps: int, optimizer: str = "adamw_cx"):
    """AdamW on a cosine one-cycle schedule, after gradient clipping.

    ``"adamw_cx"`` (the benchmark's) is complex-aware; ``"adamw"`` is plain optax.
    """
    scale_by_adam = dict(
        adamw_cx=scale_by_adam_cx(eps=protocol["adam_eps"]),
        adamw=optax.scale_by_adam(eps=protocol["adam_eps"]),
    )[optimizer]
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(protocol["max_grad_norm"]),
        scale_by_adam,
        optax.add_decayed_weights(protocol["weight_decay"]),
        optax.scale_by_learning_rate(schedule),
    )
    return tx, schedule


# %%
# Train
# -----
# GINO takes one sample at a time, but the benchmark's recipe is one
# optimiser step per batch of 8. The loss of a batch is a sum over its
# samples, so its gradient is the sum of the samples' gradients: we add 8
# per-sample gradients, then take one step. Every sample has the same
# padded graph size, so the per-sample step compiles once. After every epoch
# the test samples are scored, to draw the error curve; no epoch is selected
# by it, and the weights after the last epoch are the result.


def train(
    model, params, data, y_norm, epochs, seed, protocol=PROTOCOL, optimizer="adamw_cx"
):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    grid = data["grid"]
    bs = protocol["batch_size"]
    ntrain = len(data["y_train"])
    tx, _ = make_optimizer(protocol, -(-ntrain // bs) * epochs, optimizer)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, y_norm, grid)
    grad_step = jax.jit(jax.value_and_grad(loss_fn))
    add_grads = jax.jit(lambda a, b: jax.tree_util.tree_map(jnp.add, a, b))
    eval_step = jax.jit(loss_fn)

    @jax.jit
    def apply_grads(params, opt_state, grads):
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    def inputs(split, i):
        return model_inputs(data[f"pos_{split}"][i], data[f"graph_{split}"][i], grid)

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        order = rng.permutation(ntrain)
        train_total = 0.0
        for start in range(0, ntrain, bs):
            grads = None
            for i in order[start : start + bs]:
                loss, g = grad_step(
                    params, inputs("train", i), jnp.asarray(data["y_train"][i])
                )
                grads = g if grads is None else add_grads(grads, g)
                train_total += float(loss)
            params, opt_state = apply_grads(params, opt_state, grads)
        secs = perf_counter() - t0
        test_total = sum(
            float(eval_step(params, inputs("test", i), jnp.asarray(y)))
            for i, y in enumerate(data["y_test"])
        )
        history.append(
            dict(
                epoch=epoch,
                train_rel_l2=train_total / ntrain,
                test_rel_l2=test_total / len(data["y_test"]),
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


def evaluate(model, params, data, y_norm) -> dict:
    """Per-sample relative L2 on the test samples, and the predictions."""
    grid = data["grid"]
    predict = jax.jit(
        lambda p, inputs: y_norm.decode(model.apply(p, **inputs)[0, :, 0].reshape(grid))
    )
    preds = np.stack(
        [
            np.asarray(predict(params, model_inputs(pos, graph, grid)))
            for pos, graph in zip(data["pos_test"], data["graph_test"])
        ]
    )
    errors = np.asarray(
        rel_l2_per_sample(jnp.asarray(preds), jnp.asarray(data["y_test"]))
    )
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
# ``--smoke`` takes a few samples on every second mesh node, uses the tiny
# GINO and trains for a few epochs. Without it the script trains the
# benchmark-size GINO on the full split, on every mesh node, for 500 epochs.
# That mode is meant for a GPU: the benchmark took about 3.2 h per seed.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("PIPE_DATA_DIR"),
        help="folder with Pipe_X.npy, Pipe_Y.npy, Pipe_Q.npy (or set $PIPE_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few samples, tiny GINO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/pipe_gino_tutorial"))
    ap.add_argument("--n-train", type=int, default=32, help="samples in smoke mode")
    ap.add_argument("--n-test", type=int, default=8, help="samples in smoke mode")
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="keep every stride-th mesh node (default: 2 with --smoke, else 1)",
    )
    ap.add_argument(
        "--optimizer",
        choices=("adamw_cx", "adamw"),
        default="adamw_cx",
        help="adamw_cx (the benchmark's) or plain optax AdamW, for comparison",
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
    stride = args.stride or (2 if args.smoke else 1)
    raw = load_pipe(
        Path(args.data_dir).expanduser(), ntrain, ntest, p["n_total"], stride
    )
    x_norm = UnitGaussian(raw["x_train"], axes=(0, 1, 2))
    y_norm = UnitGaussian(raw["y_train"], axes=(0, 1, 2))
    grid = raw["y_train"].shape[1:]
    data = dict(grid=grid, y_train=raw["y_train"], y_test=raw["y_test"])
    for split in ("train", "test"):
        pos = x_norm.encode(raw[f"x_{split}"]).reshape(-1, grid[0] * grid[1], 2)
        data[f"pos_{split}"] = pos.astype(np.float32)
    print(f"data loading: {perf_counter() - t0:.1f}s")
    t0 = perf_counter()
    graphs = build_graphs(
        np.concatenate([data["pos_train"], data["pos_test"]]), size["in_gno_radius"]
    )
    data["graph_train"], data["graph_test"] = graphs[:ntrain], graphs[ntrain:]
    n_edges = [int(g["counts"].sum()) for g in graphs]
    print(
        f"graphs: {perf_counter() - t0:.1f}s | edges per sample: min {min(n_edges):,}, "
        f"max {max(n_edges):,} | padded to {len(graphs[0]['segment_ids']):,}"
    )
    model, params = build_model(
        size, data["pos_train"][0], data["graph_train"][0], grid, args.seed
    )

    params, history = train(
        model, params, data, y_norm, epochs, args.seed, optimizer=args.optimizer
    )
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
        metrics["predictions"][k],
        out_dir / "prediction.png",
        f"Pipe, GINO: test sample with the median error ({metrics['errors'][k]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
