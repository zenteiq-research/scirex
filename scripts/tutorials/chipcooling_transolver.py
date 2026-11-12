# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting the temperature of a liquid-cooled dual-chip cold plate with Transolver
=================================================================================

**Problem.** Given the six boundary-condition parameters of a liquid-cooled
cold plate that carries two chips (the heat flux of each chip, and the mass
flow rate and temperature of each of the two coolant inlets), predict the
steady temperature on the central plane of the plate. The data are 2000 CFD
simulations of Curl & Hu (2026); the model is Transolver (Wu et al. 2024),
which works on the unstructured CFD mesh directly: every node is a token,
and attention runs between a small number of learned "slices" of the nodes.

**What you will learn.**

1. How to read the chip cooling files and build the 1600 / 400 split.
2. How to turn a parametric problem into point-cloud inputs for
   ``scirex.models.Transolver``: the node coordinates as positions and the
   six parameters, repeated at every node, as features.
3. How to train with the benchmark's recipe: Adam with weight decay added
   to the gradients, gradient clipping, a linear warmup and a cosine decay.
4. How to evaluate in kelvin at the mesh nodes and plot truth, prediction
   and error.

**Caveat.** The ``--smoke`` mode trains a small Transolver on a few
simulations, on every second mesh node, for a few epochs. It checks that the
pipeline works and gives poor predictions. It is not a benchmark number. For
the benchmark protocol (1600 training simulations, 500 epochs, 1,807,561
parameters) use ``scripts/benchmarks/chip_cooling/train.py``; see the
"Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/chipcooling_transolver.py \\
        --data-dir ~/data/chip_cooling --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.Transolver`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import json
import os
import platform
import re
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
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 42

# The chip cooling benchmark's settings, the same values as PROTOCOL and
# RUNS["transolver"].opt in scripts/benchmarks/chip_cooling/_config.py.
PROTOCOL = dict(
    ntrain=1600,
    ntest=400,
    epochs=500,
    batch_size=4,
    learning_rate=1e-4,
    weight_decay=1e-6,
    grad_clip=1.0,
    warmup_epochs=50,
    lr_floor=1e-6,
)

# Model sizes. "full" is the benchmark's Transolver (1,807,561 parameters, the
# size the other three models are matched to), the same dict as
# RUNS["transolver"].arch. "smoke" is a smaller Transolver for a CPU.
SIZES = dict(
    full=dict(
        n_layers=5,
        n_hidden=256,
        n_head=8,
        mlp_ratio=1,
        dropout_rate=0.0,
        slice_num=32,
        ref=8,
        space_dim=2,
        fun_dim=6,
        out_dim=1,
        time_input=False,
        unified_pos=False,
        geometry="irregular",
    ),
)
SIZES["smoke"] = dict(SIZES["full"], n_layers=2, n_hidden=64, n_head=4, slice_num=16)

# The six parameters, in the loader's order: sorted by boundary condition,
# then by parameter name.
PARAM_LABELS = [
    "chip 1 heat flux",
    "chip 2 heat flux",
    "inlet 1 mass flow",
    "inlet 1 temperature",
    "inlet 2 mass flow",
    "inlet 2 temperature",
]
FIELD_KEY = "yz-mid|temperature"
COORD_KEY = "yz-mid|coordinates"


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
# ``model_setup.json`` holds the six parameters of all 2000 simulations under
# ``doe_configuration``; ``sim_0001.npz`` to ``sim_2000.npz`` hold one
# simulation each. Each file has several fields; the benchmark uses the
# temperature on the central ``yz`` plane, at the 11,110 nodes of the CFD
# mesh, and the coordinates of those nodes, which are the same in every file.
# The first ``ntrain`` simulations train and the last ``ntest`` test; only
# those files are read.


def load_chip_cooling(data_dir: Path, ntrain: int, ntest: int) -> dict:
    """Parameters ``(n, 6)``, temperatures ``(n, 11110)`` in K and coordinates."""
    data_dir = Path(data_dir)
    doe = json.loads((data_dir / "model_setup.json").read_text())["doe_configuration"]
    columns = [
        doe[bc][name]
        for bc in sorted(doe)
        for name in sorted(doe[bc])
        if isinstance(doe[bc][name], list)
    ]
    parameters = np.asarray(columns, dtype=np.float32).T  # (2000, 6)
    files = sorted(
        data_dir.glob("sim_*.npz"), key=lambda p: int(re.search(r"\d+", p.stem)[0])
    )
    if len(files) != len(parameters) or ntrain + ntest > len(files):
        raise ValueError(
            f"{len(files)} simulation files for {len(parameters)} parameter rows; "
            f"need ntrain + ntest <= {len(files)}"
        )
    rows = dict(train=np.arange(ntrain), test=np.arange(len(files) - ntest, len(files)))
    out = {}
    for name, idx in rows.items():
        fields = []
        for i in idx:
            with np.load(files[i], allow_pickle=False) as d:
                fields.append(np.asarray(d[FIELD_KEY], dtype=np.float32))
                coords = np.asarray(d[COORD_KEY], dtype=np.float32)
        out[f"params_{name}"] = parameters[idx]
        out[f"T_{name}"] = np.stack(fields)
    out["coords"] = coords
    print(
        f"simulations: {ntrain} train / {ntest} test | {coords.shape[0]} mesh nodes | "
        f"T range [{out['T_train'].min():.1f}, {out['T_train'].max():.1f}] K"
    )
    return out


# %%
# Inputs and normalisation
# ------------------------
# Transolver sees every mesh node as a token with a position ``x`` and
# features ``fx``. The position is the node's ``(y, z)``, scaled to [0, 1]
# with its minimum and maximum. The features are the six parameters,
# standardised one by one with the training mean and standard deviation, and
# repeated at every node: the geometry is the same in every simulation, so
# the parameters are what the model has to read. The temperature gets one mean
# and one standard deviation over all nodes of the training simulations.


def normalise(data: dict) -> dict:
    """Positions ``(n_nodes, 2)``, standardised parameters and temperatures."""
    p_mean = data["params_train"].mean(axis=0)
    p_std = data["params_train"].std(axis=0) + 1e-8
    t_mean = float(data["T_train"].mean())
    t_std = float(data["T_train"].std()) + 1e-8
    yz = data["coords"][:, 1:3]
    lo, hi = yz.min(axis=0), yz.max(axis=0)
    return dict(
        nodes=((yz - lo) / (hi - lo + 1e-8)).astype(np.float32),
        p_train=((data["params_train"] - p_mean) / p_std).astype(np.float32),
        p_test=((data["params_test"] - p_mean) / p_std).astype(np.float32),
        y_train=((data["T_train"] - t_mean) / t_std).astype(np.float32)[..., None],
        y_test=((data["T_test"] - t_mean) / t_std).astype(np.float32)[..., None],
        t_mean=t_mean,
        t_std=t_std,
    )


def tokens(nodes: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x`` ``(b, n_nodes, 2)`` and ``fx`` ``(b, n_nodes, 6)`` for a batch."""
    b, n = len(params), len(nodes)
    x = np.broadcast_to(nodes, (b, n, 2))
    fx = np.broadcast_to(params[:, None, :], (b, n, params.shape[1]))
    return x, fx


# %%
# Build the model
# ---------------
# ``geometry="irregular"`` because the nodes are an unstructured point cloud,
# ``space_dim=2`` for the positions, ``fun_dim=6`` for the features and
# ``out_dim=1`` for the temperature. Every layer softly assigns the nodes to
# ``slice_num`` slices, lets the slices attend to each other and sends the
# result back to the nodes, so the cost grows linearly with the number of
# nodes. The weights are real.


def build_model(arch: dict, nodes: np.ndarray, params: np.ndarray, seed: int):
    """Build the Transolver, initialise it on one sample, print its parameter count."""
    model = Transolver(**arch)
    _, key = jax.random.split(jax.random.PRNGKey(seed))
    x, fx = tokens(nodes, params[:1])
    init = model.init(key, x=jnp.asarray(x), fx=jnp.asarray(fx))
    print(f"Transolver | parameters: {count_model_params(init):,}")
    return model, init


# %%
# Loss
# ----
# The loss is the relative L2 error of the standardised temperature over all
# nodes of the whole batch, ``||pred - y|| / ||y||``, as in the benchmark.


def rel_l2(pred, target):
    """``||pred - target|| / ||target||`` over all entries."""
    return jnp.sqrt(jnp.sum((pred - target) ** 2) / (jnp.sum(target**2) + 1e-12))


def make_loss(model):
    """Return ``loss_fn(params, x, fx, y) -> relative L2`` of a batch."""

    def loss_fn(params, x, fx, y):
        return rel_l2(model.apply(params, x=x, fx=fx), y)

    return loss_fn


# %%
# Optimiser
# ---------
# The benchmark's Transolver recipe: the gradient norm is clipped to 1, the
# weight decay of 1e-6 is added to the gradients (an L2 penalty, not AdamW's
# decoupled decay), and Adam follows a learning rate that rises linearly from
# 0 to 1e-4 during the warmup and then falls on a cosine to 1e-6, at every
# step. The warmup is 50 of the 500 epochs; a shorter run warms up for a tenth
# of its epochs.


def make_optimizer(protocol: dict, epochs: int, steps_per_epoch: int):
    """Clipped, L2-regularised Adam with a linear warmup and a cosine decay."""
    warmup_epochs = min(protocol["warmup_epochs"], max(1, epochs // 10))
    warmup = max(1, warmup_epochs * steps_per_epoch)
    decay = max(1, (epochs - warmup_epochs) * steps_per_epoch)
    lr = protocol["learning_rate"]
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, transition_steps=warmup),
            optax.cosine_decay_schedule(
                lr, decay_steps=decay, alpha=protocol["lr_floor"] / lr
            ),
        ],
        boundaries=[warmup],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(protocol["grad_clip"]),
        optax.add_decayed_weights(protocol["weight_decay"]),
        optax.adam(schedule),
    )
    return tx, schedule


# %%
# Train
# -----
# Batches of 4 simulations in a new random order every epoch, one optimiser
# step per batch. After every epoch the test simulations are scored with the
# same loss, to draw the error curve; the weights after the last epoch are the
# result. JAX compiles the step on its first call, so the first epoch is
# slower than the others.


def train(model, params, norm, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    bs = protocol["batch_size"]
    n = len(norm["p_train"])
    tx, schedule = make_optimizer(protocol, epochs, max(1, n // bs))
    opt_state = tx.init(params)
    loss_fn = make_loss(model)

    @jax.jit
    def train_step(params, opt_state, x, fx, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, fx, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    def batch(p, y, idx):
        x, fx = tokens(norm["nodes"], p[idx])
        return jnp.asarray(x), jnp.asarray(fx), jnp.asarray(y[idx])

    rng = np.random.default_rng(seed)
    history, step = [], 0
    for epoch in range(epochs):
        t0 = perf_counter()
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, bs):
            b = batch(norm["p_train"], norm["y_train"], order[i : i + bs])
            params, opt_state, loss = train_step(params, opt_state, *b)
            losses.append(float(loss))
            step += 1
        secs = perf_counter() - t0
        n_test = len(norm["p_test"])
        test = [
            float(
                eval_step(
                    params,
                    *batch(
                        norm["p_test"],
                        norm["y_test"],
                        np.arange(i, min(i + bs, n_test)),
                    ),
                )
            )
            for i in range(0, n_test, bs)
        ]
        history.append(
            dict(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                test_loss=float(np.mean(test)),
                lr=float(schedule(step)),
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} "
            f"test_loss={history[-1]['test_loss']:.4f} lr={history[-1]['lr']:.2e} "
            f"epoch_time={secs:.1f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# The benchmark's metrics, in kelvin, over the mesh nodes of the test
# simulations: R2, RMSE and MAE pooled over all nodes, the largest absolute
# error, and the relative L2 error per simulation, averaged. A temperature
# field of about 300 to 400 K has a large norm, so its relative L2 error is
# small; the errors in kelvin are easier to read. As a baseline, the mean
# training field predicts the same temperature for every test simulation.


def field_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Metrics of ``(n, n_points)`` predictions in kelvin."""
    err = pred - true
    return dict(
        R2=float(1.0 - np.sum(err**2) / np.sum((true - true.mean()) ** 2)),
        RMSE=float(np.sqrt(np.mean(err**2))),
        MAE=float(np.mean(np.abs(err))),
        max_AE=float(np.max(np.abs(err))),
        rel_l2=float(
            np.mean(np.linalg.norm(err, axis=1) / np.linalg.norm(true, axis=1))
        ),
    )


def evaluate(model, params, norm, data) -> dict:
    """Test metrics in kelvin at the mesh nodes, a baseline and the predictions."""
    predict = jax.jit(lambda p, x, fx: model.apply(p, x=x, fx=fx))
    pred = []
    for i in range(len(norm["p_test"])):
        x, fx = tokens(norm["nodes"], norm["p_test"][i : i + 1])
        pred.append(
            np.asarray(predict(params, jnp.asarray(x), jnp.asarray(fx)))[0, :, 0]
        )
    pred = np.stack(pred) * norm["t_std"] + norm["t_mean"]
    true = data["T_test"]
    baseline = np.broadcast_to(data["T_train"].mean(axis=0), true.shape)
    return dict(
        model=field_metrics(pred, true),
        baseline=field_metrics(baseline, true),
        predictions=pred,
    )


# %%
# Plot
# ----
# The standard figure of the benchmark, for one test simulation, at the mesh
# nodes in millimetres: the input (the six parameters, the only thing that
# changes between simulations), the true and the predicted temperature on one
# colour scale, and the absolute error. We also plot the loss per epoch.


def plot_prediction(params, coords, truth, pred, path: Path, title: str) -> None:
    """Save input parameters, true temperature, predicted temperature and error."""
    z, y = coords[:, 2] * 1e3, coords[:, 1] * 1e3
    lim = (truth.min(), truth.max())
    fig, axes = plt.subplots(
        1, 4, figsize=(22, 4), gridspec_kw=dict(width_ratios=[0.8, 1, 1, 1])
    )
    axes[0].barh(PARAM_LABELS[::-1], params[::-1], color="tab:gray")
    axes[0].set_title("Input: six parameters (standardised)")
    for ax, (name, values, cmap, vlim) in zip(
        axes[1:],
        [
            ("Truth: temperature (K)", truth, "inferno", lim),
            ("Prediction: temperature (K)", pred, "inferno", lim),
            ("Absolute error (K)", np.abs(pred - truth), "magma", None),
        ],
    ):
        vmin, vmax = vlim if vlim else (None, None)
        sc = ax.scatter(z, y, c=values, s=2, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and test loss per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["test_loss"] for h in history], marker="o", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 (standardised)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few simulations, every second mesh node and the smaller
# Transolver, and trains for a few epochs. Without it the script trains the
# benchmark-size Transolver on all nodes of the full split for 500 epochs.
# That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("CHIPCOOLING_DATA_DIR"),
        help="folder with model_setup.json and sim_*.npz ($CHIPCOOLING_DATA_DIR)",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="few simulations, smaller Transolver"
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/chipcooling_transolver_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=32, help="simulations in smoke mode")
    ap.add_argument("--n-test", type=int, default=8, help="simulations in smoke mode")
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="keep every stride-th mesh node (default: 1, or 2 with --smoke)",
    )
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $CHIPCOOLING_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PROTOCOL
    ntrain, ntest = (
        (args.n_train, args.n_test) if args.smoke else (p["ntrain"], p["ntest"])
    )
    epochs = args.epochs or (5 if args.smoke else p["epochs"])
    stride = args.stride or (2 if args.smoke else 1)
    arch = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data = load_chip_cooling(Path(args.data_dir).expanduser(), ntrain, ntest)
    if stride > 1:
        data["coords"] = data["coords"][::stride]
        data["T_train"], data["T_test"] = (
            data["T_train"][:, ::stride],
            data["T_test"][:, ::stride],
        )
        print(f"stride {stride}: {len(data['coords'])} mesh nodes")
    print(f"data loading: {perf_counter() - t0:.1f}s")
    norm = normalise(data)
    model, params = build_model(arch, norm["nodes"], norm["p_train"], args.seed)

    params, history = train(model, params, norm, epochs, args.seed)
    metrics = evaluate(model, params, norm, data)
    for name in ("model", "baseline"):
        m = metrics[name]
        print(
            f"{name:8s}: R2 {m['R2']:.4f} | RMSE {m['RMSE']:.3f} K "
            f"| MAE {m['MAE']:.3f} K | max AE {m['max_AE']:.2f} K "
            f"| rel L2 {m['rel_l2']:.2e}"
        )

    k = ntest // 2
    plot_prediction(
        norm["p_test"][k],
        data["coords"],
        data["T_test"][k],
        metrics["predictions"][k],
        out_dir / "prediction.png",
        f"Chip cooling, Transolver: test simulation {k} of {ntest}",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return dict(model=metrics["model"], baseline=metrics["baseline"])


if __name__ == "__main__":
    main()
