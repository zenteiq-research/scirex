# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting the temperature of a liquid-cooled dual-chip cold plate with GINO
===========================================================================

**Problem.** Given the six boundary-condition parameters of a liquid-cooled
cold plate that carries two chips (the heat flux of each chip, and the mass
flow rate and temperature of each of the two coolant inlets), predict the
steady temperature on the central plane of the plate. The data are 2000 CFD
simulations of Curl & Hu (2026); the model is the Geometry-Informed Neural
Operator (GINO, Li et al. 2023). GINO works on the unstructured CFD mesh
itself: a graph neural operator moves information from the mesh nodes to a
regular latent grid, an FNO works on that grid, and a second graph operator
brings the result back to the nodes. No interpolation onto a grid is needed.

**What you will learn.**

1. How to read the chip cooling files and build the 1600 / 400 split.
2. How to build GINO's two radius graphs (mesh to latent grid and back) with
   ``scirex.layers.kdtree_neighbor_search``, once, because every simulation
   has the same mesh.
3. How to give six scalars to GINO: as latent features, constant over the
   latent grid.
4. How to train one simulation per step with the complex-aware ``adamw_cx``
   and a cosine learning rate stepped after every epoch.
5. How to evaluate in kelvin at the mesh nodes and plot truth, prediction
   and error.

**Caveat.** The ``--smoke`` mode trains a small GINO on a few simulations,
on every second mesh node, for a few epochs. It checks that the pipeline
works and gives poor predictions. It is not a benchmark number. For the
benchmark protocol (1600 training simulations, 500 epochs, 1,821,028
parameters) use ``scripts/benchmarks/chip_cooling/train.py``; see the
"Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/chipcooling_gino.py \\
        --data-dir ~/data/chip_cooling --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.GINO``,
# ``scirex.layers.kdtree_neighbor_search``, ``scirex.training.adamw_cx`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import json
import math
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
from scirex.layers import kdtree_neighbor_search
from scirex.models import GINO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 42

# The chip cooling benchmark's settings, the same values as PROTOCOL and
# RUNS["gino"].opt in scripts/benchmarks/chip_cooling/_config.py.
PROTOCOL = dict(
    ntrain=1600,
    ntest=400,
    epochs=500,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# Model sizes. "full" is the benchmark's GINO (1,821,028 parameters, matched to
# Transolver's 1,807,561), the same dict as RUNS["gino"].arch. "smoke" is a
# smaller GINO, with a coarser latent grid, that trains on a CPU.
SIZES = dict(
    full=dict(
        latent_h=64,
        latent_w=128,
        latent_feature_channels=6,
        gno_coord_dim=2,
        gno_coord_embed_dim=16,
        in_gno_radius=0.05,
        out_gno_radius=0.05,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        in_gno_pos_embed_type=None,
        out_gno_pos_embed_type="transformer",
        gno_embed_channels=16,
        gno_embed_max_positions=600,
        in_gno_channel_mlp_hidden_layers=(80, 80, 80),
        out_gno_channel_mlp_hidden_layers=(512, 256),
        fno_n_modes=(20, 20),
        fno_hidden_channels=30,
        fno_lifting_channel_ratio=4,
        fno_n_layers=4,
        fno_use_channel_mlp=True,
        fno_channel_mlp_expansion=0.5,
        fno_norm="group_norm",
        fno_ada_in_features=8,
        fno_factorization=None,
        fno_rank=0.8,
        fno_domain_padding=0.0,
        projection_channel_ratio=4,
    ),
)
SIZES["smoke"] = dict(
    SIZES["full"],
    latent_h=32,
    latent_w=64,
    in_gno_radius=0.08,
    out_gno_radius=0.08,
    in_gno_channel_mlp_hidden_layers=(32, 32),
    out_gno_channel_mlp_hidden_layers=(64, 64),
    fno_n_modes=(12, 12),
    fno_hidden_channels=16,
    fno_n_layers=2,
)

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
# Normalise
# ---------
# The six parameters are standardised one by one with the mean and standard
# deviation of the training simulations; the temperature gets one mean and
# one standard deviation over all nodes of the training simulations. The
# ``(y, z)`` coordinates of the nodes are scaled to [0, 1] with their minimum
# and maximum, so that the graph radius and the latent grid live in the unit
# square.


def normalise(data: dict) -> dict:
    """Standardised parameters and temperatures, and the scaled node coordinates."""
    p_mean = data["params_train"].mean(axis=0)
    p_std = data["params_train"].std(axis=0) + 1e-8
    t_mean = float(data["T_train"].mean())
    t_std = float(data["T_train"].std()) + 1e-8
    yz = data["coords"][:, 1:3]
    lo, hi = yz.min(axis=0), yz.max(axis=0)
    return dict(
        p_train=((data["params_train"] - p_mean) / p_std).astype(np.float32),
        p_test=((data["params_test"] - p_mean) / p_std).astype(np.float32),
        y_train=((data["T_train"] - t_mean) / t_std).astype(np.float32),
        y_test=((data["T_test"] - t_mean) / t_std).astype(np.float32),
        t_mean=t_mean,
        t_std=t_std,
        nodes=((yz - lo) / (hi - lo + 1e-8)).astype(np.float32),
    )


# %%
# Build the graphs
# ----------------
# GINO needs two neighbour lists. The input graph lists, for every latent grid
# point, the mesh nodes within ``in_gno_radius``; the output graph lists, for
# every mesh node, the latent points within ``out_gno_radius``. They are
# returned as flat index arrays: ``neighbors_index`` (who), ``segment_ids``
# (for which query) and ``counts`` (how many per query). The mesh is the same
# in every simulation, so the graphs are built once. ``--stride`` keeps every
# ``stride``-th mesh node, to make a CPU run cheaper.


def radius_graph(points: np.ndarray, queries: np.ndarray, radius: float) -> dict:
    """For every query, the points within ``radius``, as flat index arrays."""
    nbrs = kdtree_neighbor_search(
        data=points, queries=queries, radius=radius, return_norm=False
    )
    counts = np.diff(np.asarray(nbrs["neighbors_row_splits"]))
    return dict(
        neighbors_index=np.asarray(nbrs["neighbors_index"]),
        segment_ids=np.repeat(np.arange(len(counts)), counts),
        counts=counts,
    )


def build_geometry(nodes: np.ndarray, arch: dict) -> dict:
    """The mesh, the latent grid and the two radius graphs, as GINO's inputs."""
    h, w = arch["latent_h"], arch["latent_w"]
    gy, gz = np.meshgrid(
        np.linspace(0, 1, h, dtype=np.float32),
        np.linspace(0, 1, w, dtype=np.float32),
        indexing="ij",
    )
    latent = np.stack([gy, gz], axis=-1)  # (h, w, 2)
    flat = latent.reshape(-1, 2)
    geometry = dict(
        input_geom=nodes[None],
        latent_queries=latent[None],
        output_queries=nodes[None],
        neighbors_in=radius_graph(nodes, flat, arch["in_gno_radius"]),
        neighbors_out=radius_graph(flat, nodes, arch["out_gno_radius"]),
    )
    n_in = len(geometry["neighbors_in"]["neighbors_index"])
    n_out = len(geometry["neighbors_out"]["neighbors_index"])
    print(f"graphs: {n_in:,} mesh->latent edges | {n_out:,} latent->mesh edges")
    return geometry


def latent_features(params: np.ndarray, arch: dict) -> np.ndarray:
    """``(n, 6)`` parameters -> ``(n, h, w, 6)``, constant over the latent grid."""
    shape = (len(params), arch["latent_h"], arch["latent_w"], params.shape[1])
    return np.broadcast_to(params[:, None, None, :], shape).astype(np.float32)


# %%
# Build the model
# ---------------
# ``GINO`` takes the node coordinates (``input_geom``, ``output_queries``),
# the latent grid (``latent_queries``), the two graphs and, here, the six
# parameters as ``latent_features`` on the grid; ``in_channels=0`` because the
# nodes carry no features of their own. The FNO in the middle has complex
# spectral weights.


def gino_kwargs(arch: dict) -> dict:
    """The constructor arguments of ``GINO``: ``arch`` without the latent grid size."""
    kw = {k: v for k, v in arch.items() if k not in ("latent_h", "latent_w")}
    kw["fno_n_modes"] = tuple(kw["fno_n_modes"])
    for k in ("in_gno_channel_mlp_hidden_layers", "out_gno_channel_mlp_hidden_layers"):
        kw[k] = list(kw[k])
    return kw


def build_model(arch: dict, geometry: dict, lf: np.ndarray, seed: int):
    """Build GINO, initialise it on one sample and print the parameter count."""
    model = GINO(
        in_channels=0,
        out_channels=1,
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
        **gino_kwargs(arch),
    )
    params = model.init(jax.random.PRNGKey(seed), **geometry, latent_features=lf)
    print(f"GINO | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss of one simulation is the relative L2 error of its standardised
# temperature over all nodes, ``||pred - y|| / ||y||``, as in the benchmark.


def rel_l2(pred, target):
    """``||pred - target|| / ||target||`` over all entries."""
    return jnp.sqrt(jnp.sum((pred - target) ** 2) / (jnp.sum(target**2) + 1e-12))


def make_loss(model, geometry: dict):
    """Return ``loss_fn(params, lf, y) -> relative L2`` for one simulation."""

    def loss_fn(params, lf, y):
        out = model.apply(params, **geometry, latent_features=lf)  # (1, n_nodes, 1)
        return rel_l2(out[0, :, 0], y)

    return loss_fn


# %%
# Optimiser
# ---------
# The benchmark trains GINO with ``adamw_cx``, the AdamW that conjugates
# complex gradients (``jax.grad`` returns the conjugate of the descent
# direction optax expects), through ``optax.inject_hyperparams`` so that the
# learning rate can be changed between epochs. ``hyperparam_dtype=jnp.float32``
# keeps the injected values real: by default they take the highest dtype among
# the weights, complex64 here, and every update would turn complex (the
# benchmark's ``train.py`` still leaves it out; its updates are complex with a
# zero imaginary part).


def make_optimizer(protocol: dict):
    """``adamw_cx`` with the learning rate exposed as an injected hyperparameter."""
    return optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
        learning_rate=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"],
    )


def cosine_lr(protocol: dict, epoch: int, epochs: int) -> float:
    """Learning rate after ``epoch`` epochs: a cosine from the peak to 0."""
    return protocol["learning_rate"] * (1 + math.cos(math.pi * epoch / epochs)) / 2


# %%
# Train
# -----
# One simulation per optimiser step, in a new random order every epoch, from
# NumPy's global generator seeded once, as in the benchmark. The cosine is
# stepped after each epoch, so epoch 0 runs at the peak rate. After every
# epoch the test simulations are scored with the same loss, to draw the error
# curve; the weights after the last epoch are the result. JAX compiles the
# step on its first call, so the first epoch is slower than the others.


def train(model, params, geometry, norm, lf, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    tx = make_optimizer(protocol)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, geometry)

    @jax.jit
    def train_step(params, opt_state, lf, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, lf, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)
    np.random.seed(seed)  # noqa: NPY002  (the benchmark's sample order)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        losses = []
        for i in np.random.permutation(len(norm["y_train"])):  # noqa: NPY002
            params, opt_state, loss = train_step(
                params, opt_state, lf["train"][i : i + 1], norm["y_train"][i]
            )
            losses.append(float(loss))
        secs = perf_counter() - t0
        test = [
            float(eval_step(params, lf["test"][i : i + 1], norm["y_test"][i]))
            for i in range(len(norm["y_test"]))
        ]
        lr = cosine_lr(protocol, epoch + 1, epochs)
        opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
        history.append(
            dict(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                test_loss=float(np.mean(test)),
                lr=lr,
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} "
            f"test_loss={history[-1]['test_loss']:.4f} next_lr={lr:.2e} "
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


def evaluate(model, params, geometry, norm, lf, data) -> dict:
    """Test metrics in kelvin at the mesh nodes, a baseline and the predictions."""
    predict = jax.jit(lambda p, f: model.apply(p, **geometry, latent_features=f))
    pred = np.stack(
        [
            np.asarray(predict(params, lf["test"][i : i + 1]))[0, :, 0]
            for i in range(len(lf["test"]))
        ]
    )
    pred = pred * norm["t_std"] + norm["t_mean"]
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
# GINO, and trains for a few epochs. Without it the script trains the
# benchmark-size GINO on all nodes of the full split for 500 epochs. That mode
# is meant for a GPU.


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
        "--smoke", action="store_true", help="few simulations, smaller GINO"
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/chipcooling_gino_tutorial")
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
    epochs = args.epochs or (3 if args.smoke else p["epochs"])
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
    t0 = perf_counter()
    geometry = build_geometry(norm["nodes"], arch)
    print(f"graphs built in {perf_counter() - t0:.1f}s")
    lf = dict(
        train=latent_features(norm["p_train"], arch),
        test=latent_features(norm["p_test"], arch),
    )
    model, params = build_model(arch, geometry, lf["train"][:1], args.seed)

    params, history = train(model, params, geometry, norm, lf, epochs, args.seed)
    metrics = evaluate(model, params, geometry, norm, lf, data)
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
        f"Chip cooling, GINO: test simulation {k} of {ntest}",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return dict(model=metrics["model"], baseline=metrics["baseline"])


if __name__ == "__main__":
    main()
