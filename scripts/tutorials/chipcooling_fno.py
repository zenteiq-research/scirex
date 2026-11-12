# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting the temperature of a liquid-cooled dual-chip cold plate with FNO
==========================================================================

**Problem.** Given the six boundary-condition parameters of a liquid-cooled
cold plate that carries two chips (the heat flux of each chip, and the mass
flow rate and temperature of each of the two coolant inlets), predict the
steady temperature on the central plane of the plate. The data are 2000 CFD
simulations of Curl & Hu (2026); the model is the Fourier Neural Operator
(FNO, Li et al. 2021). The CFD mesh is unstructured, so the temperature is
first interpolated onto a regular 128 x 256 grid, and the FNO works on that
grid.

**What you will learn.**

1. How to read the chip cooling files (``model_setup.json`` and one
   ``sim_*.npz`` per simulation) and build the 1600 / 400 split.
2. How to put an unstructured CFD field on a grid for an FNO: linear
   interpolation, a nearest-neighbour fill and a mask of the cells that lie
   inside the CFD domain.
3. How to give six scalars to an operator that maps fields to fields: each
   parameter becomes a constant input channel, plus the domain mask.
4. How to train with a masked relative L2 loss, the complex-aware
   ``adamw_cx`` and a cosine learning rate set once per epoch.
5. How to evaluate in kelvin (R2, RMSE, MAE, maximum error and relative L2
   over the in-domain cells) and plot truth, prediction and error.

**Caveat.** The ``--smoke`` mode trains the FNO on a few simulations, on a
coarser grid, for a few epochs. It checks that the pipeline works and gives
poor predictions. It is not a benchmark number. For the benchmark protocol
(1600 training simulations on the 128 x 256 grid, 500 epochs) use
``scripts/benchmarks/chip_cooling/train.py``; see the "Reproduce at full
scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/chipcooling_fno.py \\
        --data-dir ~/data/chip_cooling --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.FNO``,
# ``scirex.training.adamw_cx`` and ``scirex.utils.count_model_params``. The
# data are read with NumPy and interpolated with SciPy; the loop itself is
# plain JAX and optax.

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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay

import scirex
from scirex.models import FNO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 42

# The chip cooling benchmark's settings, the same values as PROTOCOL and
# RUNS["fno"].opt in scripts/benchmarks/chip_cooling/_config.py.
PROTOCOL = dict(
    ntrain=1600,
    ntest=400,
    grid=(128, 256),
    epochs=500,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-4,
)

# The benchmark's FNO (1,829,496 parameters, matched to Transolver's
# 1,807,561). The smoke run uses it too: its parameter count does not depend
# on the grid, and on a coarser grid it trains on a CPU in minutes.
ARCH = dict(n_modes=(24, 24), hidden_channels=27)

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
# Put the fields on a grid
# ------------------------
# The FNO needs a regular grid. ``rasterise`` interpolates every temperature
# field linearly from the mesh nodes onto a ``128 x 256`` grid over the
# bounding box of the plane (``y`` across, ``z`` along). The plane is not a
# rectangle: ``domain_mask`` marks the grid cells that contain at least one
# mesh node. Cells inside the mask that the linear interpolation misses (near
# the boundary) take the nearest node's value; cells outside are set to 0 and
# never count in the loss or the metrics. This is what
# ``scirex.data.datasets.chipcooling.build_grid_cache`` does for the benchmark
# with ``scipy.interpolate.griddata``. Every simulation has the same mesh, so
# the Delaunay triangulation that linear interpolation needs is built once
# and reused, which gives the same values many times faster.


def domain_mask(coords: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray):
    """``(H, W)`` mask: True where at least one mesh node falls in the cell."""

    def edges(g):
        e = np.empty(len(g) + 1)
        e[1:-1] = 0.5 * (g[:-1] + g[1:])
        e[0], e[-1] = g[0] - 0.5 * (g[1] - g[0]), g[-1] + 0.5 * (g[-1] - g[-2])
        return e

    iy = np.searchsorted(edges(y_grid), coords[:, 1], side="right") - 1
    iz = np.searchsorted(edges(z_grid), coords[:, 2], side="right") - 1
    ok = (iy >= 0) & (iy < len(y_grid)) & (iz >= 0) & (iz < len(z_grid))
    mask = np.zeros((len(y_grid), len(z_grid)), dtype=bool)
    mask[iy[ok], iz[ok]] = True
    return mask


def rasterise(coords: np.ndarray, fields: np.ndarray, grid: tuple[int, int]) -> dict:
    """Fields ``(n, n_nodes)`` -> ``(n, H, W)`` on a grid, 0 outside the mask."""
    y, z = coords[:, 1], coords[:, 2]
    y_grid = np.linspace(y.min(), y.max(), grid[0])
    z_grid = np.linspace(z.min(), z.max(), grid[1])
    zg, yg = np.meshgrid(z_grid, y_grid)
    mask = domain_mask(coords, y_grid, z_grid)
    nodes = np.stack([z, y], axis=-1)
    tri = Delaunay(nodes)  # one triangulation for every field
    out = np.zeros((len(fields), *grid), dtype=np.float32)
    for i, f in enumerate(fields):
        vals = LinearNDInterpolator(tri, f)(zg, yg)
        missing = mask & ~np.isfinite(vals)
        if missing.any():
            vals[missing] = NearestNDInterpolator(nodes, f)(zg, yg)[missing]
        out[i] = np.where(mask, vals, 0.0)
    return dict(fields=out, mask=mask, y_grid=y_grid, z_grid=z_grid)


# %%
# Inputs and normalisation
# ------------------------
# The operator maps a field to a field, but the input here is six numbers. Each
# parameter is broadcast over the grid as one constant channel, and the domain
# mask is a seventh channel, so a sample is ``(7, 128, 256)``. The six
# parameter channels are standardised with the training mean and standard
# deviation; the mask stays 0/1. The target gets one mean and one standard
# deviation, over the in-domain cells of the training fields, and is 0
# outside the domain.


def grid_inputs(params: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(n, 6)`` parameters -> ``(n, 7, H, W)``: six constant channels and the mask."""
    x = np.zeros((len(params), 7, *mask.shape), dtype=np.float32)
    x[:, :6] = params[:, :, None, None]
    x[:, 6] = mask
    return x


def normalise(data: dict, grid: dict) -> dict:
    """Standardised inputs and targets; the statistics come from the training split."""
    mask = grid["mask"]
    x_train = grid_inputs(data["params_train"], mask)
    x_test = grid_inputs(data["params_test"], mask)
    x_mean = x_train[:, :6].mean(axis=(0, 2, 3), keepdims=True)
    x_std = x_train[:, :6].std(axis=(0, 2, 3), keepdims=True) + 1e-8
    for x in (x_train, x_test):
        x[:, :6] = (x[:, :6] - x_mean) / x_std
    y_train, y_test = grid["fields_train"], grid["fields_test"]
    y_mean = float(y_train[:, mask].mean())
    y_std = float(y_train[:, mask].std()) + 1e-8

    def encode(y):
        return np.where(mask, (y - y_mean) / y_std, 0.0)[:, None].astype(np.float32)

    return dict(
        x_train=x_train,
        x_test=x_test,
        y_train=encode(y_train),
        y_test=encode(y_test),
        y_mean=y_mean,
        y_std=y_std,
    )


# %%
# Build the model
# ---------------
# ``FNO`` is a Flax module and channels-first, so a batch is
# ``(batch, 7, 128, 256)``. The benchmark keeps the class defaults otherwise,
# including ``positional_embedding="grid"``, which appends the two grid
# coordinates as extra input channels: the parameters say how hot, the
# coordinates say where. ``n_modes=(24, 24)`` counts the modes kept along each
# axis in total. ``count_model_params`` counts a complex weight as two numbers.


def build_model(arch: dict, sample: np.ndarray, seed: int):
    """Build the FNO, initialise it on one sample and print the parameter count."""
    model = FNO(in_channels=7, out_channels=1, **arch)
    _, key = jax.random.split(jax.random.PRNGKey(seed))
    params = model.init(key, x=jnp.zeros((1, *sample.shape), dtype=jnp.float32))
    print(f"FNO {arch} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss is the relative L2 error of the standardised temperature over the
# in-domain cells of the whole batch,
# ``sqrt(sum(mask (pred - y)^2) / sum(mask y^2))``, as in the benchmark.


def masked_rel_l2(pred, target, mask):
    """Relative L2 over the batch, counting only the cells where ``mask`` is set."""
    m = mask.astype(pred.dtype)
    num = jnp.sum(((pred - target) ** 2) * m)
    return jnp.sqrt(num / (jnp.sum((target**2) * m) + 1e-12))


def make_loss(model, mask: np.ndarray):
    """Return ``loss_fn(params, x, y) -> masked relative L2``."""
    m = jnp.asarray(mask)[None, None]

    def loss_fn(params, x, y):
        return masked_rel_l2(model.apply(params, x=x), y, m)

    return loss_fn


# %%
# Optimiser
# ---------
# The spectral weights of an FNO are complex, so the benchmark trains with
# ``adamw_cx``, the AdamW that conjugates complex gradients (``jax.grad``
# returns the conjugate of the descent direction optax expects). The learning
# rate follows a cosine from 1e-3 to 0, set once at the start of every epoch
# through ``optax.inject_hyperparams``. ``hyperparam_dtype=jnp.float32`` keeps
# the injected values real: by default they take the highest dtype among the
# weights, complex64 here, and every update would turn complex (the benchmark's
# ``train.py`` still leaves it out; its updates are complex with a zero
# imaginary part). ``optimizer="adamw"`` swaps in plain ``optax.adamw``, with
# the same eps of 1e-6, to see what the conjugation changes.


def make_optimizer(protocol: dict, optimizer: str = "adamw_cx"):
    """AdamW with the learning rate exposed as an injected hyperparameter.

    ``"adamw_cx"`` (the benchmark's) is complex-aware; ``"adamw"`` is plain optax.
    """
    builder = dict(adamw_cx=adamw_cx, adamw=optax.adamw)[optimizer]
    return optax.inject_hyperparams(builder, hyperparam_dtype=jnp.float32)(
        learning_rate=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"],
        eps=1e-6,  # adamw_cx's default, which the benchmark uses
    )


def cosine_lr(protocol: dict, epoch: int, epochs: int) -> float:
    """Learning rate of ``epoch``: a cosine from the peak to 0 over ``epochs``."""
    return protocol["learning_rate"] * (1 + math.cos(math.pi * epoch / epochs)) / 2


# %%
# Train
# -----
# Batches of 8 in a new random order every epoch, one optimiser step per
# batch. The order comes from NumPy's global generator seeded once, as in the
# benchmark. After every epoch the test simulations are scored with the same
# loss, to draw the error curve; the weights after the last epoch are the
# result. JAX compiles the step on its first call, so the first epoch is
# slower than the others.


def train(
    model, params, norm, mask, epochs, seed, protocol=PROTOCOL, optimizer="adamw_cx"
):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    tx = make_optimizer(protocol, optimizer)
    opt_state = tx.init(params)
    loss_fn = make_loss(model, mask)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)
    bs = protocol["batch_size"]
    x_train, y_train = norm["x_train"], norm["y_train"]
    x_test, y_test = norm["x_test"], norm["y_test"]
    np.random.seed(seed)  # noqa: NPY002  (the benchmark's batch order)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        lr = cosine_lr(protocol, epoch, epochs)
        opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
        order = np.arange(len(x_train))
        np.random.shuffle(order)  # noqa: NPY002
        losses = []
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            params, opt_state, loss = train_step(
                params, opt_state, jnp.asarray(x_train[b]), jnp.asarray(y_train[b])
            )
            losses.append(float(loss))
        secs = perf_counter() - t0
        test = [
            float(
                eval_step(params, jnp.asarray(x_test[i : i + bs]), y_test[i : i + bs])
            )
            for i in range(0, len(x_test), bs)
        ]
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
            f"test_loss={history[-1]['test_loss']:.4f} lr={lr:.2e} "
            f"epoch_time={secs:.1f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# The benchmark's metrics, in kelvin, over the in-domain cells of the test
# simulations: R2, RMSE and MAE pooled over all cells, the largest absolute
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


def evaluate(model, params, norm, grid, batch_size: int = 8) -> dict:
    """Test metrics in kelvin on the in-domain cells, a baseline and the predictions."""
    predict = jax.jit(lambda p, x: model.apply(p, x=x))
    x_test, mask = norm["x_test"], grid["mask"]
    pred = np.concatenate(
        [
            np.asarray(predict(params, jnp.asarray(x_test[i : i + batch_size])))[:, 0]
            for i in range(0, len(x_test), batch_size)
        ]
    )
    pred = pred * norm["y_std"] + norm["y_mean"]
    true = grid["fields_test"]
    baseline = np.broadcast_to(grid["fields_train"].mean(axis=0), true.shape)
    return dict(
        model=field_metrics(pred[:, mask], true[:, mask]),
        baseline=field_metrics(baseline[:, mask], true[:, mask]),
        predictions=np.where(mask, pred, np.nan),
    )


# %%
# Plot
# ----
# The standard figure of the benchmark, for one test simulation, on the grid
# in millimetres: the input (the six parameters, the only thing the model
# sees), the true and the predicted temperature on one colour scale, and the
# absolute error. Cells outside the domain are blank. We also plot the loss
# per epoch.


def plot_prediction(params, truth, pred, grid, path: Path, title: str) -> None:
    """Save input parameters, true temperature, predicted temperature and error."""
    zg, yg = np.meshgrid(grid["z_grid"] * 1e3, grid["y_grid"] * 1e3)
    truth = np.where(grid["mask"], truth, np.nan)
    lim = (np.nanmin(truth), np.nanmax(truth))
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
        mesh = ax.pcolormesh(zg, yg, values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
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
    ax.set_ylabel("masked relative L2 (standardised)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few simulations on a 64 x 128 grid and trains for a few
# epochs. Without it the script trains on the full split on the benchmark's
# 128 x 256 grid for 500 epochs. That mode is meant for a GPU.


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
        "--smoke", action="store_true", help="few simulations, coarser grid"
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/chipcooling_fno_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=64, help="simulations in smoke mode")
    ap.add_argument("--n-test", type=int, default=16, help="simulations in smoke mode")
    ap.add_argument(
        "--grid",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="grid size (default: 128 256, or 64 128 with --smoke)",
    )
    ap.add_argument(
        "--optimizer",
        choices=("adamw_cx", "adamw"),
        default="adamw_cx",
        help="adamw_cx (the benchmark's) or plain optax AdamW, for comparison",
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
    grid_shape = tuple(args.grid or ((64, 128) if args.smoke else p["grid"]))
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data = load_chip_cooling(Path(args.data_dir).expanduser(), ntrain, ntest)
    print(f"data loading: {perf_counter() - t0:.1f}s")
    t0 = perf_counter()
    grid = rasterise(
        data["coords"], np.concatenate([data["T_train"], data["T_test"]]), grid_shape
    )
    grid["fields_train"] = grid["fields"][:ntrain]
    grid["fields_test"] = grid["fields"][ntrain:]
    print(
        f"rasterised to {grid_shape}: {grid['mask'].sum()} in-domain cells "
        f"({grid['mask'].mean():.0%}) | {perf_counter() - t0:.1f}s"
    )
    norm = normalise(data, grid)
    model, params = build_model(ARCH, norm["x_train"][0], args.seed)

    params, history = train(
        model, params, norm, grid["mask"], epochs, args.seed, optimizer=args.optimizer
    )
    metrics = evaluate(model, params, norm, grid)
    for name in ("model", "baseline"):
        m = metrics[name]
        print(
            f"{name:8s}: R2 {m['R2']:.4f} | RMSE {m['RMSE']:.3f} K "
            f"| MAE {m['MAE']:.3f} K | max AE {m['max_AE']:.2f} K "
            f"| rel L2 {m['rel_l2']:.2e}"
        )

    k = len(norm["x_test"]) // 2
    plot_prediction(
        norm["x_test"][k, :6, 0, 0],
        grid["fields_test"][k],
        metrics["predictions"][k],
        grid,
        out_dir / "prediction.png",
        f"Chip cooling, FNO: test simulation {k} of {ntest}",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return dict(model=metrics["model"], baseline=metrics["baseline"])


if __name__ == "__main__":
    main()
