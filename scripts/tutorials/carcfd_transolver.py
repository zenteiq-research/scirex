# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting the surface pressure on a car with Transolver (Car-CFD)
==================================================================

**Problem.** Given the shape of a car, predict the steady pressure at each of
the 3,586 vertices of its surface mesh. Every car is a different ShapeNet
model, simulated in the same wind tunnel at 72 km/h (Umetani & Bickel 2018).
This is the benchmark of the GINO paper (Li et al. 2023, Table 2). The model
here is Transolver (Wu et al. 2024): one token per surface vertex, and
attention over a small set of learned "slices" of similar points instead of
over all pairs of points.

**What you will learn.**

1. How to read the Car-CFD files (a triangle mesh and a pressure array per
   car) and turn each car into a point cloud of 3,586 tokens: position and
   unit normal per vertex.
2. How to build a ``scirex.models.Transolver`` for irregular geometry and
   count its parameters.
3. How to train it in plain JAX and optax with the benchmark's optimiser:
   Adam with a one-epoch linear warm-up and cosine decay, one car per step,
   a relative L2 loss on the normalised pressure.
4. How to evaluate with the de-normalised relative L2 of the GINO paper, and
   how to plot truth, prediction and error on the car surface.

**Caveat.** The ``--smoke`` mode trains a tiny Transolver on a handful of
cars for a few epochs. It checks that the pipeline works and gives poor
predictions. It is not a benchmark number. For the benchmark protocol (500
training cars, 100 epochs, 75,481,841 parameters) use
``scripts/benchmarks/car_cfd/train.py --run transolver_val``; see the
"Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/carcfd_transolver.py \\
        --data-dir ~/data/car-pressure-data/processed-car-pressure-data --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.Transolver``,
# ``scirex.training.adamw_cx`` and ``scirex.utils.count_model_params``. The
# data are read with NumPy and open3d (a core dependency of ScireX); the loop
# is plain JAX and optax.

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
import open3d as o3d
import optax
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scirex
from scirex.models import Transolver
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed

# The committed Car-CFD runs were computed with jax_enable_x64 on (an earlier
# version of the library turned it on as a side effect of importing GINO; 1.0
# does not). Setting it explicitly keeps the tutorial in the numerics of those
# runs. The data and the weights are still float32.
jax.config.update("jax_enable_x64", True)

SEED = 0

# Training settings of the benchmark's "transolver_val" run, the same values
# as PROTOCOL and RUNS["transolver_val"] in scripts/benchmarks/car_cfd/_config.py:
# the GINO paper's split and 100 epochs, Adam (adamw_cx with no weight decay;
# Transolver has only real weights, so this is plain Adam), a peak learning
# rate of 1e-4 chosen on a validation split, one epoch of linear warm-up from
# 0 and cosine decay to 0, one car per step.
PROTOCOL = dict(
    n_train=500,
    n_test=111,
    n_epochs=100,
    learning_rate=1e-4,
    warmup_epochs=1,
    weight_decay=0.0,
    adam_b2=0.999,
    adam_eps=1e-6,
)

# Model sizes. "full" is the benchmark's Transolver: the Transolver paper's
# car shape (8 layers, 8 heads, MLP ratio 2, 32 slices) widened to GINO's
# parameter budget, 75,481,841 parameters (GINO: 75,917,953). "smoke" is a
# tiny Transolver that trains on a CPU.
SIZES = dict(
    full=dict(n_hidden=1136, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32),
    smoke=dict(n_hidden=64, n_layers=2, n_head=4, mlp_ratio=2, slice_num=16),
)

# The stored pressure has 3,682 entries per car and the mesh 3,586 vertices.
# The benchmark keeps entries 0-15 and 112-3681 so that the two match; the
# source of this slice does not say why (see the Car-CFD problem card).
KEEP = np.r_[0:16, 112:3682]


def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__} | open3d {o3d.__version__}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")


# %%
# Load the data
# -------------
# The data folder holds ``train.txt`` and ``test.txt`` (comma-separated car
# ids, the GINO paper's 500 / 111 split) and one folder per car with
# ``tri_mesh.ply`` (the surface mesh) and ``press.npy`` (the pressure). The
# loader below does what the benchmark's loader does for Transolver: one
# bounding box over all loaded cars scales the vertex positions into [0, 1];
# the unit vertex normals come from open3d; the pressure is standardised with
# one scalar mean and standard deviation, fitted on all loaded cars, train
# and test together, as in neuraloperator's reference loader. (The benchmark
# loader also computes a signed distance function on a token 8^3 grid, which
# Transolver never reads, so it is left out here.)


def read_ids(data_dir: Path, split: str, n: int) -> list[str]:
    """The first ``n`` car ids of ``train.txt`` or ``test.txt``."""
    ids = (Path(data_dir) / f"{split}.txt").read_text().split(",")
    return [i.strip() for i in ids[:n]]


def car_folder(data_dir: Path, car_id: str) -> Path:
    """``data/<id>``, or ``data/data/<id>`` as in the Zenodo archive."""
    folder = Path(data_dir) / "data" / car_id
    return folder if folder.exists() else Path(data_dir) / "data" / "data" / car_id


def load_cars(data_dir: Path, n_train: int, n_test: int) -> dict:
    """Positions, normals and normalised pressure of one split."""
    ids = read_ids(data_dir, "train", n_train) + read_ids(data_dir, "test", n_test)
    meshes = []
    for car_id in ids:
        path = car_folder(data_dir, car_id) / "tri_mesh.ply"
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    lo = np.min([m.get_min_bound() for m in meshes], axis=0)
    hi = np.max([m.get_max_bound() for m in meshes], axis=0)

    cars = []
    for car_id, mesh in zip(ids, meshes):
        vertices = (np.asarray(mesh.vertices) - lo) / (hi - lo)
        cars.append(
            dict(
                id=car_id,
                vertices=vertices.astype(np.float32),
                normals=np.asarray(mesh.vertex_normals).astype(np.float32),
                triangles=np.asarray(mesh.triangles),
            )
        )
    press = np.stack(
        [np.load(car_folder(data_dir, i) / "press.npy") for i in ids]
    ).astype(np.float32)
    # One scalar mean and std over every stored entry of every loaded car. The
    # benchmark's streaming normaliser ends with sqrt(var) * N / (N - 1); the
    # same factor is used here (it changes the std by 4e-7 at full size).
    n = press.size
    mean = press.mean(dtype=np.float64)
    std = np.sqrt(press.var(dtype=np.float64)) * n / (n - 1)
    for car, p in zip(cars, press):
        car["press"] = ((p - mean) / (std + 1e-7))[KEEP].astype(np.float32)
    print(
        f"cars: {n_train} train / {n_test} test | vertices per car: "
        f"{len(cars[0]['vertices'])} | pressure mean {mean:.3f}, std {std:.3f}"
    )
    return dict(
        train=cars[:n_train],
        test=cars[n_train:],
        press_mean=float(mean),
        press_scale=float(std + 1e-7),
        box=(lo, hi),  # to draw the cars in mesh coordinates
    )


# %%
# Build the model
# ---------------
# Each car is one sequence of 3,586 tokens with six features: the position
# (3) and the unit normal (3) of the vertex. This is the Transolver paper's
# car input (position, signed distance, normal) without the signed distance,
# which is zero on the surface, where the pressure lives. ``fun_dim=0``
# because there is no input function, and ``geometry="irregular"`` because
# the points are not on a grid. ``Transolver`` is a Flax module: ``init``
# creates the parameters, ``apply`` uses them.


def transolver_input(car: dict) -> jnp.ndarray:
    """``(1, n_vertices, 6)``: position and unit normal of every vertex."""
    return jnp.asarray(np.concatenate([car["vertices"], car["normals"]], axis=-1))[None]


def build_model(size: dict, car: dict, seed: int):
    """Build the Transolver, initialise it on one car and print its size."""
    model = Transolver(space_dim=6, fun_dim=0, out_dim=1, geometry="irregular", **size)
    params = model.init(jax.random.PRNGKey(seed), x=transolver_input(car))
    print(f"Transolver {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss of one car is the relative L2 error of the normalised pressure over
# its vertices, ``LpLoss(d=2, p=2)`` in the benchmark. The test metric is the
# same error on the de-normalised pressure, averaged over the test cars.


def rel_l2(pred, true, eps: float = 1e-8):
    """``||pred - true|| / (||true|| + eps)`` over all vertices of one car."""
    diff = jnp.linalg.norm((pred - true).ravel())
    return diff / (jnp.linalg.norm(true.ravel()) + eps)


def make_loss(model):
    """Return ``loss_fn(params, x, press) -> relative L2``."""

    def loss_fn(params, x, press):
        return rel_l2(model.apply(params, x=x)[0, :, 0], press)

    return loss_fn


# %%
# Optimiser
# ---------
# Adam, built as in the benchmark: ``adamw_cx`` with no weight decay inside
# ``optax.inject_hyperparams``, with the learning rate given as a per-step
# schedule, a linear warm-up from 0 over the first epoch to the peak, then
# cosine decay to 0 at the last step. At 75 million parameters the Transolver
# paper's own peak of 1e-3 diverged; 1e-4 was chosen on a validation split
# of the training cars (see the benchmark page).
# ``hyperparam_dtype=jnp.float32`` makes no difference here (Transolver's
# weights are all real) and is kept for the models with complex weights.


def make_optimizer(protocol: dict, steps_per_epoch: int, epochs: int):
    """Adam with a linear warm-up and cosine decay, one value per step."""
    warmup = int(protocol["warmup_epochs"] * steps_per_epoch)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=protocol["learning_rate"],
        warmup_steps=warmup,
        # At least one step of decay, so that a one-epoch run is valid too.
        decay_steps=max(steps_per_epoch * epochs, warmup + 1),
        end_value=0.0,
    )

    def make_tx(learning_rate, weight_decay):
        return adamw_cx(
            learning_rate,
            weight_decay=weight_decay,
            b2=protocol["adam_b2"],
            eps=protocol["adam_eps"],
        )

    tx = optax.inject_hyperparams(make_tx, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule, weight_decay=protocol["weight_decay"]
    )
    return tx, schedule


# %%
# Train
# -----
# One car per step, in a new random order every epoch. After every epoch the
# test cars are scored with the de-normalised relative L2, to draw the error
# curve; the reported weights are those after the last epoch. JAX compiles
# the step on its first call, so the first epoch is slower than the others.


def train(model, params, data, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    tx, schedule = make_optimizer(protocol, len(data["train"]), epochs)
    opt_state = tx.init(params)
    loss_fn = make_loss(model)
    predict = make_predict(model)

    @jax.jit
    def train_step(params, opt_state, x, press):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, press)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        losses = []
        for i in rng.permutation(len(data["train"])):
            car = data["train"][i]
            params, opt_state, loss = train_step(
                params, opt_state, transolver_input(car), jnp.asarray(car["press"])
            )
            losses.append(float(loss))
        secs = perf_counter() - t0
        test = evaluate(predict, params, data)["test_rel_l2"]
        history.append(
            dict(
                epoch=epoch,
                train_rel_l2=float(np.mean(losses)),
                test_rel_l2=test,
                lr=float(schedule((epoch + 1) * len(data["train"]))),
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_rel_l2={history[-1]['train_rel_l2']:.4f} "
            f"test_rel_l2={test:.4f} epoch_time={secs:.1f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# Prediction and truth are de-normalised with the pressure mean and scale,
# and the relative L2 error is taken per car. The mean over the test cars is
# the GINO paper's metric; the median, the 90th percentile and the worst car
# show how the error is spread.


def make_predict(model):
    """The jitted forward pass, ``predict(params, x)`` -> normalised pressure."""
    return jax.jit(lambda params, x: model.apply(params, x=x)[0, :, 0])


def evaluate(predict, params, data) -> dict:
    """De-normalised relative L2 per test car, and the de-normalised predictions."""
    mean, scale = data["press_mean"], data["press_scale"]
    errors, preds = [], []
    for car in data["test"]:
        pred = np.asarray(predict(params, transolver_input(car))) * scale + mean
        true = car["press"] * scale + mean
        errors.append(float(rel_l2(jnp.asarray(pred), jnp.asarray(true))))
        preds.append(pred)
    errors = np.asarray(errors)
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
# The standard figure of the benchmark, for one test car: an input (here the
# component of the unit normal along the car's length, one of the six
# features the model reads), and the true and predicted surface pressure on
# one colour scale, with the absolute error. The cars are drawn in mesh
# coordinates, so they keep their proportions; every triangle is coloured by
# the mean of its three vertex values, seen from the benchmark's elevated
# three-quarter view. A second figure shows the error per epoch.


def draw_surface(ax, vertices, triangles, values, cmap, norm) -> None:
    """One car, upright: mesh axes (width, height, length) drawn as (x, z, y)."""
    pts = vertices[:, [0, 2, 1]]
    faces = values[triangles].mean(axis=1)
    ax.add_collection3d(
        Poly3DCollection(
            pts[triangles], facecolors=plt.get_cmap(cmap)(norm(faces)), edgecolor="none"
        )
    )
    for setter, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(pts[:, k].min(), pts[:, k].max())
    ax.set_box_aspect(np.ptp(pts, axis=0))
    ax.view_init(elev=20.0, azim=-60.0)
    ax.set_axis_off()


def plot_prediction(car, pred, true, data, path: Path, title: str) -> None:
    """Save an input, true and predicted pressure, and the absolute error."""
    lo, hi = data["box"]
    vertices = car["vertices"] * (hi - lo) + lo
    err = np.abs(pred - true)
    lim = Normalize(vmin=float(true.min()), vmax=float(true.max()))
    panels = [
        ("Input: normal along length", car["normals"][:, 2], "coolwarm",
         Normalize(vmin=-1.0, vmax=1.0)),
        ("Truth: pressure", true, "jet", lim),
        ("Prediction: pressure", pred, "jet", lim),
        ("Absolute error", err, "magma", Normalize(vmin=0.0, vmax=float(err.max()))),
    ]  # fmt: skip
    fig = plt.figure(figsize=(19, 4.6))
    for k, (name, values, cmap, norm) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, k, projection="3d")
        draw_surface(ax, vertices, car["triangles"], values, cmap, norm)
        ax.set_title(name)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and test relative L2 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        epochs,
        [h["train_rel_l2"] for h in history],
        marker="o",
        label="train (normalised)",
    )
    ax.plot(
        epochs,
        [h["test_rel_l2"] for h in history],
        marker="o",
        label="test (de-normalised)",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 (mean over cars)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few cars, uses the tiny Transolver and trains for a few
# epochs. Without it the script trains the benchmark's Transolver on the full
# 500 / 111 split for 100 epochs. That mode is meant for a GPU. ``--lr``
# replaces the peak learning rate, for the smoke runs.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("CARCFD_DATA_DIR"),
        help="folder with train.txt, test.txt and data/ (or set $CARCFD_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few cars, tiny Transolver")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/carcfd_transolver_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=8, help="cars in smoke mode")
    ap.add_argument("--n-test", type=int, default=4, help="cars in smoke mode")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $CARCFD_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = dict(PROTOCOL, learning_rate=args.lr or PROTOCOL["learning_rate"])
    n_train, n_test = (
        (args.n_train, args.n_test) if args.smoke else (p["n_train"], p["n_test"])
    )
    epochs = args.epochs or (5 if args.smoke else p["n_epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data = load_cars(Path(args.data_dir).expanduser(), n_train, n_test)
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params = build_model(size, data["train"][0], args.seed)

    params, history = train(model, params, data, epochs, args.seed, protocol=p)
    metrics = evaluate(make_predict(model), params, data)
    print(
        f"test relative L2: mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} "
        f"| p90 {metrics['p90']:.4f} | worst {metrics['worst']:.4f}"
    )

    car = data["test"][0]
    true = car["press"] * data["press_scale"] + data["press_mean"]
    plot_prediction(
        car,
        metrics["predictions"][0],
        true,
        data,
        out_dir / "prediction.png",
        f"Car-CFD, Transolver: first test car, id {car['id']} "
        f"(relative L2 {metrics['errors'][0]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
