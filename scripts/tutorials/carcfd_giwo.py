# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Predicting the surface pressure on a car with GIWO (Car-CFD)
============================================================

**Problem.** Given the shape of a car, predict the steady pressure at each of
the 3,586 vertices of its surface mesh. Every car is a different ShapeNet
model, simulated in the same wind tunnel at 72 km/h (Umetani & Bickel 2018).
This is the benchmark of the GINO paper (Li et al. 2023, Table 2). The model
here, GIWO, is GINO with one part swapped: the Fourier (FNO) layers on the
latent grid become wavelet (WNO) layers. Everything around them is GINO's
decoder-only shell: the signed distance function (SDF) of the car on a
regular latent grid is the input, and a graph kernel carries the latent
field out to the surface vertices. GIWO is not a published architecture.

**What you will learn.**

1. How to read the Car-CFD files (a triangle mesh and a pressure array per
   car) and build the inputs the benchmark uses: the SDF on a latent grid,
   with open3d, and range-normalised coordinates.
2. How to precompute GINO's per-car neighbour lists with the public
   ``scirex.layers`` search, and pad them so the training step compiles once.
3. How to build GIWO from public parts, a subclass of
   ``scirex.models.GINO`` whose latent operator is a ``scirex.models.WNO``,
   and count its parameters.
4. How to train it in plain JAX and optax with the benchmark's optimiser:
   Adam with a one-epoch linear warm-up and cosine decay, and a relative L2
   loss.
5. How to evaluate with the de-normalised relative L2 of the GINO paper, and
   how to plot truth, prediction and error on the car surface.

**Caveat.** The ``--smoke`` mode trains a tiny GIWO on a coarse latent grid,
on a handful of cars, for a few epochs. It checks that the pipeline works and
gives poor predictions. It is not a benchmark number. For the benchmark
protocol (500 training cars, 64^3 latent grid, 100 epochs, 74,197,232
parameters) use ``scripts/benchmarks/car_cfd/train.py --run giwo_val``; see the
"Reproduce at full scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/carcfd_giwo.py \\
        --data-dir ~/data/car-pressure-data/processed-car-pressure-data --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.GINO``,
# ``scirex.models.WNO``, ``scirex.layers.kdtree_neighbor_search``,
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
from scirex.layers import kdtree_neighbor_search
from scirex.models import GINO, WNO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed

# The committed Car-CFD runs were computed with jax_enable_x64 on (an earlier
# version of the library turned it on as a side effect of importing GINO; 1.0
# does not). Setting it explicitly keeps the tutorial in the numerics of those
# runs. The data and the weights are still float32.
jax.config.update("jax_enable_x64", True)

SEED = 0

# Training settings of the benchmark's "giwo_val" run, the same values as
# PROTOCOL and RUNS["giwo_val"] in scripts/benchmarks/car_cfd/_config.py: the
# GINO paper's split and 100 epochs, Adam (adamw_cx with no weight decay;
# GIWO's weights are all real, so this is plain Adam), a peak learning rate of
# 8e-3 chosen on a validation split, one epoch of linear warm-up from 0 and
# cosine decay to 0, one car per step.
PROTOCOL = dict(
    n_train=500,
    n_test=111,
    n_epochs=100,
    learning_rate=8e-3,
    warmup_epochs=1,
    weight_decay=0.0,
    adam_b2=0.999,
    adam_eps=1e-6,
)

# Model sizes. "full" is the benchmark's GIWO, sized to GINO's parameter
# budget: GINO's decoder-only shell (output graph kernel of radius 0.033)
# around a 4-layer db4 WNO with 3 decomposition levels and width 29 on the
# 64^3 latent grid, 74,197,232 parameters (GINO: 75,917,953). "smoke" is a
# tiny GIWO on an 8^3 grid that trains on a CPU: on a CPU the wavelet
# library's 3-D transform costs about 5 s per training step at 16^3 and
# 0.15 s at 8^3. Its larger radius keeps about as many latent points per
# vertex (about two grid spacings) as 0.033 does on the 64^3 grid.
SIZES = dict(
    full=dict(
        latent_res=64,
        width=29,
        wno_n_layers=4,
        wno_level=3,
        wno_projection_hidden_channels=128,
        out_gno_radius=0.033,
        out_gno_channel_mlp_hidden_layers=(512, 256),
    ),
    smoke=dict(
        latent_res=8,
        width=8,
        wno_n_layers=2,
        wno_level=1,
        wno_projection_hidden_channels=32,
        out_gno_radius=0.3,
        out_gno_channel_mlp_hidden_layers=(64, 32),
    ),
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
# loader below does what the benchmark's loader does: one bounding box over
# all loaded cars scales the coordinates into [0, 1]; the SDF of every car is
# computed on a regular grid spanning that box and mapped linearly to
# [1e-6, 1] with the minimum and maximum of the training cars; the pressure is
# standardised with one scalar mean and standard deviation, fitted on all
# loaded cars, train and test together, as in neuraloperator's reference
# loader.


def read_ids(data_dir: Path, split: str, n: int) -> list[str]:
    """The first ``n`` car ids of ``train.txt`` or ``test.txt``."""
    ids = (Path(data_dir) / f"{split}.txt").read_text().split(",")
    return [i.strip() for i in ids[:n]]


def car_folder(data_dir: Path, car_id: str) -> Path:
    """``data/<id>``, or ``data/data/<id>`` as in the Zenodo archive."""
    folder = Path(data_dir) / "data" / car_id
    return folder if folder.exists() else Path(data_dir) / "data" / "data" / car_id


def load_cars(data_dir: Path, n_train: int, n_test: int, latent_res: int) -> dict:
    """Meshes, SDF on the latent grid and normalised pressure of one split."""
    ids = read_ids(data_dir, "train", n_train) + read_ids(data_dir, "test", n_test)
    meshes = []
    for car_id in ids:
        mesh = o3d.io.read_triangle_mesh(
            str(car_folder(data_dir, car_id) / "tri_mesh.ply")
        )
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    lo = np.min([m.get_min_bound() for m in meshes], axis=0)
    hi = np.max([m.get_max_bound() for m in meshes], axis=0)
    axes = [np.linspace(lo[k], hi[k], latent_res) for k in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).astype(np.float32)

    cars, raw_press = [], []
    for car_id, mesh in zip(ids, meshes):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        cars.append(
            dict(
                id=car_id,
                vertices=((np.asarray(mesh.vertices) - lo) / (hi - lo)).astype(
                    np.float32
                ),
                normals=np.asarray(mesh.vertex_normals).astype(np.float32),
                triangles=np.asarray(mesh.triangles),
                sdf=scene.compute_signed_distance(grid).numpy(),
            )
        )
        raw_press.append(np.load(car_folder(data_dir, car_id) / "press.npy"))

    d_min = min(c["sdf"].min() for c in cars[:n_train])
    d_max = max(c["sdf"].max() for c in cars[:n_train])
    press = np.stack(raw_press).astype(np.float32)
    # One scalar mean and std over every stored entry of every loaded car. The
    # benchmark's streaming normaliser ends with sqrt(var) * N / (N - 1); the
    # same factor is used here (it changes the std by 4e-7 at full size).
    n = press.size
    mean = press.mean(dtype=np.float64)
    std = np.sqrt(press.var(dtype=np.float64)) * n / (n - 1)
    for car, p in zip(cars, press):
        sdf = (car.pop("sdf") - d_min) / (d_max - d_min)
        car["sdf"] = ((1 - 1e-6) * sdf + 1e-6)[..., None]
        car["press"] = ((p - mean) / (std + 1e-7))[KEEP].astype(np.float32)
    print(
        f"cars: {n_train} train / {n_test} test | vertices per car: "
        f"{len(cars[0]['vertices'])} | latent grid {latent_res}^3 | "
        f"pressure mean {mean:.3f}, std {std:.3f}"
    )
    return dict(
        train=cars[:n_train],
        test=cars[n_train:],
        latent_queries=((grid - lo) / (hi - lo)).astype(np.float32),
        press_mean=float(mean),
        press_scale=float(std + 1e-7),
        box=(lo, hi),  # to draw the cars in mesh coordinates
        sdf_zero=float((1 - 1e-6) * (0 - d_min) / (d_max - d_min) + 1e-6),
    )


# %%
# Neighbour lists
# ---------------
# GINO's output graph kernel sums, for every surface vertex, over the latent
# grid points within ``out_gno_radius``. A car sits at its own place in the
# fixed grid, so every car has its own lists. They need a KD-tree search that
# cannot run inside a jitted step, so they are built once, before training,
# with the public ``kdtree_neighbor_search``. Each list is then padded to the
# longest one: padded edges point at grid point 0 and belong to segment
# ``n_vertices``, one past the last vertex, so the sum over ``n_vertices``
# segments drops them, and every car has the same shapes, so ``jax.jit``
# compiles the step once.


def attach_neighbors(
    cars: list[dict], latent_queries: np.ndarray, radius: float
) -> int:
    """Store padded output-kernel neighbour lists in every car; return their length."""
    grid = latent_queries.reshape(-1, 3)
    raw = [kdtree_neighbor_search(grid, c["vertices"], radius) for c in cars]
    max_edges = max(len(r["neighbors_index"]) for r in raw)
    for car, r in zip(cars, raw):
        counts = np.diff(r["neighbors_row_splits"])
        n_vertices, pad = len(counts), max_edges - len(r["neighbors_index"])
        car["neighbors_out"] = dict(
            neighbors_index=np.concatenate(
                [r["neighbors_index"], np.zeros(pad, np.int32)]
            ),
            segment_ids=np.concatenate(
                [np.repeat(np.arange(n_vertices, dtype=np.int32), counts),
                 np.full(pad, n_vertices, np.int32)]
            ),
            counts=counts.astype(np.int32),
        )  # fmt: skip
    return max_edges


# %%
# Build the model
# ---------------
# GIWO is built as a subclass of the public ``GINO``, so the decoder-only
# shell is GINO's own code: ``use_in_gno=False`` (no input graph kernel; the
# SDF on the latent grid, one channel, is the whole latent input), the output
# graph kernel and the projection head. Only ``latent_embedding``, the step
# on the latent grid, is replaced: GINO's lifting and FNO layers become one
# channels-last ``WNO`` (a lifting layer, ``wno_n_layers`` wavelet layers and
# a two-layer projection back to ``fno_hidden_channels``), computed in
# float32 as the rest of the model. GINO's own lifting and FNO layers are
# still declared by ``GINO.setup`` but never called, so they own no
# parameters. A wavelet layer's weights are shaped by the coarsest subband of
# the whole grid, so on the 64^3 grid the width and the number of levels
# trade off: width 29 at 3 levels (14^3 subbands) is the closest to GINO's
# budget (see the benchmark page).


class DecoderGIWO(GINO, name="carcfd_tutorial_giwo"):
    """GINO's decoder-only shell around a WNO on the latent grid."""

    wno_size: tuple = (64, 64, 64)
    wno_level: int = 3
    wno_n_layers: int = 4
    wno_projection_hidden_channels: int = 128

    def setup(self) -> None:
        super().setup()
        self.wno = WNO(
            in_channels=self.latent_feature_channels,
            out_channels=self.fno_hidden_channels,
            hidden_channels=self.fno_hidden_channels,
            n_layers=self.wno_n_layers,
            level=self.wno_level,
            size=tuple(self.wno_size),
            wavelet="db4",
            mode="symmetric",
            compute_dtype=jnp.float32,
            projection_hidden_channels=self.wno_projection_hidden_channels,
        )

    def latent_embedding(self, in_p, ada_in=None):
        """``(batch, n1, n2, n3, c)`` -> ``(batch, hidden, n1, n2, n3)``, as GINO's."""
        del ada_in
        return jnp.moveaxis(self.wno(in_p), -1, 1)


def gino_inputs(car: dict, latent_queries: np.ndarray) -> dict:
    """The keyword arguments of ``GINO.__call__`` for one car (batch of one)."""
    v = jnp.asarray(car["vertices"])[None]
    return dict(
        input_geom=v,
        latent_queries=jnp.asarray(latent_queries)[None],
        output_queries=v,
        latent_features=jnp.asarray(car["sdf"])[None],
        neighbors_out={k: jnp.asarray(a) for k, a in car["neighbors_out"].items()},
    )


def build_model(size: dict, car: dict, latent_queries: np.ndarray, seed: int):
    """Build GIWO, initialise it on one car and print its size."""
    model = DecoderGIWO(
        in_channels=0,
        out_channels=1,
        latent_feature_channels=1,
        use_in_gno=False,
        fno_hidden_channels=size["width"],
        fno_channel_mlp_expansion=1.0,
        fno_resolution_scaling_factor=1,
        out_gno_radius=size["out_gno_radius"],
        out_gno_channel_mlp_hidden_layers=size["out_gno_channel_mlp_hidden_layers"],
        wno_size=(size["latent_res"],) * 3,
        wno_level=size["wno_level"],
        wno_n_layers=size["wno_n_layers"],
        wno_projection_hidden_channels=size["wno_projection_hidden_channels"],
    )
    params = model.init(jax.random.PRNGKey(seed), **gino_inputs(car, latent_queries))
    print(f"GIWO {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# The loss of one car is the relative L2 error of the normalised pressure over
# its vertices, ``LpLoss(d=2, p=2)`` in the benchmark. The test metric is the
# same error on the de-normalised pressure, averaged over the test cars.


def rel_l2(pred, true, eps: float = 1e-8):
    """``||pred - true|| / (||true|| + eps)`` over all vertices of one car."""
    return jnp.linalg.norm((pred - true).ravel()) / (
        jnp.linalg.norm(true.ravel()) + eps
    )


def make_loss(model):
    """Return ``loss_fn(params, car_inputs, press) -> relative L2``."""

    def loss_fn(params, inputs, press):
        out = model.apply(params, **inputs)[0, :, 0]
        return rel_l2(out, press)

    return loss_fn


# %%
# Optimiser
# ---------
# Adam, built as in the benchmark: ``adamw_cx`` with no weight decay inside
# ``optax.inject_hyperparams``, with the learning rate given as a per-step
# schedule, a linear warm-up from 0 over the first epoch to the peak, then
# cosine decay to 0 at the last step. The peak of 8e-3 was chosen on a
# validation split of the training cars; 1.6e-2 diverged (see the benchmark
# page). ``hyperparam_dtype=jnp.float32`` makes no difference here (GIWO's
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
    q = data["latent_queries"]
    tx, _ = make_optimizer(protocol, len(data["train"]), epochs)
    opt_state = tx.init(params)
    loss_fn = make_loss(model)
    predict = make_predict(model)

    @jax.jit
    def train_step(params, opt_state, inputs, press):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, press)
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
                params, opt_state, gino_inputs(car, q), jnp.asarray(car["press"])
            )
            losses.append(float(loss))
        secs = perf_counter() - t0
        test = evaluate(predict, params, data)["test_rel_l2"]
        history.append(
            dict(
                epoch=epoch,
                train_rel_l2=float(np.mean(losses)),
                test_rel_l2=test,
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
    """The jitted forward pass, ``predict(params, inputs)`` -> normalised pressure."""
    return jax.jit(lambda params, inputs: model.apply(params, **inputs)[0, :, 0])


def evaluate(predict, params, data) -> dict:
    """De-normalised relative L2 per test car, and the de-normalised predictions."""
    q, mean, scale = data["latent_queries"], data["press_mean"], data["press_scale"]
    errors, preds = [], []
    for car in data["test"]:
        pred = np.asarray(predict(params, gino_inputs(car, q))) * scale + mean
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
# The standard figure of the benchmark, for one test car: the input (here the
# normalised SDF on the latent grid's middle plane across the car's width,
# which is what the model reads, with the car's outline where the SDF is 0),
# and the true and predicted surface pressure on one colour scale, with the
# absolute error. The cars are drawn in mesh coordinates, so they keep their
# proportions; every triangle is coloured by the mean of its three vertex
# values, seen from the benchmark's elevated three-quarter view. A second
# figure shows the error per epoch.


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
    """Save the SDF input, true and predicted pressure, and the absolute error."""
    lo, hi = data["box"]
    fig = plt.figure(figsize=(19, 4.6))
    ax = fig.add_subplot(1, 4, 1)
    sdf = car["sdf"][car["sdf"].shape[0] // 2, :, :, 0]  # (height, length)
    extent = (lo[2], hi[2], lo[1], hi[1])
    im = ax.imshow(sdf, origin="lower", cmap="coolwarm", extent=extent)
    ax.contour(sdf, levels=[data["sdf_zero"]], colors="k", extent=extent)
    ax.set_title("Input: SDF, middle plane\n(black line: SDF = 0)")
    ax.set_xlabel("length (mesh units)")
    ax.set_ylabel("height (mesh units)")
    fig.colorbar(im, ax=ax, shrink=0.5)
    vertices = car["vertices"] * (hi - lo) + lo
    lim = Normalize(vmin=float(true.min()), vmax=float(true.max()))
    err = np.abs(pred - true)
    panels = [
        ("Truth: pressure", true, "jet", lim),
        ("Prediction: pressure", pred, "jet", lim),
        ("Absolute error", err, "magma", Normalize(vmin=0.0, vmax=float(err.max()))),
    ]
    for k, (name, values, cmap, norm) in enumerate(panels, start=2):
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
# ``--smoke`` takes a few cars, uses the tiny GIWO on an 8^3 grid and trains
# for a few epochs. Without it the script trains the benchmark's GIWO on the
# full 500 / 111 split for 100 epochs. That mode is meant for a GPU. ``--lr``
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
    ap.add_argument("--smoke", action="store_true", help="few cars, tiny GIWO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="peak learning rate")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/carcfd_giwo_tutorial")
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
    data = load_cars(
        Path(args.data_dir).expanduser(), n_train, n_test, size["latent_res"]
    )
    edges = attach_neighbors(
        data["train"] + data["test"], data["latent_queries"], size["out_gno_radius"]
    )
    print(
        f"data loading and neighbour search: {perf_counter() - t0:.1f}s | "
        f"edges per car: {edges:,}"
    )
    model, params = build_model(
        size, data["train"][0], data["latent_queries"], args.seed
    )

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
        f"Car-CFD, GIWO: first test car, id {car['id']} "
        f"(relative L2 {metrics['errors'][0]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
