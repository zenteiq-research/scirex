<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Tutorial: Car-CFD (Car Surface Pressure) with GINO

In this tutorial you train a [Geometry-Informed Neural Operator](../../theory_guide/operator_learning.md#gino)
(GINO) on **Car-CFD**, the car benchmark of the GINO paper ([[1]](#ref1),
Table 2): given the shape of a car, predict the steady pressure at each of
the 3,586 vertices of its surface mesh. Every car is a different ShapeNet
model in the same wind tunnel. GINO puts the car's signed distance function
(SDF) on a regular latent grid, processes it with an FNO, and carries the
result out to the surface vertices with a graph kernel. The whole tutorial is
one script, [`scripts/tutorials/carcfd_gino.py`](../../../scripts/tutorials/carcfd_gino.py). It uses only the public
ScireX API ([`scirex.models.GINO`](../../api/models.md#scirex.models.GINO), [`scirex.layers.kdtree_neighbor_search`](../../api/layers.md#scirex.layers.kdtree_neighbor_search),
[`scirex.training.adamw_cx`](../../api/training.md#scirex.training.adamw_cx), [`scirex.utils.count_model_params`](../../api/utils.md#scirex.utils.count_model_params)), NumPy and
open3d for the data, and plain JAX and optax for the training loop. The code
on this page is copied from that script.

## Summary

| | |
|---|---|
| Problem | Steady external flow around a car ([[2]](#ref2)): car geometry in, surface pressure at 3,586 mesh vertices out |
| Model | GINO, [`scirex.models.GINO`](../../api/models.md#scirex.models.GINO), decoder-only ("GINO (decoder)" in the paper). Benchmark size: 64³ latent grid, 4 FNO layers, width 64, 16³ modes, output graph kernel of radius 0.033, 75,917,953 parameters. Smoke size: 16³ grid, 2 FNO layers, width 16, 8³ modes, radius 0.13, 358,337 parameters |
| Data | neuraloperator's processed Car-CFD (Zenodo record 13936501, CC BY 4.0, 198 MB), 611 cars. Benchmark: the paper's 500 train / 111 test. Smoke: the first 8 train / first 4 test cars |
| Optimiser | Adam ([`adamw_cx`](../../api/training.md#scirex.training.adamw_cx), no weight decay, β₂ 0.999, ε 1e-6), learning rate 2.5e-4 halved at epoch 50, 100 epochs, one car per step |
| Metric | Relative L2 error of the de-normalised surface pressure per car, averaged over the test cars (the paper's metric) |
| Full-run values | 8.37% (final epoch; best epoch 8.31%). Paper ([[1]](#ref1), GINO decoder): 7.12% |

## Setup

Install the library from a clone of the repository (see [Install](../../install/index.md)):

```bash
pip install -e ".[all]"
```

### Data

The dataset is not bundled. Download neuraloperator's processed version of
the [[2]](#ref2) simulations from Zenodo (record 13936501, CC BY 4.0,
198 MB) and unpack it:

```bash
mkdir -p ~/data/car-pressure-data && cd ~/data/car-pressure-data
wget -O processed-car-pressure-data.tar.gz \
    https://zenodo.org/api/records/13936501/files/processed-car-pressure-data.tar.gz/content
tar xzf processed-car-pressure-data.tar.gz
```

This gives `processed-car-pressure-data/` with `train.txt` and `test.txt` (the
paper's 500 / 111 split, comma-separated car ids) and
`data/data/<id>/tri_mesh.ply` and `press.npy` for each of the 611 cars. Pass
that folder as `--data-dir`, or set `CARCFD_DATA_DIR`. The benchmark's own
loader takes the folder *above* it as `--data-root`. ScireX does not
redistribute the data; see the [dataset guide](../../dataset_guide/car_cfd.md).

For the smoke run you do not need to copy anything: the script reads only the
first `--n-train` training and first `--n-test` test cars.

## The problem

Each car is one simulation by [[2]](#ref2): the flow around the car
at an inlet speed of 20 m/s (72 km/h), Reynolds number about 5×10⁶, solved
with the Reynolds-averaged equations and the k-ε turbulence model, and the
last 4 of 10 simulated seconds averaged. In its laminar form the flow obeys

$$
(\mathbf u \cdot \nabla)\,\mathbf u = -\nabla p + \nu \nabla^2 \mathbf u, \qquad \nabla \cdot \mathbf u = 0 .
$$

Only the car surface and the pressure on it are in the data. The model never
sees the equation: it learns the map from the car's shape to its surface
pressure, $\mathcal G: \Gamma \mapsto p|_\Gamma$, from 500 solved cars. The
[problem card](../../benchmarks/car_cfd/README.md) has every parameter and the
open questions, for example the unit of the stored pressure and the
length unit of the meshes.

Here the car enters as its SDF $s$ on a regular $R^3$ grid spanning one
bounding box of all cars, linearly mapped to $[10^{-6}, 1]$ with the minimum
and maximum of the training cars. The pressure is standardised with one
scalar mean and standard deviation, $\tilde p = (p - \mu)/(\sigma + 10^{-7})$.
The model is the map

$$
\mathcal G_\theta : s \in \mathbb R^{R\times R\times R} \;\mapsto\; \hat{\tilde p} \in \mathbb R^{3586} ,
$$

evaluated at the car's own vertices. The training loss of one car is the
relative L2 error of the normalised pressure, and the test metric is the same
error on the de-normalised pressure, averaged over the $N_\mathrm{test}$ test
cars:

$$
\mathcal L = \frac{\lVert \hat{\tilde p} - \tilde p\rVert_2}{\lVert \tilde p\rVert_2 + \varepsilon}, \qquad
E_\mathrm{test} = \frac{1}{N_\mathrm{test}}\sum_{k=1}^{N_\mathrm{test}} \frac{\lVert \hat p_k - p_k\rVert_2}{\lVert p_k\rVert_2 + \varepsilon}, \qquad \varepsilon = 10^{-8} .
$$

**GINO in one paragraph.** A car is an irregular surface, but an FNO needs a
regular grid. GINO does both: it represents the geometry on a fixed latent
grid (here by the SDF, which is negative inside the car and positive
outside), lets an FNO mix information over the whole grid in Fourier space,
and then a graph kernel, a learned integral over the grid points within a
radius of each vertex, carries the latent field to the irregular surface. The
paper's best variant, used here, has no input graph kernel at all
("decoder-only"): the SDF on the grid is the whole input. The
[theory guide](../../theory_guide/operator_learning.md#gino) has the details.

## Step-by-step walkthrough

### 1. Setup

Imports, the precision setting, the seed, the training settings and the two
model sizes. `PROTOCOL` holds the GINO paper's training settings (the same
values as in [`scripts/benchmarks/car_cfd/_config.py`](../../../scripts/benchmarks/car_cfd/_config.py)). `SIZES["full"]` is the
benchmark's GINO; `SIZES["smoke"]` is small enough for a CPU.

<!-- tutorial-snippet -->
```python
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
from scirex.models import GINO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
```

The committed Car-CFD runs were computed with JAX's 64-bit mode on. An earlier version of the
library switched it on as a side effect of importing [`GINO`](../../api/models.md#scirex.models.GINO); 1.0 does not.
The script sets it explicitly, so that it computes as those runs did. Data and weights stay float32.

<!-- tutorial-snippet -->
```python
# The committed Car-CFD runs were computed with jax_enable_x64 on (an earlier
# version of the library turned it on as a side effect of importing GINO; 1.0
# does not). Setting it explicitly keeps the tutorial in the numerics of those
# runs. The data and the weights are still float32.
jax.config.update("jax_enable_x64", True)
```

<!-- tutorial-snippet -->
```python
SEED = 0

# Training settings of the GINO paper (Table 2, Sec. 4.2), the same values as
# PROTOCOL and RUNS["paper"] in scripts/benchmarks/car_cfd/_config.py: Adam
# (adamw_cx with no weight decay), learning rate 2.5e-4 halved at epoch 50,
# 100 epochs, one car per step.
PROTOCOL = dict(
    n_train=500,
    n_test=111,
    n_epochs=100,
    learning_rate=2.5e-4,
    lr_step=50,
    lr_gamma=0.5,
    weight_decay=0.0,
    adam_b2=0.999,
    adam_eps=1e-6,
)

# Model sizes. "full" is the benchmark's "paper" run: decoder-only GINO on a
# 64^3 latent grid, 4 FNO layers of width 64 with 16^3 modes, output graph
# kernel of radius 0.033 (75,917,953 parameters). "smoke" is a tiny GINO on a
# 16^3 grid that trains on a CPU; its larger radius keeps about as many
# latent points per vertex as 0.033 does on the 64^3 grid.
SIZES = dict(
    full=dict(
        latent_res=64,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_n_layers=4,
        out_gno_radius=0.033,
        out_gno_channel_mlp_hidden_layers=(512, 256),
    ),
    smoke=dict(
        latent_res=16,
        fno_n_modes=(8, 8, 8),
        fno_hidden_channels=16,
        fno_n_layers=2,
        out_gno_radius=0.13,
        out_gno_channel_mlp_hidden_layers=(64, 32),
    ),
)

# The stored pressure has 3,682 entries per car and the mesh 3,586 vertices.
# The benchmark keeps entries 0-15 and 112-3681 so that the two match; the
# source of this slice does not say why (see the Car-CFD problem card).
KEEP = np.r_[0:16, 112:3682]
```

`print_environment` records the library versions, the devices JAX sees and the seed;
see [how the tutorials work](../index.md#how-the-tutorials-work).

<!-- tutorial-snippet -->
```python
def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__} | open3d {o3d.__version__}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")
```

### 2. Load the data

`read_ids` and `car_folder` find the cars. `load_cars` does what the
benchmark's loader does: one bounding box over all loaded cars scales the
vertices into [0, 1]; the SDF of each car is computed by open3d's ray-casting
scene on a regular grid spanning that box and mapped to [1e-6, 1] with the
training cars' minimum and maximum; the pressure is standardised with one
scalar mean and standard deviation of all loaded cars, train and test
together, as in neuraloperator's reference loader. The stored pressure has
3,682 entries per car and the mesh 3,586 vertices; the benchmark keeps
entries 0 to 15 and 112 to 3,681 (`KEEP`).

<!-- tutorial-snippet -->
```python
def read_ids(data_dir: Path, split: str, n: int) -> list[str]:
    """The first ``n`` car ids of ``train.txt`` or ``test.txt``."""
    ids = (Path(data_dir) / f"{split}.txt").read_text().split(",")
    return [i.strip() for i in ids[:n]]
```

<!-- tutorial-snippet -->
```python
def car_folder(data_dir: Path, car_id: str) -> Path:
    """``data/<id>``, or ``data/data/<id>`` as in the Zenodo archive."""
    folder = Path(data_dir) / "data" / car_id
    return folder if folder.exists() else Path(data_dir) / "data" / "data" / car_id
```

<!-- tutorial-snippet -->
```python
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
```

### 3. Neighbour lists

The output graph kernel sums, for every vertex, over the grid points within
`out_gno_radius`. Each car sits at its own place in the grid, so each needs
its own lists, and a KD-tree search cannot run inside a jitted step: they are
built once, before training. Padding every list to the longest one, with
dummy edges that point at grid point 0 and belong to an extra segment that
the sum drops, gives every car the same shapes, so the step compiles once.

<!-- tutorial-snippet -->
```python
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
```

### 4. Build the model

`use_in_gno=False` removes the input graph kernel: the SDF (one channel,
`latent_feature_channels=1`) is the whole input of the FNO, and there is no
input function (`in_channels=0`). The other settings are those the
benchmark's config resolves to: the reference FNO (no factorisation, full
rank, no normalisation) with a channel-MLP expansion of 1. The FNO's spectral
weights are complex; [`count_model_params`](../../api/utils.md#scirex.utils.count_model_params) counts each as two real numbers.

<!-- tutorial-snippet -->
```python
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
```

<!-- tutorial-snippet -->
```python
def build_model(size: dict, car: dict, latent_queries: np.ndarray, seed: int):
    """Build the decoder-only GINO, initialise it on one car, print its size."""
    model = GINO(
        in_channels=0,
        out_channels=1,
        latent_feature_channels=1,
        use_in_gno=False,
        fno_n_modes=size["fno_n_modes"],
        fno_hidden_channels=size["fno_hidden_channels"],
        fno_n_layers=size["fno_n_layers"],
        fno_channel_mlp_expansion=1.0,
        fno_resolution_scaling_factor=1,
        out_gno_radius=size["out_gno_radius"],
        out_gno_channel_mlp_hidden_layers=size["out_gno_channel_mlp_hidden_layers"],
    )
    params = model.init(jax.random.PRNGKey(seed), **gino_inputs(car, latent_queries))
    print(f"GINO {size} | parameters: {count_model_params(params):,}")
    return model, params
```

### 5. Loss

The loss from [The problem](#the-problem): the relative L2 error of the
normalised pressure of one car, [`LpLoss(d=2, p=2)`](../../api/losses.md#scirex.losses.LpLoss) in the benchmark.

<!-- tutorial-snippet -->
```python
def rel_l2(pred, true, eps: float = 1e-8):
    """``||pred - true|| / (||true|| + eps)`` over all vertices of one car."""
    return jnp.linalg.norm((pred - true).ravel()) / (
        jnp.linalg.norm(true.ravel()) + eps
    )
```

<!-- tutorial-snippet -->
```python
def make_loss(model):
    """Return ``loss_fn(params, car_inputs, press) -> relative L2``."""

    def loss_fn(params, inputs, press):
        out = model.apply(params, **inputs)[0, :, 0]
        return rel_l2(out, press)

    return loss_fn
```

### 6. Optimiser

The paper's optimiser is Adam, built as [`adamw_cx`](../../api/training.md#scirex.training.adamw_cx) with no weight decay. The
FNO's spectral weights are complex, and `jax.grad` returns the conjugate of
the steepest-ascent direction for a complex weight; [`adamw_cx`](../../api/training.md#scirex.training.adamw_cx) conjugates it
back before the Adam update. `optax.inject_hyperparams` keeps the learning
rate in the optimiser state, so that StepLR can halve it at epoch 50 without
recompiling. By default it gives the injected values the "highest" dtype of
the parameters, which is complex64 here: `hyperparam_dtype=jnp.float32` keeps
the learning rate real. The benchmark's `CarCFDModelWrapper` omits it; its
learning rate is then complex with a zero imaginary part, which gives the same
updates but makes JAX warn ("Casting from complex to real") on every step.

<!-- tutorial-snippet -->
```python
def make_optimizer(protocol: dict):
    """Adam (``adamw_cx``) with the learning rate as an injected hyperparameter."""

    def make_tx(learning_rate, weight_decay):
        return adamw_cx(
            learning_rate,
            weight_decay=weight_decay,
            b2=protocol["adam_b2"],
            eps=protocol["adam_eps"],
        )

    return optax.inject_hyperparams(make_tx, hyperparam_dtype=jnp.float32)(
        learning_rate=protocol["learning_rate"],
        weight_decay=protocol["weight_decay"],
    )
```

<!-- tutorial-snippet -->
```python
def step_lr(protocol: dict, epoch: int) -> float:
    """StepLR: ``learning_rate * lr_gamma ** (epoch // lr_step)``."""
    return protocol["learning_rate"] * protocol["lr_gamma"] ** (
        epoch // protocol["lr_step"]
    )
```

### 7. Train

One car per step, in a new random order every epoch, with the StepLR learning
rate written into the optimiser state before each epoch. After every epoch the
test cars are scored, to draw the error curve; no epoch is selected by it, and
the reported weights are those after the last epoch. JAX compiles the step on
its first call, so the first epoch is slower than the others.

<!-- tutorial-snippet -->
```python
def train(model, params, data, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    q = data["latent_queries"]
    tx = make_optimizer(protocol)
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
        opt_state.hyperparams["learning_rate"] = jnp.asarray(
            step_lr(protocol, epoch), dtype=jnp.float32
        )
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
```

### 8. Evaluate

`make_predict` jits the forward pass once. `evaluate` de-normalises
prediction and truth and takes the relative L2 error per car: the mean is the
paper's metric, and the median, the 90th percentile and the worst car show how
it is spread.

<!-- tutorial-snippet -->
```python
def make_predict(model):
    """The jitted forward pass, ``predict(params, inputs)`` -> normalised pressure."""
    return jax.jit(lambda params, inputs: model.apply(params, **inputs)[0, :, 0])
```

<!-- tutorial-snippet -->
```python
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
```

### 9. Plot

The standard figure for this benchmark has four panels for one test car: an
input, the true surface pressure, the predicted surface pressure (same colour
scale as the truth) and the absolute error. The input panel shows the SDF on
the latent grid's middle plane across the car's width, which is what GINO
reads, with the car's outline where the SDF is zero. The cars are drawn in
mesh coordinates from the benchmark's three-quarter view, every triangle
coloured by the mean of its three vertex values. A second figure shows the
error per epoch.

<!-- tutorial-snippet -->
```python
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
```

<!-- tutorial-snippet -->
```python
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
```

<!-- tutorial-snippet -->
```python
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
```

The complete script is [`scripts/tutorials/carcfd_gino.py`](../../../scripts/tutorials/carcfd_gino.py).

## Run it

Smoke run (CPU, about 15 s, no GPU needed):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/tutorials/carcfd_gino.py \
    --data-dir ~/data/car-pressure-data/processed-car-pressure-data --smoke
```

Full protocol (500 / 111 cars, benchmark-size GINO, 100 epochs; use a GPU):

```bash
python scripts/tutorials/carcfd_gino.py \
    --data-dir ~/data/car-pressure-data/processed-car-pressure-data
```

The other flags are `--epochs`, `--lr` (the starting learning rate), `--seed`,
`--out-dir` (default `outputs/carcfd_gino_tutorial`) and, for the smoke run,
`--n-train` and `--n-test`.

This is the output of the smoke run on a CPU (smoke run, CPU, 8 train / 4
test cars, not a benchmark number). The wall time was 14.9 s,
measured with `time`, on a machine shared with other jobs (load average
4.25 at the start); library warnings are left out:

```text
python 3.12.3 | numpy 1.26.4
scirex 1.0.0rc1 | open3d 0.19.0
jax 0.6.2 | flax 0.10.7 | optax 0.2.8
devices: [CpuDevice(id=0)]
SEED = 0
cars: 8 train / 4 test | vertices per car: 3586 | latent grid 16^3 | pressure mean -37.211, std 46.393
data loading and neighbour search: 0.2s | edges per car: 105,380
GINO {'latent_res': 16, 'fno_n_modes': (8, 8, 8), 'fno_hidden_channels': 16, 'fno_n_layers': 2, 'out_gno_radius': 0.13, 'out_gno_channel_mlp_hidden_layers': (64, 32)} | parameters: 358,337
epoch 0: train_rel_l2=0.9903 test_rel_l2=0.7594 epoch_time=1.8s
epoch 1: train_rel_l2=0.9601 test_rel_l2=0.7366 epoch_time=1.1s
epoch 2: train_rel_l2=0.9252 test_rel_l2=0.7062 epoch_time=1.1s
epoch 3: train_rel_l2=0.8687 test_rel_l2=0.6717 epoch_time=1.1s
epoch 4: train_rel_l2=0.8272 test_rel_l2=0.6370 epoch_time=1.1s
test relative L2: mean 0.6370 | median 0.6324 | p90 0.7150 | worst 0.7453
figures in outputs/carcfd_gino_tutorial | total time 13.3s
```

## Results

### Full run

The next figures were not made by this script. They come from the benchmark
runner (`train.py` and `infer.py`) and are the committed results of the **full
100-epoch run** `paper` (500 training and 111 test cars, benchmark-size GINO,
one NVIDIA RTX PRO 6000 Blackwell). `infer.py` draws one test car (the figure title names it):
truth, prediction and absolute error. The prediction reproduces the high
stagnation pressure on the front face and the suction around its edges; the
error is concentrated on the front face.

![Full 100-epoch run: surface pressure of one test car, truth, prediction and absolute error](../../assets/benchmarks/car_cfd_gino/contour.png)

Training and test relative L2 per epoch. The test error levels off at about
8.4% in the second half of training while the training error keeps falling,
with a visible step when the learning rate is halved at epoch 50:

![Full 100-epoch run: training and test relative L2 per epoch](../../assets/benchmarks/car_cfd_gino/error_curve.png)

Seconds per epoch of the same run (the first epoch includes compilation; in
`metrics.json`, the first epoch took 29.2 s and the mean is 17.3 s):

![Full 100-epoch run: compute cost per epoch](../../assets/benchmarks/car_cfd_gino/compute_cost.png)

| Test relative L2, 111 test cars | Train | Test | Best test (epoch) |
|---|---|---|---|
| GINO (decoder), paper [[1]](#ref1) | 6.37% | 7.12% | |
| GINO (encoder-decoder), paper | 7.95% | 9.47% | |
| **`paper` run of this repository (this tutorial's full mode)** | 3.55% | **8.37%** | 8.31% (91) |

| Model, same protocol and split, about 75 M parameters each | Test relative L2 |
|---|---|
| **GINO** | **8.37%** |
| [Transolver](../Others/CarCFD_Transolver.md) | 6.81% |
| [GIWO](../Others/CarCFD_GIWO.md) | 12.01% |

The train column is the relative L2 of the normalised pressure that the model
trains on, a different metric from the test column.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'open3d'`.** open3d is an
  optional dependency (the `open3d` and `benchmark` extras): install it with
  `pip install -e ".[open3d]"`, or `pip install "scirex[open3d]"` for a pip install.
- **`FileNotFoundError: ... train.txt`.** `--data-dir` must be the
  `processed-car-pressure-data` folder itself, the one that holds `train.txt`,
  `test.txt` and `data/`, not the folder above it.
- **`tar: Ignoring unknown extended header keyword ...`.** The Zenodo archive
  was made on macOS; tar skips the extra metadata. The files are fine.

## Next steps

- Compare with the other architectures on the same benchmark:
  [Transolver](../Others/CarCFD_Transolver.md), which reads the surface points directly,
  and [GIWO](../Others/CarCFD_GIWO.md), which replaces GINO's FNO by a wavelet operator;
  the [comparison page](../../benchmarks/car_cfd/results/comparative.md) puts the three
  side by side.
- Read the [Car-CFD results](../../benchmarks/car_cfd/results/comparative.md#gino) for the
  settings the paper states and the demo config does not, and the
  [problem card](../../benchmarks/car_cfd/README.md) for every parameter.
- Learn how GINO works in the [theory guide](../../theory_guide/operator_learning.md#gino) and see
  its arguments in the [API reference](../../api/models.md); the optimiser is in
  the [training API](../../api/training.md).
- Try GINO on another car dataset with a different protocol:
  [ShapeNetCar](../../benchmarks/shapenetcar/results/comparative.md#gino).

## References

1. <span id="ref1"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>

2. <span id="ref2"></span>Umetani and Bickel, *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*, ACM Transactions on Graphics 37(4), 2018. <https://doi.org/10.1145/3197517.3201325>
