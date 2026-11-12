<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Tutorial: Navier-Stokes 2D (Turbulence) with GINO

In this tutorial you train a [Geometry-Informed Neural Operator](../../theory_guide/operator_learning.md#gino)
(GINO [[3]](#ref3)) on the **Navier-Stokes 2D** benchmark: given ten frames of the
vorticity of a 2D turbulent flow, predict the next ten, one frame at a time,
each prediction fed back as the newest input. GINO is built for irregular
geometries; here every flow lives on the same regular grid, which keeps the
graph part simple and lets you compare it with the grid-only models. The
whole tutorial is one script, [`scripts/tutorials/navierstokes_gino.py`](../../../scripts/tutorials/navierstokes_gino.py). It
uses only the public ScireX API ([`scirex.models.GINO`](../../api/models.md#scirex.models.GINO),
[`scirex.layers.build_neighbor_graph`](../../api/layers.md#scirex.layers.build_neighbor_graph), [`scirex.layers.edge_index_to_neighbors`](../../api/layers.md#scirex.layers.edge_index_to_neighbors),
[`scirex.training.adamw_cx`](../../api/training.md#scirex.training.adamw_cx), [`scirex.utils.count_model_params`](../../api/utils.md#scirex.utils.count_model_params)), SciPy to read
the data and plain JAX and optax. The code on this page is copied from that
script.

## Summary

| | |
|---|---|
| Problem | 2D incompressible Navier-Stokes in vorticity form on a periodic 64 × 64 grid, viscosity 1e-5: 10 frames in, the next 10 out, one at a time |
| Model | GINO, [`scirex.models.GINO`](../../api/models.md#scirex.models.GINO). Benchmark size: `in_gno_radius=out_gno_radius=0.033, fno_n_modes=(16, 16), fno_hidden_channels=97, fno_n_layers=4`, 11,297,041 parameters. Smoke size: same radius, `fno_n_modes=(12, 12), fno_hidden_channels=16, fno_n_layers=2`, 390,539 parameters |
| Data | `NavierStokes_V1e-5_N1200_T20.mat` (FNO [[1]](#ref1) data release, about 413 MB), 1200 flows of 20 frames. Benchmark: first 1000 train / last 200 test. Smoke: 20 train / 4 test |
| Optimiser | Complex-aware AdamW ([`adamw_cx`](../../api/training.md#scirex.training.adamw_cx)), eps 1e-8, weight decay 1e-5, no gradient clipping, cosine one-cycle learning rate with peak 1e-3, batch size 2 |
| Metric | Relative L2 error of each test flow over its 10 rolled-out frames, averaged over the test flows |
| Full-run values | 0.0666 (one seed). Transolver [[2]](#ref2) paper, its own model: 0.0900. Copying the last input frame: 0.7481 |

## Setup

The setup is the same as in the [FNO tutorial](../FNO/NavierStokes_FNO.md#setup).

## The problem

The problem, the data and the metric are the same as in the [FNO tutorial](../FNO/NavierStokes_FNO.md#the-problem).

**GINO in one paragraph.** GINO has three parts. A graph neural operator
(GNO) integrates the input features over a ball of radius `in_gno_radius`
around each point of a regular latent grid; an FNO processes the latent grid;
a second GNO integrates the result back onto the output points. The graph
layers let it read and write any point cloud, and the FNO in the middle gives
it the global reach of a spectral method. Here input points, latent grid and
output points are all the 64 × 64 grid, so the graph layers act as learned
local filters before and after the FNO. The [theory guide](../../theory_guide/operator_learning.md#gino)
has the details.

## Step-by-step walkthrough

The explanations of the steps that are the same for every model are in the [FNO tutorial](../FNO/NavierStokes_FNO.md#step-by-step-walkthrough). This page keeps the code and the text that is specific to the model.

### 1. Setup

Imports, the seed, the training settings and the two model sizes. `PROTOCOL`
holds the benchmark's training settings (the same values as in
[`scripts/benchmarks/navier_stokes/_config.py`](../../../scripts/benchmarks/navier_stokes/_config.py), the Transolver paper's recipe,
which every model on this benchmark shares): 500 epochs, batch size 2, AdamW
with a peak learning rate of 1e-3 on a one-cycle schedule, weight decay 1e-5
and no gradient clipping. `SIZES["full"]` is the benchmark's GINO;
`SIZES["smoke"]` is small enough for a CPU.

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
import optax
import scipy.io

import scirex
from scirex.layers import build_neighbor_graph, edge_index_to_neighbors
from scirex.models import GINO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)
```

<!-- tutorial-snippet -->
```python
SEED = 0
DATA_FILE = "NavierStokes_V1e-5_N1200_T20.mat"

# Training settings of the Navier-Stokes benchmark, the same values as
# PROTOCOL in scripts/benchmarks/navier_stokes/_config.py (the Transolver
# paper's recipe, used by every model).
PROTOCOL = dict(
    ntrain=1000,
    ntest=200,
    t_in=10,
    t_out=10,
    epochs=500,
    batch_size=2,
    learning_rate=1e-3,
    weight_decay=1e-5,
    adam_eps=1e-8,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
)

# Model sizes. "full" is the benchmark's GINO (11,297,041 parameters, matched
# to Transolver's 11,232,321). "smoke" is a tiny GINO that trains on a CPU.
SIZES = dict(
    full=dict(
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        fno_n_modes=(16, 16),
        fno_hidden_channels=97,
        fno_n_layers=4,
    ),
    smoke=dict(
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        fno_n_modes=(12, 12),
        fno_hidden_channels=16,
        fno_n_layers=2,
    ),
)

# The neighbour lists of GINO's radius graph, built once per (grid, radius).
GRAPHS: dict = {}
```

<!-- tutorial-snippet -->
```python
def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")
```

### 2. Load the data

<!-- tutorial-snippet -->
```python
def load_ns(data_path: Path, ntrain: int, ntest: int, t_in: int, t_out: int) -> dict:
    """Read the split. History ``(n, 64, 64, t_in)``, targets ``(n, 64, 64, t_out)``."""
    u = scipy.io.loadmat(str(data_path), variable_names=["u"])["u"]
    n, h, w, t = u.shape
    if ntrain + ntest > n or t_in + t_out > t:
        raise ValueError(
            f"need {ntrain} + {ntest} <= {n} samples, {t_in} + {t_out} <= {t} frames"
        )
    out = {}
    for name, part in (("train", u[:ntrain]), ("test", u[n - ntest :])):
        part = part.astype(np.float32)
        out[f"a_{name}"] = part[..., :t_in]
        out[f"u_{name}"] = part[..., t_in : t_in + t_out]
    print(
        f"samples: {ntrain} train / {ntest} test | grid {h}x{w} | "
        f"{t_in} frames in, {t_out} out | "
        f"vorticity range [{out['a_train'].min():.2f}, {out['a_train'].max():.2f}]"
    )
    return out
```

### 3. Build the model

Every flow lives on the same 64 × 64 grid, so the input points, the latent
grid of the FNO block and the output points are all the grid points in
$[0, 1]^2$, in the same order as a `reshape` of the frame. One radius graph
from the grid to itself serves both graph layers: with radius 0.033 each
point sees its neighbours up to two cells away (the grid spacing is 1/63).
[`build_neighbor_graph`](../../api/layers.md#scirex.layers.build_neighbor_graph) builds the graph once per grid and radius, and
[`edge_index_to_neighbors`](../../api/layers.md#scirex.layers.edge_index_to_neighbors) turns it into the neighbour lists that [`GINO`](../../api/models.md#scirex.models.GINO)
expects; `GRAPHS` (in step 1) caches them. The ten history frames are the
features of every point, so `in_channels` and `fno_in_channels` are both 10.
`make_step` wraps the layout, so the rest of the script only sees
channels-last histories `(batch, 64, 64, 10)`.

<!-- tutorial-snippet -->
```python
def grid_points(grid: tuple[int, int]) -> np.ndarray:
    """``(H * W, 2)`` coordinates in [0, 1]^2, in the order of ``reshape``."""
    xs = np.linspace(0.0, 1.0, grid[0], dtype=np.float32)
    ys = np.linspace(0.0, 1.0, grid[1], dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)
```

<!-- tutorial-snippet -->
```python
def grid_graph(grid: tuple[int, int], radius: float) -> dict:
    """Neighbour lists of the radius graph from the grid to itself."""
    key = (tuple(grid), float(radius))
    if key not in GRAPHS:
        pts = grid_points(grid)
        edges = build_neighbor_graph(pts, pts, radius, "radius")
        GRAPHS[key] = edge_index_to_neighbors(
            jnp.asarray(edges), num_query=pts.shape[0]
        )
    return GRAPHS[key]
```

<!-- tutorial-snippet -->
```python
def gino_inputs(model, fx) -> dict:
    """``GINO``'s keyword arguments for a channels-last history ``fx``."""
    b, h, w, t = fx.shape
    pts = jnp.asarray(grid_points((h, w)))[None]
    graph = grid_graph((h, w), model.in_gno_radius)
    return dict(
        input_geom=pts,
        latent_queries=pts.reshape(1, h, w, 2),
        output_queries=pts,
        x=fx.reshape(b, h * w, t),
        neighbors_in=graph,
        neighbors_out=graph,
    )
```

<!-- tutorial-snippet -->
```python
def build_model(size: dict, t_in: int, sample: np.ndarray, seed: int):
    """Build the GINO and its graph, initialise it and print the parameter count."""
    model = GINO(
        in_channels=t_in, out_channels=1, gno_coord_dim=2, fno_in_channels=t_in, **size
    )
    inputs = gino_inputs(model, jnp.asarray(sample)[None])
    params = model.init(jax.random.PRNGKey(seed), **inputs)
    print(f"GINO {size} | parameters: {count_model_params(params):,}")
    return model, params
```

<!-- tutorial-snippet -->
```python
def make_step(model):
    """``step(params, history) -> next frame``, both channels-last."""

    def step(params, fx):
        out = model.apply(params, **gino_inputs(model, fx))
        return out.reshape(*fx.shape[:3], 1)

    return step
```

### 4. Loss

<!-- tutorial-snippet -->
```python
def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all remaining axes."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.linalg.norm(true.reshape(b, -1), axis=1)
```

### 5. Teacher forcing and rollout

<!-- tutorial-snippet -->
```python
def teacher_forced(step, params, fx, yy):
    """Loss (summed over steps and batch) with the true history at every step."""
    loss = 0.0
    for t in range(yy.shape[-1]):
        pred = step(params, fx)
        loss = loss + jnp.sum(rel_l2_per_sample(pred, yy[..., t : t + 1]))
        fx = jnp.concatenate([fx[..., 1:], yy[..., t : t + 1]], axis=-1)
    return loss
```

<!-- tutorial-snippet -->
```python
def rollout(step, params, fx, t_out: int):
    """``t_out`` frames, each step fed the model's own previous predictions."""
    preds = []
    for _ in range(t_out):
        pred = step(params, fx)
        preds.append(pred)
        fx = jnp.concatenate([fx[..., 1:], pred], axis=-1)
    return jnp.concatenate(preds, axis=-1)
```

### 6. Optimiser for complex weights

The FNO block of GINO has complex spectral weights. For a real loss $L$ and a
complex weight $w$, `jax.grad` returns $\partial L/\partial w$, the complex
conjugate of the steepest-ascent direction that every optax transform assumes
it is given, so plain `optax.adamw` moves their imaginary parts the wrong
way. [`adamw_cx`](../../api/training.md#scirex.training.adamw_cx) conjugates complex gradients first and keeps Adam's second
moment real; real weights, which are most of GINO's, are updated exactly as
by AdamW.

`optax.inject_hyperparams` lets the learning rate follow the one-cycle
schedule. By default it casts the hyperparameters to the highest dtype among
the parameters, which is complex64 here. Every update would then be complex,
the real weights' included, and `apply_updates` would cast it back with a
deprecation warning that JAX will turn into an error.
`hyperparam_dtype=jnp.float32` keeps them real. `optimizer="adamw"` swaps in
plain optax AdamW for comparison.

<!-- tutorial-snippet -->
```python
def make_optimizer(protocol: dict, total_steps: int, optimizer: str = "adamw_cx"):
    """AdamW on a cosine one-cycle schedule (``"adamw_cx"`` or plain ``"adamw"``)."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    builder = dict(adamw_cx=adamw_cx, adamw=optax.adamw)[optimizer]
    tx = optax.inject_hyperparams(builder, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule,
        weight_decay=protocol["weight_decay"],
        eps=protocol["adam_eps"],
    )
    return tx, schedule
```

### 7. Train

<!-- tutorial-snippet -->
```python
def train(model, params, data, epochs, seed, protocol=PROTOCOL, optimizer="adamw_cx"):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    a_train, u_train = data["a_train"], data["u_train"]
    a_test, u_test = data["a_test"], data["u_test"]
    bs, t_out = protocol["batch_size"], protocol["t_out"]
    ntrain = len(a_train)
    tx, _ = make_optimizer(protocol, -(-ntrain // bs) * epochs, optimizer)
    opt_state = tx.init(params)
    step = make_step(model)

    @jax.jit
    def train_step(params, opt_state, fx, yy):
        loss, grads = jax.value_and_grad(lambda p: teacher_forced(step, p, fx, yy))(
            params
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def test_error(params, fx, yy):
        return jnp.sum(rel_l2_per_sample(rollout(step, params, fx, t_out), yy))

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        order = rng.permutation(ntrain)
        train_total = 0.0
        for i in range(0, ntrain, bs):
            idx = order[i : i + bs]
            params, opt_state, loss = train_step(
                params, opt_state, jnp.asarray(a_train[idx]), jnp.asarray(u_train[idx])
            )
            train_total += float(loss)
        secs = perf_counter() - t0
        test_total = sum(
            float(test_error(params, a_test[i : i + bs], u_test[i : i + bs]))
            for i in range(0, len(a_test), bs)
        )
        history.append(
            dict(
                epoch=epoch,
                train_step_rel_l2=train_total / ntrain / t_out,
                test_rel_l2=test_total / len(a_test),
                secs=secs,
            )
        )
        print(
            f"epoch {epoch}: train_step_rel_l2={history[-1]['train_step_rel_l2']:.4f} "
            f"test_rel_l2={history[-1]['test_rel_l2']:.4f} epoch_time={secs:.1f}s"
        )
    return params, history
```

### 8. Evaluate

<!-- tutorial-snippet -->
```python
def evaluate(model, params, data, t_out: int, batch_size: int = 2) -> dict:
    """Rollout error per test flow and per time step, and the predictions."""
    step = make_step(model)
    predict = jax.jit(lambda p, fx: rollout(step, p, fx, t_out))
    a_test, u_test = data["a_test"], data["u_test"]
    preds = np.concatenate(
        [
            np.asarray(predict(params, jnp.asarray(a_test[i : i + batch_size])))
            for i in range(0, len(a_test), batch_size)
        ]
    )
    copy_last = np.repeat(a_test[..., -1:], t_out, axis=-1)

    def scores(pred):
        whole = np.asarray(rel_l2_per_sample(jnp.asarray(pred), jnp.asarray(u_test)))
        steps = [
            float(np.mean(rel_l2_per_sample(pred[..., t], u_test[..., t])))
            for t in range(t_out)
        ]
        return whole, steps

    errors, per_step = scores(preds)
    baseline, baseline_per_step = scores(copy_last)
    return dict(
        test_rel_l2=float(errors.mean()),
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        worst=float(errors.max()),
        copy_last_frame=float(baseline.mean()),
        per_step=per_step,
        copy_last_frame_per_step=baseline_per_step,
        errors=errors,
        predictions=preds,
    )
```

### 9. Plot

<!-- tutorial-snippet -->
```python
def plot_prediction(history_frame, truth, pred, path: Path, title: str) -> None:
    """Save input, true vorticity, predicted vorticity and absolute error."""
    lim = float(np.abs(truth).max())
    panels = [
        ("Input: vorticity at t = 10", history_frame, "RdBu_r", (-lim, lim)),
        ("Truth: vorticity at t = 20", truth, "RdBu_r", (-lim, lim)),
        ("Prediction: vorticity at t = 20", pred, "RdBu_r", (-lim, lim)),
        ("Absolute error at t = 20", np.abs(pred - truth), "magma", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for ax, (name, values, cmap, vlim) in zip(axes, panels):
        vmin, vmax = vlim if vlim else (None, None)
        im = ax.imshow(values.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

<!-- tutorial-snippet -->
```python
def plot_rollout(per_step: list[float], baseline: list[float], path: Path) -> None:
    """Save the mean relative L2 of every rolled-out frame, with the baseline."""
    times = np.arange(11, 11 + len(per_step))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, per_step, marker="o", label="model (rollout)")
    ax.plot(times, baseline, marker="s", label="copy the last input frame")
    ax.set_xlabel("time step t")
    ax.set_ylabel("relative L2 of the frame (mean over test flows)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

<!-- tutorial-snippet -->
```python
def plot_loss(history: list[dict], path: Path) -> None:
    """Save the teacher-forced training error and the rollout test error per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_step_rel_l2"] for h in history], marker="o",
            label="train (teacher forced, per step)")  # fmt: skip
    ax.plot(epochs, [h["test_rel_l2"] for h in history], marker="o",
            label="test (rollout, 10 frames)")  # fmt: skip
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 (mean over flows)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
```

The complete script is [`scripts/tutorials/navierstokes_gino.py`](../../../scripts/tutorials/navierstokes_gino.py).

## Run it

Smoke run (CPU, no GPU needed):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/tutorials/navierstokes_gino.py --data-dir ~/data/navier_stokes --smoke
```

Full protocol (1000 training flows, benchmark-size GINO, 500 epochs; use a
GPU):

```bash
python scripts/tutorials/navierstokes_gino.py --data-dir ~/data/navier_stokes
```

The other flags are `--epochs`, `--seed`, `--out-dir` (default
`outputs/navierstokes_gino_tutorial`), `--optimizer` ([`adamw_cx`](../../api/training.md#scirex.training.adamw_cx), the default, or `adamw`) and, for the
smoke run, `--n-train` and `--n-test`.

This is the output of the smoke run on a CPU (smoke run, CPU, 20 train / 4
test flows, not a benchmark number). The wall time was 67.2 s,
measured with `time`; library deprecation warnings are left out:

```text
python 3.12.3 | numpy 1.26.4
scirex 1.0.0rc1
jax 0.6.2 | flax 0.10.7 | optax 0.2.8
devices: [CpuDevice(id=0)]
SEED = 0
samples: 20 train / 4 test | grid 64x64 | 10 frames in, 10 out | vorticity range [-2.03, 1.91]
data loading: 0.2s
GINO {'in_gno_radius': 0.033, 'out_gno_radius': 0.033, 'fno_n_modes': (12, 12), 'fno_hidden_channels': 16, 'fno_n_layers': 2} | parameters: 390,539
epoch 0: train_step_rel_l2=0.9933 test_rel_l2=0.9673 epoch_time=22.8s
epoch 1: train_step_rel_l2=0.7694 test_rel_l2=0.6900 epoch_time=14.6s
epoch 2: train_step_rel_l2=0.5381 test_rel_l2=0.7012 epoch_time=14.3s
test relative L2 (rollout): mean 0.7012 | median 0.6881 | p90 0.8179 | worst 0.8617
copy the last input frame: 0.7329
per step: 0.436 0.489 0.567 0.608 0.647 0.700 0.772 0.790 0.777 0.789
figures in outputs/navierstokes_gino_tutorial | total time 65.8s
```

## Results

### Full run

The next figures were not made by this script. They come from the benchmark
runner (`train.py` and `infer.py`) and are the committed results of the
**full 500-epoch run** (1000 training and 200 test flows, benchmark-size
GINO, seed 0, one NVIDIA RTX PRO 6000 Blackwell). `infer.py` draws
one test flow (the figure title names it and its relative L2 error): the
prediction, the ground truth and the absolute error at the last step,
$t = 20$.

![Full 500-epoch run: prediction for one test flow at t = 20](../../assets/benchmarks/navier_stokes_gino/contour.png)

Error of every rolled-out frame, mean over the test flows:

![Full 500-epoch run: error per rollout step](../../assets/benchmarks/navier_stokes_gino/rollout.png)

Training and test error per epoch:

![Full 500-epoch run: training and test error per epoch](../../assets/benchmarks/navier_stokes_gino/error_curve.png)

Seconds per epoch (in `metrics.json`, the first epoch took 83.5 s and
the median epoch 19.7 s):

![Full 500-epoch run: compute cost per epoch](../../assets/benchmarks/navier_stokes_gino/compute_cost.png)

| Model, same data, recipe and seed | Parameters | Mean | Median | 90th percentile | Worst | Training time |
|---|---|---|---|---|---|---|
| **GINO (this tutorial)** | 11,297,041 | 0.0666 | 0.0603 | 0.1157 | 0.2485 | 3.9 h |
| [FNO](../../benchmarks/navier_stokes/results/comparative.md#fno) | 11,183,467 | 0.0705 | 0.0631 | 0.1228 | 0.2779 | 2.5 h |
| [Transolver](../../benchmarks/navier_stokes/results/comparative.md#transolver) | 11,232,321 | 0.0872 | 0.0796 | 0.1298 | 0.3617 | 25.2 h |
| [WNO](../../benchmarks/navier_stokes/results/comparative.md#wno) | 11,312,957 | 0.3198 | 0.3050 | 0.4536 | 0.6159 | 8.7 h |
| Copy the last input frame | | 0.7481 | | | | |

Error of each rolled-out frame, mean over the 200 test flows:

| Time t | 11 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|
| GINO | 0.0203 | 0.0232 | 0.0290 | 0.0406 | 0.0658 | 0.1206 |
| FNO | 0.0192 | 0.0230 | 0.0295 | 0.0426 | 0.0702 | 0.1275 |
| Transolver | 0.0109 | 0.0192 | 0.0367 | 0.0579 | 0.0927 | 0.1501 |
| WNO | 0.0614 | 0.0929 | 0.1770 | 0.2741 | 0.3720 | 0.4665 |
| Copy last frame | 0.2037 | 0.3593 | 0.6116 | 0.7807 | 0.8718 | 0.8554 |

## Troubleshooting

The common problems are listed in the [FNO tutorial](../FNO/NavierStokes_FNO.md#troubleshooting).

## Next steps

- Compare with the other architectures on the same data:
  [Transolver](../../benchmarks/navier_stokes/results/comparative.md#transolver),
  [GINO](../../benchmarks/navier_stokes/results/comparative.md#gino),
  [FNO](../../benchmarks/navier_stokes/results/comparative.md#fno),
  [WNO](../../benchmarks/navier_stokes/results/comparative.md#wno) and the
  [comparison page](../../benchmarks/navier_stokes/results/comparative.md).
- Read the [Navier-Stokes 2D results](../../benchmarks/navier_stokes/results/comparative.md#gino) for the
  checkpoint, the compute cost and the notes on the run, and the
  [dataset guide](../../dataset_guide/navier_stokes.md) for the file.
- Learn how GINO works in the [theory guide](../../theory_guide/operator_learning.md#gino) and
  see its arguments in the [API reference](../../api/models.md).
- Try a steady problem on a deformed mesh: the [Pipe FNO tutorial](../Others/Pipe_FNO.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>

3. <span id="ref3"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>
