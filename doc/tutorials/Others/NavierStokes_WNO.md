<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Tutorial: Navier-Stokes 2D (Turbulence) with WNO

In this tutorial you train a [Wavelet Neural Operator](../../theory_guide/operator_learning.md#wno)
(WNO [[3]](#ref3)) on the **Navier-Stokes 2D** benchmark: given ten frames of the vorticity
of a 2D turbulent flow, predict the next ten, one frame at a time, each
prediction fed back as the newest input. The whole tutorial is one script,
[`scripts/tutorials/navierstokes_wno.py`](../../../scripts/tutorials/navierstokes_wno.py). It uses only the public ScireX API
([`scirex.models.WNO`](../../api/models.md#scirex.models.WNO), [`scirex.utils.count_model_params`](../../api/utils.md#scirex.utils.count_model_params)), SciPy to read the
data and plain JAX and optax. The code on this page is copied from that
script.

## Summary

| | |
|---|---|
| Problem | 2D incompressible Navier-Stokes in vorticity form on a periodic 64 × 64 grid, viscosity 1e-5: 10 frames in, the next 10 out, one at a time |
| Model | WNO, [`scirex.models.WNO`](../../api/models.md#scirex.models.WNO), float32. Benchmark size: `hidden_channels=60, n_layers=4, level=3, wavelet="db4"`, grid channels appended, 11,312,957 parameters. Smoke size: `hidden_channels=12, n_layers=2`, same wavelet and level, 228,053 parameters |
| Data | `NavierStokes_V1e-5_N1200_T20.mat` (FNO [[1]](#ref1) data release, about 413 MB), 1200 flows of 20 frames. Benchmark: first 1000 train / last 200 test. Smoke: 20 train / 4 test |
| Optimiser | AdamW (`optax.adamw`), eps 1e-8, weight decay 1e-5, no gradient clipping, cosine one-cycle learning rate with peak 1e-3, batch size 2 |
| Metric | Relative L2 error of each test flow over its 10 rolled-out frames, averaged over the test flows |
| Full-run values | 0.3198 (one seed). Transolver [[2]](#ref2) paper, its own model: 0.0900. Copying the last input frame: 0.7481 |

## Setup

The setup is the same as in the [FNO tutorial](../FNO/NavierStokes_FNO.md#setup).

## The problem

The problem, the data and the metric are the same as in the [FNO tutorial](../FNO/NavierStokes_FNO.md#the-problem).

**WNO in one paragraph.** A WNO layer works like an FNO layer with the
Fourier transform replaced by a discrete wavelet transform: it decomposes the
hidden channels `level` times with a wavelet (here `db4`), multiplies the
coarsest approximation and detail coefficients by learned weights and
transforms back, plus a pointwise linear map. Wavelets are local in both
space and scale, so the layer can in principle treat sharp, local features
better than a truncated Fourier series. Its weights are all real. The
[theory guide](../../theory_guide/operator_learning.md#wno) has the details.

## Step-by-step walkthrough

The explanations of the steps that are the same for every model are in the [FNO tutorial](../FNO/NavierStokes_FNO.md#step-by-step-walkthrough). This page keeps the code and the text that is specific to the model.

### 1. Setup

Imports, the seed, the training settings and the two model sizes. `PROTOCOL`
holds the benchmark's training settings (the same values as in
[`scripts/benchmarks/navier_stokes/_config.py`](../../../scripts/benchmarks/navier_stokes/_config.py), the Transolver paper's recipe,
which every model on this benchmark shares): 500 epochs, batch size 2, AdamW
with a peak learning rate of 1e-3 on a one-cycle schedule, weight decay 1e-5
and no gradient clipping. `SIZES["full"]` is the benchmark's WNO;
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
from scirex.models import WNO
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

# Model sizes. "full" is the benchmark's WNO (11,312,957 parameters, matched
# to Transolver's 11,232,321). "smoke" is a tiny WNO that trains on a CPU.
SIZES = dict(
    full=dict(hidden_channels=60, n_layers=4, level=3, wavelet="db4"),
    smoke=dict(hidden_channels=12, n_layers=2, level=3, wavelet="db4"),
)
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

[`WNO`](../../api/models.md#scirex.models.WNO) is a Flax module: the parameters are a separate pytree, created by
`init` and used by `apply`. It is channels-last: a batch of histories is
`(batch, 64, 64, 10)`, the ten frames being the input channels, and the
output is `(batch, 64, 64, 1)`. `use_grid=True`, the default, appends the $x$
and $y$ coordinates as two more channels. `level=3` decomposes the field
three times with the `db4` wavelet (symmetric padding, WNO's default).
`compute_dtype=jnp.float32` keeps the computation in float32 like the other
models of the benchmark; WNO's own default is bfloat16. `make_step` is a thin
wrapper, so that the rest of the script is the same for every model.

<!-- tutorial-snippet -->
```python
def build_model(size: dict, t_in: int, sample: np.ndarray, seed: int):
    """Build the WNO, initialise it on one history and print the parameter count."""
    model = WNO(
        in_channels=t_in,
        out_channels=1,
        size=sample.shape[:2],
        compute_dtype=jnp.float32,
        **size,
    )
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(sample)[None])
    print(f"WNO {size} | parameters: {count_model_params(params):,}")
    return model, params
```

<!-- tutorial-snippet -->
```python
def make_step(model):
    """``step(params, history) -> next frame``, both channels-last."""

    def step(params, fx):
        return model.apply(params, fx)  # (batch, H, W, 1)

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

### 6. Optimiser

WNO's weights are all real, so plain `optax.adamw` is right for it.
`optax.inject_hyperparams` lets the learning rate follow the one-cycle
schedule. By default it casts the hyperparameters to the highest dtype among
the parameters. For WNO that is float32 anyway, but for a model with complex
weights (FNO, GINO) it would be complex64 and would make every update
complex; `hyperparam_dtype=jnp.float32` keeps them real either way, so the
same optimiser code is safe for every model.

<!-- tutorial-snippet -->
```python
def make_optimizer(protocol: dict, total_steps: int):
    """AdamW on a cosine one-cycle schedule."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    tx = optax.inject_hyperparams(optax.adamw, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule,
        weight_decay=protocol["weight_decay"],
        eps=protocol["adam_eps"],
    )
    return tx, schedule
```

### 7. Train

<!-- tutorial-snippet -->
```python
def train(model, params, data, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    a_train, u_train = data["a_train"], data["u_train"]
    a_test, u_test = data["a_test"], data["u_test"]
    bs, t_out = protocol["batch_size"], protocol["t_out"]
    ntrain = len(a_train)
    tx, _ = make_optimizer(protocol, -(-ntrain // bs) * epochs)
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

The complete script is [`scripts/tutorials/navierstokes_wno.py`](../../../scripts/tutorials/navierstokes_wno.py).

## Run it

Smoke run (CPU, no GPU needed):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python scripts/tutorials/navierstokes_wno.py --data-dir ~/data/navier_stokes --smoke
```

Full protocol (1000 training flows, benchmark-size WNO, 500 epochs; use a
GPU):

```bash
python scripts/tutorials/navierstokes_wno.py --data-dir ~/data/navier_stokes
```

The other flags are `--epochs`, `--seed`, `--out-dir` (default
`outputs/navierstokes_wno_tutorial`) and, for the
smoke run, `--n-train` and `--n-test`.

This is the output of the smoke run on a CPU (smoke run, CPU, 20 train / 4
test flows, not a benchmark number). The wall time was 21.2 s,
measured with `time`; library deprecation warnings are left out:

```text
python 3.12.3 | numpy 1.26.4
scirex 1.0.0rc1
jax 0.6.2 | flax 0.10.7 | optax 0.2.8
devices: [CpuDevice(id=0)]
SEED = 0
samples: 20 train / 4 test | grid 64x64 | 10 frames in, 10 out | vorticity range [-2.03, 1.91]
data loading: 0.2s
WNO {'hidden_channels': 12, 'n_layers': 2, 'level': 3, 'wavelet': 'db4'} | parameters: 228,053
epoch 0: train_step_rel_l2=0.9828 test_rel_l2=0.9059 epoch_time=6.4s
epoch 1: train_step_rel_l2=0.7544 test_rel_l2=0.8455 epoch_time=2.4s
epoch 2: train_step_rel_l2=0.6528 test_rel_l2=0.8177 epoch_time=2.5s
test relative L2 (rollout): mean 0.8177 | median 0.8305 | p90 0.8931 | worst 0.9034
copy the last input frame: 0.7329
per step: 0.551 0.621 0.691 0.791 0.841 0.874 0.875 0.841 0.848 0.901
figures in outputs/navierstokes_wno_tutorial | total time 20.0s
```

## Results

### Full run

The next figures were not made by this script. They come from the benchmark
runner (`train.py` and `infer.py`) and are the committed results of the
**full 500-epoch run** (1000 training and 200 test flows, benchmark-size
WNO, seed 0, one NVIDIA RTX PRO 6000 Blackwell). `infer.py` draws
one test flow (the figure title names it and its relative L2 error): the
prediction, the ground truth and the absolute error at the last step,
$t = 20$.

![Full 500-epoch run: prediction for one test flow at t = 20](../../assets/benchmarks/navier_stokes_wno/contour.png)

Error of every rolled-out frame, mean over the test flows:

![Full 500-epoch run: error per rollout step](../../assets/benchmarks/navier_stokes_wno/rollout.png)

Training and test error per epoch:

![Full 500-epoch run: training and test error per epoch](../../assets/benchmarks/navier_stokes_wno/error_curve.png)

Seconds per epoch (in `metrics.json`, the first epoch took 57.1 s and
the median epoch 65.8 s):

![Full 500-epoch run: compute cost per epoch](../../assets/benchmarks/navier_stokes_wno/compute_cost.png)

| Model, same data, recipe and seed | Parameters | Mean | Median | 90th percentile | Worst | Training time |
|---|---|---|---|---|---|---|
| [GINO](../../benchmarks/navier_stokes/results/comparative.md#gino) | 11,297,041 | 0.0666 | 0.0603 | 0.1157 | 0.2485 | 3.9 h |
| [FNO](../../benchmarks/navier_stokes/results/comparative.md#fno) | 11,183,467 | 0.0705 | 0.0631 | 0.1228 | 0.2779 | 2.5 h |
| [Transolver](../../benchmarks/navier_stokes/results/comparative.md#transolver) | 11,232,321 | 0.0872 | 0.0796 | 0.1298 | 0.3617 | 25.2 h |
| **WNO (this tutorial)** | 11,312,957 | 0.3198 | 0.3050 | 0.4536 | 0.6159 | 8.7 h |
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
- Read the [Navier-Stokes 2D results](../../benchmarks/navier_stokes/results/comparative.md#wno) for the
  checkpoint, the compute cost and the notes on the run, and the
  [dataset guide](../../dataset_guide/navier_stokes.md) for the file.
- Learn how WNO works in the [theory guide](../../theory_guide/operator_learning.md#wno) and
  see its arguments in the [API reference](../../api/models.md).
- Try a steady problem on a deformed mesh: the [Pipe FNO tutorial](../Others/Pipe_FNO.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>

3. <span id="ref3"></span>Tripura and Chakraborty, *Wavelet Neural Operator for Solving Parametric Partial Differential Equations in Computational Mechanics Problems*, Computer Methods in Applied Mechanics and Engineering 404, 115783 (2023). <https://arxiv.org/abs/2205.02191>
