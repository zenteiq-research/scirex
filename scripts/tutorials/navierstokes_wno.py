# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Forecasting 2D turbulence with WNO (Navier-Stokes)
==================================================

**Problem.** Given the vorticity of a 2D incompressible flow on a 64x64
periodic grid at ten time steps, predict the next ten, one at a time, each
prediction fed back as the newest input. The flow is the vorticity form of
the Navier-Stokes equations at viscosity 1e-5 (Li et al. 2021); the model is
the Wavelet Neural Operator, which replaces the Fourier transform of an
FNO layer by a discrete wavelet transform, local in both space and scale.

**What you will learn.**

1. How to read ``NavierStokes_V1e-5_N1200_T20.mat`` and build the 1000 / 200
   split of the benchmark: frames 1-10 in, frames 11-20 out.
2. How to train a one-step model with teacher forcing and evaluate it with an
   autoregressive rollout.
3. How to build ``scirex.models.WNO`` with its own grid channels, in
   float32, and count its parameters.
4. Why ``optax.inject_hyperparams`` should be told to keep the
   hyperparameters float32.
5. How to report the rollout error per test sample and per time step, against
   the copy-the-last-frame baseline, and how to plot the fields.

**Caveat.** The ``--smoke`` mode trains a tiny WNO on a few samples for a
few epochs. It checks that the pipeline works and gives poor predictions. It
is not a benchmark number. For the benchmark protocol (1000 training
samples, 500 epochs, 11,312,957 parameters) use
``scripts/benchmarks/navier_stokes/train.py``; see the "Reproduce at full
scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/navierstokes_wno.py \\
        --data-dir ~/data/navier_stokes --smoke
"""
# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.WNO`` and
# ``scirex.utils.count_model_params``. The data is read with SciPy, and the
# loop itself is plain JAX and optax.

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
# The file holds ``u``, the vorticity, as ``(1200, 64, 64, 20)``: 1200 flows,
# a 64x64 grid, 20 time steps (t = 1 .. 20). The first ``ntrain`` flows train
# and the last ``ntest`` test. Frames 1-10 are the input history, frames
# 11-20 the targets. The vorticity is used as it is, without normalisation,
# as in the benchmark.


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


# %%
# Build the model
# ---------------
# ``WNO`` is channels-last: a batch of histories is ``(batch, 64, 64, t_in)``,
# the ten frames being the input channels, and the output is the next frame,
# ``(batch, 64, 64, 1)``. ``use_grid=True`` (the default) appends the x and y
# coordinates as two more channels. ``level=3`` decomposes the 64x64 field
# three times with the ``db4`` wavelet, down to 8x8 approximation
# coefficients. ``compute_dtype=jnp.float32`` keeps the computation in
# float32, like the other models of the benchmark; WNO's own default is
# bfloat16. ``step`` maps a history to the next frame, so the training loop
# does not depend on the model.


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


def make_step(model):
    """``step(params, history) -> next frame``, both channels-last."""

    def step(params, fx):
        return model.apply(params, fx)  # (batch, H, W, 1)

    return step


# %%
# Teacher forcing and rollout
# ---------------------------
# The model predicts one frame from the ten before it. In training each of
# the ten steps gets the true history (teacher forcing): the steps do not
# depend on each other's predictions, and the loss is the relative L2 error
# of every predicted frame, summed over the steps and the batch. In
# evaluation each step gets the model's own earlier predictions
# (autoregressive rollout), which is how the model would be used, so errors
# can grow along the rollout.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all remaining axes."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.linalg.norm(true.reshape(b, -1), axis=1)


def teacher_forced(step, params, fx, yy):
    """Loss (summed over steps and batch) with the true history at every step."""
    loss = 0.0
    for t in range(yy.shape[-1]):
        pred = step(params, fx)
        loss = loss + jnp.sum(rel_l2_per_sample(pred, yy[..., t : t + 1]))
        fx = jnp.concatenate([fx[..., 1:], yy[..., t : t + 1]], axis=-1)
    return loss


def rollout(step, params, fx, t_out: int):
    """``t_out`` frames, each step fed the model's own previous predictions."""
    preds = []
    for _ in range(t_out):
        pred = step(params, fx)
        preds.append(pred)
        fx = jnp.concatenate([fx[..., 1:], pred], axis=-1)
    return jnp.concatenate(preds, axis=-1)


# %%
# Optimiser
# ---------
# WNO's weights are all real, so plain ``optax.adamw`` is right for
# it. ``optax.inject_hyperparams`` lets the learning rate follow the
# one-cycle schedule. By default it casts the hyperparameters to the highest
# dtype of the parameters; that is float32 here, but it would be complex64
# for a model with complex weights (FNO, GINO) and would then make every
# update complex. ``hyperparam_dtype=jnp.float32`` keeps them real either way.


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


# %%
# Train
# -----
# Batches of 2 flows in a new random order every epoch, one optimiser step
# per batch, no gradient clipping. After every epoch the test flows are
# rolled out and scored, to draw the error curve; no epoch is selected by it,
# and the weights after the last epoch are the result. JAX compiles the step
# on its first call, so the first epoch is slower than the others.


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


# %%
# Evaluate
# --------
# The benchmark's metric is the relative L2 error of each test flow over all
# ten rolled-out frames at once, averaged over the flows. The error of each
# frame on its own shows how it grows along the rollout. Copying the last
# input frame ten times is the baseline that any useful model must beat.


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


# %%
# Plot
# ----
# The standard figure of the benchmark, for one test flow: the input (the
# last frame of the history, t = 10), the true and the predicted vorticity
# at the end of the rollout (t = 20) on one colour scale, and the absolute
# error. A second figure shows the error of every rolled-out frame against
# the copy-the-last-frame baseline, and a third the error per epoch.


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


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few flows, uses the tiny WNO and trains for a few
# epochs. Without it the script trains the benchmark-size WNO on the full
# split for 500 epochs. That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("NS_DATA_DIR"),
        help=f"folder with {DATA_FILE} (or set $NS_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few flows, tiny WNO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/navierstokes_wno_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=20, help="flows in smoke mode")
    ap.add_argument("--n-test", type=int, default=4, help="flows in smoke mode")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $NS_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = PROTOCOL
    ntrain, ntest = (
        (args.n_train, args.n_test) if args.smoke else (p["ntrain"], p["ntest"])
    )
    epochs = args.epochs or (3 if args.smoke else p["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data_path = Path(args.data_dir).expanduser() / DATA_FILE
    data = load_ns(data_path, ntrain, ntest, p["t_in"], p["t_out"])
    print(f"data loading: {perf_counter() - t0:.1f}s")
    model, params = build_model(size, p["t_in"], data["a_train"][0], args.seed)

    params, history = train(model, params, data, epochs, args.seed)
    metrics = evaluate(model, params, data, p["t_out"])
    print(
        f"test relative L2 (rollout): mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} | p90 {metrics['p90']:.4f} | "
        f"worst {metrics['worst']:.4f}"
    )
    print(f"copy the last input frame: {metrics['copy_last_frame']:.4f}")
    print("per step: " + " ".join(f"{e:.3f}" for e in metrics["per_step"]))

    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        data["a_test"][k, ..., -1],
        data["u_test"][k, ..., -1],
        metrics["predictions"][k, ..., -1],
        out_dir / "prediction.png",
        f"Navier-Stokes, WNO: test flow with the median error "
        f"({metrics['errors'][k]:.2%})",
    )
    plot_rollout(
        metrics["per_step"],
        metrics["copy_last_frame_per_step"],
        out_dir / "rollout.png",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    skip = ("errors", "predictions", "per_step", "copy_last_frame_per_step")
    return {k: v for k, v in metrics.items() if k not in skip}


if __name__ == "__main__":
    main()
