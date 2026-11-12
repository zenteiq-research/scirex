# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Solving 1D Poisson with FNO
===========================

**Problem.** Given a source term ``f(x)`` on the unit interval, predict the
solution ``u(x)`` of ``-u''(x) = f(x)`` with ``u(0) = u(1) = 0``. Every
source is a random sine series, so the exact solution is known in closed form
and the data are generated in a second, with nothing to download. The model
is the Fourier Neural Operator (FNO, Li et al. 2021) in one dimension.

This is the worked example of the "Add a new example" guide
(``doc/tutorials/NewExample/``): it has every piece a new example needs (a
data loader, a config that rejects unknown keys, a model, training, a
checkpoint and inference from that checkpoint) and runs on a CPU in well
under two minutes.

**What you will learn.**

1. How to generate a dataset with an exact solution, so the error of the
   model is the only error.
2. How to keep every setting in one config whose keys are checked.
3. How to build ``scirex.models.FNO`` for a 1D field and train it with the
   complex-aware AdamW ``scirex.training.adamw_cx``.
4. How to save a checkpoint, load it again and evaluate it, the way a
   benchmark's ``train.py`` and ``infer.py`` split the work.

Usage, from the repository root::

    python scripts/tutorials/poisson1d_fno.py --smoke
    python scripts/tutorials/poisson1d_fno.py
    python scripts/tutorials/poisson1d_fno.py --infer
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.FNO``,
# ``scirex.training.adamw_cx`` and ``scirex.utils.count_model_params``. The
# loop is plain JAX and optax.

from __future__ import annotations

import argparse
import json
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
from scirex.models import FNO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# Every setting of the example, in one place. ``make_config`` accepts only
# these keys, so a typo such as ``--set epoch=10`` fails instead of being
# ignored.
DEFAULT_CONFIG = dict(
    # problem
    n_points=128,  # grid points on [0, 1], both ends included
    n_terms=16,  # sine terms in every source f
    n_train=1000,
    n_test=200,
    # model
    modes=16,
    hidden_channels=32,
    n_layers=4,
    # training recipe
    learning_rate=3e-3,
    weight_decay=1e-4,
    batch_size=50,
    epochs=100,
)

# The --smoke run: fewer samples and epochs, the same model.
SMOKE = dict(n_train=200, n_test=50, epochs=10)


def make_config(**overrides) -> dict:
    """``DEFAULT_CONFIG`` with ``overrides`` applied. Unknown keys are an error."""
    unknown = sorted(set(overrides) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(
            f"unknown config key(s) {unknown}; allowed keys: {sorted(DEFAULT_CONFIG)}"
        )
    return {**DEFAULT_CONFIG, **overrides}


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
# Generate the data
# -----------------
# Each source is ``f(x) = sum_k c_k sin(k pi x)`` for ``k = 1..n_terms``, with
# ``c_k ~ N(0, 1) / k`` so that high frequencies are weaker. Since
# ``-(sin(k pi x))'' = (k pi)^2 sin(k pi x)``, the exact solution is
# ``u(x) = sum_k c_k / (k pi)^2 sin(k pi x)``, and it is zero at both ends.
# Training and test sources come from different seeds.


def poisson1d_dataset(n_samples: int, n_points: int, n_terms: int, seed: int) -> dict:
    """Sources ``f`` and exact solutions ``u``, each ``(n_samples, 1, n_points)``."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_points)
    k = np.arange(1, n_terms + 1)
    c = rng.normal(size=(n_samples, n_terms)) / k
    basis = np.sin(np.pi * k[:, None] * x[None, :])  # (n_terms, n_points)
    f = c @ basis
    u = (c / (np.pi * k) ** 2) @ basis
    return dict(
        x=x.astype(np.float32),
        f=f[:, None].astype(np.float32),
        u=u[:, None].astype(np.float32),
    )


def load_data(config: dict, seed: int) -> dict:
    """The train and test split of the example."""
    train = poisson1d_dataset(
        config["n_train"], config["n_points"], config["n_terms"], seed
    )
    test = poisson1d_dataset(
        config["n_test"], config["n_points"], config["n_terms"], seed + 1
    )
    print(
        f"samples: {config['n_train']} train / {config['n_test']} test | "
        f"grid {config['n_points']} points | "
        f"u range [{train['u'].min():.4f}, {train['u'].max():.4f}]"
    )
    return dict(
        x=train["x"],
        f_train=train["f"],
        u_train=train["u"],
        f_test=test["f"],
        u_test=test["u"],
    )


# %%
# Build the model
# ---------------
# ``FNO`` is channels-first, so a batch is ``(batch, 1, n_points)``: one input
# channel, the source. The length of ``n_modes`` sets the dimension, so
# ``n_modes=(16,)`` is a 1D FNO. By default FNO appends the x coordinate as one
# more channel. ``count_model_params`` counts a complex weight as two.


def build_model(config: dict, seed: int):
    """Build the FNO, initialise it and print the parameter count."""
    model = FNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=config["hidden_channels"],
        n_modes=(config["modes"],),
        n_layers=config["n_layers"],
    )
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    sample = jnp.zeros((1, 1, config["n_points"]), jnp.float32)
    params = model.init(init_key, sample)
    print(f"FNO 1D | parameters: {count_model_params(params):,}")
    return model, params, key


# %%
# Loss
# ----
# The loss of a batch is the mean over its samples of the relative L2 error.
# It does not depend on the scale of ``u``, so the targets need no
# normalisation. The metric is the same per-sample relative L2, averaged over
# the test samples.


def rel_l2_per_sample(pred, true):
    """``||pred - true|| / ||true||`` per sample, over all grid points."""
    b = pred.shape[0]
    diff = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    return diff / jnp.maximum(jnp.linalg.norm(true.reshape(b, -1), axis=1), 1e-12)


def make_loss(model):
    """Return ``loss_fn(params, f, u) -> mean relative L2``."""

    def loss_fn(params, f, u):
        return jnp.mean(rel_l2_per_sample(model.apply(params, f), u))

    return loss_fn


# %%
# Optimiser
# ---------
# FNO's spectral weights are complex. ``adamw_cx`` is AdamW that handles
# complex gradients correctly; for real weights it is plain AdamW. The
# learning rate follows a cosine decay to zero over the whole run.
# ``hyperparam_dtype`` keeps the injected hyperparameters float32 instead of
# the parameters' complex64.


def make_optimizer(config: dict, steps_per_epoch: int):
    """``adamw_cx`` on a cosine learning-rate decay."""
    schedule = optax.cosine_decay_schedule(
        init_value=config["learning_rate"],
        decay_steps=steps_per_epoch * config["epochs"],
    )
    return optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule, weight_decay=config["weight_decay"]
    )


# %%
# Train
# -----
# Batches in a new random order every epoch, one optimiser step per batch;
# the last incomplete batch is dropped. The test samples are scored at the
# end of every epoch, to draw the error curve; no epoch is selected by it.
# JAX compiles the step on its first call, so the first epoch is slower.


def train(model, params, key, data, config):
    """Train for ``config["epochs"]`` epochs. Return the parameters and the history."""
    f_train, u_train = jnp.asarray(data["f_train"]), jnp.asarray(data["u_train"])
    bs = config["batch_size"]
    n_train = f_train.shape[0]
    steps_per_epoch = max(n_train // bs, 1)
    tx = make_optimizer(config, steps_per_epoch)
    opt_state = tx.init(params)
    loss_fn = make_loss(model)
    predict = jax.jit(model.apply)  # compiled once, reused every epoch

    @jax.jit
    def train_step(params, opt_state, f, u):
        loss, grads = jax.value_and_grad(loss_fn)(params, f, u)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = []
    for epoch in range(config["epochs"]):
        t0 = perf_counter()
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n_train)
        total = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * bs : (b + 1) * bs]
            params, opt_state, loss = train_step(
                params, opt_state, f_train[idx], u_train[idx]
            )
            total += float(loss)
        test = evaluate(predict, params, data)["test_rel_l2"]
        history.append(
            dict(epoch=epoch, train_loss=total / steps_per_epoch, test_rel_l2=test)
        )
        print(
            f"epoch {epoch}: train_loss={total / steps_per_epoch:.4f} "
            f"test_rel_l2={test:.4f} epoch_time={perf_counter() - t0:.2f}s"
        )
    return params, history


# %%
# Evaluate
# --------
# The relative L2 error of every test sample. The mean is the example's
# number; the median, the 90th percentile and the worst sample show how the
# error is spread. ``predict`` is ``jax.jit(model.apply)``, compiled once by
# the caller: wrapping ``model.apply`` in ``jax.jit`` inside this function
# would compile it again on every call.


def evaluate(predict, params, data) -> dict:
    """Per-sample relative L2 on the test samples, and the predictions."""
    preds = np.asarray(predict(params, jnp.asarray(data["f_test"])))
    errors = np.asarray(
        rel_l2_per_sample(jnp.asarray(preds), jnp.asarray(data["u_test"]))
    )
    return dict(
        test_rel_l2=float(errors.mean()),
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        worst=float(errors.max()),
        errors=errors,
        predictions=preds[:, 0],
    )


# %%
# Save and load a checkpoint
# --------------------------
# ``FNO.save_checkpoint`` writes the weights and the constructor arguments;
# ``FNO.from_checkpoint`` rebuilds the same model from them. The config is
# saved next to them as JSON, so inference regenerates exactly the same test
# split. This is the line between a benchmark's ``train.py`` and
# ``infer.py``. ``from_checkpoint`` returns the weights as NumPy arrays, and
# FNO's spectral layer refuses those outside ``jax.jit``, so they are turned
# back into JAX arrays.


def save_run(model, params, config: dict, seed: int, folder: Path) -> None:
    """Write the checkpoint and the config of a run to ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(folder, "fno", params)
    (folder / "config.json").write_text(
        json.dumps(dict(config=config, seed=seed), indent=2)
    )


def load_run(folder: Path):
    """Return ``(model, params, config, seed)`` saved by ``save_run``."""
    saved = json.loads((folder / "config.json").read_text())
    config = make_config(**saved["config"])
    _, template, _ = build_model(config, saved["seed"])
    model, params = FNO.from_checkpoint(folder, "fno", template)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    return model, params, config, saved["seed"]


# %%
# Plot
# ----
# One test sample: the source, the true and the predicted solution on the
# same axes, and the error. We also plot the loss and the test error per
# epoch.


def plot_prediction(x, f, truth, pred, path: Path, title: str) -> None:
    """Save the source, the true and predicted solution and the error."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    axes[0].plot(x, f, color="tab:gray")
    axes[0].set_title("Input: source f")
    axes[1].plot(x, truth, label="truth", linewidth=2)
    axes[1].plot(x, pred, "--", label="prediction")
    axes[1].set_title("Solution u")
    axes[1].legend()
    axes[2].plot(x, pred - truth, color="tab:red")
    axes[2].set_title("Error: prediction - truth")
    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training loss and the test relative L2 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["test_rel_l2"] for h in history], label="test")
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
# Without ``--infer`` the script trains, saves the run to ``--out-dir``, loads
# it back and evaluates the loaded model, so the saved checkpoint is checked
# on every run. With ``--infer`` it only loads and evaluates. ``--set
# key=value`` overrides one config key; unknown keys are refused.


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ``["epochs=5", "learning_rate=1e-3"]`` into a dict of numbers."""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = float(value) if any(c in value for c in ".e") else int(value)
    return out


def main(argv: list[str] | None = None) -> dict:
    """Run the example end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--smoke", action="store_true", help="fewer samples and epochs")
    ap.add_argument("--infer", action="store_true", help="evaluate a saved run only")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/poisson1d_fno_tutorial")
    )
    args = ap.parse_args(argv)

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()

    if args.infer:
        model, params, config, seed = load_run(out_dir)
        data = load_data(config, seed)
    else:
        overrides = {**(SMOKE if args.smoke else {}), **parse_overrides(args.set)}
        config = make_config(**overrides)
        data = load_data(config, args.seed)
        model, params, key = build_model(config, args.seed)
        params, history = train(model, params, key, data, config)
        save_run(model, params, config, args.seed, out_dir)
        plot_loss(history, out_dir / "loss.png")
        model, params, _, _ = load_run(out_dir)

    metrics = evaluate(jax.jit(model.apply), params, data)
    print(
        f"test relative L2: mean {metrics['test_rel_l2']:.4f} | "
        f"median {metrics['median']:.4f} | p90 {metrics['p90']:.4f} | "
        f"worst {metrics['worst']:.4f}"
    )
    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        data["x"],
        data["f_test"][k, 0],
        data["u_test"][k, 0],
        metrics["predictions"][k],
        out_dir / "prediction.png",
        f"1D Poisson, FNO: test sample with the median error "
        f"({metrics['errors'][k]:.2%})",
    )
    result = {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"outputs in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return result


if __name__ == "__main__":
    main()
