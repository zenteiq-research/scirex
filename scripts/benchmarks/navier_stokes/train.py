# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Navier-Stokes benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` with the recipe of the Transolver
paper's ``exp_ns.py`` and saves one checkpoint per run, holding the weights
after the last epoch. The test relative L2 of the autoregressive rollout is
recorded every epoch for the error curve only; it does not choose which
weights are saved.

Recipe (``_config.PROTOCOL``)::

    data           1000 train / 200 test samples, 64x64, nu = 1e-5
    task           10 frames in, next frame out, rolled out 10 steps
    optimiser      AdamW, weight decay 1e-5, eps 1e-8, no gradient clipping
                   (Transolver: beta1 cycled 0.95 -> 0.85 -> 0.95;
                   FNO, GINO: complex-aware adamw_cx for the spectral weights)
    lr schedule    cosine one-cycle, peak 1e-3 (pct_start 0.3, div 25, 1e4)
    loss           relative L2 per sample and step, summed over the batch
                   and the 10 teacher-forced steps
    epochs         500, batch size 2, seed 0

Usage (from the repo root)::

    export NS_DATA_DIR=/path/to/dir   # NavierStokes_V1e-5_N1200_T20.mat
    python scripts/benchmarks/navier_stokes/train.py --run transolver
    python scripts/benchmarks/navier_stokes/train.py --run all

Runs that already have a complete checkpoint are skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from scirex.training import adamw_cx  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.navier_stokes._checkpoint import (  # noqa: E402
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.navier_stokes._config import (  # noqa: E402
    DATA_FILE,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.navier_stokes._data import (  # noqa: E402
    load_split,
    rel_l2_sum,
    rollout,
    teacher_forced,
)
from scripts.benchmarks.navier_stokes._models import get_spec  # noqa: E402

jax.config.update("jax_enable_x64", False)

OPTIMIZERS = {"adamw": optax.adamw, "adamw_cx": adamw_cx}


def momentum_schedule(protocol: dict, total_steps: int) -> optax.Schedule:
    """beta1 of ``OneCycleLR(cycle_momentum=True)``: cosine from 0.95 down to
    0.85 over the warm-up, then back to 0.95, inverse to the learning rate."""
    warm = protocol["pct_start"] * total_steps

    def schedule(step):
        up = jnp.minimum(step / warm, 1.0)
        down = jnp.clip((step - warm) / (total_steps - warm), 0.0, 1.0)
        descending = 0.85 + 0.05 * (1 + jnp.cos(jnp.pi * up))
        ascending = 0.95 - 0.05 * (1 + jnp.cos(jnp.pi * down))
        return jnp.where(step < warm, descending, ascending)

    return schedule


def make_optimizer(
    protocol: dict,
    total_steps: int,
    cycle_momentum: bool = False,
    optimizer: str = "adamw",
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """AdamW (``optimizer``: ``"adamw"`` or the complex-aware ``"adamw_cx"``)
    on a cosine one-cycle schedule, optionally with beta1 cycling."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["lr"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    kwargs = dict(
        learning_rate=schedule,
        weight_decay=protocol["weight_decay"],
        eps=protocol["adam_eps"],
    )
    if cycle_momentum:
        kwargs["b1"] = momentum_schedule(protocol, total_steps)
    # inject_hyperparams otherwise takes the params' highest dtype, which is
    # complex64 for FNO and GINO: lr, betas, eps and decay would turn complex
    # and every update, the real weights' included, with them.
    tx = optax.inject_hyperparams(OPTIMIZERS[optimizer], hyperparam_dtype=jnp.float32)(
        **kwargs
    )
    if protocol["max_grad_norm"] is not None:
        tx = optax.chain(optax.clip_by_global_norm(protocol["max_grad_norm"]), tx)
    return tx, schedule


def train_one(run: RunConfig, data: dict, run_dir: Path, protocol: dict) -> dict:
    """Trains ``run`` on ``data`` (from :func:`load_split`) and saves it.

    Returns the ``training`` summary stored in the checkpoint's config.
    """
    p = protocol
    tag = run.name
    spec = get_spec(run.model)
    grid = data["grid"]
    a_train, u_train = data["a_train"], data["u_train"]
    a_test, u_test = data["a_test"], data["u_test"]
    ntrain, ntest = a_train.shape[0], a_test.shape[0]
    bs, t_out = p["batch_size"], p["t_out"]

    model = spec.build(run.arch, grid, p["t_in"])
    params = spec.init(
        model, jax.random.PRNGKey(p["seed"]), jnp.asarray(a_train[:bs]), grid, p["seed"]
    )

    steps_per_epoch = -(-ntrain // bs)
    tx, schedule = make_optimizer(
        p, steps_per_epoch * run.epochs, run.cycle_momentum, run.optimizer
    )
    opt_state = tx.init(params)

    def loss_fn(params, fx, yy):
        return teacher_forced(lambda h: spec.apply(model, params, h, grid), fx, yy)

    @jax.jit
    def train_step(params, opt_state, fx, yy):
        (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, fx, yy)
        updates, opt_state = tx.update(grads, opt_state, params)
        return (
            optax.apply_updates(params, updates),
            opt_state,
            loss,
            rel_l2_sum(pred, yy),
        )

    @jax.jit
    def eval_step(params, fx, yy):
        pred = rollout(lambda h: spec.apply(model, params, h, grid), fx, t_out)
        return rel_l2_sum(pred, yy)

    rng = np.random.default_rng(p["seed"])
    history: list[dict] = []
    t_start = perf_counter()
    for epoch in range(run.epochs):
        t0 = perf_counter()
        order = rng.permutation(ntrain)
        step_total, full_total = 0.0, 0.0
        for i in range(0, ntrain, bs):
            idx = order[i : i + bs]
            params, opt_state, loss, full = train_step(
                params, opt_state, jnp.asarray(a_train[idx]), jnp.asarray(u_train[idx])
            )
            step_total += float(loss)
            full_total += float(full)
        test_total = 0.0
        for i in range(0, ntest, bs):
            sl = slice(i, min(i + bs, ntest))
            test_total += float(
                eval_step(params, jnp.asarray(a_test[sl]), jnp.asarray(u_test[sl]))
            )
        secs = perf_counter() - t0
        history.append(
            dict(
                epoch=epoch,
                # teacher-forced relative L2 of the 10 predicted frames
                train_loss=full_total / ntrain,
                # the training objective, per sample and step
                train_step_loss=step_total / ntrain / t_out,
                test_rel_l2=test_total / ntest,
                lr=float(schedule((epoch + 1) * steps_per_epoch)),
                secs=secs,
            )
        )
        if epoch % 10 == 0 or epoch == run.epochs - 1:
            print(
                f"[{tag}] epoch {epoch:3d}  train={history[-1]['train_loss']:.5f}  "
                f"test={history[-1]['test_rel_l2']:.5f}  {secs:.1f}s"
            )
            sys.stdout.flush()

    training = dict(
        {**p, "optimizer": run.optimizer},
        epochs_run=run.epochs,
        train_seconds=perf_counter() - t_start,
        final_test_rel_l2=history[-1]["test_rel_l2"],
    )
    save_checkpoint(run_dir, run, params, grid, p["t_in"], history, training)
    print(
        f"[{tag}] final test rel. L2 {history[-1]['test_rel_l2']:.5f}; "
        f"{training['train_seconds'] / 60:.1f} min -> {run_dir}"
    )
    return training


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run",
        nargs="+",
        required=True,
        metavar="NAME",
        help=f"run(s) to train, or 'all'. Choices: {', '.join(RUNS)}",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch budget (for quick checks only)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="retrain runs that already exist"
    )
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()

    names = list(RUNS) if args.run == ["all"] else args.run
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        ap.error(f"unknown run(s) {unknown}; choose from {list(RUNS)}")

    require_gpu(args.allow_cpu)
    data = load_split(args.data_dir / DATA_FILE, PROTOCOL)

    for i, name in enumerate(names, 1):
        run = RUNS[name]
        run_dir = args.ckpt_root / name
        if args.epochs is not None and args.epochs != run.epochs:
            # A quick check must not replace the published run's checkpoint,
            # which infer.py reads from <ckpt_root>/<name>.
            run = RunConfig(**{**run.__dict__, "epochs": args.epochs})
            run_dir = args.ckpt_root / f"{name}-{args.epochs}ep"
        if is_complete(run_dir) and not args.overwrite:
            saved = json.loads((run_dir / "config.json").read_text())["run"]
            if saved.get("optimizer", "adamw") != run.optimizer:
                raise SystemExit(
                    f"{run_dir} was trained with {saved.get('optimizer', 'adamw')!r}, "
                    f"but run {name!r} now uses {run.optimizer!r}; retrain it "
                    "with --overwrite"
                )
            print(f"=== [{i}/{len(names)}] {name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(names)}] {name}")
        train_one(run, data, run_dir, PROTOCOL)


if __name__ == "__main__":
    main()
