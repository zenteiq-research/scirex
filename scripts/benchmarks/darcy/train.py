# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Darcy flow benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` and saves one checkpoint per run, holding
the weights after the last epoch. Every ``PROTOCOL["eval_every"]`` epochs the
test error is recorded for the error curve; it does not choose which weights
are reported.

Each model keeps its own recipe (``RunConfig.opt``); the loop is the same for
all four:

    fno         AdamW (adamw_cx), lr 1e-3 halved every 100 epochs, batch 20
    wno         AdamW, weight decay 1e-6, lr 1e-3 halved every 50 epochs,
                batch 20
    gino        AdamW (adamw_cx), weight decay 1e-4, cosine one-cycle,
                clip 1.0, batch 16, 2000 epochs, rel. L2 + 0.1 x gradient loss
    transolver  AdamW, weight decay 1e-5, cosine one-cycle, clip 0.1, batch 4,
                rel. L2 + 0.1 x gradient loss, seed 2

Usage (from the repo root)::

    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth{1,2}.mat
    python scripts/benchmarks/darcy/train.py --run transolver
    python scripts/benchmarks/darcy/train.py --run all
    # also save the frames of animations/darcy_flow/evolution.gif:
    python scripts/benchmarks/darcy/train.py --run transolver --animate

Runs that already have a complete checkpoint are skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from scirex.training import adamw_cx  # noqa: E402
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import save_params  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.darcy._checkpoint import (  # noqa: E402
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.darcy._config import (  # noqa: E402
    ANIM_EPOCHS,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.darcy._data import (  # noqa: E402
    batched_rel_l2_loss,
    gradient_regularization_loss,
    load_split,
    rel_l2,
)
from scripts.benchmarks.darcy._models import prepare  # noqa: E402

OPTIMIZERS = {"adamw": optax.adamw, "adamw_cx": adamw_cx}


def make_optimizer(opt: dict, steps_per_epoch: int) -> optax.GradientTransformation:
    """The run's optimiser: a step-decay or cosine one-cycle learning rate,
    optionally behind global-norm gradient clipping."""
    if opt["schedule"] == "step":
        schedule = optax.exponential_decay(
            init_value=opt["lr"],
            transition_steps=max(steps_per_epoch * opt["decay_epochs"], 1),
            decay_rate=opt["decay_rate"],
            staircase=True,
        )
    elif opt["schedule"] == "onecycle":
        schedule = optax.cosine_onecycle_schedule(
            transition_steps=steps_per_epoch * opt["epochs"], peak_value=opt["lr"]
        )
    else:
        raise ValueError(f"unknown schedule {opt['schedule']!r}")
    tx = optax.inject_hyperparams(OPTIMIZERS[opt["optimizer"]])(
        learning_rate=schedule, weight_decay=opt["weight_decay"], eps=1e-8
    )
    if opt["grad_clip"] is not None:
        tx = optax.chain(optax.clip_by_global_norm(opt["grad_clip"]), tx)
    return tx


def train_one(
    run: RunConfig,
    data: dict,
    run_dir: Path,
    protocol: dict,
    animate: bool = False,
) -> dict:
    """Trains ``run`` on ``data`` and saves it to ``run_dir``. Returns the
    training summary."""
    opt = run.opt
    prep = prepare(run.model, run.arch, data)
    n_train, n_test = data["train_a"].shape[0], data["test_a"].shape[0]
    n_side = data["train_a"].shape[1]
    n_points = n_side * n_side
    dx = 1.0 / n_side
    train_u = data["train_u"].reshape(n_train, n_points)
    test_u = data["test_u"].reshape(n_test, n_points)
    batch_size, epochs = opt["batch_size"], opt["epochs"]
    grad_w = opt["grad_loss_weight"]

    key = jax.random.PRNGKey(run.seed)
    key, init_key = jax.random.split(key)
    params = prep.init(init_key)

    def loss_fn(p: Any, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pred = prep.predict(p, x)
        loss = batched_rel_l2_loss(pred, u)
        if grad_w == 0.0:
            return loss
        b = pred.shape[0]
        grad_loss = gradient_regularization_loss(
            pred.reshape(b, n_side, n_side, 1), u.reshape(b, n_side, n_side, 1), dx
        )
        return loss + grad_w * grad_loss

    predict_jit = jax.jit(prep.predict)

    def test_predict(p: Any) -> np.ndarray:
        return predict_in_batches(predict_jit, p, jnp.asarray(prep.test_x), batch_size)

    def test_rel_l2(p: Any) -> float:
        pred = test_predict(p)
        return float(np.mean([rel_l2(pred[i], test_u[i]) for i in range(n_test)]))

    steps_per_epoch = max(n_train // batch_size, 1)
    tx = make_optimizer(opt, steps_per_epoch)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(p: Any, s: Any, x: jnp.ndarray, u: jnp.ndarray):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, u)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    train_x, train_u_j = jnp.asarray(prep.train_x), jnp.asarray(train_u)
    history: dict[str, list] = dict(
        epoch=[], step=[], train_loss=[], test_rel_l2=[], secs_per_epoch=[]
    )
    frames: list[np.ndarray] = []
    best_err, best_step = float("inf"), -1
    t0 = time.time()
    t_prev, prev_epoch, step = t0, 0, 0
    print(
        f"[{run.name}] {epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"batch {batch_size}"
    )
    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n_train)
        epoch_loss = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, train_x[idx], train_u_j[idx]
            )
            epoch_loss += float(loss)
            step += 1
        epoch_loss /= steps_per_epoch

        if animate and epoch + 1 in ANIM_EPOCHS:
            frames.append(np.asarray(test_predict(params)).reshape(n_test, n_points))

        if epoch % protocol["eval_every"] == 0 or epoch == epochs - 1:
            test_err = test_rel_l2(params)
            if test_err < best_err:
                best_err, best_step = test_err, step
                save_params(params, run_dir / "best_params.msgpack")
            now = time.time()
            per_epoch_s = (now - t_prev) / max(epoch - prev_epoch, 1)
            t_prev, prev_epoch = now, epoch
            for k, v in zip(
                history,
                (epoch, step, epoch_loss, test_err, per_epoch_s),
                strict=True,
            ):
                history[k].append(v)
            print(
                f"[{run.name}] epoch {epoch:4d}/{epochs}  train_loss={epoch_loss:.4f}"
                f"  test_rel_l2={test_err:.4f}  ~{per_epoch_s:.2f}s/epoch"
            )

    training = dict(
        n_train=n_train,
        n_test=n_test,
        steps_run=step,
        train_seconds=time.time() - t0,
        final_test_rel_l2=history["test_rel_l2"][-1],
        best_test_rel_l2=best_err,
        best_step=best_step,
    )
    if frames:
        final_err = np.linalg.norm(frames[-1] - test_u, axis=1) / np.linalg.norm(
            test_u, axis=1
        )
        case = int(np.argsort(final_err)[n_test // 2])  # the median test case
        run_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            run_dir / "frames.npz",
            pred_frames=np.stack([f[case].reshape(n_side, n_side, 1) for f in frames]),
            truth_frame=data["test_u"][case][..., None],
            epochs=np.asarray([e for e in ANIM_EPOCHS if e <= epochs]),
            case=case,
        )
    save_checkpoint(
        run_dir, run, protocol, params, prep.normalisation, history, training
    )
    print(
        f"[{run.name}] done: final test rel. L2 {training['final_test_rel_l2']:.5f}; "
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
        "--animate",
        action="store_true",
        help="also save frames.npz for animate.py (does not change training)",
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
    data = load_split(args.data_dir, PROTOCOL)

    for i, name in enumerate(names, 1):
        run = RUNS[name]
        run_dir = args.ckpt_root / name
        if args.epochs is not None and args.epochs != run.opt["epochs"]:
            # A quick check must not replace the published run's checkpoint,
            # which infer.py reads from <ckpt_root>/<name>.
            run = replace(run, opt={**run.opt, "epochs": args.epochs})
            run_dir = args.ckpt_root / f"{name}-{args.epochs}ep"
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(names)}] {name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(names)}] {name}")
        train_one(run, data, run_dir, PROTOCOL, animate=args.animate)


if __name__ == "__main__":
    main()
