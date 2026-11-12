# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Chip cooling benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` and saves one checkpoint per run, holding
the weights after the last epoch. The test loss is recorded every epoch for
the error curve only; it does not choose which weights are saved.

Each model keeps the recipe it was published with (``RunConfig.opt``); the
four training functions below differ only where those recipes do:

    fno         AdamW (adamw_cx), lr 1e-3 on a per-epoch cosine, batch 8
    wno         Adam + weight decay 1e-4, lr 1e-3 on a per-step cosine, batch 8
    gino        AdamW (adamw_cx), lr 1e-3 on a per-epoch cosine, 1 sample/step
    transolver  Adam + weight decay 1e-6, clip 1.0, 50-epoch warmup to 1e-4,
                then cosine to 1e-6, batch 4

All of them: 1600 train / 400 test samples, 500 epochs, seed 42, relative L2
loss on the normalised temperature (over the in-domain cells for FNO/WNO).

Usage (from the repo root)::

    export CHIPCOOLING_DATA_DIR=/path/to/1_Dataset   # model_setup.json, sim_*.npz
    python scripts/benchmarks/chip_cooling/train.py --run fno
    python scripts/benchmarks/chip_cooling/train.py --run all

Runs that already have a complete checkpoint are skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from scirex.training import adamw_cx  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.chip_cooling._checkpoint import (  # noqa: E402
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.chip_cooling._config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.chip_cooling._data import (  # noqa: E402
    grid_split,
    load_raw,
    mesh_split,
)
from scripts.benchmarks.chip_cooling._models import get_spec  # noqa: E402

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def relative_l2(pred, target, mask=None):
    """``||pred - target|| / ||target||`` over the whole batch, counting only
    the cells where ``mask`` is set (broadcast against ``pred``)."""
    m = 1.0 if mask is None else mask.astype(pred.dtype)
    num = jnp.sum(((pred - target) ** 2) * m)
    den = jnp.sum((target**2) * m) + 1e-12
    return jnp.sqrt(num / den)


def mean_abs(pred, target, mask=None):
    """Mean absolute error; with a mask, the error summed over the batch per
    in-domain cell (the published runs' logged MAE)."""
    if mask is None:
        return jnp.mean(jnp.abs(pred - target))
    m = mask.astype(pred.dtype)
    return jnp.sum(jnp.abs(pred - target) * m) / (jnp.sum(m) + 1e-12)


HISTORY_KEYS = (
    "loss",
    "mae",
    "val_loss",
    "val_mae",
    "train_err",
    "test_l2",
    "time",
    "lr",
)


def _new_history() -> dict:
    return {k: [] for k in HISTORY_KEYS}


def _log(history, tag, epoch, tl, tm, vl, vm, secs, lr, train_err=True):
    for k, v in (
        ("loss", tl),
        ("mae", tm),
        ("val_loss", vl),
        ("val_mae", vm),
        ("time", secs),
        ("lr", lr),
    ):
        history[k].append(v)
    if train_err:
        history["train_err"].append(tl)
        history["test_l2"].append(vl)
    print(
        f"[{tag}] epoch {epoch:3d}  {secs:.2f}s  lr={lr:.2e}  loss={tl:.4f}  "
        f"MAE={tm:.4f}  test_l2={vl:.4f}  test_MAE={vm:.4f}",
        flush=True,
    )


def _minibatches(n, batch_size, shuffle):
    """Index batches, shuffled with NumPy's global generator.

    FNO, WNO and GINO were published with the global generator seeded once
    (``np.random.seed``); a ``Generator`` would draw a different order.
    """
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)  # noqa: NPY002
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


# ---------------------------------------------------------------------------
# One training function per published recipe
# ---------------------------------------------------------------------------


def train_fno(spec, model, run, prepared, data, epochs, seed):
    opt = run.opt
    np.random.seed(seed)  # noqa: NPY002
    params = spec.init(model, run.arch, prepared, seed)
    tx = optax.inject_hyperparams(adamw_cx)(
        learning_rate=opt["lr"], weight_decay=opt["weight_decay"]
    )
    opt_state = tx.init(params)
    mask = jnp.asarray(data["mask"])[None, None]

    def loss_and_mae(p, x, y):
        pred = model.apply(p, x=x)
        return relative_l2(pred, y, mask), mean_abs(pred, y, mask)

    @jax.jit
    def train_step(params, opt_state, x, y):
        (loss, mae), grads = jax.value_and_grad(loss_and_mae, has_aux=True)(
            params, x, y
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, mae

    eval_step = jax.jit(loss_and_mae)
    x_tr, y_tr = prepared["x_train"], prepared["y_train"]
    x_te, y_te = prepared["x_test"], prepared["y_test"]
    history = _new_history()
    for epoch in range(epochs):
        t0 = time.time()
        lr = float(opt["lr"] * (1 + math.cos(math.pi * epoch / epochs)) / 2)
        opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
        t_l, t_m = [], []
        for b in _minibatches(len(x_tr), opt["batch_size"], True):
            params, opt_state, loss, mae = train_step(
                params, opt_state, jnp.asarray(x_tr[b]), jnp.asarray(y_tr[b])
            )
            t_l.append(float(loss))
            t_m.append(float(mae))
        v_l, v_m = [], []
        for b in _minibatches(len(x_te), opt["batch_size"], False):
            loss, mae = eval_step(params, jnp.asarray(x_te[b]), jnp.asarray(y_te[b]))
            v_l.append(float(loss))
            v_m.append(float(mae))
        _log(
            history,
            run.name,
            epoch,
            *map(np.mean, (t_l, t_m, v_l, v_m)),
            time.time() - t0,
            lr,
        )
    return params, history


def train_wno(spec, model, run, prepared, data, epochs, seed):
    from flax.training import train_state

    opt = run.opt
    np.random.seed(seed)  # noqa: NPY002
    x_tr, y_tr = prepared["x_train"], prepared["y_train"]
    x_te, y_te = prepared["x_test"], prepared["y_test"]
    # The schedule advances once per batch, so it spans every optimiser step.
    steps_per_epoch = math.ceil(len(x_tr) / opt["batch_size"])
    schedule = optax.cosine_decay_schedule(
        init_value=opt["lr"], decay_steps=max(1, epochs * steps_per_epoch), alpha=0.0
    )
    tx = optax.chain(
        optax.add_decayed_weights(opt["weight_decay"]), optax.adam(schedule)
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=spec.init(model, run.arch, prepared, seed), tx=tx
    )
    mask = jnp.asarray(data["mask"])[None, :, :, None]

    def loss_and_mae(params, x, y):
        pred = state.apply_fn({"params": params}, x)
        return relative_l2(pred, y, mask), mean_abs(pred, y, mask)

    @jax.jit
    def train_step(state, x, y):
        (loss, mae), grads = jax.value_and_grad(loss_and_mae, has_aux=True)(
            state.params, x, y
        )
        return state.apply_gradients(grads=grads), loss, mae

    eval_step = jax.jit(loss_and_mae)
    history = _new_history()
    for epoch in range(epochs):
        t0 = time.time()
        t_l, t_m = [], []
        for b in _minibatches(len(x_tr), opt["batch_size"], True):
            state, loss, mae = train_step(
                state,
                jnp.asarray(x_tr[b], dtype=jnp.float32),
                jnp.asarray(y_tr[b], dtype=jnp.float32),
            )
            t_l.append(float(loss))
            t_m.append(float(mae))
        v_l, v_m = [], []
        for b in _minibatches(len(x_te), opt["batch_size"], False):
            loss, mae = eval_step(
                state.params,
                jnp.asarray(x_te[b], dtype=jnp.float32),
                jnp.asarray(y_te[b], dtype=jnp.float32),
            )
            v_l.append(float(loss))
            v_m.append(float(mae))
        _log(
            history,
            run.name,
            epoch,
            *map(np.mean, (t_l, t_m, v_l, v_m)),
            time.time() - t0,
            float(schedule(state.step)),
        )
    return state.params, history


def train_gino(spec, model, run, prepared, data, epochs, seed):
    opt = run.opt
    np.random.seed(seed)  # noqa: NPY002
    params = spec.init(model, run.arch, prepared, seed)
    tx = optax.inject_hyperparams(adamw_cx)(
        learning_rate=opt["lr"], weight_decay=opt["weight_decay"]
    )
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, inputs, y):
        def forward(p):
            out = model.apply(p, **inputs)
            return relative_l2(out, y), mean_abs(out, y)

        (loss, mae), grads = jax.value_and_grad(forward, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return loss, mae, optax.apply_updates(params, updates), opt_state

    @jax.jit
    def eval_step(params, inputs, y):
        out = model.apply(params, **inputs)
        return out, mean_abs(out, y)

    lf_tr, y_tr = prepared["lf_train"], prepared["y_train"]
    lf_te, y_te = prepared["lf_test"], prepared["y_test"]
    history = {k: [] for k in ("loss", "val_loss", "mae", "val_mae", "time", "lr")}
    lr = opt["lr"]
    for epoch in range(epochs):
        t0 = time.time()
        t_l, t_m = [], []
        for i in np.random.permutation(len(lf_tr)):  # noqa: NPY002
            loss, mae, params, opt_state = train_step(
                params, opt_state, spec.inputs(prepared, lf_tr[i : i + 1]), y_tr[i]
            )
            t_l.append(float(loss))
            t_m.append(float(mae))
        v_l, v_m = [], []
        for i in range(len(lf_te)):
            pred, mae = eval_step(
                params, spec.inputs(prepared, lf_te[i : i + 1]), y_te[i]
            )
            v_l.append(float(relative_l2(pred, jnp.array(y_te[i]))))
            v_m.append(float(mae))
        # The cosine is stepped after the epoch: epoch 0 runs at the peak
        # rate and the recorded lr is the next epoch's.
        lr = opt["lr"] * (1 + math.cos(math.pi * (epoch + 1) / epochs)) / 2
        opt_state.hyperparams["learning_rate"] = jnp.array(float(lr))
        _log(
            history,
            run.name,
            epoch,
            *map(np.mean, (t_l, t_m, v_l, v_m)),
            time.time() - t0,
            float(lr),
            train_err=False,
        )
    return params, history


def train_transolver(spec, model, run, prepared, data, epochs, seed):
    opt = run.opt
    x_tr, fx_tr, y_tr = prepared["x_train"], prepared["fx_train"], prepared["y_train"]
    x_te, fx_te, y_te = prepared["x_test"], prepared["fx_test"], prepared["y_test"]
    bs = opt["batch_size"]
    steps_per_epoch = max(1, len(x_tr) // bs)
    warmup_steps = max(1, opt["warmup_epochs"] * steps_per_epoch)
    decay_steps = max(1, (epochs - opt["warmup_epochs"]) * steps_per_epoch)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=opt["lr"], transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=opt["lr"],
                decay_steps=decay_steps,
                alpha=opt["lr_floor"] / opt["lr"],
            ),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(opt["grad_clip"]),
        optax.add_decayed_weights(opt["weight_decay"]),
        optax.adam(schedule),
    )
    params = spec.init(model, run.arch, prepared, seed)
    opt_state = tx.init(params)

    def loss_and_mae(p, x, fx, y):
        pred = model.apply(p, x=x, fx=fx)
        return relative_l2(pred, y), mean_abs(pred, y)

    @jax.jit
    def train_step(params, opt_state, x, fx, y):
        (loss, mae), grads = jax.value_and_grad(loss_and_mae, has_aux=True)(
            params, x, fx, y
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, mae

    eval_step = jax.jit(loss_and_mae)
    rng = np.random.default_rng(seed)

    def batches(x, fx, y, shuffle):
        idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
        for start in range(0, len(x), bs):
            b = idx[start : start + bs]
            yield (jnp.asarray(a[b], dtype=jnp.float32) for a in (x, fx, y))

    history = _new_history()
    step = 0
    for epoch in range(epochs):
        t0 = time.time()
        t_l, t_m = [], []
        for xb, fxb, yb in batches(x_tr, fx_tr, y_tr, True):
            params, opt_state, loss, mae = train_step(params, opt_state, xb, fxb, yb)
            t_l.append(float(loss))
            t_m.append(float(mae))
            step += 1
        v_l, v_m = [], []
        for xb, fxb, yb in batches(x_te, fx_te, y_te, False):
            loss, mae = eval_step(params, xb, fxb, yb)
            v_l.append(float(loss))
            v_m.append(float(mae))
        _log(
            history,
            run.name,
            epoch,
            *map(np.mean, (t_l, t_m, v_l, v_m)),
            time.time() - t0,
            float(schedule(step)),
        )
    return params, history


TRAINERS = dict(
    fno=train_fno, wno=train_wno, gino=train_gino, transolver=train_transolver
)


def load_split(kind: str, raw: dict, protocol: dict, cache_dir: Path) -> dict:
    """The grid or mesh view of the data (see ``_data.py``)."""
    if kind == "grid":
        return grid_split(raw, protocol, cache_dir)
    return mesh_split(raw, protocol)


def train_one(
    run: RunConfig,
    raw: dict,
    run_dir: Path,
    protocol: dict,
    cache_dir: Path,
) -> dict:
    """Trains ``run`` and saves it to ``run_dir``. Returns the training summary."""
    spec = get_spec(run.model)
    data = load_split(spec.kind, raw, protocol, cache_dir)
    prepared = spec.prepare(run.arch, data)
    model = spec.build(run.arch, protocol["grid"])
    t_start = time.time()
    params, history = TRAINERS[run.model](
        spec, model, run, prepared, data, run.epochs, protocol["seed"]
    )
    training = dict(
        ntrain=protocol["ntrain"],
        ntest=protocol["ntest"],
        seed=protocol["seed"],
        epochs_run=run.epochs,
        train_seconds=time.time() - t_start,
        final_val_loss=history["val_loss"][-1],
    )
    save_checkpoint(run_dir, run, params, data["normalisation"], history, training)
    print(
        f"[{run.name}] done: final test loss {history['val_loss'][-1]:.5f} "
        f"(normalised); {training['train_seconds'] / 60:.1f} min -> {run_dir}"
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
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
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
    raw = load_raw(args.data_dir)

    for i, name in enumerate(names, 1):
        run = RUNS[name]
        run_dir = args.ckpt_root / name
        if args.epochs is not None and args.epochs != run.epochs:
            # A quick check must not replace the published run's checkpoint,
            # which infer.py reads from <ckpt_root>/<name>.
            run = RunConfig(**{**run.__dict__, "epochs": args.epochs})
            run_dir = args.ckpt_root / f"{name}-{args.epochs}ep"
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(names)}] {name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(names)}] {name}")
        train_one(run, raw, run_dir, PROTOCOL, args.cache_dir)


if __name__ == "__main__":
    main()
