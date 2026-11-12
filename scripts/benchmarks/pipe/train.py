# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Pipe flow benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` with the Transolver Pipe recipe, once per
seed, and saves one checkpoint per run and seed, holding the weights after the
last epoch. The test relative L2 is recorded every epoch for the error curve
only; it does not choose which weights are saved.

``--run transolver_exp_pipe`` instead runs ``_transolver.py``, the port of the
paper's own ``exp_pipe.py``, with the settings of the JAX-vs-PyTorch
comparison (``_config.TRANSOLVER_ARGS``).

Recipe (``_config.PROTOCOL``)::

    data           1000 train / 200 test samples, full 129x129 mesh
    input          raw (x, y) coordinates, unit-Gaussian normalised
    optimiser      AdamW (complex-aware, eps 1e-8), weight decay 1e-5,
                   gradient-norm clip 0.1
    lr schedule    cosine one-cycle, peak 1e-3 (pct_start 0.3, div 25, 1e4)
    loss           relative L2 per sample, summed over the batch
    epochs         500, batch size 8, seeds 0, 1 and 2

Usage (from the repo root)::

    export PIPE_DATA_DIR=/path/to/pipe     # Pipe_X.npy, Pipe_Y.npy, Pipe_Q.npy
    python scripts/benchmarks/pipe/train.py --run fno
    python scripts/benchmarks/pipe/train.py --run transolver --seed 1
    python scripts/benchmarks/pipe/train.py --run all

Runs that already have a complete checkpoint are skipped unless
``--overwrite`` is given.

``--animate`` also records each run's prediction on one fixed test sample at
log-spaced epochs, saved as ``frames.npz`` in the run's (seed's) checkpoint
folder;
``animate.py`` renders one of them as a prediction / ground truth / error
GIF (``animations/pipe/evolution.gif`` is Transolver's).
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from scirex.training.optim import scale_by_adam_cx  # noqa: E402
from scripts.benchmarks._animate import log_spaced_epochs  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.pipe import _transolver  # noqa: E402
from scripts.benchmarks.pipe._checkpoint import (  # noqa: E402
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.pipe._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    PROTOCOL,
    RUNS,
    TRANSOLVER_ARGS,
    RunConfig,
    protocol_for,
)
from scripts.benchmarks.pipe._data import load_split, rel_l2_sum  # noqa: E402
from scripts.benchmarks.pipe._models import get_spec  # noqa: E402

jax.config.update("jax_enable_x64", False)


def make_optimizer(
    protocol: dict, total_steps: int
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Complex-aware AdamW on a cosine one-cycle schedule, after
    gradient-norm clipping."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["lr"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    if protocol["optimizer"] != "adamw_cx":
        raise ValueError(f"unsupported optimizer {protocol['optimizer']!r}")
    # adamw_cx with a schedule: torch.optim.AdamW, decay scaled by the lr.
    tx = optax.chain(
        scale_by_adam_cx(eps=protocol["adam_eps"]),
        optax.add_decayed_weights(protocol["weight_decay"]),
        optax.scale_by_learning_rate(schedule),
    )
    if protocol["max_grad_norm"] is not None:
        tx = optax.chain(optax.clip_by_global_norm(protocol["max_grad_norm"]), tx)
    return tx, schedule


def train_one(
    run: RunConfig,
    data: dict,
    run_dir: Path,
    protocol: dict,
    seed: int,
    anim_frames: int = 0,
    anim_sample: int = 0,
) -> dict:
    """Trains ``run`` on ``data`` (from :func:`load_split`) with ``seed`` and
    saves it.

    With ``anim_frames > 0``, also saves ``run_dir/frames.npz``: the
    prediction on test sample ``anim_sample`` at about that many log-spaced
    epochs.

    Returns the ``training`` summary stored in the checkpoint's config.
    """
    p = protocol
    tag = f"{run.name} seed {seed}"
    spec = get_spec(run.model)
    grid = data["grid"]
    x_train = spec.prepare(run.arch, data["x_train"], grid)
    x_test = spec.prepare(run.arch, data["x_test"], grid)
    y_train, y_test = data["y_train"], data["y_test"]
    y_normalizer = data["y_normalizer"]
    ntrain, ntest = y_train.shape[0], y_test.shape[0]
    # Every model takes one optimiser step per batch of p["batch_size"]
    # samples. A model fed fewer samples at a time (bs) sums the gradients of
    # its micro-batches first; the loss is a sum over the batch, so that is
    # exactly the gradient of the whole batch.
    batch_size = p["batch_size"]
    bs = spec.batch_size or batch_size
    if batch_size % bs:
        raise ValueError(f"batch size {batch_size} is not a multiple of {bs}")

    model = spec.build(run.arch, grid)
    params = model.init(jax.random.PRNGKey(seed), **spec.take(x_train, np.arange(bs)))
    params = spec.post_init(params, run.arch, seed)

    steps_per_epoch = -(-ntrain // batch_size)
    tx, schedule = make_optimizer(p, steps_per_epoch * run.epochs)
    opt_state = tx.init(params)

    def loss_fn(params, inputs, y):
        out = spec.apply(model, params, inputs)
        return rel_l2_sum(y_normalizer.decode(out), y)

    @jax.jit
    def train_step(params, opt_state, inputs, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    grad_step = jax.jit(jax.value_and_grad(loss_fn))
    add_grads = jax.jit(lambda a, b: jax.tree_util.tree_map(jnp.add, a, b))

    @jax.jit
    def apply_grads(params, opt_state, grads):
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    eval_step = jax.jit(loss_fn)

    anim_epochs = (
        set(log_spaced_epochs(run.epochs, anim_frames)) if anim_frames else set()
    )
    anim_inputs = spec.take(x_test, np.array([anim_sample]))
    pred_frames: list[np.ndarray] = []

    rng = np.random.default_rng(seed)
    history: list[dict] = []
    t_start = perf_counter()
    for epoch in range(run.epochs):
        t0 = perf_counter()
        order = rng.permutation(ntrain)
        train_total = 0.0
        for i in range(0, ntrain, batch_size):
            idx = order[i : i + batch_size]
            if bs == batch_size:
                params, opt_state, loss = train_step(
                    params,
                    opt_state,
                    spec.take(x_train, idx),
                    jnp.asarray(y_train[idx]),
                )
                train_total += float(loss)
                continue
            grads = None
            for j in range(0, len(idx), bs):
                sub = idx[j : j + bs]
                loss, g = grad_step(
                    params, spec.take(x_train, sub), jnp.asarray(y_train[sub])
                )
                grads = g if grads is None else add_grads(grads, g)
                train_total += float(loss)
            params, opt_state = apply_grads(params, opt_state, grads)
        test_total = 0.0
        for i in range(0, ntest, bs):
            idx = np.arange(i, min(i + bs, ntest))
            test_total += float(
                eval_step(params, spec.take(x_test, idx), jnp.asarray(y_test[idx]))
            )
        secs = perf_counter() - t0
        if epoch in anim_epochs:
            out = y_normalizer.decode(spec.apply(model, params, anim_inputs))
            pred_frames.append(np.asarray(out, dtype=np.float32)[0, :, 0])
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_total / ntrain,
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
        **p,
        seed=seed,
        batch_size_used=bs,
        samples_per_step=batch_size,
        epochs_run=run.epochs,
        train_seconds=perf_counter() - t_start,
        final_test_rel_l2=history[-1]["test_rel_l2"],
    )
    save_checkpoint(
        run_dir,
        run,
        params,
        grid,
        data["x_normalizer"],
        y_normalizer,
        history,
        training,
    )
    if pred_frames:
        np.savez_compressed(
            run_dir / "frames.npz",
            pred_frames=np.stack(pred_frames),
            truth_frame=y_test[anim_sample, :, 0],
            xy=data["xy_test"][anim_sample],
            epochs=np.array(sorted(anim_epochs)),
            grid=np.array(grid),
            sample=anim_sample,
        )
    print(
        f"[{tag}] final test rel. L2 {history[-1]['test_rel_l2']:.5f}; "
        f"{training['train_seconds'] / 60:.1f} min -> {run_dir}"
    )
    return training


#: ``_transolver.py`` (the paper-port Transolver), outside ``RUNS``.
EXP_PIPE = "transolver_exp_pipe"
CHOICES = (*RUNS, EXP_PIPE)


def train_transolver(args: argparse.Namespace) -> None:
    """Runs ``_transolver.py`` with the comparison's settings, saving to
    ``<ckpt_root>/transolver_exp_pipe`` (``<ckpt_root>/transolver_exp_pipe-<N>ep``
    with ``--epochs N``)."""
    run_dir = args.ckpt_root / EXP_PIPE
    argv = list(TRANSOLVER_ARGS)
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]  # the last --epochs wins
        run_dir = args.ckpt_root / f"{EXP_PIPE}-{args.epochs}ep"
    epochs = args.epochs or int(TRANSOLVER_ARGS[TRANSOLVER_ARGS.index("--epochs") + 1])
    log = run_dir / "loss_curve.csv"  # one row per finished epoch
    if (
        log.exists()
        and len(log.read_text().splitlines()) - 1 >= epochs
        and not args.overwrite
    ):
        print(f"[{EXP_PIPE}] {run_dir} is complete, skipping (--overwrite to retrain)")
        return
    argv += [
        "--data_path",
        str(args.data_dir),
        "--ckpt-dir",
        str(run_dir),
        "--output-dir",
        str(run_dir),
    ]
    if args.animate:
        argv += [
            "--anim-frames",
            str(args.anim_frames),
            "--anim-sample",
            str(args.anim_sample),
        ]
    _transolver.main(argv)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run",
        nargs="+",
        required=True,
        metavar="NAME",
        help=f"run(s) to train, or 'all'. Choices: {', '.join(CHOICES)}",
    )
    ap.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=list(PROTOCOL["seeds"]),
        help=f"seed(s) to train (default: {list(PROTOCOL['seeds'])}; "
        f"not used by {EXP_PIPE})",
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
    ap.add_argument(
        "--animate",
        action="store_true",
        help="also save frames.npz for animate.py (one extra forward pass "
        "per captured epoch)",
    )
    ap.add_argument(
        "--anim-frames",
        type=int,
        default=48,
        help="target frame count with --animate (log-spaced epochs)",
    )
    ap.add_argument(
        "--anim-sample",
        type=int,
        default=0,
        help="test sample to animate; keep it the same for every model",
    )
    args = ap.parse_args()

    names = list(RUNS) if args.run == ["all"] else args.run
    unknown = [n for n in names if n not in CHOICES]
    if unknown:
        ap.error(f"unknown run(s) {unknown}; choose from {list(CHOICES)}")

    require_gpu(args.allow_cpu)
    data = load_split(args.data_dir, PROTOCOL) if set(names) & set(RUNS) else None

    jobs = [
        (name, seed)
        for name in names
        for seed in ([None] if name == EXP_PIPE else args.seed)
    ]
    for i, (name, seed) in enumerate(jobs, 1):
        if name == EXP_PIPE:
            print(f"=== [{i}/{len(jobs)}] {name}")
            train_transolver(args)
            continue
        run = RUNS[name]
        run_dir = args.ckpt_root / name
        if args.epochs is not None and args.epochs != run.epochs:
            # A quick check must not replace the published run's checkpoint,
            # which infer.py reads from <ckpt_root>/<name>/seed<seed>.
            run = RunConfig(**{**run.__dict__, "epochs": args.epochs})
            run_dir = args.ckpt_root / f"{name}-{args.epochs}ep"
        run_dir = run_dir / f"seed{seed}"
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(jobs)}] {name} seed {seed}: exists, skipping")
            continue
        print(f"=== [{i}/{len(jobs)}] {name} seed {seed}")
        train_one(
            run,
            data,
            run_dir,
            protocol_for(run),
            seed,
            anim_frames=args.anim_frames if args.animate else 0,
            anim_sample=args.anim_sample,
        )


if __name__ == "__main__":
    main()
