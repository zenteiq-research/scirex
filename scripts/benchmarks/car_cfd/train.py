# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-CFD benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` (GINO, or Transolver/GIWO for the
``transolver*``/``giwo*`` runs) under the GINO paper's protocol (``_config.PROTOCOL``)
and saves one checkpoint per run. Every epoch is
followed by a pass over the 111 test cars with the paper's de-normalised
relative L2, recorded in ``history.json``; ``infer.py`` turns that into the
figures and ``metrics.json``.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/car_cfd/train.py \\
        --run paper

    # Fast smoke-scale run:
    python scripts/benchmarks/car_cfd/train.py --run paper \\
        --epochs 1 --n-train 4 --n-test 2 --ckpt-root /tmp/car_cfd_smoke

Runs that already have a complete checkpoint are skipped, so an interrupted
``--run all`` can be restarted. ``--run`` takes one or more names, e.g.
``--run paper transolver_val giwo_val`` trains the three headline runs that
``infer.py`` needs.

``--seed N`` (default 0) seeds both the model initialisation and the
training-set shuffle; a non-zero seed saves to ``<run>_seed<N>``.
``--val N`` holds out the last ``N`` training cars and scores them every
epoch in place of the 111 test cars, so hyperparameters can be chosen
without scoring the test set; it saves to ``<run>_val<N>``, which ``infer.py``
ignores. The pressure normaliser is unchanged by ``--val``: it is one scalar
mean/std that ``CarCFDDatasetjax`` fits on all ``--n-train`` training cars
and all ``--n-test`` test cars together.
"""

from __future__ import annotations

import os

# Must be set before JAX initialises the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.experimental.training import Trainer  # noqa: E402
from scirex.losses import LpLoss  # noqa: E402
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.car_cfd._checkpoint import (  # noqa: E402
    count_params,
    is_complete,
    save_checkpoint,
    save_progress,
)
from scripts.benchmarks.car_cfd._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    PROTOCOL,
    RUNS,
    RunConfig,
    build_config,
    lr_at,
)
from scripts.benchmarks.car_cfd._data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    load_data_module,
)
from scripts.benchmarks.car_cfd._models import (  # noqa: E402
    GNO_ARCHITECTURES,
    build,
)


class _NoOpScheduler:
    """StepLR is applied by :func:`train_one` itself (``model.set_lr``);
    the Trainer's own scheduler hook is disabled."""

    def step(self, metric=None):
        return None


def build_trainer(model, processor, n_epochs: int) -> Trainer:
    trainer = Trainer(
        model=model,
        n_epochs=n_epochs,
        data_processor=processor,
        device="cuda",
        wandb_log=False,
        verbose=False,
    )
    trainer.regularizer = None
    trainer.scheduler = _NoOpScheduler()
    trainer.optimizer = model
    return trainer


def train_one(
    run: RunConfig,
    data_root: Path,
    run_dir: Path,
    n_train: int,
    n_test: int,
    ckpt_every: int,
    download: bool = True,
    seed: int = PROTOCOL["seed"],
    n_val: int = 0,
) -> dict:
    """Trains ``run`` and saves its checkpoint to ``run_dir``.

    ``seed`` seeds the initialisation and the shuffle. ``n_val > 0`` scores
    the last ``n_val`` training cars instead of the test set (see the module
    docstring); the model then trains on ``n_train - n_val`` cars.

    Returns the ``training`` summary stored in the checkpoint's config.
    """
    tag = run.name
    config = build_config(run, n_train, n_test)
    if run.architecture in GNO_ARCHITECTURES:
        wno = " ".join(f"{k}={v}" for k, v in run.model.items() if k.startswith("wno_"))
        arch = (
            f"out_gno_radius={config.model.get('out_gno_radius', 0.033)} "
            f"use_in_gno={config.model.use_in_gno} "
            f"factorization={config.model.fno_factorization} "
            f"rank={config.model.fno_rank} norm={config.model.fno_norm} "
            f"hidden={config.model.fno_hidden_channels} {wno}"
        )
    else:
        arch = " ".join(f"{k}={v}" for k, v in run.model.items())
    if run.schedule is not None:
        schedule = f"schedule={run.schedule}"
    else:
        schedule = f"step={run.lr_step} gamma={run.lr_gamma}"
    if run.batch_size != 1 and run.architecture in GNO_ARCHITECTURES:
        raise ValueError(f"run {tag!r}: GINO's neighbour lists need batch_size=1")
    print(
        f"[{tag}] {run.architecture} query_res={run.sdf_query_resolution} {arch} "
        f"lr={run.learning_rate} {schedule} "
        f"wd={run.weight_decay} batch={run.batch_size} clip={run.grad_clip} "
        f"b2={run.adam_b2} eps={run.adam_eps} epochs={run.n_epochs}",
        flush=True,
    )

    # The loader's shuffle uses Python's global RNG.
    random.seed(seed)
    data_module = load_data_module(
        data_root, run.sdf_query_resolution, n_train, n_test, download=download
    )
    if n_val:
        cars = data_module.train_data.data_list
        data_module.train_data.data_list = cars[:-n_val]
        data_module.test_data.data_list = cars[-n_val:]
        print(
            f"[{tag}] validation: training on {len(cars) - n_val} cars, "
            f"scoring the held-out last {n_val} (not the test set)",
            flush=True,
        )
    model, processor = build(
        run, config, data_module, run_dir / "neighbor_logs", seed=seed
    )

    train_loader = data_module.train_loader(batch_size=run.batch_size, shuffle=True)
    test_loader = data_module.test_loader(batch_size=1, shuffle=False)
    trainer = build_trainer(model, processor, run.n_epochs)
    l2loss = LpLoss(d=2, p=2)

    history: list[dict] = []
    t_start = perf_counter()
    for epoch in range(run.n_epochs):
        if run.schedule is None:
            lr = lr_at(run, epoch)
            # The optimiser state only exists after the first step, and at
            # epoch 0 the StepLR factor is 1 anyway.
            if model.initialized:
                model.set_lr(lr)
        # avg_loss is per car; the Trainer's train_err is per batch, and
        # LpLoss sums over the batch. The two agree at batch_size=1.
        _, train_err, _, epoch_time = trainer.train_one_epoch(
            epoch, train_loader, l2loss
        )
        if run.schedule is not None:
            # The per-step schedule's value at the epoch's last step.
            lr = float(model.opt_state.hyperparams["learning_rate"])
        if epoch == 0:
            print(f"[{tag}] params={count_params(model.params):,}", flush=True)
        t0 = perf_counter()
        eval_metrics = trainer.evaluate_all(
            epoch=epoch,
            eval_losses={"l2": l2loss},
            test_loaders={"test": test_loader},
            eval_modes={},
        )
        test_err = float(eval_metrics["test_l2"])
        history.append(
            dict(
                epoch=epoch,
                train_err=float(train_err),
                test_err=test_err,
                lr=lr,
                secs=float(epoch_time),
                test_secs=perf_counter() - t0,
            )
        )
        print(
            f"[{tag}] epoch {epoch:3d}  train={train_err:.4e}  "
            f"test_l2={test_err:.4e}  lr={lr:.3e}  {epoch_time:.1f}s",
            flush=True,
        )
        if ckpt_every and (epoch + 1) % ckpt_every == 0:
            save_progress(run_dir, model.params, history)
            print(f"[{tag}] [ckpt] epoch {epoch} -> {run_dir}", flush=True)

    best = min(history, key=lambda h: h["test_err"])
    training = {
        **PROTOCOL,
        "seed": seed,
        "n_val": n_val,
        "n_epochs": run.n_epochs,
        "n_train": n_train,
        "n_test": n_test,
        "train_seconds": perf_counter() - t_start,
        "final_test_err": history[-1]["test_err"],
        "best_test_err": best["test_err"],
        "best_epoch": best["epoch"],
    }
    save_checkpoint(run_dir, run, model.params, history, training)
    print(
        f"[{tag}] final test {history[-1]['test_err']:.4%}  "
        f"best {best['test_err']:.4%} (epoch {best['epoch']}); "
        f"{training['train_seconds'] / 60:.1f} min -> {run_dir}",
        flush=True,
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
        help="one or more named runs to train (space-separated), or 'all' for "
        f"every run. infer.py needs paper, transolver_val and giwo_val. "
        f"Choices: {', '.join(RUNS)}",
    )
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="use the data already under --data-root (e.g. a read-only mount)",
    )
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--n-train", type=int, default=PROTOCOL["n_train"])
    ap.add_argument("--n-test", type=int, default=PROTOCOL["n_test"])
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch budget (for quick checks)",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=10,
        help="save weights + history every N epochs (0 disables)",
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
        "--seed",
        type=int,
        default=PROTOCOL["seed"],
        help="initialisation + shuffle seed; non-zero saves to <run>_seed<N>",
    )
    ap.add_argument(
        "--val",
        type=int,
        default=0,
        metavar="N",
        help="hold out the last N training cars and score them instead of "
        "the test set; saves to <run>_val<N>",
    )
    args = ap.parse_args()

    require_gpu(args.allow_cpu)

    names = list(RUNS) if args.run == ["all"] else args.run
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        ap.error(f"unknown run(s) {unknown}; choose from {list(RUNS)}")

    for i, name in enumerate(names, 1):
        run = RUNS[name]
        if args.epochs is not None:
            run = replace(run, n_epochs=args.epochs)
        run_dir = args.ckpt_root / name
        if args.val:
            run_dir = run_dir.with_name(f"{run_dir.name}_val{args.val}")
        if args.seed != PROTOCOL["seed"]:
            run_dir = run_dir.with_name(f"{run_dir.name}_seed{args.seed}")
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(names)}] {name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(names)}] {name}")
        train_one(
            run,
            args.data_root,
            run_dir,
            args.n_train,
            args.n_test,
            args.ckpt_every,
            download=not args.no_download,
            seed=args.seed,
            n_val=args.val,
        )


if __name__ == "__main__":
    main()
