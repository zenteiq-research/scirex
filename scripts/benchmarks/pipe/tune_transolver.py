# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Hyperparameter search for Transolver on the Pipe benchmark.

Ray Tune with Optuna's TPE sampler over Transolver's optimiser settings only.
The architecture stays at the paper's hyperparameters (n_hidden=128,
n_layers=8, n_heads=8, mlp_ratio=2, slice_num=64, ref=8), so every trial has
the same 3,073,985 parameters and only the training recipe changes.

Searched: learning rate, weight decay, gradient-norm clip, the three
cosine one-cycle schedule settings, and batch size. Dropout is not
searched: ``_transolver.py`` never calls the
model with ``deterministic=False``, so a dropout rate there has no effect.

Each trial runs ``_transolver.py`` unchanged, as a
subprocess that preallocates its own share of the GPU (a JAX process cannot
release device memory mid-run, so trials are isolated per process). Trials
use the GPUs in ``CUDA_VISIBLE_DEVICES`` at launch (default: GPU 0). Trials are
ranked on the lowest test relative L2 within a short proxy budget; the
benchmark has no validation split, so the chosen setting should be confirmed
with a full-length run and reported at its last epoch.

Requires the ``tune`` extra (``pip install -e ".[tune]"``).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/pipe/tune_transolver.py \\
        --data-path /path/to/pipe --study-dir /path/to/study
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = Path(__file__).resolve().parent / "_transolver.py"
SEED = 0

# The paper's launch-script architecture (Transolver_Pipe.sh in the
# reference repo), held fixed across trials.
FIXED_ARCH = {
    "n-hidden": 128,
    "n-layers": 8,
    "n-heads": 8,
    "mlp_ratio": 2,
    "slice_num": 64,
    "ref": 8,
    "unified_pos": 0,
}


def build_search_space() -> dict:
    return {
        "lr": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "max_grad_norm": tune.uniform(0.05, 1.0),
        "pct_start": tune.uniform(0.1, 0.4),
        "div_factor": tune.loguniform(5.0, 100.0),
        "final_div_factor": tune.loguniform(1e2, 1e5),
        "batch_size": tune.choice([4, 8, 16]),
    }


def trial_command(config: dict, args: argparse.Namespace, out_dir: Path) -> list:
    """The training command for one trial."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data_path",
        str(args.data_path),
        "--epochs",
        str(args.proxy_epochs),
        "--seed",
        str(SEED),
        "--output-dir",
        str(out_dir),
        "--ckpt-dir",
        str(out_dir / "checkpoints"),
        "--lr",
        str(config["lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--max_grad_norm",
        str(config["max_grad_norm"]),
        "--pct-start",
        str(config["pct_start"]),
        "--div-factor",
        str(config["div_factor"]),
        "--final-div-factor",
        str(config["final_div_factor"]),
        "--batch-size",
        str(config["batch_size"]),
    ]
    for flag, value in FIXED_ARCH.items():
        cmd += [f"--{flag}", str(value)]
    return cmd


def best_test_error(loss_curve: Path) -> tuple[float, int]:
    """Lowest test relative L2 in a trial's loss_curve.csv, and its epoch."""
    best, best_epoch = float("inf"), -1
    with open(loss_curve) as f:
        for row in csv.DictReader(f):
            if float(row["rel_err"]) < best:
                best, best_epoch = float(row["rel_err"]), int(row["epoch"])
    return best, best_epoch


def make_trainable(args: argparse.Namespace, visible_devices: str):
    def run_trial(config: dict) -> None:
        trial_id = tune.get_context().get_trial_id()
        out_dir = args.study_dir / "trials" / f"trial_{trial_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        # Ray may clear CUDA_VISIBLE_DEVICES in workers that requested no GPU,
        # so restore the devices the search was launched with.
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
        # Each trial preallocates its share of the GPU, which also caps it.
        # (The fraction is ignored with on-demand allocation, i.e. with
        # XLA_PYTHON_CLIENT_PREALLOCATE=false or ALLOCATOR=platform.)
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu_mem_fraction)

        t0 = time.time()
        with open(out_dir / "stdout.log", "w") as log:
            proc = subprocess.run(
                trial_command(config, args, out_dir),
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        loss_curve = out_dir / "loss_curve.csv"
        if proc.returncode == 0 and loss_curve.exists():
            rel_err, best_epoch = best_test_error(loss_curve)
        else:
            rel_err, best_epoch = float("inf"), -1
            print(f"[trial {trial_id}] failed (rc={proc.returncode}); see {out_dir}")
        tune.report(
            {
                "rel_err": rel_err,
                "best_epoch": best_epoch,
                "wall_sec": time.time() - t0,
            }
        )

    return run_trial


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--proxy-epochs", type=int, default=100)
    ap.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=0.22,
        help="share of the GPU each trial preallocates (4 trials x 0.22 = 0.88)",
    )
    args = ap.parse_args()
    args.study_dir = args.study_dir.resolve()
    args.study_dir.mkdir(parents=True, exist_ok=True)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    ray.init(num_cpus=2 * args.max_concurrent, num_gpus=0, include_dashboard=False)
    tuner = tune.Tuner(
        make_trainable(args, visible_devices),
        param_space=build_search_space(),
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric="rel_err", mode="min", seed=SEED),
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
        ),
        run_config=tune.RunConfig(
            name="transolver_optuna_search",
            storage_path=str(args.study_dir / "ray_results"),
        ),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="rel_err", mode="min")
    summary = {
        "best_config": best.config,
        "best_rel_err": best.metrics.get("rel_err"),
        "best_epoch": best.metrics.get("best_epoch"),
        "fixed_arch": FIXED_ARCH,
        "proxy_epochs": args.proxy_epochs,
        "num_samples": args.num_samples,
    }
    (args.study_dir / "best_config.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
