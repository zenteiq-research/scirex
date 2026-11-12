# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Darcy flow benchmark, step 2 of 2: evaluate the saved checkpoints.

Loads every run in ``_config.RUNS`` that has a checkpoint, predicts the test
set with its last-epoch weights and writes, for each,
``doc/assets/benchmarks/darcy_<run>/``:

* ``contour.png``: permeability, prediction, ground truth and absolute error
  on the first six test samples;
* ``error_curve.png`` and ``compute_cost.png``: the training history;
* ``metrics.json``: the run, its test relative L2 and its history.

When all four runs are present, ``darcy_comparison/`` gets the same three
figures of every model side by side. No training happens here.

Usage (from the repo root)::

    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth{1,2}.mat
    python scripts/benchmarks/darcy/infer.py
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._cli import (  # noqa: E402
    add_standard_args,
    checkpoint_dir,
    select_run,
)
from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks._plot_utils import (  # noqa: E402
    plot_compute_cost,
    plot_error_curves,
    plot_multi_instance_grid,
    save_metrics_json,
)
from scripts.benchmarks._provenance import attach  # noqa: E402
from scripts.benchmarks.darcy._checkpoint import (  # noqa: E402
    check_normalisation,
    is_complete,
    load_config,
    load_history,
    load_params_into,
)
from scripts.benchmarks.darcy._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    MODEL_LABELS,
    PAPER,
    PROTOCOL,
    RUNS,
)
from scripts.benchmarks.darcy._data import load_split, rel_l2  # noqa: E402
from scripts.benchmarks.darcy._models import prepare  # noqa: E402

N_CONTOUR_ROWS = 6

#: Each paper's name for the benchmark, used in the figure titles.
SOURCE = {
    "fno": "Li et al. Darcy",
    "wno": "Tripura & Chakraborty Darcy",
    "gino": "Darcy",
    "transolver": "Wu et al. Darcy",
}


def evaluate(run_dir: Path, data: dict) -> dict:
    """The run in ``run_dir``, its last-epoch test predictions ``(n, H * W)``
    and per-sample test relative L2 errors."""
    run, config = load_config(run_dir)
    prep = prepare(run.model, run.arch, data)
    check_normalisation(config["normalisation"], prep.normalisation, run.name)
    params = load_params_into(run_dir, prep.init(jax.random.PRNGKey(0)))
    pred = predict_in_batches(
        jax.jit(prep.predict),
        params,
        jnp.asarray(prep.test_x),
        run.opt["batch_size"],
    )
    n_test = data["test_u"].shape[0]
    truth = data["test_u"].reshape(n_test, -1)
    errs = np.array([rel_l2(pred[i], truth[i]) for i in range(n_test)])
    return dict(
        run=run,
        config=config,
        pred=np.asarray(pred),
        errs=errs,
        n_params=int(count_model_params(params)),
    )


def write_assets(
    res: dict, data: dict, history: dict, out: Path, run_dir: Path | None = None
) -> None:
    """``contour.png``, ``error_curve.png``, ``compute_cost.png`` and
    ``metrics.json`` of one run, into ``out``."""
    run, errs = res["run"], res["errs"]
    label, source = MODEL_LABELS[run.model], SOURCE[run.model]
    n_side = data["test_a"].shape[1]
    out.mkdir(parents=True, exist_ok=True)

    rows = min(N_CONTOUR_ROWS, len(errs))
    plot_multi_instance_grid(
        [data["test_a"][i] for i in range(rows)],
        [res["pred"][i].reshape(n_side, n_side) for i in range(rows)],
        [data["test_u"][i] for i in range(rows)],
        out / "contour.png",
        row_labels=[f"rel-L2={errs[i]:.3f}" for i in range(rows)],
        title=(
            "Data-driven predictions vs Li et al.'s real solved data, "
            f"held-out a(x) ({label}, s={n_side})"
        ),
    )
    if history.get("step"):
        loss = "rel-L2"
        if run.opt["grad_loss_weight"]:
            loss += f" + {run.opt['grad_loss_weight']}*grad-reg"
        plot_error_curves(
            history["step"],
            history["train_loss"],
            history["test_rel_l2"],
            out / "error_curve.png",
            title=(
                f"{source} (real s=421 data, downsampled to s=85): "
                f"{label} data-driven loss & rel-L2"
            ),
            train_label=f"train loss ({loss})",
        )
        plot_compute_cost(
            history["step"],
            history["secs_per_epoch"],
            out / "compute_cost.png",
            title=(
                f"{source} (real s=421 data, downsampled to s=85): "
                f"{label} data-driven per-epoch time"
            ),
        )

    training = res["config"]["training"]
    metrics = dict(
        problem=f"darcy_{run.model}",
        run=run.name,
        model=label,
        arch=run.arch,
        opt=run.opt,
        seed=run.seed,
        protocol=res["config"]["protocol"],
        n_params=res["n_params"],
        test_rel_l2_mean=float(errs.mean()),
        test_rel_l2_std=float(errs.std()),
        steps_run=training["steps_run"],
        train_seconds=training["train_seconds"],
        history=history,
        **({"paper": PAPER[run.model]} if run.model in PAPER else {}),
    )
    save_metrics_json(
        attach(metrics, [run_dir] if run_dir else []), out / "metrics.json"
    )


def write_comparison(out_dir: Path, models: list[str]) -> None:
    """``darcy_comparison/``: each figure of every model, side by side."""
    layouts = {
        "contour.png": (1, 4),
        "error_curve.png": (2, 2),
        "compute_cost.png": (2, 2),
    }
    for name, (n_rows, n_cols) in layouts.items():
        images = [plt.imread(out_dir / f"darcy_{m}" / name) for m in models]
        h, w = images[0].shape[:2]
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * w / 150, n_rows * (h / 150 + 0.4)),
            squeeze=False,
        )
        for ax, image, model in zip(axes.ravel(), images, models, strict=True):
            ax.imshow(image)
            ax.set_title(MODEL_LABELS[model], fontsize=18, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        path = out_dir / "darcy_comparison" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    add_standard_args(ap, out_flag="--out-dir", out_default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()
    only = select_run(ap, args, RUNS)

    require_gpu(args.allow_cpu)
    splits: dict[str, dict] = {}  # each checkpoint's own split, loaded once
    done = []
    for name in [only] if only else RUNS:
        run_dir = checkpoint_dir(args, name)
        if not is_complete(run_dir):
            print(f"[{name}] no checkpoint in {run_dir}, skipping")
            continue
        protocol = load_config(run_dir)[1].get("protocol", PROTOCOL)
        key = json.dumps(protocol, sort_keys=True)
        if key not in splits:
            splits[key] = load_split(args.data_dir, protocol)
        data = splits[key]
        res = evaluate(run_dir, data)
        write_assets(
            res,
            data,
            load_history(run_dir),
            args.out_dir / f"darcy_{res['run'].model}",
            run_dir,
        )
        done.append(res["run"].model)
        print(
            f"[{name}] test rel. L2 {res['errs'].mean():.5f} "
            f"({res['n_params']:,} params)"
        )
    if not done:
        raise SystemExit(
            "no usable checkpoint found (a Darcy run needs params and config.json, "
            f"under {args.checkpoint or args.ckpt_root}); nothing evaluated"
        )
    if set(done) == set(MODEL_LABELS):
        write_comparison(args.out_dir, list(MODEL_LABELS))
        print(f"[infer] wrote {args.out_dir / 'darcy_comparison'}")


if __name__ == "__main__":
    main()
