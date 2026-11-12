# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""AirfRANS benchmark, step 1 of 2: train and save checkpoints.

Trains the runs in ``_config.RUNS`` and saves one checkpoint per run, holding
the EMA of the weights after the last epoch. The validation loss on the 80
official validation cases is logged every ``val_every`` epochs for the error
curve only; it does not choose which weights are saved.

Recipe (``_config.PROTOCOL``), official Transolver AirfRANS unless marked +::

    data         720 train / 80 validation / 200 test cases (manifest.json)
    input        7 official node features + log10 wall distance + log10 Re (+)
    target       (Ux, Uy)/U_inf, p/U_inf^2, nut/U_inf, normalised (+)
    sampling     32,000 random nodes per case per step, batch size 1
    loss         volume MSE + surface MSE (normalised, mean over channels)
    optimiser    Adam, OneCycleLR (peak 1e-3, pct 0.3, div 25, final div 1e4,
                 beta1 0.95 -> 0.85 -> 0.95)
    stability    EMA of the weights, decay 0.9995 (+); gradient clip at
                 global norm 10 (+)
    precision    FP32 matmuls (+; XLA's default on this GPU is TF32)
    epochs       500 (+; official 398), seed 0

Usage (from the repo root)::

    export AIRFRANS_DATA_DIR=/path/to/AirfRANS/Dataset   # manifest.json, case folders
    python scripts/benchmarks/airfrans/train.py --run transolver
    python scripts/benchmarks/airfrans/train.py --run gino  # GINO's own _gino.py

The first run reads every case's VTU/VTP files once and caches the node
features under ``AIRFRANS_CACHE_DIR`` (about 4 minutes). Runs that already
have a complete checkpoint are skipped unless ``--overwrite`` is given.
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

from scripts.benchmarks._device import require_gpu  # noqa: E402
from scripts.benchmarks.airfrans import _gino  # noqa: E402
from scripts.benchmarks.airfrans._checkpoint import (  # noqa: E402
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.airfrans._config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    GINO_ARGS,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.airfrans._data import (  # noqa: E402
    load_case,
    load_split,
    model_stats,
    official_output_stats,
    prepare_case,
)
from scripts.benchmarks.airfrans._models import get_spec  # noqa: E402

# The scirex package can switch x64 on at import; float32 is intended.
jax.config.update("jax_enable_x64", False)


def onecycle(total_steps: int, p: dict):
    """``torch.optim.lr_scheduler.OneCycleLR`` (two phases, cosine), as two
    functions of the 0-based optimiser step: learning rate and Adam's beta1."""
    init_lr = p["lr"] / p["div_factor"]
    min_lr = init_lr / p["final_div_factor"]
    end1, end2 = float(p["pct_start"] * total_steps) - 1, float(total_steps - 1)

    def cos(a, b, t):
        return b + (a - b) / 2.0 * (jnp.cos(jnp.pi * t) + 1)

    def lr(k):
        k = jnp.asarray(k, jnp.float32)
        return jnp.where(
            k <= end1,
            cos(init_lr, p["lr"], k / end1),
            cos(p["lr"], min_lr, (k - end1) / (end2 - end1)),
        )

    def beta1(k):
        k = jnp.asarray(k, jnp.float32)
        lo, hi = p["base_momentum"], p["max_momentum"]
        return jnp.where(
            k <= end1, cos(hi, lo, k / end1), cos(lo, hi, (k - end1) / (end2 - end1))
        )

    return lr, beta1


def make_optimizer() -> optax.GradientTransformation:
    """Adam with the learning rate and beta1 injected at every step. Adam in
    the complex-aware form would only matter for complex weights; the runs
    here have none."""
    return optax.inject_hyperparams(optax.adam)(
        learning_rate=1e-3, b1=0.95, b2=0.999, eps=1e-8
    )


def region_losses(out: jnp.ndarray, y: jnp.ndarray, surf: jnp.ndarray):
    """Official ``MSE_weighted`` terms: per-channel MSE on non-wall and on wall
    nodes, each averaged over the 4 channels."""
    se = (out - y) ** 2
    w = surf[:, None].astype(jnp.float32)
    loss_surf = jnp.mean(jnp.sum(se * w, 0) / jnp.maximum(w.sum(), 1.0))
    loss_vol = jnp.mean(jnp.sum(se * (1 - w), 0) / jnp.maximum((1 - w).sum(), 1.0))
    return loss_vol, loss_surf


def train_one(
    run: RunConfig,
    split,
    data_dir: Path,
    cache_dir: Path,
    run_dir: Path,
    p: dict,
    max_train: int | None = None,
) -> dict:
    """Trains ``run`` and saves it. Returns the ``training`` summary."""
    spec = get_spec(run.model)
    train_ids, val_ids = (
        split.train[:max_train],
        split.val[: (8 if max_train else None)],
    )
    raw_train = [load_case(data_dir, cache_dir, c) for c in train_ids]
    stats = model_stats(raw_train)
    official_out = official_output_stats(raw_train)

    def prepare(raw):
        d = prepare_case(raw, stats)
        d["extra"] = spec.extra(run.arch, raw)
        return d

    TR = [prepare(r) for r in raw_train]
    del raw_train
    VA = [prepare(load_case(data_dir, cache_dir, c)) for c in val_ids]

    model = spec.build(run.arch)
    rng = np.random.default_rng(p["seed"])
    n_sub = p["subsample"]

    def sample(d):
        idx = rng.choice(len(d["x"]), min(n_sub, len(d["x"])), replace=False)
        return (
            jnp.asarray(d["x"][idx]),
            jnp.asarray(d["pos"][idx]),
            d["extra"],
            jnp.asarray(d["y"][idx]),
            jnp.asarray(d["surf"][idx]),
        )

    x0, pos0, extra0, _, _ = sample(TR[0])
    params = spec.init(model, jax.random.PRNGKey(p["seed"]), x0, pos0, extra0)
    ema = params
    total_steps = (len(TR) // p["batch_size"] + 1) * run.epochs  # as torch OneCycleLR
    lr_f, b1_f = onecycle(total_steps, p)
    tx = make_optimizer()
    opt = tx.init(params)
    add = p["additions"]
    decay, clip = add["ema_decay"], add["grad_clip_global_norm"]
    surf_w = p["surf_weight"]

    def loss_fn(params, x, pos, extra, y, surf):
        lv, ls = region_losses(spec.apply(model, params, x, pos, extra), y, surf)
        return lv + surf_w * ls, (lv, ls)

    @jax.jit
    def step(params, ema, opt, x, pos, extra, y, surf, k):
        (_, (lv, ls)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, pos, extra, y, surf
        )
        norm = optax.global_norm(grads)
        grads = jax.tree_util.tree_map(
            lambda g: g * jnp.minimum(1.0, clip / (norm + 1e-12)), grads
        )
        opt.hyperparams["learning_rate"] = lr_f(k)
        opt.hyperparams["b1"] = b1_f(k)
        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)
        ema = jax.tree_util.tree_map(
            lambda e, q: decay * e + (1 - decay) * q, ema, params
        )
        return params, ema, opt, lv, ls, norm

    @jax.jit
    def evaluate(params, x, pos, extra, y, surf):
        return loss_fn(params, x, pos, extra, y, surf)[1]

    history: list[dict] = []
    k = 0
    t_start = perf_counter()
    for epoch in range(run.epochs):
        t0 = perf_counter()
        lvs, lss, norms = [], [], []
        for i in rng.permutation(len(TR)):
            params, ema, opt, lv, ls, gn = step(
                params, ema, opt, *sample(TR[i]), jnp.asarray(k, jnp.int32)
            )
            k += 1
            lvs.append(lv)
            lss.append(ls)
            norms.append(gn)
        row = dict(
            epoch=epoch,
            train_vol=float(jnp.mean(jnp.stack(lvs))),
            train_surf=float(jnp.mean(jnp.stack(lss))),
            grad_norm_median=float(jnp.median(jnp.stack(norms))),
            lr=float(lr_f(k)),
            secs=perf_counter() - t0,
        )
        if epoch % p["val_every"] == p["val_every"] - 1 or epoch in (0, run.epochs - 1):
            v0 = perf_counter()
            vals = [
                [float(v) for v in evaluate(ema, *sample(d))]
                for _ in range(p["val_repeats"])
                for d in VA
            ]
            row["val_vol"], row["val_surf"] = (float(v) for v in np.mean(vals, 0))
            row["val_secs"] = perf_counter() - v0
        history.append(row)
        if epoch % 10 == 0 or epoch == run.epochs - 1:
            msg = f"train vol {row['train_vol']:.5f} surf {row['train_surf']:.5f}"
            if "val_vol" in row:
                msg += (
                    f"  val (EMA) vol {row['val_vol']:.5f} surf {row['val_surf']:.5f}"
                )
            print(
                f"[{run.name}] epoch {epoch:3d}  {msg}  {row['secs']:.1f}s", flush=True
            )

    training = dict(
        **{key: v for key, v in p.items() if key != "additions"},
        additions=add,
        epochs_run=run.epochs,
        steps=k,
        train_cases=len(TR),
        val_cases=len(VA),
        train_seconds=perf_counter() - t_start,
        weights="EMA of the weights after the last epoch",
    )
    save_checkpoint(run_dir, run, ema, stats, official_out, history, training)
    print(f"[{run.name}] {training['train_seconds'] / 3600:.2f} h -> {run_dir}")
    return training


CHOICES = (*RUNS, "gino")


def train_gino(args: argparse.Namespace) -> None:
    """Runs GINO's own trainer, ``_gino.py``, with the published settings,
    saving to ``<ckpt_root>/gino``. GINO follows the recipe the Transolver
    paper applies to its baselines, not :data:`PROTOCOL`; see ``_gino.py``."""
    run_dir = args.ckpt_root / "gino"
    epochs = args.epochs or int(GINO_ARGS[GINO_ARGS.index("--epochs") + 1])
    log = run_dir / "loss_curve.csv"  # one row per finished epoch
    if (
        (run_dir / "params_final.msgpack").exists()
        and log.exists()
        and len(log.read_text().splitlines()) - 1 >= epochs
        and not args.overwrite
    ):
        print(f"[gino] checkpoint exists at {run_dir}, skipped (use --overwrite)")
        return
    _gino.main(
        [
            *GINO_ARGS,
            "--epochs",
            str(epochs),
            "--data-dir",
            str(args.data_dir),
            "--cache-dir",
            str(args.cache_dir / "gino"),
            "--output-dir",
            str(run_dir),
        ]
    )


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
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch budget (quick checks only)",
    )
    ap.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="use the first N training cases (quick checks only)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is an error",
    )
    args = ap.parse_args()
    require_gpu(args.allow_cpu)
    jax.config.update(
        "jax_default_matmul_precision", PROTOCOL["additions"]["matmul_precision"]
    )
    names = list(CHOICES) if args.run == ["all"] else args.run
    unknown = [n for n in names if n not in CHOICES]
    if unknown:
        raise SystemExit(
            f"unknown run(s) {unknown}; choose from {', '.join(CHOICES)} or 'all'"
        )
    split = load_split(args.data_dir, PROTOCOL["n_val"])
    for name in names:
        if name == "gino":
            train_gino(args)
            continue
        run = RUNS[name]
        if args.epochs is not None:
            run = RunConfig(run.name, run.model, run.arch, args.epochs)
        run_dir = args.ckpt_root / name
        if is_complete(run_dir) and not args.overwrite:
            print(f"[{name}] checkpoint exists at {run_dir}, skipped (use --overwrite)")
            continue
        train_one(
            run, split, args.data_dir, args.cache_dir, run_dir, PROTOCOL, args.max_train
        )


if __name__ == "__main__":
    main()
