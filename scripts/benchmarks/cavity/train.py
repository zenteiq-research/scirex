# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Lid-driven cavity benchmark, step 1 of 2: train and save checkpoints.

Trains FNO, WNO, GINO and Transolver under the protocol of Costa Rocha et al.
2023 (Appl. Sci. 13(5), 3165, https://doi.org/10.3390/app13053165) and saves
one checkpoint per run. The test case (Re = 10,000) is never loaded here;
``infer.py`` evaluates the saved checkpoints on it.

Protocol (the paper's Sections 2.2 to 2.4, see ``_config.PROTOCOL``)::

    optimiser      Adam, learning rate 1e-3
    lr schedule    halve after 10 epochs without validation improvement
    early stopping after 50 epochs without validation improvement
    epochs         up to 300 (50 for the architecture search)
    loss           MSE + 0.1 * mass-conservation term
    selection      the weights with the lowest validation loss are saved

Usage (from the repo root)::

    # One named run, or all of them one after another
    python scripts/benchmarks/cavity/train.py --run fno_3to1
    python scripts/benchmarks/cavity/train.py --run all

    # The paper's architecture search: 10 random sizes x 50 epochs per model
    python scripts/benchmarks/cavity/train.py --search all

Runs that already have a complete checkpoint are skipped, so an interrupted
``--run all`` can be restarted. The raw CSVs are read from ``--data-dir``
(default: the ``CAVITY_DATA_DIR`` environment variable) and converted once to
``--cache-dir``.
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
from scripts.benchmarks.cavity._checkpoint import (  # noqa: E402
    count_params,
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.cavity._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    DEFAULT_DATA_DIR,
    MODELS,
    PROTOCOL,
    RUNS,
    SEARCH,
    RunConfig,
    sample_search_configs,
)
from scripts.benchmarks.cavity._data import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    TRAIN_CASES,
    build_train_val,
    denormalise,
    ensure_prepared,
)
from scripts.benchmarks.cavity._models import build_model  # noqa: E402
from scripts.benchmarks.cavity._physics import make_loss  # noqa: E402


def train_one(
    run: RunConfig,
    cache_dir: Path,
    run_dir: Path,
    protocol: dict,
    anim_out: Path | None = None,
    anim_every: int = 1,
) -> dict:
    """Trains ``run`` and saves its checkpoint to ``run_dir``.

    When ``anim_out`` is given, also captures the model's prediction on one
    fixed held-out validation sample every ``anim_every`` epochs and writes
    a prediction/truth/abs-error evolution GIF there (see
    ``scripts/benchmarks/_animate.py``) -- a training diagnostic, not part
    of the checkpoint itself.

    Returns the ``training`` summary stored in the checkpoint's config.
    """
    p = protocol
    tag = run.name
    data = build_train_val(
        cache_dir, run.n_in, run.n_out, val_frac=p["val_frac"], seed=p["seed"]
    )
    mean, std = data["mean"], data["std"]
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_va, Y_va = data["X_val"], data["Y_val"]
    print(f"[{tag}] train={X_tr.shape} val={X_va.shape}  mean={mean} std={std}")

    model = build_model(run.model, n_out=run.n_out, **run.arch)
    params = model.init(
        jax.random.PRNGKey(p["seed"]), jnp.asarray(X_tr[: p["batch_size"]])
    )
    print(f"[{tag}] {run.model} {run.arch}  params={count_params(params):,}")

    tx = optax.inject_hyperparams(optax.adam)(learning_rate=p["lr"])
    opt_state = tx.init(params)
    loss_fn = make_loss(model, data["normalizer"], p["mc_weight"])

    @jax.jit
    def train_step(params, opt_state, xb, yb):
        (loss, (data_loss, mc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xb, yb
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, data_loss, mc

    @jax.jit
    def val_loss_fn(params, xb, yb):
        return loss_fn(params, xb, yb)[0]

    bs = p["batch_size"]
    n_train = X_tr.shape[0]
    rng = np.random.default_rng(p["seed"])
    best_val, best_epoch, best_params = np.inf, 0, params
    since_improve = since_lr_drop = 0
    lr = p["lr"]
    history: list[dict] = []
    t_start = perf_counter()

    anim_x0 = anim_truth = None
    pred_frames: list[np.ndarray] = []
    pred_epochs: list[int] = []
    if anim_out is not None:
        anim_x0 = jnp.asarray(X_va[:1])
        anim_truth = np.asarray(denormalise(Y_va[:1], data["normalizer"]))[0, 0]

    for epoch in range(1, run.epochs + 1):
        t0 = perf_counter()
        perm = rng.permutation(n_train)
        tot = tot_d = tot_mc = 0.0
        nb = 0
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            params, opt_state, loss, dl, mc = train_step(
                params, opt_state, jnp.asarray(X_tr[idx]), jnp.asarray(Y_tr[idx])
            )
            tot += float(loss)
            tot_d += float(dl)
            tot_mc += float(mc)
            nb += 1

        vt, vnb = 0.0, 0
        for i in range(0, X_va.shape[0], bs):
            vt += float(
                val_loss_fn(
                    params, jnp.asarray(X_va[i : i + bs]), jnp.asarray(Y_va[i : i + bs])
                )
            )
            vnb += 1
        train_loss, val_loss = tot / nb, vt / vnb

        if anim_out is not None and epoch % anim_every == 0:
            pred0 = np.asarray(model.apply(params, anim_x0))
            pred_frames.append(np.asarray(denormalise(pred0, data["normalizer"]))[0, 0])
            pred_epochs.append(epoch)

        if val_loss < best_val - 1e-9:
            best_val, best_epoch = val_loss, epoch
            best_params = jax.tree_util.tree_map(lambda a: a.copy(), params)
            since_improve = since_lr_drop = 0
        else:
            since_improve += 1
            since_lr_drop += 1

        # ReduceLROnPlateau(factor, patience)
        if since_lr_drop >= p["lr_patience"]:
            lr *= p["lr_factor"]
            opt_state.hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
            since_lr_drop = 0
            print(f"[{tag}] epoch {epoch}: lr -> {lr:.3e}")

        secs = perf_counter() - t0
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=lr,
                data_loss=tot_d / nb,
                mc=tot_mc / nb,
                secs=secs,
            )
        )
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[{tag}] epoch {epoch:3d}  train={train_loss:.5e}  "
                f"val={val_loss:.5e}  lr={lr:.2e}  {secs:.1f}s"
            )
            sys.stdout.flush()

        if since_improve >= p["early_stop_patience"]:
            print(f"[{tag}] early stop at epoch {epoch}")
            break

    training = dict(
        **p,
        epochs_max=run.epochs,
        epochs_run=len(history),
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        train_seconds=perf_counter() - t_start,
        train_cases=list(TRAIN_CASES),
        n_train_windows=int(n_train),
        n_val_windows=int(X_va.shape[0]),
    )
    save_checkpoint(run_dir, run, best_params, mean, std, history, training)
    print(
        f"[{tag}] best val loss {best_val:.5e} at epoch {best_epoch}; "
        f"{len(history)} epochs in {training['train_seconds'] / 60:.1f} min "
        f"-> {run_dir}"
    )

    if anim_out is not None and pred_frames:
        from scripts.benchmarks._animate import save_prediction_evolution_gif

        # Cached alongside the GIF so a cosmetic-only re-render (title, fps,
        # colours) never needs a full retrain -- see _animate.py's
        # ``rebuild_from_cache``.
        np.savez(
            anim_out.with_name("frames.npz"),
            pred_frames=np.stack(pred_frames),
            truth_frame=anim_truth,
            epochs=np.asarray(pred_epochs),
        )
        save_prediction_evolution_gif(
            pred_frames,
            anim_truth,
            pred_epochs,
            anim_out,
            title="Lid-Driven Cavity",
        )

    return training


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--run",
        nargs="+",
        metavar="NAME",
        help=f"named run(s) to train, or 'all'. Choices: {', '.join(RUNS)}",
    )
    what.add_argument(
        "--search",
        nargs="+",
        metavar="MODEL",
        help=f"architecture search for these models, or 'all' ({', '.join(MODELS)})",
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
        "--animate",
        action="store_true",
        help="capture a prediction/truth/abs-error evolution GIF per run "
        "(see scripts/benchmarks/_animate.py); off by default since it adds "
        "one extra forward pass per captured epoch",
    )
    ap.add_argument(
        "--anim-every",
        type=int,
        default=1,
        help="capture a frame every this many epochs (with --animate); "
        "ignored if --anim-frames is given",
    )
    ap.add_argument(
        "--anim-frames",
        type=int,
        default=None,
        help="target total frame count (with --animate): computes the "
        "capture cadence as epochs // anim-frames, overriding --anim-every",
    )
    ap.add_argument(
        "--anim-root",
        type=Path,
        default=_REPO_ROOT / "animations" / "cavity",
        help="directory --animate writes evolution.gif (+ frames.npz cache) into",
    )
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit a CPU run; by default a missing GPU is a hard error",
    )
    args = ap.parse_args()

    require_gpu(args.allow_cpu)

    if args.run:
        names = list(RUNS) if args.run == ["all"] else args.run
        unknown = [n for n in names if n not in RUNS]
        if unknown:
            ap.error(f"unknown run(s) {unknown}; choose from {list(RUNS)}")
        runs = [(RUNS[n], args.ckpt_root / n) for n in names]
    else:
        models = list(MODELS) if args.search == ["all"] else args.search
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            ap.error(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
        runs = [
            (r, args.ckpt_root / "search" / r.name)
            for m in models
            for r in sample_search_configs(m, SEARCH["n_configs"], SEARCH["seed"])
        ]

    ensure_prepared(TRAIN_CASES, args.data_dir, args.cache_dir)

    for i, (run, run_dir) in enumerate(runs, 1):
        if args.epochs is not None:
            run = RunConfig(**{**run.__dict__, "epochs": args.epochs})
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(runs)}] {run.name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(runs)}] {run.name}")
        anim_out = args.anim_root / "evolution.gif" if args.animate else None
        anim_every = args.anim_every
        if args.animate and args.anim_frames is not None:
            anim_every = max(1, run.epochs // args.anim_frames)
        train_one(
            run,
            args.cache_dir,
            run_dir,
            PROTOCOL,
            anim_out=anim_out,
            anim_every=anim_every,
        )


if __name__ == "__main__":
    main()
