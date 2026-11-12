# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-Design-ShapeNetCar benchmark, step 1 of 2: train and save checkpoints.

Trains Transolver under the Transolver paper's own protocol for this
benchmark (``_config.PROTOCOL``, confirmed against the official
``Car-Design-ShapeNetCar/main.py``/``train.py``) and saves one checkpoint per
run. ``infer.py`` evaluates the saved checkpoints, including the paper's Table
3-style relative-L2 and drag-coefficient metrics.

Every car mesh has a different point count, so this trains one gradient step
per sample (``batch_size=1``, matching the reference) rather than through the
batched ``scirex.experimental.training.Trainer`` -- see this package's
``__init__.py``.

Usage (from the repo root)::

    python scripts/benchmarks/shapenetcar/train.py --run transolver \\
        --data-dir ~/data/mlcfd_data/training_data

    # Fast smoke-scale run against a small fixture:
    python scripts/benchmarks/shapenetcar/train.py --run transolver \\
        --data-dir /path/to/small/fixture --epochs 2

Runs that already have a complete checkpoint are skipped, so an interrupted
``--run all`` can be restarted.
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
from scripts.benchmarks.shapenetcar import _gino_geometry  # noqa: E402
from scripts.benchmarks.shapenetcar._checkpoint import (  # noqa: E402
    count_params,
    is_complete,
    save_checkpoint,
)
from scripts.benchmarks.shapenetcar._config import (  # noqa: E402
    DEFAULT_CKPT_ROOT,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.shapenetcar._data import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    MODEL_KWARGS,
    load_split,
)
from scripts.benchmarks.shapenetcar._models import build_model  # noqa: E402
from scripts.benchmarks.shapenetcar._physics import make_loss  # noqa: E402

# `scirex` (via FNO -> src/scirex/layers/blocks/spectral_convolution.py) pulls
# in tensorly's jax backend, which unconditionally forces
# jax_enable_x64=True as an import-time side effect (needed for tensorly's
# own numerics, not Transolver). That flag is process-global; restore
# float32 here, at the script that actually needs it -- see
# spectral_convolution.py's own comment on this.
jax.config.update("jax_enable_x64", False)


def _run_epoch(
    train_step, eval_step, params, opt_state, samples, rng, train: bool, tag: str
):
    order = rng.permutation(len(samples)) if train else np.arange(len(samples))
    press_losses, velo_losses = [], []
    n_total = len(order)
    t0 = perf_counter()
    for i, idx in enumerate(order):
        s = samples[idx]
        x = jnp.asarray(s["x"])[None, :, :]
        y = jnp.asarray(s["y"])
        surf = jnp.asarray(s["surf"])
        neighbors_in = s.get("neighbors_in")
        neighbors_out = s.get("neighbors_out")

        if train:
            params, opt_state, loss_press, loss_velo = train_step(
                params, opt_state, x, y, surf, neighbors_in, neighbors_out
            )
        else:
            loss_press, loss_velo = eval_step(
                params, x, y, surf, neighbors_in, neighbors_out
            )

        press_losses.append(loss_press)
        velo_losses.append(loss_velo)

        # Point count varies per sample, so jax.jit recompiles on every new
        # shape -- this can make individual samples within an epoch take far
        # longer than the eventual steady-state per-sample cost, especially
        # early on. Print periodic progress so that gap doesn't look like a
        # hang.
        if (i + 1) % 25 == 0 or (i + 1) == n_total:
            split = "train" if train else "val"
            elapsed = perf_counter() - t0
            print(f"[{tag}] {split} sample {i + 1}/{n_total} ({elapsed:.1f}s elapsed)")
            sys.stdout.flush()

    press_mean = float(jnp.mean(jnp.stack(press_losses)))
    velo_mean = float(jnp.mean(jnp.stack(velo_losses)))
    return params, opt_state, press_mean, velo_mean


def train_one(
    run: RunConfig,
    data_dir: Path,
    cache_dir: Path | None,
    fold_id: int,
    run_dir: Path,
    p: dict,
    anim_out: Path | None = None,
    anim_every: int = 1,
) -> dict:
    """Trains ``run`` and saves its checkpoint to ``run_dir``.

    Returns the ``training`` summary stored in the checkpoint's config.
    """
    tag = run.name
    data = load_split(data_dir, fold_id, cache_dir, use_cache=True)
    train_samples, val_samples = data["train"], data["val"]
    print(f"[{tag}] n_train={len(train_samples)}  n_val={len(val_samples)}")

    # GINO/GIWO's geometry is per-sample (see _gino_geometry.py's module
    # docstring) and needs a concrete-numpy KD-tree search that cannot run
    # inside a jitted forward pass, so it is computed once here rather than
    # per training step. Transolver's samples are left untouched (no
    # "neighbors_in"/"neighbors_out" keys), and _physics.make_loss's loss_fn
    # forwards them to model.apply only when present. train and val samples
    # are padded together (one call) so train_step/eval_step share one
    # compiled shape across both -- see attach_neighbors's own docstring.
    # GIWO (GNOBlock-in -> WNO -> GNOBlock-out) reads this same fixed latent
    # grid and the same two GNO radii GINO does, so it needs the identical
    # geometry call -- see _models.GIWOCarWrapper.
    if run.model in ("gino", "giwo"):
        _gino_geometry.attach_neighbors(
            train_samples + val_samples,
            run.arch["latent_grid"],
            run.arch["in_gno_radius"],
            run.arch["out_gno_radius"],
        )

    model = build_model(run.model, run.arch, MODEL_KWARGS)
    x0 = jnp.asarray(train_samples[0]["x"])[None, :, :]
    init_kwargs = {}
    if run.model in ("gino", "giwo"):
        init_kwargs = dict(
            neighbors_in=train_samples[0]["neighbors_in"],
            neighbors_out=train_samples[0]["neighbors_out"],
        )
    params = model.init(jax.random.PRNGKey(p["seed"]), x=x0, fx=None, **init_kwargs)
    print(f"[{tag}] {run.model} {run.arch}  params={count_params(params):,}")

    # Fixed held-out sample for the training-progress animation (same held-out
    # sample throughout, matching infer.py's own val_samples[0] convention) --
    # captured as surface pressure only (the "Surf" metric this benchmark
    # reports), since that's the field this benchmark's own contour.png
    # visualizes.
    anim_sample0 = anim_x0 = anim_truth = anim_pos = None
    pred_frames: list[np.ndarray] = []
    pred_epochs: list[int] = []
    if anim_out is not None:
        anim_sample0 = val_samples[0]
        anim_x0 = jnp.asarray(anim_sample0["x"])[None, :, :]
        surf0 = anim_sample0["surf"]
        anim_pos = np.asarray(anim_sample0["pos"])[surf0]
        _, _, mean_out, std_out = data["coef_norm"]
        y0_denorm = np.asarray(anim_sample0["y"]) * std_out + mean_out
        anim_truth = y0_denorm[surf0, -1]
        anim_extra = {}
        if "neighbors_in" in anim_sample0:
            anim_extra = dict(
                neighbors_in=anim_sample0["neighbors_in"],
                neighbors_out=anim_sample0["neighbors_out"],
            )

    # Reference uses plain torch.optim.Adam with a one-cycle LR schedule.
    total_steps = (len(train_samples) + 1) * run.n_epochs
    lr_schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=p["learning_rate"],
        pct_start=p["pct_start"],
        div_factor=p["div_factor"],
        final_div_factor=p["final_div_factor"],
    )
    tx = optax.adam(learning_rate=lr_schedule)
    opt_state = tx.init(params)

    loss_fn = make_loss(model, p["press_weight"])

    @jax.jit
    def train_step(params, opt_state, x, y, surf, neighbors_in, neighbors_out):
        (_, (loss_press, loss_velo)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y, surf, neighbors_in, neighbors_out
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss_press, loss_velo

    @jax.jit
    def eval_step(params, x, y, surf, neighbors_in, neighbors_out):
        _, (loss_press, loss_velo) = loss_fn(
            params, x, y, surf, neighbors_in=neighbors_in, neighbors_out=neighbors_out
        )
        return loss_press, loss_velo

    rng = np.random.default_rng(p["seed"])
    history: list[dict] = []
    t_start = perf_counter()
    for epoch in range(run.n_epochs):
        t0 = perf_counter()
        params, opt_state, train_press, train_velo = _run_epoch(
            train_step,
            eval_step,
            params,
            opt_state,
            train_samples,
            rng,
            train=True,
            tag=tag,
        )
        train_loss = train_velo + p["press_weight"] * train_press
        epoch_time = perf_counter() - t0

        val_t0 = perf_counter()
        params, opt_state, val_press, val_velo = _run_epoch(
            train_step,
            eval_step,
            params,
            opt_state,
            val_samples,
            rng,
            train=False,
            tag=tag,
        )
        val_time = perf_counter() - val_t0
        val_loss = val_velo + p["press_weight"] * val_press

        if anim_out is not None and epoch % anim_every == 0:
            out0 = np.asarray(model.apply(params, x=anim_x0, fx=None, **anim_extra)[0])
            out0_denorm = out0 * std_out + mean_out
            pred_frames.append(out0_denorm[surf0, -1])
            pred_epochs.append(epoch)

        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                train_press_mse=train_press,
                train_velo_mse=train_velo,
                secs=epoch_time,
                val_loss=val_loss,
                val_press_mse=val_press,
                val_velo_mse=val_velo,
                val_secs=val_time,
            )
        )
        print(
            f"[{tag}] epoch {epoch}: train_loss={train_loss:.6f} "
            f"train_press_mse={train_press:.6f} train_velo_mse={train_velo:.6f} "
            f"epoch_time_sec={epoch_time:.6f} val_loss={val_loss:.6f} "
            f"val_press_mse={val_press:.6f} val_velo_mse={val_velo:.6f} "
            f"val_time_sec={val_time:.6f}"
        )
        sys.stdout.flush()

    training = {
        **p,
        "n_epochs": run.n_epochs,  # overrides p["n_epochs"]: the run's actual budget
        "train_seconds": perf_counter() - t_start,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "fold_id": fold_id,
        "best_val_loss": history[-1]["val_loss"] if history else None,
    }
    save_checkpoint(run_dir, run, params, data["coef_norm"], history, training)
    mins = training["train_seconds"] / 60
    print(
        f"[{tag}] final val loss {history[-1]['val_loss']:.5e}; "
        f"{len(history)} epochs in {mins:.1f} min -> {run_dir}"
    )

    if anim_out is not None and pred_frames:
        from scripts.benchmarks._animate import save_point_cloud_evolution_gif_3d

        anim_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            anim_out.with_name("frames.npz"),
            pred_frames=np.stack(pred_frames),
            truth_frame=anim_truth,
            epochs=np.asarray(pred_epochs),
            pos=anim_pos,
        )
        save_point_cloud_evolution_gif_3d(
            anim_pos,
            pred_frames,
            anim_truth,
            pred_epochs,
            anim_out,
            title="Car-Design-ShapeNetCar",
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
        help=f"named run(s) to train, or 'all'. Choices: {', '.join(RUNS)}",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--fold-id", type=int, default=PROTOCOL["fold_id"])
    ap.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epoch budget (for quick checks)",
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
        help="capture a fixed held-out sample's surface pressure every "
        "--anim-every (or --anim-frames-derived) epochs and write an "
        "evolution.gif under --anim-root",
    )
    ap.add_argument(
        "--anim-every",
        type=int,
        default=1,
        help="... ignored if --anim-frames is given",
    )
    ap.add_argument(
        "--anim-frames",
        type=int,
        default=None,
        help="target total frame count (with --animate): computes the "
        "capture cadence as n_epochs // anim-frames, overriding --anim-every",
    )
    ap.add_argument(
        "--anim-root",
        type=Path,
        default=_REPO_ROOT / "animations" / "shapenetcar",
        help="directory --animate writes evolution.gif (+ frames.npz cache) into",
    )
    args = ap.parse_args()

    require_gpu(args.allow_cpu)

    names = list(RUNS) if args.run == ["all"] else args.run
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        ap.error(f"unknown run(s) {unknown}; choose from {list(RUNS)}")

    data_dir = args.data_dir.expanduser()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir else None

    for i, name in enumerate(names, 1):
        run = RUNS[name]
        if args.epochs is not None:
            run = RunConfig(**{**run.__dict__, "n_epochs": args.epochs})
        run_dir = args.ckpt_root / name
        if is_complete(run_dir) and not args.overwrite:
            print(f"=== [{i}/{len(names)}] {run.name}: checkpoint exists, skipping")
            continue
        print(f"=== [{i}/{len(names)}] {run.name}")
        anim_out = args.anim_root / "evolution.gif" if args.animate else None
        anim_every = args.anim_every
        if args.animate and args.anim_frames is not None:
            anim_every = max(1, run.n_epochs // args.anim_frames)
        train_one(
            run,
            data_dir,
            cache_dir,
            args.fold_id,
            run_dir,
            PROTOCOL,
            anim_out=anim_out,
            anim_every=anim_every,
        )


if __name__ == "__main__":
    main()
