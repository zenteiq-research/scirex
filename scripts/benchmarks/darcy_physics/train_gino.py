# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only GINO on the exact Darcy benchmark from Li et al. (ICLR 2021) /
Wu et al. (Transolver, ICML 2024, Table 2): the label-free counterpart of
the data-driven benchmark (``darcy/``) at its parameter-matched size (#362).

Everything except the training signal is that script's protocol:

- Data: Li et al.'s ``piececonst_r421_N1024_smooth1.mat`` (first 1000
  instances, training) and ``smooth2.mat`` (first 200, test), strided to
  s=85, loaded with the same loader (``darcy._data.load_split``).
- Model: ``GINODarcy2d`` with an FNO latent of 31 channels, 5 layers and
  (16, 16) modes, input geometry, latent grid and output queries on the
  85 x 85 nodes, GNO radii ``0.16 * 16 / 84`` (in) and ``0.175 * 16 / 84``
  (out) -- 2,801,340 real parameters (complex weights counted as two),
  matched to Transolver's 2,826,945.
- Model input: the coefficient at the nodes with the same per-pixel
  normalizer, fitted on the training coefficients only (released plus any
  sampled extra fields; never on test data or solutions).
- Optimizer: complex-aware AdamW (``adamw_cx``), peak lr 1e-3, weight decay
  1e-4, cosine one-cycle schedule, global-norm gradient clipping 1.0,
  batch size 16, 124,000 steps at 1000 fields (2000 epochs; 124,800-125,000
  for the larger sets). No loss scale: the data-driven recipe is kept,
  including the clipping (see ``check_gino_physics_gradient_scale.py``).

The loss and its label-free choices are shared by every backbone
(``_ritz``):
the Q4 Ritz energy of ``-div(a grad u) = 1`` on the 85 x 85 benchmark nodes
with the majority element coefficient; the exact boundary condition
``u = (c N(a) + m) sin(pi x) sin(pi y)`` with ``c``, ``m`` from the Q4
torsion function over ``sqrt(a_lo a_hi)``; scoring at the released sample
points through the fixed cell-centre map; the finite-element contractions at
full float32 precision inside the script; and ``--extra-coefficients K``,
which adds the first ``K`` fields of the same fixed, seed-independent pool
of coefficients drawn from the data's own law (no solutions), so training
sets are nested. Lower ``--epochs`` with the set size to keep the step
budget (the one-cycle schedule follows the run length).
"""

import argparse
import hashlib
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import optax  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex import get_model_jax  # noqa: E402
from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    BatchedEnergyFunctionalLossQ4,
    build_grid_aligned_q4_mesh,
    kappa_fn,
    source_fn,
)
from scirex.training import adamw_cx  # noqa: E402
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import save_params  # noqa: E402
from scripts.benchmarks._plot_utils import (  # noqa: E402
    plot_compute_cost,
    plot_multi_instance_grid,
    save_metrics_json,
)
from scripts.benchmarks.darcy._data import (  # noqa: E402
    PixelNormalizer,
    load_split,
    rel_l2,
)
from scripts.benchmarks.darcy._models import (  # noqa: E402
    _build_neighbors,
    _GINODarcyConfig,
    build_grid_geometry,
)
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PROTOCOL,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    CELL_MODES,
    FE_PRECISION,
    cell_centre_interpolation,
    plot_training_curves,
    sample_extra_coefficients,
    sin_boundary_factor,
    split_coefficient,
    torsion_output_scale,
)

# Again after every import: tensorly (imported by the spectral layers) turns
# jax_enable_x64 on as a side effect.
jax.config.update("jax_enable_x64", False)

CKPT_ROOT = (
    _REPO_ROOT / "scripts" / "benchmarks" / "checkpoints" / "darcy_gino_physics_s85"
)


def git_commit() -> str:
    """The commit the run came from, with ``-dirty`` if the tree has changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], cwd=_REPO_ROOT
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def script_sha256() -> str:
    """SHA-256 of this script's source: identifies the code that produced a
    run even after the branch is rebased."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_model(
    n_side: int, hidden_channels: int = 31, modes: int = 16, n_layers: int = 5
) -> tuple[Any, dict, float, float]:
    """The data-driven benchmark's GINO (#362; 2,801,340 real parameters at the
    defaults and ``n_side=85``) on the ``n_side x n_side`` nodes.

    Returns the model, its geometry keyword arguments (input geometry, latent
    grid and output queries on the nodes, flattened row-major as the data and
    the Q4 mesh) and the in / out GNO radii.
    """
    resolution = n_side - 1
    in_gno_radius = 0.16 * 16.0 / resolution
    out_gno_radius = 0.175 * 16.0 / resolution
    grid_xy = build_grid_geometry(n_side)
    config = _GINODarcyConfig().to_dict()
    config["model"]["in_gno_radius"] = in_gno_radius
    config["model"]["out_gno_radius"] = out_gno_radius
    config["model"]["fno_hidden_channels"] = hidden_channels
    config["model"]["fno_n_layers"] = n_layers
    config["model"]["fno_n_modes"] = [modes, modes]
    gino_kwargs = dict(
        input_geom=grid_xy[None, ...],
        latent_queries=grid_xy.reshape(n_side, n_side, 2)[None, ...],
        output_queries=grid_xy[None, ...],
        neighbors_in=_build_neighbors(grid_xy, grid_xy, in_gno_radius),
        neighbors_out=_build_neighbors(grid_xy, grid_xy, out_gno_radius),
    )
    return get_model_jax(config), gino_kwargs, in_gno_radius, out_gno_radius


def make_predict_nodes(
    model: Any,
    gino_kwargs: dict,
    output_scale: float,
    output_offset: float,
    boundary_factor: jnp.ndarray,
) -> Callable:
    """Encoded ``(B, N)`` coefficient -> ``(B, N)`` nodal solution in physical
    units, zero on the boundary."""

    def predict_nodes(p: Any, a_batch_enc: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(p, x=a_batch_enc[..., None], **gino_kwargs)[..., 0]
        return (output_scale * out + output_offset) * boundary_factor

    return predict_nodes


def make_loss_fn(predict_nodes: Callable, physics_loss: Callable) -> Callable:
    """Ritz energy of the predicted nodal solution, with the finite-element
    contractions at ``FE_PRECISION``."""

    def loss_fn(
        p: Any, a_batch_enc: jnp.ndarray, cells_batch: jnp.ndarray
    ) -> jnp.ndarray:
        u = predict_nodes(p, a_batch_enc)
        with jax.default_matmul_precision(FE_PRECISION):
            return physics_loss(u, cells_batch)

    return loss_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument(
        "--extra-coefficients",
        type=int,
        default=0,
        help="Add this many coefficient fields from the fixed sampled pool to "
        "the --n-train released ones (no solutions). Lower --epochs to keep "
        "the step budget (the one-cycle schedule follows the run length).",
    )
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--loss-scale",
        type=float,
        default=1.0,
        help="Constant multiplier on the loss before backpropagation (logged "
        "energies stay unscaled). With Adam and no gradient clipping this is "
        "equivalent to dividing Adam's eps by the same factor.",
    )
    ap.add_argument("--hidden-channels", type=int, default=31)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample-rate", type=int, default=5)
    ap.add_argument("--cell-mode", type=str, default="majority", choices=CELL_MODES)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-grid-rows", type=int, default=6)
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        default=1240,
        help="Log the training energy (mean over the window) and the test error "
        "every this many optimizer steps, and at the last step: 100 points at "
        "1k fields, 101 for the larger sets.",
    )
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    if args.extra_coefficients < 0:
        ap.error("--extra-coefficients must be >= 0")
    commit = git_commit()

    print(
        f"[darcy_gino_physics_s85] loading Li et al.'s s=421 Darcy .mat data "
        f"from {args.data_dir}...",
        flush=True,
    )
    split = load_split(
        args.data_dir,
        dict(
            PROTOCOL,
            n_train=args.n_train,
            n_test=args.n_test,
            subsample_rate=1,
        ),
    )
    train_a_native, train_u_native = split["train_a"], split["train_u"]
    test_a_native, test_u_native = split["test_a"], split["test_u"]
    del split
    del train_u_native  # the training solutions are never used

    train_a, train_cells = split_coefficient(
        train_a_native, args.subsample_rate, args.cell_mode
    )
    a_lo, a_hi = float(train_a_native.min()), float(train_a_native.max())
    n_released = train_a.shape[0]
    if args.extra_coefficients > 0:
        extra_a, extra_cells = sample_extra_coefficients(
            args.extra_coefficients,
            train_a_native.shape[1],
            args.subsample_rate,
            args.cell_mode,
        )
        train_a = np.concatenate([train_a.astype(np.float32), extra_a])
        train_cells = np.concatenate([train_cells.astype(np.float32), extra_cells])
        print(
            f"[darcy_gino_physics_s85] training on {n_released} released + "
            f"{args.extra_coefficients} sampled coefficient fields (no solutions)",
            flush=True,
        )
    test_a, _ = split_coefficient(test_a_native, args.subsample_rate, args.cell_mode)
    test_u = test_u_native[:, :: args.subsample_rate, :: args.subsample_rate]
    del train_a_native, test_a_native, test_u_native
    n_side = train_a.shape[1]
    n_points = n_side * n_side
    test_u_flat = test_u.reshape(test_u.shape[0], n_points)

    xy_np, quad_np, boundary_np = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy_np), jnp.asarray(quad_np))
        output_scale, output_offset = torsion_output_scale(
            galerkin, ~boundary_np, a_lo, a_hi
        )
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    boundary_factor = jnp.asarray(sin_boundary_factor(xy_np), dtype=jnp.float32)
    to_points = jnp.asarray(
        cell_centre_interpolation(n_side, args.subsample_rate), dtype=jnp.float32
    )
    print(
        f"[darcy_gino_physics_s85] label-free output scale c={output_scale:.5f} "
        f"offset m={output_offset:.5f}",
        flush=True,
    )

    # Per-pixel input normalization fitted on the training coefficients
    # (released + sampled extra fields), as in the data-driven GINO script.
    # GINO input: (B, N) nodal values.
    a_normalizer = PixelNormalizer(train_a.reshape(train_a.shape[0], n_points))
    train_a_enc = a_normalizer.encode(train_a.reshape(-1, n_points)).astype(np.float32)
    test_a_enc = a_normalizer.encode(test_a.reshape(-1, n_points)).astype(np.float32)

    model, gino_kwargs, in_gno_radius, out_gno_radius = build_model(
        n_side, args.hidden_channels, args.modes, args.n_layers
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    model_params = model.init(
        init_key, x=jnp.asarray(train_a_enc[:1])[..., None], **gino_kwargs
    )
    n_params_real = count_model_params(model_params)
    print(f"[darcy_gino_physics_s85] n_params_real: {n_params_real}", flush=True)

    predict_nodes = make_predict_nodes(
        model, gino_kwargs, output_scale, output_offset, boundary_factor
    )
    loss_fn = make_loss_fn(predict_nodes, physics_loss)

    @jax.jit
    def predict_both(p: Any, a_batch_enc: jnp.ndarray) -> jnp.ndarray:
        """``(B, 2, N)``: prediction at the released sample points (index 0,
        the benchmark metric) and at the Q4 nodes (index 1)."""
        u = predict_nodes(p, a_batch_enc)
        mapped = jnp.einsum(
            "ij,bjk,lk->bil",
            to_points,
            u.reshape(-1, n_side, n_side),
            to_points,
            precision="highest",
        )
        return jnp.stack([mapped.reshape(-1, n_points), u], axis=1)

    def test_errors(p: Any) -> tuple[list[float], list[float], np.ndarray]:
        pred = np.asarray(
            predict_in_batches(
                predict_both, p, jnp.asarray(test_a_enc), args.batch_size
            )
        )
        mapped = [rel_l2(pred[i, 0], test_u_flat[i]) for i in range(args.n_test)]
        nodes = [rel_l2(pred[i, 1], test_u_flat[i]) for i in range(args.n_test)]
        return mapped, nodes, pred[:, 0]

    n_train = train_a.shape[0]
    steps_per_epoch = max(n_train // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    steps_hist: list[int] = []
    train_hist: list[float] = []
    test_hist: list[float] = []
    time_hist: list[float] = []
    final_ckpt_path = (
        CKPT_ROOT / Path(args.out_dir).name / "gino_physics_s85_final.msgpack"
    )

    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps, peak_value=args.lr
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.inject_hyperparams(adamw_cx)(
            learning_rate=schedule, weight_decay=args.weight_decay, eps=1e-8
        ),
    )
    opt_state = tx.init(model_params)

    @jax.jit
    def train_step(p: Any, opt_state: object, a_enc: jnp.ndarray, cells: jnp.ndarray):
        loss, grads = jax.value_and_grad(
            lambda q: args.loss_scale * loss_fn(q, a_enc, cells)
        )(p)
        updates, opt_state = tx.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss / args.loss_scale

    train_a_j = jnp.asarray(train_a_enc)
    train_cells_j = jnp.asarray(train_cells, dtype=jnp.float32)
    print(
        f"[darcy_gino_physics_s85] training {args.epochs} epochs "
        f"({steps_per_epoch} steps/epoch, {total_steps} steps total, "
        f"batch={args.batch_size}), Q4 Ritz energy only",
        flush=True,
    )
    t0 = t_prev = time.time()
    prev_step, step = 0, 0
    window_loss, window_steps = 0.0, 0
    for epoch in range(args.epochs):
        key, sub = jax.random.split(key)
        perm = jax.random.permutation(sub, n_train)
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            model_params, opt_state, loss = train_step(
                model_params, opt_state, train_a_j[idx], train_cells_j[idx]
            )
            window_loss += float(loss)
            window_steps += 1
            step += 1
            if step % args.eval_every_steps and step != total_steps:
                continue
            test_err = float(np.mean(test_errors(model_params)[0]))
            steps_hist.append(step)
            train_hist.append(window_loss / window_steps)
            test_hist.append(test_err)
            now = time.time()
            per_epoch_s = (now - t_prev) / ((step - prev_step) / steps_per_epoch)
            time_hist.append(per_epoch_s)
            t_prev, prev_step = now, step
            print(
                f"[darcy_gino_physics_s85] step {step:6d}/{total_steps} "
                f"(epoch {epoch:4d}/{args.epochs})  "
                f"energy={window_loss / window_steps:.4e}  "
                f"test_rel_l2={test_err:.4f}  "
                f"elapsed={now - t0:.0f}s",
                flush=True,
            )
            window_loss, window_steps = 0.0, 0
    total_time_s, steps_run = time.time() - t0, step
    save_params(model_params, final_ckpt_path)

    test_errs, test_errs_nodes, test_pred = test_errors(model_params)
    final_mean, final_std = float(np.mean(test_errs)), float(np.std(test_errs))
    print(
        f"[darcy_gino_physics_s85] final test rel_l2: mean={final_mean:.4f} "
        f"std={final_std:.4f} over {args.n_test} instances (at the Q4 nodes: "
        f"{np.mean(test_errs_nodes):.4f})",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if steps_hist:
        plot_training_curves(
            steps_hist,
            train_hist,
            test_hist,
            out_dir / "error_curve.png",
            title="GINO physics only, Darcy s=85 (Li et al. released data)",
        )
        plot_compute_cost(
            steps_hist,
            time_hist,
            out_dir / "compute_cost.png",
            title="GINO physics-only per-epoch time, Darcy s=85",
        )
    n_rows = min(args.n_grid_rows, args.n_test)
    plot_multi_instance_grid(
        [test_a[i] for i in range(n_rows)],
        [test_pred[i].reshape(n_side, n_side) for i in range(n_rows)],
        [test_u_flat[i].reshape(n_side, n_side) for i in range(n_rows)],
        out_dir / "contour.png",
        row_labels=[f"rel-L2={test_errs[i]:.3f}" for i in range(n_rows)],
        title=(
            "GINO physics-only predictions vs Li et al.'s released solutions, "
            f"held-out a(x) (s={n_side})"
        ),
    )

    metrics = {
        "problem": "darcy_gino_physics_s85",
        "backbone": "GINO (GINODarcy2d)",
        "data_source": (
            f"Li et al.'s piececonst_r421_N1024 .mat files, native s=421, "
            f"subsample_rate={args.subsample_rate}; first {n_released} training "
            f"coefficients + {args.extra_coefficients} sampled from the data's "
            "law, training solutions unused"
        ),
        "loss": "Q4 Ritz energy functional (physics only, no labels)",
        "cell_mode": args.cell_mode,
        "boundary_condition": "exact, (c * N(a) + m) * sin(pi x) sin(pi y)",
        "output_scale_c": output_scale,
        "output_offset_m": output_offset,
        "metric_points": (
            "released sample points (native cell centres), fixed linear map"
        ),
        "fe_precision": FE_PRECISION,
        "resolution": n_side,
        "n_train": n_train,
        "n_train_released": n_released,
        "extra_coefficients": args.extra_coefficients,
        "n_test": args.n_test,
        "batch_size": args.batch_size,
        "fno_hidden_channels": args.hidden_channels,
        "fno_modes": args.modes,
        "fno_n_layers": args.n_layers,
        "in_gno_radius": in_gno_radius,
        "out_gno_radius": out_gno_radius,
        "n_params_real": n_params_real,
        "optimizer": "adamw_cx (complex-aware AdamW), eps 1e-8, cosine one-cycle",
        "grad_clip": args.grad_clip,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "loss_scale": args.loss_scale,
        "seed": args.seed,
        "input_normalization": "per-pixel UnitGaussianNormalizer on the coefficient",
        "epochs": args.epochs,
        "steps_run": steps_run,
        "eval_every_steps": args.eval_every_steps,
        "checkpoint_final": final_ckpt_path.relative_to(_REPO_ROOT).as_posix(),
        "final_test_rel_l2_mean": final_mean,
        "final_test_rel_l2_std": final_std,
        "final_test_rel_l2_nodes_mean": float(np.mean(test_errs_nodes)),
        "total_time_s": total_time_s,
        "history": {
            "steps": steps_hist,
            "train_ritz_energy": train_hist,
            "test_rel_l2": test_hist,
        },
        "git_commit": commit,
        "script_sha256": script_sha256(),
        "spec_reference": "doc/assets/benchmarks/darcy_q4_s85_spec_floor/metrics.json",
        "datadriven_reference": (
            "doc/assets/benchmarks/darcy_gino" " (#362, 1000 labels)"
        ),
    }
    save_metrics_json(metrics, out_dir / "metrics.json")
    print(f"[darcy_gino_physics_s85] done, wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
