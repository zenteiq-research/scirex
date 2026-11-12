# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Transolver on the exact Darcy benchmark from Wu et al.
(Transolver, ICML 2024, arXiv:2402.02366, Table 2 "Darcy"): the label-free
counterpart of the data-driven benchmark (``darcy/``).

Everything except the training signal is that script's published protocol,
reused rather than re-implemented:

- Data: Li et al.'s ``piececonst_r421_N1024_smooth1.mat`` (first 1000
  instances, training) and ``smooth2.mat`` (first 200, test), strided to
  s=85, loaded with the same loader (``darcy._data.load_split``).
- Model: the same :class:`~scirex.models.Transolver`
  call (structured_2d, 8 layers, 8 heads, 128 channels, 64 slices, ref 8,
  unified positions) -- 2,826,945 parameters.
- Model input: the s=85 coefficient field with the same global-scalar
  normalizer (statistics of the training *coefficients* only).
- Optimizer and loop: AdamW (lr 1e-3, weight decay 1e-5, eps 1e-8), cosine
  one-cycle schedule, global-norm gradient clipping 0.1, batch size 4,
  500 epochs of 250 steps with a fresh shuffle each epoch.
- Metric: mean per-instance relative L2 against the released test
  solutions over all 85 x 85 points.

What changes is the loss. The training solutions are dropped right after
loading and never used: the network is trained only on the Q4 Galerkin
(Ritz) energy ``J(u) = 0.5 * int(a |grad u|^2) - int(f u)`` of
``-div(a grad u) = 1``
(:class:`~scirex.losses.darcy_piecewise_coefficient.BatchedEnergyFunctionalLossQ4`),
whose minimiser is the Q4 finite-element solution. The choices this needs
use no label information:

- One coefficient value per Q4 element, the majority of the native s=421
  nodes it covers (``_ritz.node_to_cell_coefficient``); see
  ``check_floor.py`` for the comparison of cell rules.
- ``u = 0`` on the boundary is exact: ``u = (c * N(a) + m) * sin(pi x)
  sin(pi y)``.
- The output scale ``c`` and offset ``m`` replace the solution statistics a
  data-driven model normalises with. They come from one Q4 solve of the
  torsion problem ``-Lap(phi) = 1`` (``phi = 0`` on the boundary) divided by
  the effective coefficient of a 50/50 two-phase medium in 2-D,
  ``sqrt(a_lo * a_hi)`` (Keller 1964; Dykhne 1971): ``m`` and ``c`` are the
  mean and standard deviation of ``phi / sqrt(a_lo * a_hi)`` over the nodes.
- The network predicts on the Q4 nodes ``j / 84``, where the loss acts. Li
  et al.'s generator stores its solutions at the native cell centres, half a
  native cell inside those nodes, so for scoring the prediction is moved
  there by a fixed linear interpolation
  (``_ritz.cell_centre_interpolation``). With it an exact Q4 solve
  scores 0.31% against the released test solutions (0.60% at the nodes):
  the floor of this formulation.

Unlabelled scaling (``--extra-coefficients K``): the physics loss needs
coefficient fields only, so the released training fields can be
supplemented by fields drawn from the data's own law
(:func:`~scirex.losses.darcy_piecewise_coefficient.sample_piecewise_coefficient`:
thresholded Gaussian random field, ``C = (-Delta + 9I)^-2``, native s=421).
The extra fields come from one fixed, seed-independent pool, so the training
sets are nested (released 1000 + the first K pool fields); no solutions are
generated. ``check_sampler.py`` compares the pool's
statistics with the released fields. The input normalizer is fitted on
everything the model trains on (released plus sampled coefficients; still
label-free). Keep the step budget by lowering ``--epochs`` with the set
size.

The finite-element contractions (geometry and energy) run at full float32
precision inside this script (``jax.default_matmul_precision("highest")``):
on GPU, JAX defaults float32 matmuls to TF32, which perturbs the Q4
Jacobians by up to ~5%. The network keeps the default precision.
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

# Same guard as the data-driven benchmark (darcy/): keep Transolver in
# float32 even if an import flips jax_enable_x64 on.
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import optax  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    BatchedEnergyFunctionalLossQ4,
    build_grid_aligned_q4_mesh,
    kappa_fn,
    source_fn,
)
from scirex.models import Transolver  # noqa: E402
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import save_params  # noqa: E402
from scripts.benchmarks._plot_utils import (  # noqa: E402
    plot_compute_cost,
    plot_multi_instance_grid,
    save_metrics_json,
)
from scripts.benchmarks.darcy._data import (  # noqa: E402
    GlobalNormalizer,
    load_split,
    rel_l2,
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

# Again after every import: an import further down the scirex package
# (tensorly) turns jax_enable_x64 on as a side effect.
jax.config.update("jax_enable_x64", False)

CKPT_ROOT = (
    _REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "checkpoints"
    / "darcy_transolver_physics_s85"
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
    n_side: int,
    n_hidden: int = 128,
    n_layers: int = 8,
    n_head: int = 8,
    slice_num: int = 64,
    ref: int = 8,
    mlp_ratio: int = 1,
    kernel: int = 3,
) -> Transolver:
    """The data-driven benchmark's Transolver (2,826,945 parameters at the
    defaults and ``n_side=85``)."""
    return Transolver(
        space_dim=2,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        mlp_ratio=mlp_ratio,
        fun_dim=1,
        out_dim=1,
        slice_num=slice_num,
        ref=ref,
        unified_pos=True,
        geometry="structured_2d",
        H=n_side,
        W=n_side,
        kernel=kernel,
        dropout_rate=0.0,
        time_input=False,
    )


def make_predict_nodes(
    model: Transolver,
    n_points: int,
    output_scale: float,
    output_offset: float,
    boundary_factor: jnp.ndarray,
) -> Callable[[Any, jnp.ndarray], jnp.ndarray]:
    """Encoded ``(B, N, 1)`` coefficient -> ``(B, N)`` nodal solution in
    physical units: ``(c N(a) + m) * boundary_factor``, zero on the boundary."""
    # unified_pos + structured_2d recompute positions from H/W; this
    # placeholder only supplies the batch dimension.
    x_placeholder = jnp.zeros((1, n_points, 2), dtype=jnp.float32)

    def predict_nodes(p: Any, a_batch_enc: jnp.ndarray) -> jnp.ndarray:
        b = a_batch_enc.shape[0]
        x_b = jnp.broadcast_to(x_placeholder, (b, n_points, 2))
        out = model.apply(p, x=x_b, fx=a_batch_enc)  # (B, N, 1)
        return (output_scale * out[:, :, 0] + output_offset) * boundary_factor

    return predict_nodes


def make_loss_fn(
    predict_nodes: Callable[[Any, jnp.ndarray], jnp.ndarray],
    physics_loss: BatchedEnergyFunctionalLossQ4,
) -> Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Mean Q4 Ritz energy of the prediction, FE contractions at full
    precision."""

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
        "the step budget. See the module docstring.",
    )
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-grad-norm", type=float, default=0.1)
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--slice-num", type=int, default=64)
    ap.add_argument("--ref", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=int, default=1)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample-rate", type=int, default=5)
    ap.add_argument(
        "--cell-mode",
        type=str,
        default="majority",
        choices=CELL_MODES,
        help="How each Q4 cell's coefficient is taken from the native s=421 "
        "nodes it covers -- see _ritz.py and "
        "check_floor.py.",
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-grid-rows", type=int, default=6)
    ap.add_argument(
        "--eval-every-steps",
        type=int,
        default=1250,
        help="Log the training energy (mean over the window) and the test error "
        "every this many optimizer steps: ~100 points per 125k-step run, "
        "whatever the number of training fields.",
    )
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    if args.extra_coefficients < 0:
        ap.error("--extra-coefficients must be >= 0")
    commit = git_commit()

    print(
        f"[darcy_transolver_physics_s85] loading Li et al.'s s=421 Darcy .mat "
        f"data from {args.data_dir}...",
        flush=True,
    )
    # Native resolution: the per-cell coefficient needs every s=421 node.
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
    # The training solutions are never used.
    del train_u_native

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
            f"[darcy_transolver_physics_s85] training on {n_released} released + "
            f"{args.extra_coefficients} sampled coefficient fields (no solutions)",
            flush=True,
        )
    test_a, _ = split_coefficient(test_a_native, args.subsample_rate, args.cell_mode)
    test_u = test_u_native[:, :: args.subsample_rate, :: args.subsample_rate]
    del train_a_native, test_a_native, test_u_native
    n_side = train_a.shape[1]
    n_points = n_side * n_side
    test_u_flat = test_u.reshape(test_u.shape[0], n_points)
    print(
        f"[darcy_transolver_physics_s85] data ready: train_a={train_a.shape} "
        f"train_cells={train_cells.shape} test_a={test_a.shape} "
        f"test_u={test_u.shape} (s={n_side}, cell_mode={args.cell_mode})",
        flush=True,
    )

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
        f"[darcy_transolver_physics_s85] label-free output scale c={output_scale:.5f} "
        f"offset m={output_offset:.5f} (torsion / sqrt({a_lo:g} * {a_hi:g}))",
        flush=True,
    )

    # Input normalization from the training coefficients only (released +
    # sampled extra fields), as in the data-driven script.
    a_normalizer = GlobalNormalizer(train_a[..., None])
    train_a_enc = a_normalizer.encode(train_a[..., None]).reshape(-1, n_points, 1)
    test_a_enc = a_normalizer.encode(test_a[..., None]).reshape(-1, n_points, 1)

    model = build_model(
        n_side,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        n_head=args.n_head,
        slice_num=args.slice_num,
        ref=args.ref,
        mlp_ratio=args.mlp_ratio,
        kernel=args.kernel,
    )
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    model_params = model.init(
        init_key,
        x=jnp.zeros((1, n_points, 2), dtype=jnp.float32),
        fx=jnp.asarray(train_a_enc[:1]),
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
    print(f"[darcy_transolver_physics_s85] n_params: {n_params}", flush=True)

    predict_nodes = make_predict_nodes(
        model, n_points, output_scale, output_offset, boundary_factor
    )
    loss_fn = make_loss_fn(predict_nodes, physics_loss)

    @jax.jit
    def predict_both(p: Any, a_batch_enc: jnp.ndarray) -> jnp.ndarray:
        """``(B, 2, N)``: the prediction at the released sample points
        (index 0, the benchmark metric) and at the Q4 nodes (index 1)."""
        u = predict_nodes(p, a_batch_enc)
        grid = u.reshape(-1, n_side, n_side)
        mapped = jnp.einsum(
            "ij,bjk,lk->bil", to_points, grid, to_points, precision="highest"
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
    # Namespaced by --out-dir's own name so different seeds/configurations
    # never overwrite each other's checkpoints.
    ckpt_dir = CKPT_ROOT / Path(args.out_dir).name
    final_ckpt_path = ckpt_dir / "transolver_physics_s85_final.msgpack"

    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps, peak_value=args.lr
    )
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = tx.init(model_params)

    @jax.jit
    def train_step(
        p: Any,
        opt_state: object,
        a_batch_enc: jnp.ndarray,
        cells_batch: jnp.ndarray,
    ):
        loss, grads = jax.value_and_grad(loss_fn)(p, a_batch_enc, cells_batch)
        updates, opt_state = tx.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    train_a_j = jnp.asarray(train_a_enc)
    train_cells_j = jnp.asarray(train_cells, dtype=jnp.float32)

    t0 = time.time()
    t_prev = t0
    prev_step = 0
    window_loss, window_steps = 0.0, 0
    print(
        f"[darcy_transolver_physics_s85] training {args.epochs} epochs "
        f"({steps_per_epoch} steps/epoch, {total_steps} steps total, "
        f"batch={args.batch_size}), Q4 Ritz energy only",
        flush=True,
    )
    step = 0
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
            eta_min = (total_steps - step) / steps_per_epoch * per_epoch_s / 60.0
            print(
                f"[darcy_transolver_physics_s85] step {step:6d}/{total_steps} "
                f"(epoch {epoch:4d}/{args.epochs})  "
                f"energy={window_loss / window_steps:.4e}  "
                f"test_rel_l2={test_err:.4f}  "
                f"~{per_epoch_s:.2f}s/epoch  elapsed={now - t0:.0f}s  "
                f"eta={eta_min:.1f}min",
                flush=True,
            )
            window_loss, window_steps = 0.0, 0
    total_time_s = time.time() - t0
    steps_run = step

    save_params(model_params, final_ckpt_path)
    print(
        f"[darcy_transolver_physics_s85] saved final checkpoint: {final_ckpt_path}",
        flush=True,
    )

    test_errs, test_errs_nodes, test_pred = test_errors(model_params)
    final_mean, final_std = float(np.mean(test_errs)), float(np.std(test_errs))
    print(
        f"[darcy_transolver_physics_s85] final test rel_l2: mean={final_mean:.4f} "
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
            title="Transolver physics only, Darcy s=85 (Li et al. released data)",
        )
        plot_compute_cost(
            steps_hist,
            time_hist,
            out_dir / "compute_cost.png",
            title=(
                "Wu et al. Darcy (real s=421 data, downsampled to s=85): "
                "Transolver physics-only per-epoch time"
            ),
        )
    n_rows = min(args.n_grid_rows, args.n_test)
    plot_multi_instance_grid(
        [test_a[i] for i in range(n_rows)],
        [test_pred[i].reshape(n_side, n_side) for i in range(n_rows)],
        [test_u_flat[i].reshape(n_side, n_side) for i in range(n_rows)],
        out_dir / "contour.png",
        row_labels=[f"rel-L2={test_errs[i]:.3f}" for i in range(n_rows)],
        title=(
            "Physics-only predictions vs Li et al.'s released solutions, "
            f"held-out a(x) (Transolver, s={n_side})"
        ),
    )

    metrics = {
        "problem": "darcy_transolver_physics_s85",
        "domain": (
            "Wu et al. Transolver paper's Darcy benchmark, Li et al.'s own "
            "real released data"
        ),
        "backbone": "Transolver (general-purpose, geometry=structured_2d)",
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
        "output_scale_source": f"Q4 torsion solve / sqrt({a_lo:g} * {a_hi:g})",
        "metric_points": (
            "released sample points (native cell centres), fixed linear map"
        ),
        "fe_precision": FE_PRECISION,
        "resolution": n_side,
        "n_points": n_points,
        "n_train": n_train,
        "n_train_released": n_released,
        "extra_coefficients": args.extra_coefficients,
        "n_test": args.n_test,
        "batch_size": args.batch_size,
        "n_hidden": args.n_hidden,
        "n_layers": args.n_layers,
        "n_head": args.n_head,
        "slice_num": args.slice_num,
        "ref": args.ref,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "input_normalization": (
            "global-scalar UnitTransformer on the coefficient, eps=1e-8"
        ),
        "n_params": n_params,
        "epochs": args.epochs,
        "steps_run": steps_run,
        "eval_every_steps": args.eval_every_steps,
        "git_commit": commit,
        "script_sha256": script_sha256(),
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
        "spec_reference": "doc/assets/benchmarks/darcy_q4_s85_spec_floor/metrics.json",
        "paper_reported_rel_l2": 0.0057,
        "paper_reference": (
            "Wu, H. et al. Transolver: A Fast Transformer Solver for PDEs on "
            "General Geometries. ICML 2024 / arXiv:2402.02366, Table 2, "
            "'DARCY' column, 'TRANSOLVER (OURS)' row: 0.57% (data-driven)."
        ),
    }
    save_metrics_json(metrics, out_dir / "metrics.json")
    print(
        f"[darcy_transolver_physics_s85] done, wrote results to {out_dir}", flush=True
    )


if __name__ == "__main__":
    main()
