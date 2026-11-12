# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Faithful port of the Transolver paper's OWN 2D Navier-Stokes Pipe-flow
benchmark
(https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/exp_pipe.py,
read 2026-09-15), using this repo's JAX/Flax ``scirex.models.
transolver.Transolver`` (``geometry="structured_2d"``) in place of the
reference's PyTorch ``model.Transolver_Structured_Mesh_2D``.

Every hyperparameter and the training strategy itself are copied 1:1 from
``exp_pipe.py`` so a run of this script is directly comparable to the
paper's own numbers on this benchmark, "same trainable parameters, same
number of epochs":

  Architecture (``Transolver_Structured_Mesh_2D.Model``, ``scirex.config.models.
  TransolverPipe``): space_dim=2, fun_dim=0, out_dim=1, n_hidden=64,
  n_layers=3, n_head=4, mlp_ratio=1, slice_num=32, ref=8, dropout=0.0,
  unified_pos=False, geometry=structured_2d, H=W=129 (s1/s2 at the default
  --downsamplex/--downsampley of 1). Weight init additionally matched to
  the reference's own ``Model.initialize_weights`` -- every ``nn.Linear``
  kernel ~ trunc_normal(std=0.02) (numerically == N(0, 0.02^2): at this std
  the default a=-2/b=2 truncation bounds are ~100 std-devs out and never
  actually clip), bias=0, EXCEPT ``in_project_slice`` (orthogonally
  initialized in both implementations) and every ``nn.Conv2d`` kernel
  (``_init_weights`` only matches on ``nn.Linear``, so PyTorch's default
  Conv2d init -- kaiming_uniform -- applies instead; Flax's own default
  Conv kernel init, lecun_normal, is left untouched here to mirror that gap
  rather than closing it). See ``_match_official_weight_init`` below.

  Optimizer: ``torch.optim.AdamW(lr=1e-3, weight_decay=1e-5)`` -> here
  ``optax.adamw`` (same defaults otherwise: betas=(0.9,0.999), eps=1e-8).
  This benchmark's own recipe genuinely uses decoupled weight decay, unlike
  this repo's ShapeNetCar/AirfRANS Transolver recipes (plain Adam, no
  decay) -- don't copy this script's optimizer choice over to those.

  LR schedule: ``torch.optim.lr_scheduler.OneCycleLR(max_lr=1e-3,
  epochs=500, steps_per_epoch=len(train_loader))`` with every other kwarg
  left at PyTorch's own default (``pct_start=0.3``, ``div_factor=25``,
  ``final_div_factor=1e4``, ``anneal_strategy='cos'``) -> here
  ``optax.cosine_onecycle_schedule`` with those same three numbers,
  ``transition_steps = epochs * steps_per_epoch``. ``scheduler.step()`` is
  called once per TRAINING batch only (never during eval), matching one
  ``tx.update`` per training step here.

  Batching: unlike this repo's ShapeNetCar/AirfRANS Transolver scripts
  (batch_size=1, variable mesh size per sample), every Pipe sample lives on
  the SAME fixed 129x129 grid, so real mini-batching (batch_size=8,
  reshuffled every epoch, matching ``DataLoader(shuffle=True)``) is used
  here via a single ``jax.jit``-ed batched train step -- much closer to the
  reference's own per-batch loop than a per-sample Python loop would be.

  Data pipeline: ``UnitTransformer``-equivalent normalization (mean/std
  over BOTH the batch and point dimensions, computed once from x_train/
  y_train only) -- ``_UnitTransformer`` below. Positions (x) are normalized
  for both train and test using TRAIN statistics; targets (y) are only
  ever compared in physical units (the reference's own
  encode-then-immediately-decode of y_train is an algebraic no-op given
  ``UnitTransformer.decode`` exactly inverts ``encode``, so this script
  skips that round trip and compares the model's decoded output directly
  against the raw y -- see ``_compute_losses``'s docstring for the exact
  correspondence). Model receives ONLY the normalized positions as ``x``,
  called with ``fx=None`` (fun_dim=0), matching the reference's own
  ``model(x, None)``.

  Loss: ``utils.testloss.TestLoss(size_average=False)`` -- relative L2
  per sample (``||pred-true||_2 / ||true||_2``, flattened over all 16641
  points), SUMMED (not averaged) over the batch. The per-epoch number this
  script prints (``train_loss``/``rel_err``) divides that accumulated sum
  by ntrain/ntest at the end of the epoch, exactly matching
  ``train_loss = train_loss / ntrain`` / ``rel_err /= ntest`` in
  ``exp_pipe.py`` -- both give the mean per-sample relative-L2 error over
  the epoch, but the quantity actually differentiated at each training
  step is the batch SUM, not its mean (this changes the effective
  gradient scale by a factor of batch_size relative to a naive
  mean-reduced loss, so it matters for an apples-to-apples comparison).

  No gradient clipping (``--max_grad_norm`` reference default: ``None``).

KNOWN DATA CAVEAT -- read before trusting numbers from this script:
``Dataset/Pipe_Q.npy`` in this repo is truncated (its .npy header declares
shape (2310, 3, 129, 129) float64, ~880MB of array data, but the file on
disk is only ~63MB -- confirmed corrupt/incomplete, not a format
difference). This script therefore falls back to
``Dataset/Pipe_Q_interp_x.npy`` (1200, 129, 129) float64 -- already
sliced to the reference's own N=1200 and, judging by its value range
matching exp_pipe.py's own plotting clim (0, 0.3) for this field almost
exactly (measured max ~0.293 either way), almost certainly a previously
-cached ``Pipe_Q.npy[:1200, 0]`` (channel 0 is exactly what
``exp_pipe.py`` itself selects: ``output = np.load(OUTPUT_Sigma)[:, 0]``).
It is NOT byte-identical to the truncated file's own first-sample channel-0
slice (checked directly), so it is not proven to be pixel-identical to
whatever the original full Pipe_Q.npy contained -- likely a resampled/
interpolated variant aligned onto the Pipe_X/Pipe_Y grid (hence the
"interp_x" name) rather than raw solver output. If an intact Pipe_Q.npy
becomes available, prefer it via ``--q-path`` (or just replace the file at
its default path) -- this script auto-detects and uses it over the
fallback in that case.

Outputs (``--output-dir`` / ``--ckpt-dir``, both
``scripts/benchmarks/checkpoints/pipe/transolver_exp_pipe/`` by default):
    loss_curve.csv, the checkpoints and, with ``--anim-frames``, frames.npz

Usage: ``train.py --run transolver_exp_pipe`` runs it with the settings of the
JAX-vs-PyTorch comparison (``_config.TRANSOLVER_ARGS``)::

    python scripts/benchmarks/pipe/train.py --run transolver_exp_pipe --data-dir Dataset

The benchmark's own Transolver run, trained like every other model by the
shared runner, is ``train.py --run transolver``.

It also runs on its own, with every flag of the reference ``exp_pipe.py``::

    python scripts/benchmarks/pipe/_transolver.py --data_path Dataset
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpy.lib.format as npy_format  # noqa: E402
import optax  # noqa: E402

from scirex import get_model_jax  # noqa: E402
from scirex.config.transolver_pipe_config import Default  # noqa: E402
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks._animate import log_spaced_epochs  # noqa: E402

jax.config.update("jax_enable_x64", False)

OUTPUT_DIR = _REPO_ROOT / "scripts/benchmarks/checkpoints/pipe/transolver_exp_pipe"
CKPT_DIR = OUTPUT_DIR


# ── Reference-matching normalizer ───────────────────────────────────────────


class _UnitTransformer:
    """Ports ``utils.normalizer.UnitTransformer`` exactly: mean/std computed
    over dims (0, 1) of the fitting tensor (batch AND point dims), kept as a
    broadcastable (1, 1, ...) array. ``eps=1e-8`` added to std, matching the
    reference."""

    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=(0, 1), keepdims=True)
        self.std = x.std(axis=(0, 1), keepdims=True) + 1e-8

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jnp.asarray(self.std, dtype=x.dtype) + jnp.asarray(
            self.mean, dtype=x.dtype
        )


# ── Data loading (ports exp_pipe.py's own data pipeline) ───────────────────


def _npy_shape(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with open(path, "rb") as f:
        ver = npy_format.read_magic(f)
        shape, _fortran, dtype = npy_format._read_array_header(f, ver)
    return shape, dtype


def _load_output_field(data_dir: Path, q_path: Path | None, n_total: int) -> np.ndarray:
    """Returns the (n_total, 129, 129) scalar output field, matching
    ``output = np.load(OUTPUT_Sigma)[:, 0]`` in ``exp_pipe.py``. Falls back
    to ``Pipe_Q_interp_x.npy`` if ``Pipe_Q.npy`` is missing/truncated -- see
    this module's docstring's "KNOWN DATA CAVEAT" section."""
    q_path = q_path or (data_dir / "Pipe_Q.npy")
    try:
        declared_shape, _dtype = _npy_shape(q_path)
        expected_bytes = int(np.prod(declared_shape)) * np.dtype(_dtype).itemsize
        actual_bytes = q_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise OSError(
                f"{q_path} is truncated: header declares shape "
                f"{declared_shape} ({expected_bytes:,} bytes of array data) "
                f"but the file is only {actual_bytes:,} bytes on disk."
            )
        output = np.load(q_path)[:, 0]
        print(f"[transolver_pipe_official] loaded output field from {q_path}")
        return output.astype(np.float64)
    except Exception as e:
        # failure (missing file, truncation, corrupt header) falls back the
        # same way.
        fallback_path = data_dir / "Pipe_Q_interp_x.npy"
        print(
            f"[transolver_pipe_official] WARNING: could not use {q_path} "
            f"({e}). Falling back to {fallback_path} -- see this script's "
            "module docstring's KNOWN DATA CAVEAT before trusting results.",
            file=sys.stderr,
        )
        output = np.load(fallback_path).astype(np.float64)
        if output.shape[0] < n_total:
            raise RuntimeError(
                f"{fallback_path} has only {output.shape[0]} samples, need "
                f"n_total={n_total}."
            ) from e
        return output


def load_pipe_data(
    data_dir: Path,
    q_path: Path | None,
    ntrain: int,
    ntest: int,
    n_total: int,
    downsamplex: int,
    downsampley: int,
) -> dict:
    inputX = np.load(data_dir / "Pipe_X.npy").astype(np.float64)
    inputY = np.load(data_dir / "Pipe_Y.npy").astype(np.float64)
    inp = np.stack([inputX, inputY], axis=-1)  # (Nfull, 129, 129, 2)

    output = _load_output_field(data_dir, q_path, n_total)  # (Nfull_or_N, 129, 129)

    r1, r2 = downsamplex, downsampley
    s1 = ((129 - 1) // r1) + 1
    s2 = ((129 - 1) // r2) + 1

    inp_n = inp[:n_total]
    out_n = output[:n_total]

    x_train = inp_n[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = out_n[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = inp_n[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = out_n[-ntest:, ::r1, ::r2][:, :s1, :s2]

    x_train = x_train.reshape(ntrain, -1, 2).astype(np.float32)
    x_test = x_test.reshape(ntest, -1, 2).astype(np.float32)
    y_train = y_train.reshape(ntrain, -1).astype(np.float32)
    y_test = y_test.reshape(ntest, -1).astype(np.float32)

    return dict(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, s1=s1, s2=s2
    )


# ── Official weight init (matches Model.initialize_weights) ────────────────


def _match_official_weight_init(params, seed: int = 0, n_layers: int | None = None):
    """Reinitializes every non-orthogonal 2D Dense "kernel" leaf to
    N(0, 0.02^2) (the reference's ``trunc_normal_(std=0.02)``, numerically
    a plain Gaussian at this std -- see module docstring). Detects the one
    orthogonally-initialized layer per block (``in_project_slice``)
    mathematically (Gram matrix ~= identity) rather than by Flax's
    auto-generated submodule name, and leaves Conv kernels (ndim=4)
    untouched, matching the reference's ``isinstance(m, nn.Linear)``-only
    check.

    The Gram-matrix check is only meaningful for kernels with
    ``min(r, c) > 1`` -- ``in_project_slice`` is always a genuine
    rectangular orthogonal matrix (e.g. (16, 64)), never rank-1. For a
    (n, 1) or (1, n) kernel (the model's final (n_hidden, 1) output
    projection), the check degenerates to "is this vector's squared norm
    within 0.01 of 1", a coincidental ~5-7% per-seed false positive rather
    than a real orthogonality test -- such kernels are always reinitialized,
    never treated as orthogonal. If ``n_layers`` is given (one
    ``in_project_slice`` per Transolver block), the kept-count is asserted
    against it so any future misdetection fails loudly instead of silently
    leaving a layer on Flax's default init."""
    base_key = jax.random.PRNGKey(seed)
    counter = [0]
    n_reinit = [0]
    n_kept = [0]

    def _reinit_leaf(path, leaf):
        path_key = getattr(path[-1], "key", None)
        if path_key != "kernel" or leaf.ndim != 2:
            return leaf
        r, c = leaf.shape
        if min(r, c) > 1:
            if r >= c:
                is_orthogonal = jnp.allclose(leaf.T @ leaf, jnp.eye(c), atol=1e-2)
            else:
                is_orthogonal = jnp.allclose(leaf @ leaf.T, jnp.eye(r), atol=1e-2)
        else:
            is_orthogonal = False
        if bool(is_orthogonal):
            n_kept[0] += 1
            return leaf
        counter[0] += 1
        key = jax.random.fold_in(base_key, counter[0])
        n_reinit[0] += 1
        return jax.random.normal(key, leaf.shape, dtype=leaf.dtype) * 0.02

    new_params = jax.tree_util.tree_map_with_path(_reinit_leaf, params)
    print(
        f"[transolver_pipe_official] weight init: reinitialized "
        f"{n_reinit[0]} Dense kernel(s) to N(0, 0.02^2) (official recipe's "
        f"trunc_normal_(std=0.02)), kept {n_kept[0]} orthogonal-initialized "
        f"kernel(s) unchanged"
    )
    if n_layers is not None and n_kept[0] != n_layers:
        raise RuntimeError(
            f"[transolver_pipe_official] expected exactly {n_layers} "
            f"orthogonal in_project_slice kernel(s) (one per Transolver "
            f"block), found {n_kept[0]}. Orthogonal-layer detection is "
            "likely misfiring for this config -- check before trusting "
            "this run's weight init."
        )
    return new_params


# ── Loss (matches utils.testloss.TestLoss(size_average=False).rel) ─────────


def _rel_l2_sum(pred: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Per-sample relative L2 error, SUMMED over the batch -- matches
    ``TestLoss(size_average=False)`` (default d=2, p=2, reduction=True):
    ``torch.sum(||pred-true||_2 / ||true||_2)`` over the batch dim, each
    norm flattening all non-batch dims first."""
    b = pred.shape[0]
    diff_norm = jnp.linalg.norm(pred.reshape(b, -1) - true.reshape(b, -1), axis=1)
    true_norm = jnp.linalg.norm(true.reshape(b, -1), axis=1)
    return jnp.sum(diff_norm / true_norm)


# ── CLI (mirrors exp_pipe.py's argparse 1:1) ────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Training Transolver on the Pipe benchmark")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--n-hidden", type=int, default=64, help="hidden dim")
    ap.add_argument("--n-layers", type=int, default=3, help="layers")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max_grad_norm", type=float, default=None)
    ap.add_argument(
        "--pct-start",
        type=float,
        default=0.3,
        help="cosine_onecycle_schedule pct_start (exp_pipe.py OneCycleLR default).",
    )
    ap.add_argument(
        "--div-factor",
        type=float,
        default=25.0,
        help="cosine_onecycle_schedule div_factor (exp_pipe.py OneCycleLR default).",
    )
    ap.add_argument(
        "--final-div-factor",
        type=float,
        default=10_000.0,
        help="cosine_onecycle_schedule final_div_factor (exp_pipe.py OneCycleLR "
        "default).",
    )
    ap.add_argument("--downsamplex", type=int, default=1)
    ap.add_argument("--downsampley", type=int, default=1)
    ap.add_argument("--mlp_ratio", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--unified_pos", type=int, default=0)
    ap.add_argument("--ref", type=int, default=8)
    ap.add_argument("--slice_num", type=int, default=32)
    ap.add_argument("--save_name", type=str, default="pipe_UniPDE")
    ap.add_argument(
        "--data_path",
        type=str,
        default="Dataset",
        help="Directory containing Pipe_X.npy/Pipe_Y.npy/Pipe_Q.npy "
        "(exp_pipe.py's --data_path).",
    )
    ap.add_argument(
        "--q-path",
        type=str,
        default=None,
        help="Explicit path to the output-field .npy (overrides the "
        "Pipe_Q.npy/Pipe_Q_interp_x.npy auto-detection).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--anim-frames",
        type=int,
        default=0,
        help="Also save <ckpt-dir>/frames.npz: the prediction on test sample "
        "--anim-sample at about this many log-spaced epochs, for "
        "scripts/benchmarks/pipe/animate.py. 0 (default) disables it.",
    )
    ap.add_argument("--anim-sample", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument(
        "--skip-official-init",
        action="store_true",
        help="Skip the trunc_normal(0.02) reinit and keep Flax's own "
        "lecun_normal Dense init (ablation flag, not part of the official "
        "recipe).",
    )
    return ap


def build_config(args: argparse.Namespace) -> dict:
    config = Default()
    config.opt.n_epochs = args.epochs
    config.opt.learning_rate = args.lr
    config.opt.weight_decay = args.weight_decay
    config.opt.batch_size = args.batch_size
    config.opt.max_grad_norm = args.max_grad_norm
    config.opt.pct_start = args.pct_start
    config.opt.div_factor = args.div_factor
    config.opt.final_div_factor = args.final_div_factor
    config.data.data_dir = args.data_path
    config.data.downsamplex = args.downsamplex
    config.data.downsampley = args.downsampley
    config.model.n_hidden = args.n_hidden
    config.model.n_layers = args.n_layers
    config.model.n_head = args.n_heads
    config.model.mlp_ratio = args.mlp_ratio
    config.model.dropout_rate = args.dropout
    config.model.unified_pos = bool(args.unified_pos)
    config.model.ref = args.ref
    config.model.slice_num = args.slice_num
    return config.to_dict()


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = build_config(args)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else OUTPUT_DIR
    ckpt_dir = Path(args.ckpt_dir).expanduser() if args.ckpt_dir else CKPT_DIR

    print(f"[transolver_pipe_official] jax devices: {jax.devices()}")
    print(f"[transolver_pipe_official] config: {config}")
    sys.stdout.flush()

    data_dir = Path(config.data.data_dir).expanduser()
    q_path = Path(args.q_path).expanduser() if args.q_path else None
    data = load_pipe_data(
        data_dir,
        q_path,
        ntrain=config.data.ntrain,
        ntest=config.data.ntest,
        n_total=config.data.n_total,
        downsamplex=config.data.downsamplex,
        downsampley=config.data.downsampley,
    )
    x_train_raw, y_train_raw = data["x_train"], data["y_train"]
    x_test_raw, y_test_raw = data["x_test"], data["y_test"]
    s1, s2 = data["s1"], data["s2"]
    ntrain, ntest = x_train_raw.shape[0], x_test_raw.shape[0]
    print(
        f"[transolver_pipe_official] x_train={x_train_raw.shape} "
        f"y_train={y_train_raw.shape} x_test={x_test_raw.shape} "
        f"y_test={y_test_raw.shape}  H=W={s1}"
    )

    x_normalizer = _UnitTransformer(x_train_raw)
    y_normalizer = _UnitTransformer(y_train_raw)
    x_train = x_normalizer.encode(x_train_raw)
    x_test = x_normalizer.encode(x_test_raw)
    # Reference encodes y_train then immediately decodes it back inside the
    # training loop before computing the loss -- an algebraic no-op (decode
    # exactly inverts encode), so y_train is used here in its raw (physical
    # -unit) form directly; only the model's raw output is ever decoded
    # (see _compute_losses). See module docstring for the full argument.
    y_train = y_train_raw

    config.model.height = s1
    config.model.width = s2
    model = get_model_jax(config)

    key = jax.random.PRNGKey(args.seed)
    x0 = jnp.asarray(x_train[: config.opt.batch_size])
    params = model.init(key, x=x0, fx=None)
    if not args.skip_official_init:
        params = _match_official_weight_init(
            params, seed=args.seed, n_layers=config.model.n_layers
        )
    n_params = int(count_model_params(params))
    print(f"[transolver_pipe_official] model has {n_params:,} trainable params")
    sys.stdout.flush()

    batch_size = config.opt.batch_size
    steps_per_epoch = -(-ntrain // batch_size)  # ceil division, drop_last=False
    n_epochs = config.opt.n_epochs
    total_steps = steps_per_epoch * n_epochs

    lr_schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=config.opt.learning_rate,
        pct_start=config.opt.pct_start,
        div_factor=config.opt.div_factor,
        final_div_factor=config.opt.final_div_factor,
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=config.opt.weight_decay)
    if config.opt.max_grad_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(config.opt.max_grad_norm), tx)
    opt_state = tx.init(params)

    def _compute_losses(params, x, y):
        """out = model(x_normalized_pos); loss = rel_l2_sum(decode(out), y)
        with y already in physical units -- see module docstring for why
        this is exactly equivalent to the reference's own
        encode-then-decode-then-compare of y."""
        out = model.apply(params, x=x, fx=None)[..., 0]  # (B, Npts)
        out = y_normalizer.decode(out)
        return _rel_l2_sum(out, y)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(_compute_losses)(params, x, y)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def eval_step(params, x, y):
        return _compute_losses(params, x, y)

    def run_train_epoch(params, opt_state, rng):
        order = rng.permutation(ntrain)
        total = 0.0
        for start in range(0, ntrain, batch_size):
            idx = order[start : start + batch_size]
            x = jnp.asarray(x_train[idx])
            y = jnp.asarray(y_train[idx])
            params, opt_state, loss = train_step(params, opt_state, x, y)
            total += float(loss)
        return params, opt_state, total / ntrain

    def run_eval():
        total = 0.0
        for start in range(0, ntest, batch_size):
            x = jnp.asarray(x_test[start : start + batch_size])
            y = jnp.asarray(y_test_raw[start : start + batch_size])
            total += float(eval_step(params, x, y))
        return total / ntest

    anim_epochs = (
        set(log_spaced_epochs(n_epochs, args.anim_frames))
        if args.anim_frames
        else set()
    )
    anim_x = jnp.asarray(x_test[args.anim_sample : args.anim_sample + 1])
    pred_frames: list[np.ndarray] = []

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    print(
        f"[transolver_pipe_official] training {n_epochs} epochs, "
        f"{steps_per_epoch} steps/epoch ({total_steps} total steps)"
    )
    sys.stdout.flush()

    best_rel_err = float("inf")
    best_epoch = -1

    for ep in range(n_epochs):
        t0 = perf_counter()
        params, opt_state, train_loss = run_train_epoch(params, opt_state, rng)
        rel_err = run_eval()
        epoch_time = perf_counter() - t0

        if ep in anim_epochs:
            out = y_normalizer.decode(model.apply(params, x=anim_x, fx=None)[..., 0])
            pred_frames.append(np.asarray(out, dtype=np.float32)[0])

        print(f"Epoch {ep} Train loss : {train_loss:.5f}")
        print(f"rel_err:{rel_err}")
        sys.stdout.flush()

        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_epoch = ep
            model.save_checkpoint(
                ckpt_dir, "transolver_pipe_official_best", jax.device_get(params)
            )

        rows.append(
            dict(
                epoch=ep,
                train_loss=train_loss,
                rel_err=rel_err,
                learning_rate=float(lr_schedule((ep + 1) * steps_per_epoch)),
                epoch_time_sec=epoch_time,
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "loss_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        if ep % config.opt.ckpt_iter == 0 or ep == n_epochs - 1:
            model.save_checkpoint(
                ckpt_dir, "transolver_pipe_official", jax.device_get(params)
            )
            print(f"[transolver_pipe_official] checkpoint saved to {ckpt_dir}")
            sys.stdout.flush()

    if pred_frames:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ckpt_dir / "frames.npz",
            pred_frames=np.stack(pred_frames),
            truth_frame=y_test_raw[args.anim_sample],
            xy=x_test_raw[args.anim_sample],
            epochs=np.array(sorted(anim_epochs)),
            grid=np.array([s1, s2]),
            sample=args.anim_sample,
        )

    print(
        f"[transolver_pipe_official] done. best rel_err={best_rel_err:.5f} "
        f"at epoch {best_epoch}. loss_curve.csv -> "
        f"{output_dir / 'loss_curve.csv'}"
    )


if __name__ == "__main__":
    main()
