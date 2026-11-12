# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Gradient scale of the physics-only GINO at initialisation: why it uses no
loss scale.

On one batch (the first ``--batch-size`` released training coefficients) and
the initial parameters (``--seed``), reports for the Q4 Ritz-energy gradient:
the fraction of real gradient components (complex weights counted as two)
below AdamW's ``eps = 1e-8``, their median magnitude, and the global gradient
norm with and without a loss scale equal to the interior node count (6889),
next to the clipping threshold of the training script (1.0).

Usage:
    python scripts/benchmarks/darcy_physics/check_gino_gradient_scale.py --out-dir DIR
"""

import argparse
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    BatchedEnergyFunctionalLossQ4,
    build_grid_aligned_q4_mesh,
    kappa_fn,
    source_fn,
)
from scripts.benchmarks.darcy._data import PixelNormalizer, load_split  # noqa: E402
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PROTOCOL,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    FE_PRECISION,
    sin_boundary_factor,
    split_coefficient,
    torsion_output_scale,
)
from scripts.benchmarks.darcy_physics.train_gino import (  # noqa: E402
    build_model,
    make_loss_fn,
    make_predict_nodes,
)

jax.config.update("jax_enable_x64", False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    train_a_native = load_split(
        args.data_dir, dict(PROTOCOL, n_train=1000, n_test=1, subsample_rate=1)
    )["train_a"]
    train_a, train_cells = split_coefficient(train_a_native, 5, "majority")
    a_lo, a_hi = float(train_a_native.min()), float(train_a_native.max())
    n_side = train_a.shape[1]
    n_points = n_side * n_side
    xy, quad, boundary = build_grid_aligned_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        scale, offset = torsion_output_scale(galerkin, ~boundary, a_lo, a_hi)
    physics_loss = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)
    factor = jnp.asarray(sin_boundary_factor(xy), dtype=jnp.float32)
    normalizer = PixelNormalizer(train_a.reshape(train_a.shape[0], n_points))
    a_enc = jnp.asarray(
        normalizer.encode(train_a.reshape(-1, n_points)).astype(np.float32)
    )[: args.batch_size]
    cells = jnp.asarray(train_cells[: args.batch_size], dtype=jnp.float32)

    model, kwargs, _, _ = build_model(n_side)
    params = model.init(jax.random.PRNGKey(args.seed), x=a_enc[:1, :, None], **kwargs)
    loss_fn = make_loss_fn(
        make_predict_nodes(model, kwargs, scale, offset, factor), physics_loss
    )
    grads = jax.grad(loss_fn)(params, a_enc, cells)
    parts = []
    for g in jax.tree_util.tree_leaves(grads):
        g = np.asarray(g)
        parts.append(
            np.abs(np.concatenate([g.real.ravel(), g.imag.ravel()]))
            if np.iscomplexobj(g)
            else np.abs(g.ravel())
        )
    mag = np.concatenate(parts)
    norm = float(np.sqrt(np.sum(mag**2)))
    interior = int((~boundary).sum())
    result = dict(
        batch_size=args.batch_size,
        seed=args.seed,
        n_real_components=int(mag.size),
        fraction_below_eps=float(np.mean(mag < args.eps)),
        eps=args.eps,
        median_abs_grad=float(np.median(mag)),
        global_norm=norm,
        global_norm_scaled_by_interior_nodes=norm * interior,
        interior_nodes=interior,
        grad_clip=args.grad_clip,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
