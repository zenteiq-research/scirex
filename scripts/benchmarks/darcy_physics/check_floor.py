# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Darcy benchmark, formulation floor: how close does an *exact*
solve of our Q4 discretization get to the published solutions?

A physics-only model never sees the released solutions -- its loss drives it
towards the Q4 Galerkin solution of ``-div(a grad u) = 1``, ``u = 0`` on the
boundary. That solution is then scored against Li et al.'s ``sol``, which came
from their own solver on the native 421 x 421 grid. The gap between the two
discrete solutions is what a perfectly trained physics-only model (one that
reaches the minimiser of its loss for every input) scores on this metric --
the reference point for any physics-only result. It is not a strict bound: a
model's own error could happen to point towards Li et al.'s solution, though
nothing in the loss pushes it that way. This script measures the gap
directly: no model, no training -- one sparse
FEM solve per test instance with the same mesh, quadrature and solver
(:func:`~scirex.losses.darcy_piecewise_coefficient.assemble_and_solve_q4`)
the physics loss is built on.

Data and metric are the published protocol, unchanged: the first ``--n-test``
(200) instances of ``piececonst_r421_N1024_smooth2.mat``, loaded with the
data-driven benchmark's loader (``darcy._data.load_split``) and scored with
its per-instance relative L2 over all grid nodes.

Reported per configuration (mesh resolution x cell-coefficient mode, see
``_ritz.py``):

- resolution 84 (s=85, the benchmark grid) with each cell mode, on all test
  instances -- the score of a perfectly trained physics-only model on the
  benchmark grid;
- resolution 420 (Li et al.'s native grid) on the first ``--n-test-fine``
  instances -- whether the Q4 formulation reproduces the released solutions
  at all once mesh coarsening is taken out.

Each floor is reported twice: at the Q4 nodes, and at the points where the
released solutions are sampled (native cell centres, half a native cell
inside the nodes), reached by the fixed linear map
:func:`._ritz.cell_centre_interpolation` that training also uses for
scoring.

Quadrature check (benchmark grid, independent float64 assembler): with one
coefficient value per element, the bilinear stiffness integrand is at most
quadratic in each direction on square cells, so 2 x 2 Gauss points (the loss's
rule) integrate it exactly. The script verifies this (3 x 3 vs 2 x 2
stiffness) and also reports the floor when the native coefficient is instead
sampled at each Gauss point (2 x 2 and 3 x 3), i.e. when the jumps inside an
element are resolved by the quadrature.

The finite-element contractions run at full float32 precision inside this
script (``jax.default_matmul_precision("highest")``): on GPU, JAX defaults
float32 matmuls to TF32, which perturbs the Q4 Jacobians by up to ~5%.

Usage (from the repo root)::

    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth2.mat
    python scripts/benchmarks/darcy_physics/check_floor.py

Written to ``doc/assets/benchmarks/darcy_q4_s85_spec_floor/metrics.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.losses import Q4Galerkin  # noqa: E402
from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    assemble_and_solve_q4,
    build_grid_aligned_q4_mesh,
)
from scripts.benchmarks._plot_utils import save_metrics_json  # noqa: E402
from scripts.benchmarks.darcy._data import load_split, rel_l2  # noqa: E402
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    FLOOR_OUT_DIR,
    PROTOCOL,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    CELL_MODES,
    cell_centre_interpolation,
    node_to_cell_coefficient,
)


def q4_floor(
    a_node: np.ndarray, u_node: np.ndarray, resolution: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Q4 solve for each instance vs. the released solution on the
    strided ``resolution + 1`` grid. ``a_node``/``u_node`` shape
    ``(N, n, n)`` on the native grid. Returns per-instance relative L2 at the
    nodes and at the released sample points (cell-centre map)."""
    xy, quad, boundary_mask = build_grid_aligned_q4_mesh(resolution)
    galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    interior_mask = ~boundary_mask
    a_cells = node_to_cell_coefficient(a_node, resolution, mode)
    rate = (a_node.shape[1] - 1) // resolution
    truth = u_node[:, ::rate, ::rate].reshape(a_node.shape[0], -1)

    m = cell_centre_interpolation(resolution + 1, rate)
    errs = np.empty(a_node.shape[0])
    errs_mapped = np.empty(a_node.shape[0])
    t0 = time.time()
    for i in range(a_node.shape[0]):
        u = assemble_and_solve_q4(
            galerkin, jnp.asarray(a_cells[i], dtype=jnp.float32), interior_mask
        )
        errs[i] = rel_l2(u, truth[i])
        u_grid = np.asarray(u, dtype=np.float64).reshape(resolution + 1, resolution + 1)
        errs_mapped[i] = rel_l2((m @ u_grid @ m.T).reshape(-1), truth[i])
        if i == 0 or (i + 1) % 20 == 0 or i == a_node.shape[0] - 1:
            print(
                f"[check_floor] res={resolution} mode={mode} "
                f"{i + 1}/{a_node.shape[0]}  running mean rel_l2="
                f"{errs[: i + 1].mean():.4%}, mapped "
                f"{errs_mapped[: i + 1].mean():.4%}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return errs, errs_mapped


def summarize(errs: np.ndarray) -> dict:
    return {
        "n_instances": int(errs.size),
        "rel_l2_mean": float(errs.mean()),
        "rel_l2_median": float(np.median(errs)),
        "rel_l2_p95": float(np.percentile(errs, 95)),
        "rel_l2_max": float(errs.max()),
        "rel_l2_per_instance": errs.tolist(),
    }


def _gauss_q4(
    n: int, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n x n Gauss rule on a square Q4 element of side h: per Gauss point,
    ``w det(J) grad N_i . grad N_j`` (Q, 4, 4), ``sum_q w det(J) N_i`` (4,),
    and the point offsets inside the element (Q,), (Q,)."""
    ref = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    x, w = np.polynomial.legendre.leggauss(n)
    xi, eta = (g.ravel() for g in np.meshgrid(x, x, indexing="ij"))
    wq = np.outer(w, w).ravel() * h * h / 4.0
    dn = np.stack(
        [
            0.25 * ref[:, 0][None] * (1 + ref[:, 1][None] * eta[:, None]),
            0.25 * ref[:, 1][None] * (1 + ref[:, 0][None] * xi[:, None]),
        ],
        axis=-1,
    ) * (2.0 / h)
    n_q = (
        0.25
        * (1 + ref[:, 0][None] * xi[:, None])
        * (1 + ref[:, 1][None] * eta[:, None])
    )
    gg = np.einsum("qid,qjd->qij", dn, dn) * wq[:, None, None]
    return gg, (n_q * wq[:, None]).sum(0), (xi + 1) * h / 2, (eta + 1) * h / 2


def q4_solve_gauss(
    coef_q: np.ndarray, resolution: int, n_gauss: int
) -> tuple[np.ndarray, object]:
    """Float64 Q4 solve of ``-div(a grad u) = 1``, ``u = 0`` on the boundary,
    with an ``n_gauss x n_gauss`` rule and coefficient ``coef_q`` (T, Q) given
    at every Gauss point of every element (element ``(i, j)`` = cell
    ``[i h, (i+1) h] x [j h, (j+1) h]``). Returns nodal u (x-major) and K."""
    s, h = resolution + 1, 1.0 / resolution
    gg, fn, _, _ = _gauss_q4(n_gauss, h)
    ix, iy = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    n00 = (ix * s + iy).ravel()
    conn = np.stack([n00, n00 + s, n00 + s + 1, n00 + 1], axis=-1)
    k_e = np.einsum("tq,qij->tij", coef_q, gg)
    k = coo_matrix(
        (
            k_e.ravel(),
            (np.repeat(conn, 4, axis=1).ravel(), np.tile(conn, (1, 4)).ravel()),
        ),
        shape=(s * s, s * s),
    ).tocsr()
    f = np.zeros(s * s)
    np.add.at(f, conn.ravel(), np.tile(fn, conn.shape[0]))
    bx, by = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    inner = ~((bx == 0) | (bx == resolution) | (by == 0) | (by == resolution)).ravel()
    u = np.zeros(s * s)
    idx = np.where(inner)[0]
    u[idx] = splu(k[idx][:, idx].tocsc()).solve(f[idx])
    return u, k


def coefficient_at_gauss(
    a_node: np.ndarray, cells: np.ndarray | None, resolution: int, n_gauss: int
) -> np.ndarray:
    """(T, Q) coefficient: the element value ``cells`` repeated at every Gauss
    point, or (``cells=None``) the native coefficient sampled at each Gauss
    point (nearest native cell centre ``(k + 1/2) / n_native``)."""
    h = 1.0 / resolution
    _, _, gx, gy = _gauss_q4(n_gauss, h)
    if cells is not None:
        return np.repeat(cells.reshape(-1, 1), gx.size, axis=1)
    n_native = a_node.shape[0]
    ix, iy = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    px = ix.reshape(-1, 1) * h + gx[None]
    py = iy.reshape(-1, 1) * h + gy[None]
    kx = np.clip(np.floor(px * n_native).astype(int), 0, n_native - 1)
    ky = np.clip(np.floor(py * n_native).astype(int), 0, n_native - 1)
    return a_node[kx, ky]


def quadrature_check(a_node: np.ndarray, u_node: np.ndarray, resolution: int) -> dict:
    """Stiffness exactness of the 2 x 2 rule and floors with the coefficient
    per element (majority) or sampled at the Gauss points."""
    rate = (a_node.shape[1] - 1) // resolution
    s = resolution + 1
    truth = u_node[:, ::rate, ::rate].reshape(a_node.shape[0], -1)
    m = cell_centre_interpolation(s, rate)
    maj = node_to_cell_coefficient(a_node, resolution, "majority")
    _, k2 = q4_solve_gauss(
        coefficient_at_gauss(a_node[0], maj[0], resolution, 2), resolution, 2
    )
    _, k3 = q4_solve_gauss(
        coefficient_at_gauss(a_node[0], maj[0], resolution, 3), resolution, 3
    )
    k_diff = float(abs(k3 - k2).max() / abs(k2).max())
    print(
        "[check_floor] quadrature: max|K_3x3 - K_2x2| / max|K| = "
        f"{k_diff:.2e} "
        "(majority coefficient)",
        flush=True,
    )
    out = {"stiffness_3x3_vs_2x2_max_rel_diff": k_diff}
    for name, n_gauss, per_element in (
        ("majority_2x2", 2, True),
        ("majority_3x3", 3, True),
        ("gauss_sampled_2x2", 2, False),
        ("gauss_sampled_3x3", 3, False),
    ):
        errs, errs_mapped = np.empty(a_node.shape[0]), np.empty(a_node.shape[0])
        for i in range(a_node.shape[0]):
            coef = coefficient_at_gauss(
                a_node[i], maj[i] if per_element else None, resolution, n_gauss
            )
            u, _ = q4_solve_gauss(coef, resolution, n_gauss)
            u_grid = u.reshape(s, s)
            errs[i] = rel_l2(u_grid.reshape(-1), truth[i])
            errs_mapped[i] = rel_l2((m @ u_grid @ m.T).reshape(-1), truth[i])
        out[name] = {"nodes": summarize(errs), "mapped": summarize(errs_mapped)}
        print(
            f"[check_floor] quadrature {name:<18} mapped mean="
            f"{errs_mapped.mean():.4%}  nodes mean={errs.mean():.4%}",
            flush=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--n-test", type=int, default=PROTOCOL["n_test"])
    ap.add_argument("--resolution", type=int, default=84)
    ap.add_argument(
        "--n-test-fine",
        type=int,
        default=20,
        help="Instances for the native-resolution (420-cell) check; each "
        "solve has ~177k unknowns. 0 skips it.",
    )
    ap.add_argument("--out-dir", type=Path, default=FLOOR_OUT_DIR)
    args = ap.parse_args()

    test_file = PROTOCOL["test_file"]
    print(
        f"[check_floor] loading {test_file} (native s=421) from {args.data_dir}...",
        flush=True,
    )
    # load_split always reads the training file too; one sample is enough.
    split = load_split(
        args.data_dir,
        dict(PROTOCOL, n_train=1, n_test=args.n_test, subsample_rate=1),
    )
    a_node = split["test_a"].astype(np.float64)
    u_node = split["test_u"].astype(np.float64)
    values = np.unique(a_node)
    print(
        f"[check_floor] coeff shape={a_node.shape} values={values.tolist()}",
        flush=True,
    )
    if values.size != 2:
        raise ValueError(
            f"expected a two-valued piecewise-constant coefficient, got {values}"
        )
    native_cells = a_node.shape[1] - 1

    configs = [(args.resolution, mode, args.n_test) for mode in CELL_MODES]
    if args.n_test_fine:
        configs += [
            (native_cells, mode, min(args.n_test_fine, args.n_test))
            for mode in ("corner", "mean")
        ]

    results = {}
    with jax.default_matmul_precision("highest"):
        for resolution, mode, n in configs:
            errs, errs_mapped = q4_floor(a_node[:n], u_node[:n], resolution, mode)
            results[f"res{resolution}_{mode}"] = {
                "resolution": resolution,
                "s": resolution + 1,
                "cell_mode": mode,
                **summarize(errs),
                "mapped": summarize(errs_mapped),
            }
    quadrature = quadrature_check(a_node, u_node, args.resolution)

    print("[check_floor] summary (exact Q4 solve vs. released sol):")
    for name, r in results.items():
        print(
            f"  {name:<18} n={r['n_instances']:<4} mean={r['rel_l2_mean']:.4%}  "
            f"median={r['rel_l2_median']:.4%}  p95={r['rel_l2_p95']:.4%}  "
            f"max={r['rel_l2_max']:.4%}  | mapped mean="
            f"{r['mapped']['rel_l2_mean']:.4%}",
            flush=True,
        )

    metrics = {
        "problem": "darcy_q4_s85_spec_floor",
        "description": (
            "Exact Q4 Galerkin solve of -div(a grad u) = 1, u = 0 on the "
            "boundary, scored against Li et al.'s released solutions with the "
            "benchmark's per-instance relative L2: the score of a perfectly "
            "trained physics-only model (the exact minimiser of a loss on "
            "this discretization)."
        ),
        "data_source": (
            f"Li et al.'s {test_file}, first {args.n_test} instances, native s=421"
        ),
        "coefficient_values": values.tolist(),
        "results": results,
        "quadrature": quadrature,
    }
    out_path = save_metrics_json(metrics, args.out_dir / "metrics.json")
    print(f"[check_floor] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
