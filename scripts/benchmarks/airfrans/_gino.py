# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""GINO on AirfRANS (``--task full``, 800/200), scored with the Transolver paper's
Table 3 protocol (arXiv 2402.02366).

Training uses the recipe the Transolver paper applies to every AirfRANS baseline
(thuml/Transolver ``Airfoil-Design-AirfRANS``):

* a fresh random 32,000-point subset of every case each epoch, one case per step;
* loss = mean-over-channels MSE on volume points + ``--weight`` x the same on surface
  points, on train-normalised fields;
* Adam, one-cycle schedule (peak 1e-3, pct_start 0.3, div 25, final div 1e4) over
  ``(n_train + 1) * epochs`` steps, 398 epochs.

The model is GINO (``scirex.models.GINO``) on a 64 x 32 latent grid over
x in [-2, 4], y in [-1.5, 1.5] (chords), sized to Transolver's AirfRANS model
(3,893,572 parameters). GINO choices:

* both neighbour graphs are rebuilt on the GPU every step, because the point subset
  changes every step: for each query, the ``k`` nearest sources within the radius;
* input GNO radius 0.16 (about 1.7 latent spacings, the GINO paper's default ratio)
  and up to 128 neighbours, since a 32k subset puts thousands of points within the
  radius of latent nodes near the airfoil;
* optional ``--latent-sdf``: the distance from each latent node to the airfoil
  surface, concatenated to the input-GNO output (the GINO paper's SDF input);
* optional ``--decoder-point-features``: each output point's own 7 input features
  (position, SDF, normal, inflow) concatenated to the output-GNO result before the
  projection, so the decoder sees where the wall is; nearly all of the volume error
  otherwise sits within 0.005 chord of the wall, far below the ~0.095-chord latent
  spacing. With ``--out-gno-mlp 512 362`` the size stays matched (3,894,069);
* output queries outside the latent box (about 0.9 % of mesh points) are clamped to
  the box;
* complex spectral weights are updated with ``scale_by_adam_cx``.

The final model is evaluated (no checkpoint selection on the test set) with the
official ``utils/metrics.py`` protocol:

* full-mesh prediction by repeated random 32k-point passes until every point has
  been covered, averaging overlapping predictions;
* wall points are the mesh points that coincide with ``_aerofoil.vtp`` (the
  official ``U_x == 0``); U and nut are set to zero there;
* Volume / Surf: MSE of the normalised fields over the volume / wall points, mean
  over the 4 channels, averaged over cases (the official ``Results_test``; the
  Transolver paper's caption calls these relative L2). With U and nut zero on the
  wall, Surf is the wall-pressure MSE / 4;
* C_L: mean |(C_L - C_L_pred) / C_L|, with both coefficients integrated from
  pressure and wall shear stress on ``_aerofoil.vtp``; rho_L: Spearman correlation
  of true and predicted C_L over the test cases.

Usage: ``train.py --run gino`` trains the published model
(``_config.GINO_ARGS``) and ``infer.py`` scores it. On its own::

    python scripts/benchmarks/airfrans/_gino.py --data-dir <AirfRANS/Dataset> \\
        --cache-dir <cache> --output-dir <out> [--latent-sdf] [--epochs 398]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from scirex.data import load_airfrans  # noqa: E402
from scirex.data.datasets.airfrans import (  # noqa: E402
    OUTPUT_FIELDS,
    get_volume_triangles,
)
from scirex.models import GINO  # noqa: E402
from scirex.training.optim import scale_by_adam_cx  # noqa: E402
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks._plot_utils import save_metrics_json  # noqa: E402

BOX = ((-2.0, 4.0), (-1.5, 1.5))
NU = 1.56e-5  # AirfRANS kinematic viscosity
WALL_TOL = 1e-7  # wall points match exactly; the first off-wall layer is >1.7e-6 away
TRANSOLVER_PARAMS = 3_893_572


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--task", default="full")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--output-dir", default="Output_airfrans_gino")
    ap.add_argument("--epochs", type=int, default=398)
    ap.add_argument("--n-points", type=int, default=32_000)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--final-div-factor", type=float, default=1e4)
    ap.add_argument("--val-every", type=int, default=20, help="monitoring only")
    ap.add_argument("--latent-res", type=int, nargs=2, default=(64, 32))
    ap.add_argument("--in-radius", type=float, default=0.16)
    ap.add_argument("--out-radius", type=float, default=0.15)
    ap.add_argument("--in-k", type=int, default=128)
    ap.add_argument("--out-k", type=int, default=32)
    ap.add_argument("--latent-sdf", action="store_true")
    ap.add_argument(
        "--decoder-point-features",
        action="store_true",
        help="give the decoder each output point's own 7 input features",
    )
    ap.add_argument("--fno-modes", type=int, nargs=2, default=(21, 16))
    ap.add_argument("--fno-hidden", type=int, default=48)
    ap.add_argument("--fno-layers", type=int, default=4)
    ap.add_argument("--fno-lifting-ratio", type=int, default=4)
    ap.add_argument(
        "--out-gno-mlp",
        type=int,
        nargs="+",
        default=(512, 364),
        help="output-GNO kernel MLP hidden widths (used to match Transolver's size)",
    )
    ap.add_argument(
        "--matmul-precision",
        choices=("default", "high", "highest"),
        default="default",
        help="jax_default_matmul_precision; 'default' is TF32 on GPU",
    )
    ap.add_argument("--eval-cases", type=int, default=None, help="limit Table 3 eval")
    ap.add_argument("--eval-only", default=None, help="params .msgpack to evaluate")
    ap.add_argument("--count-params", action="store_true")
    ap.add_argument(
        "--figures-from",
        default=None,
        help="with --eval-only: redraw the figures from this metrics.json instead "
        "of re-scoring the test set",
    )
    ap.add_argument(
        "--contour-case",
        default=None,
        help="test case id for contour.png (default: the median Volume error case)",
    )
    ap.add_argument(
        "--history",
        default=None,
        help="loss_curve.csv for the figures (default: <output-dir>/loss_curve.csv)",
    )
    ap.add_argument(
        "--loader-surf-mask",
        action="store_true",
        help="use the loader's surf mask (includes near-wall points) instead of "
        "the exact wall points",
    )
    return ap


def build_model(args) -> GINO:
    return GINO(
        in_channels=7,
        out_channels=4,
        gno_coord_dim=2,
        fno_in_channels=7,  # the 'linear' input GNO keeps the input width
        latent_feature_channels=1 if args.latent_sdf else None,
        gno_use_open3d=False,
        in_gno_radius=args.in_radius,
        out_gno_radius=args.out_radius,
        fno_n_modes=tuple(args.fno_modes),
        fno_hidden_channels=args.fno_hidden,
        fno_n_layers=args.fno_layers,
        fno_lifting_channel_ratio=args.fno_lifting_ratio,
        out_gno_channel_mlp_hidden_layers=tuple(args.out_gno_mlp),
        out_query_feature_channels=7 if args.decoder_point_features else None,
    )


def latent_grid(res) -> np.ndarray:
    gx = np.linspace(*BOX[0], res[0], dtype=np.float32)
    gy = np.linspace(*BOX[1], res[1], dtype=np.float32)
    return np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)


def radius_neighbors(sources, queries, radius, k, chunk=512) -> dict:
    """k nearest sources within ``radius`` of each query, in IntegralTransform's
    fixed-shape layout; empty slots get segment id ``n_queries`` and are dropped."""
    n_q = queries.shape[0]
    k = min(k, sources.shape[0])
    pad = (-n_q) % chunk
    qp = jnp.concatenate([queries, jnp.zeros((pad, queries.shape[1]), queries.dtype)])

    def block(qb):
        d2 = jnp.sum((qb[:, None, :] - sources[None, :, :]) ** 2, axis=-1)
        neg, idx = jax.lax.top_k(-d2, k)
        return idx, -neg <= radius * radius

    idx, valid = jax.lax.map(block, qp.reshape(-1, chunk, queries.shape[1]))
    idx, valid = idx.reshape(-1, k)[:n_q], valid.reshape(-1, k)[:n_q]
    seg = jnp.where(valid, jnp.arange(n_q, dtype=jnp.int32)[:, None], n_q)
    return {
        "neighbors_index": jnp.where(valid, idx, 0).reshape(-1).astype(jnp.int32),
        "segment_ids": seg.reshape(-1).astype(jnp.int32),
        "counts": valid.sum(axis=1).astype(jnp.int32),
    }


def make_forward(model: GINO, lq: jnp.ndarray, args):
    lq_flat = lq.reshape(-1, 2)
    lo = jnp.array([BOX[0][0], BOX[1][0]], jnp.float32)
    hi = jnp.array([BOX[0][1], BOX[1][1]], jnp.float32)

    def forward(params, x, pos, lat):
        out_q = jnp.clip(pos, lo, hi)
        nbrs_in = radius_neighbors(pos, lq_flat, args.in_radius, args.in_k)
        nbrs_out = radius_neighbors(lq_flat, out_q, args.out_radius, args.out_k)
        out = model.apply(
            params,
            input_geom=pos[None],
            latent_queries=lq[None],
            output_queries=out_q[None],
            x=x[None],
            latent_features=lat[None] if args.latent_sdf else None,
            neighbors_in=nbrs_in,
            neighbors_out=nbrs_out,
            output_query_features=x[None] if args.decoder_point_features else None,
        )
        return out[0]

    return forward


def wall_mask(sample, data_dir) -> np.ndarray:
    """Mesh points that coincide with ``_aerofoil.vtp`` points (the no-slip wall).

    The loader's ``surf`` (distance < 1e-5 to an airfoil point) also catches the
    first off-wall layers: AirfRANS wall cells are ~2e-6 chord thick, so it marks
    about 5x the true wall points, where the velocity is not zero.
    """
    cid = sample["case_id"]
    foil = _read_vtk(Path(data_dir) / cid / f"{cid}_aerofoil.vtp")
    from vtk.util.numpy_support import vtk_to_numpy

    d, _ = cKDTree(vtk_to_numpy(foil.GetPoints().GetData())[:, :2]).query(sample["pos"])
    return d < WALL_TOL


def latent_sdf(sample, lq: np.ndarray) -> np.ndarray:
    """Distance from every latent node to the nearest airfoil-surface mesh point."""
    d, _ = cKDTree(sample["pos"][sample["surf"]]).query(lq.reshape(-1, 2))
    return d.reshape(*lq.shape[:-1], 1).astype(np.float32)


def region_losses(out, y, surf, weight):
    d2 = (out - y) ** 2
    s = surf.astype(out.dtype)[:, None]
    l_surf = jnp.mean(jnp.sum(d2 * s, 0) / (jnp.sum(s) + 1e-8))
    l_vol = jnp.mean(jnp.sum(d2 * (1 - s), 0) / (jnp.sum(1 - s) + 1e-8))
    return l_vol + weight * l_surf, (l_vol, l_surf)


# ---------------------------------------------------------------- Table 3 metrics
def _read_vtk(path):
    import vtk

    reader = (
        vtk.vtkXMLUnstructuredGridReader()
        if str(path).endswith(".vtu")
        else vtk.vtkXMLPolyDataReader()
    )
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def _surface_velocity_gradient(internal, U: np.ndarray, surf: np.ndarray):
    """dU_i/dx_j (2 x 2) at the wall points, from the volume mesh (pyvista's
    compute_derivative, i.e. vtkGradientFilter, as in the official code)."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    grid = vtk.vtkUnstructuredGrid()
    grid.ShallowCopy(internal)
    u3 = np.zeros((U.shape[0], 3), np.float64)
    u3[:, :2] = U
    arr = numpy_to_vtk(u3, deep=True)
    arr.SetName("Upred")
    grid.GetPointData().AddArray(arr)
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(grid)
    gf.SetInputArrayToProcess(0, 0, 0, 0, "Upred")
    gf.SetResultArrayName("grad")
    gf.Update()
    g = vtk_to_numpy(gf.GetOutput().GetPointData().GetArray("grad"))
    return g.reshape(-1, 3, 3)[surf, :2, :2]


def lift_drag(internal, aerofoil, U, p, surf, u_inf, angle_deg):
    """Official Compute_coefficients: integrate pressure and wall shear stress over
    the airfoil cells; returns (C_D, C_L)."""
    from vtk.util.numpy_support import vtk_to_numpy

    mesh_pts = vtk_to_numpy(internal.GetPoints().GetData())[surf, :2]
    foil_pts = vtk_to_numpy(aerofoil.GetPoints().GetData())[:, :2]
    tree = cKDTree(mesh_pts)
    dist, order = tree.query(foil_pts)
    assert dist.max() < 1e-6, "airfoil points do not coincide with wall mesh points"

    jac = _surface_velocity_gradient(internal, U, surf)[order]
    p_s = p[surf][order]
    normals = vtk_to_numpy(aerofoil.GetPointData().GetArray("Normals"))[:, :2]
    S = 0.5 * (jac + jac.transpose(0, 2, 1))
    S = S - np.trace(S, axis1=1, axis2=2)[:, None, None] * np.eye(2)[None] / 3
    wss = ((2 * NU * S) * (-normals)[:, None, :]).sum(axis=2)

    lines = vtk_to_numpy(aerofoil.GetLines().GetData()).reshape(-1, 3)[:, 1:]
    cd = aerofoil.GetCellData()
    length = (
        vtk_to_numpy(cd.GetArray("Length"))
        if cd.GetArray("Length") is not None
        else np.linalg.norm(foil_pts[lines[:, 0]] - foil_pts[lines[:, 1]], axis=1)
    )
    p_c = p_s[lines].mean(axis=1)
    n_c = normals[lines].mean(axis=1)
    wss_c = wss[lines].mean(axis=1)
    wss_int = (wss_c * length[:, None]).sum(0)
    wp_int = (-p_c[:, None] * n_c * length[:, None]).sum(0)
    force = wss_int - wp_int
    a = np.deg2rad(angle_deg)
    basis = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return 2 * (basis @ force) / u_inf**2


def predict_full_mesh(forward_jit, params, s, lat, n_points, rng):
    """Official Infer_test: random n-point passes until every point is covered."""
    n = s["x"].shape[0]
    acc = np.zeros(s["y"].shape, np.float64)
    hits = np.zeros(n, np.int64)
    while (hits == 0).any():
        idx = rng.choice(n, size=min(n_points, n), replace=False)
        out = forward_jit(
            params, jnp.asarray(s["x"][idx]), jnp.asarray(s["pos"][idx]), lat
        )
        acc[idx] += np.asarray(out)
        hits[idx] += 1
    return (acc / hits[:, None]).astype(np.float32)


def evaluate_table3(forward_jit, params, test, lats, coef_norm, args, data_dir):
    _, _, mean_out, std_out = coef_norm
    zero_norm = (0.0 - mean_out) / (std_out + 1e-8)
    rng = np.random.default_rng(0)
    vol, srf, cl_true, cl_pred, cd_true, cd_pred, ids = [], [], [], [], [], [], []
    vol_case = []
    cases = test if args.eval_cases is None else test[: args.eval_cases]
    for i, s in enumerate(cases):
        cid = s["case_id"]
        pred = predict_full_mesh(forward_jit, params, s, lats[cid], args.n_points, rng)
        surf = s["surf"]
        pred[surf, 0:2] = zero_norm[0:2]
        pred[surf, 3] = zero_norm[3]
        d2 = (pred - s["y"]) ** 2
        vol.append(d2[~surf].mean(0))
        vol_case.append(float(d2[~surf].mean()))
        srf.append(d2[surf].mean(0))

        internal = _read_vtk(Path(data_dir) / cid / f"{cid}_internal.vtu")
        aerofoil = _read_vtk(Path(data_dir) / cid / f"{cid}_aerofoil.vtp")
        u_inf, angle = float(cid.split("_")[2]), float(cid.split("_")[3])
        y_true = s["y"] * (std_out + 1e-8) + mean_out
        y_pred = pred * (std_out + 1e-8) + mean_out
        ct = lift_drag(
            internal, aerofoil, y_true[:, :2], y_true[:, 2], surf, u_inf, angle
        )
        cp = lift_drag(
            internal, aerofoil, y_pred[:, :2], y_pred[:, 2], surf, u_inf, angle
        )
        cd_true.append(ct[0])
        cd_pred.append(cp[0])
        cl_true.append(ct[1])
        cl_pred.append(cp[1])
        ids.append(cid)
        if (i + 1) % 20 == 0:
            print(f"[eval] {i + 1}/{len(cases)} cases", flush=True)

    vol, srf = np.mean(vol, 0), np.mean(srf, 0)
    cl_true, cl_pred = np.array(cl_true), np.array(cl_pred)
    cd_true, cd_pred = np.array(cd_true), np.array(cd_pred)
    return {
        "n_cases": len(ids),
        "volume": float(vol.mean()),
        "surf": float(srf.mean()),
        "volume_mse_per_channel": dict(zip(OUTPUT_FIELDS, vol.tolist())),
        "surf_mse_per_channel": dict(zip(OUTPUT_FIELDS, srf.tolist())),
        "C_L_rel_err": float(np.mean(np.abs((cl_true - cl_pred) / cl_true))),
        "rho_L": float(spearmanr(cl_true, cl_pred)[0]),
        "C_D_rel_err": float(np.mean(np.abs((cd_true - cd_pred) / cd_true))),
        "rho_D": float(spearmanr(cd_true, cd_pred)[0]),
        "per_case": {
            "case_id": ids,
            "volume": vol_case,
            "C_L_true": cl_true.tolist(),
            "C_L_pred": cl_pred.tolist(),
        },
    }


# ---------------------------------------------------------------------- figures
# Same layout as the Transolver AirfRANS benchmark's figures.
WINDOW = (-0.5, 1.5, -0.45, 0.45)  # x / c, y / c
CP_RANGE = (-3.0, 1.0)
MODEL = "GINO"


def _save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_contour(path, s, pred, coef_norm, data_dir, label, cl_true, cl_pred):
    """|U|/U_inf and Cp (truth, prediction, absolute error) around the airfoil and
    -Cp on its surface, for one test case (normalised ``pred``, wall imposed)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.tri import Triangulation

    _, _, mean_out, std_out = coef_norm
    cid = s["case_id"]
    u_inf, angle = float(cid.split("_")[2]), float(cid.split("_")[3])
    fields = []
    for y in (s["y"], pred):
        y = y * (std_out + 1e-8) + mean_out
        fields.append(
            (np.linalg.norm(y[:, :2], axis=1) / u_inf, y[:, 2] / (0.5 * u_inf**2))
        )
    tri = Triangulation(
        s["pos"][:, 0],
        s["pos"][:, 1],
        get_volume_triangles(Path(data_dir) / cid / f"{cid}_internal.vtu"),
    )
    fig = plt.figure(figsize=(14, 9.5))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.9))
    u_max = float(np.percentile(fields[0][0], 99.9))
    for row, name in enumerate(("|U| / U$_\\infty$", "Cp")):
        truth, model = fields[0][row], fields[1][row]
        if row == 0:
            kw = dict(cmap="viridis", levels=np.linspace(0.0, u_max, 101))
            clip = (0.0, u_max)
        else:
            kw = dict(
                cmap="RdBu_r",
                norm=TwoSlopeNorm(0.0, *CP_RANGE),
                levels=np.linspace(*CP_RANGE, 101),
            )
            clip = CP_RANGE
        err = np.abs(model - truth)
        panels = (
            ("Ground truth", np.clip(truth, *clip), kw),
            (f"{MODEL} prediction", np.clip(model, *clip), kw),
            (
                "Absolute error",
                err,
                dict(
                    cmap="magma", levels=np.linspace(0, np.percentile(err, 99.9), 101)
                ),
            ),
        )
        for col, (title, val, style) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            ax.set_facecolor("#404040")  # the airfoil interior has no mesh
            m = ax.tricontourf(tri, val, **style)
            fig.colorbar(m, ax=ax, shrink=0.9)
            ax.set_xlim(WINDOW[0], WINDOW[1])
            ax.set_ylim(WINDOW[2], WINDOW[3])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}: {name}", fontsize=11)
    ax = fig.add_subplot(grid[2, :])
    wall = s["surf"]
    order = np.argsort(s["pos"][wall, 0])
    x = s["pos"][wall, 0][order]
    ax.plot(
        x, -fields[0][1][wall][order], ".", ms=3, color="black", label="Ground truth"
    )
    ax.plot(
        x,
        -fields[1][1][wall][order],
        ".",
        ms=2,
        color="tab:purple",
        label=f"{MODEL} prediction",
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("-Cp on the airfoil surface")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(
        f"Test case {cid} ({label}): "
        f"U$_\\infty$ = {u_inf:.1f} m/s, angle of attack {angle:.2f}°, "
        f"$C_L$ true {cl_true:.4f} / predicted {cl_pred:.4f}",
        fontsize=12,
    )
    _save(fig, path)


def _read_history(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_error_curve(history, path):
    import matplotlib.pyplot as plt

    ep = [int(r["epoch"]) for r in history]
    test = [r for r in history if r["test_loss"]]
    te = [int(r["epoch"]) for r in test]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(
        ep, [float(r["train_vol"]) for r in history], lw=1.2, label="train, volume MSE"
    )
    ax.plot(
        ep,
        [float(r["train_surf"]) for r in history],
        lw=1.2,
        label="train, surface MSE",
    )
    ax.plot(
        te,
        [float(r["test_vol"]) for r in test],
        "o-",
        ms=4,
        label="test (200 cases, monitoring only), volume MSE",
    )
    ax.plot(
        te,
        [float(r["test_surf"]) for r in test],
        "s-",
        ms=4,
        label="test (200 cases, monitoring only), surface MSE",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalised targets, log scale)")
    ax.set_title(f"{MODEL} on AirfRANS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_compute_cost(history, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        [int(r["epoch"]) for r in history],
        [float(r["sec"]) for r in history],
        color="tab:purple",
        lw=1.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Wall time per epoch (s, log scale)", fontsize=12)
    ax.set_title(f"Per-epoch compute cost ({MODEL} on AirfRANS)", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, path)


def make_figures(res, forward_jit, params, test, lats, coef_norm, args, out_dir):
    """contour.png, error_curve.png and compute_cost.png next to metrics.json."""
    import matplotlib

    matplotlib.use("Agg")
    per_case = res["per_case"]
    vols = np.array(per_case["volume"])
    if args.contour_case:
        k = per_case["case_id"].index(args.contour_case)
        label = f"Volume error {vols[k]:.2e}"
    else:
        k = int(np.argsort(vols)[len(vols) // 2])
        label = f"median Volume error {vols[k]:.2e}"
    by_id = {s["case_id"]: s for s in test}
    s = by_id[per_case["case_id"][k]]
    pred = predict_full_mesh(
        forward_jit,
        params,
        s,
        lats[s["case_id"]],
        args.n_points,
        np.random.default_rng(0),
    )
    _, _, mean_out, std_out = coef_norm
    zero_norm = (0.0 - mean_out) / (std_out + 1e-8)
    pred[s["surf"], 0:2] = zero_norm[0:2]
    pred[s["surf"], 3] = zero_norm[3]
    plot_contour(
        out_dir / "contour.png",
        s,
        pred,
        coef_norm,
        args.data_dir,
        label,
        per_case["C_L_true"][k],
        per_case["C_L_pred"][k],
    )
    hist = Path(args.history) if args.history else out_dir / "loss_curve.csv"
    if hist.exists():
        history = _read_history(hist)
        plot_error_curve(history, out_dir / "error_curve.png")
        plot_compute_cost(history, out_dir / "compute_cost.png")


# ------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.figures_from and not args.eval_only:
        parser.error("--figures-from needs --eval-only (the model to draw)")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    model = build_model(args)
    lq_np = latent_grid(args.latent_res)
    lq = jnp.asarray(lq_np)
    forward = make_forward(model, lq, args)
    dummy_lat = jnp.zeros((*lq_np.shape[:-1], 1), jnp.float32)

    def init(n):
        key = jax.random.PRNGKey(0)
        pos = jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)
        x = jnp.zeros((n, 7), jnp.float32)
        return model.init(
            key,
            input_geom=pos[None],
            latent_queries=lq[None],
            output_queries=pos[None],
            x=x[None],
            latent_features=dummy_lat[None] if args.latent_sdf else None,
            neighbors_in=radius_neighbors(
                pos, lq.reshape(-1, 2), args.in_radius, args.in_k
            ),
            neighbors_out=radius_neighbors(
                lq.reshape(-1, 2), pos, args.out_radius, args.out_k
            ),
            output_query_features=x[None] if args.decoder_point_features else None,
        )

    if args.count_params:
        n = int(count_model_params(jax.eval_shape(lambda: init(4096))))
        print(f"{n:,} params ({n / TRANSOLVER_PARAMS - 1:+.2%} vs Transolver)")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[airfrans_gino] devices={jax.devices()}  args={vars(args)}", flush=True)
    data = load_airfrans(
        Path(args.data_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        max_cases=args.max_cases,
        task=args.task,
    )
    train, test = data["train"], data["test"]
    if not args.loader_surf_mask:
        for s in train + test:
            s["surf"] = wall_mask(s, args.data_dir)
    lats = {
        s["case_id"]: jnp.asarray(
            latent_sdf(s, lq_np) if args.latent_sdf else dummy_lat
        )
        for s in train + test
    }
    forward_jit = jax.jit(forward)

    if args.eval_only:
        params = serialization.from_bytes(
            init(args.n_points), Path(args.eval_only).read_bytes()
        )
    else:
        params = init(args.n_points)
        n_params = int(count_model_params(params))
        print(
            f"[airfrans_gino] n_train={len(train)} n_test={len(test)} "
            f"params={n_params:,} "
            f"({n_params / TRANSOLVER_PARAMS - 1:+.2%} vs Transolver)",
            flush=True,
        )
        total_steps = (len(train) + 1) * args.epochs
        schedule = optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=args.learning_rate,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=args.final_div_factor,
        )
        tx = optax.chain(
            scale_by_adam_cx(b1=0.9, b2=0.999, eps=1e-8),
            optax.scale_by_learning_rate(schedule),
        )
        opt_state = tx.init(params)

        def loss_fn(p, x, y, surf, pos, lat):
            return region_losses(forward(p, x, pos, lat), y, surf, args.weight)

        @jax.jit
        def train_step(p, o, x, y, surf, pos, lat):
            (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
                p, x, y, surf, pos, lat
            )
            u, o = tx.update(g, o, p)
            return optax.apply_updates(p, u), o, loss, aux

        eval_step = jax.jit(loss_fn)
        rng = np.random.default_rng(0)

        def batch(s):
            idx = rng.choice(s["x"].shape[0], size=args.n_points, replace=False)
            return (
                jnp.asarray(s["x"][idx]),
                jnp.asarray(s["y"][idx]),
                jnp.asarray(s["surf"][idx]),
                jnp.asarray(s["pos"][idx]),
                lats[s["case_id"]],
            )

        log = out_dir / "loss_curve.csv"
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_vol",
                    "train_surf",
                    "test_loss",
                    "test_vol",
                    "test_surf",
                    "lr",
                    "sec",
                ]
            )
        for epoch in range(args.epochs):
            t0 = perf_counter()
            stats = []
            for i in rng.permutation(len(train)):
                params, opt_state, loss, (lv, ls) = train_step(
                    params, opt_state, *batch(train[i])
                )
                stats.append((loss, lv, ls))
            tr = np.asarray(jnp.stack([jnp.stack(s) for s in stats])).mean(0)
            te = ["", "", ""]
            if epoch % args.val_every == 0 or epoch == args.epochs - 1:
                ev = [eval_step(params, *batch(s)) for s in test]
                te = (
                    np.asarray(jnp.stack([jnp.stack([l, a[0], a[1]]) for l, a in ev]))
                    .mean(0)
                    .tolist()
                )
            lr = float(schedule((epoch + 1) * (len(train) + 1)))
            row = [epoch, *tr.tolist(), *te, lr, perf_counter() - t0]
            with open(log, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(
                f"epoch {epoch}: train={tr[0]:.5g} (vol {tr[1]:.5g}, surf {tr[2]:.5g})"
                + (f" test={te[0]:.5g}" if te[0] != "" else "")
                + f" {row[-1]:.1f}s",
                flush=True,
            )
            if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
                (out_dir / "params_final.msgpack").write_bytes(
                    serialization.to_bytes(jax.device_get(params))
                )

    if args.figures_from:
        res = json.loads(Path(args.figures_from).read_text())
    else:
        t0 = perf_counter()
        res = evaluate_table3(
            forward_jit, params, test, lats, data["coef_norm"], args, args.data_dir
        )
        res["eval_sec"] = perf_counter() - t0
        res["args"] = vars(args)
        save_metrics_json(res, out_dir / "metrics.json")
    make_figures(res, forward_jit, params, test, lats, data["coef_norm"], args, out_dir)
    print(
        f"[table3] Volume={res['volume']:.4f} Surf={res['surf']:.4f} "
        f"C_L={res['C_L_rel_err']:.4f} rho_L={res['rho_L']:.4f} "
        "(paper GINO 0.0297/0.0482/0.1821/0.9958, "
        "Transolver 0.0037/0.0142/0.1030/0.9978)",
        flush=True,
    )


if __name__ == "__main__":
    main()
