# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License (the training
# recipe and the metrics); see THIRD_PARTY_NOTICES.md.

"""
Predicting the flow around an airfoil with GINO (AirfRANS)
==========================================================

**Problem.** Given the CFD mesh of one airfoil case (about 180,000 nodes) and
its inflow, predict the steady velocity ``(Ux, Uy)``, pressure ``p`` and
turbulent viscosity ``nut`` at every node. The 1,000 cases of AirfRANS
(Bonnet et al. 2022) are steady 2D RANS simulations with the k-omega SST
model, each on its own mesh. The model is the Geometry-Informed Neural
Operator (GINO, Li et al. 2023): a graph neural operator lifts the mesh onto a
regular latent grid, an FNO processes the grid, and a second graph operator
brings the result back to every mesh node.

**What you will learn.**

1. How to load AirfRANS with ``scirex.data.load_airfrans`` and the official
   "full" split, and how to make a small split for a CPU.
2. How to build ``scirex.models.GINO`` with a latent grid, a signed-distance
   latent channel and per-node decoder features, and count its parameters.
3. How to build the two neighbour graphs (mesh to grid and grid to mesh) for
   every random subset of nodes, inside the jitted step.
4. How to train with the complex-aware ``scale_by_adam_cx`` on a one-cycle
   schedule, which GINO's complex FNO weights need.
5. How to score with the official metrics: Volume and Surf (normalised MSE),
   the lift coefficient C_L and Spearman's rho of C_L; and how to plot the
   pressure coefficient.

**Caveat.** The ``--smoke`` mode trains a tiny GINO on a few cases for a few
epochs. It checks that the pipeline works and gives poor predictions. It is
not a benchmark number. For the benchmark protocol (800 training cases, 500
epochs, 3,894,069 parameters) use ``scripts/benchmarks/airfrans/train.py
--run gino``; see the "Reproduce at full scale" section of the tutorial page.

Usage, from the repository root (needs the ``benchmark`` extra for ``vtk``)::

    python scripts/tutorials/airfrans_gino.py \\
        --data-dir ~/data/AirfRANS/Dataset --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.data.load_airfrans``,
# ``scirex.models.GINO``, ``scirex.training.optim.scale_by_adam_cx`` and
# ``scirex.utils.count_model_params``. The loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import flax
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import vtk
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import scirex
from scirex.data import load_airfrans
from scirex.models import GINO
from scirex.training.optim import scale_by_adam_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# The model is meant to run in float32, also if float64 was enabled earlier
# in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0
NU = 1.56e-5  # kinematic viscosity of the AirfRANS simulations
# The latent grid's box, Transolver's reference box: x in [-2, 4],
# y in [-1.5, 1.5] chords.
BOX = ((-2.0, 4.0), (-1.5, 1.5))
# Wall nodes coincide exactly with airfoil points; the first off-wall layer is
# more than 1.7e-6 chord away.
WALL_TOL = 1e-7

# Training settings of the benchmark's GINO run (scripts/benchmarks/airfrans/
# _gino.py with _config.GINO_ARGS): the recipe the Transolver paper applies to
# its AirfRANS baselines, for 500 epochs.
PROTOCOL = dict(
    n_points=32_000,
    epochs=500,
    learning_rate=1e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    surf_weight=1.0,
    test_every=20,
)

# Model sizes. "full" is the benchmark's GINO (3,894,069 parameters, matched
# to the paper's Transolver, 3,893,572). "smoke" is a small GINO for a CPU.
SIZES = dict(
    full=dict(
        latent_res=(128, 64),
        in_radius=0.16,
        out_radius=0.15,
        in_k=128,
        out_k=32,
        fno_modes=(21, 16),
        fno_hidden=48,
        fno_layers=4,
        fno_lifting_ratio=4,
        out_gno_mlp=(512, 362),
    ),
    smoke=dict(
        latent_res=(64, 32),
        in_radius=0.16,
        out_radius=0.15,
        in_k=32,
        out_k=16,
        fno_modes=(16, 12),
        fno_hidden=16,
        fno_layers=2,
        fno_lifting_ratio=2,
        out_gno_mlp=(64, 64),
    ),
)


def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__} | vtk {vtk.vtkVersion.GetVTKVersion()}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")


# %%
# Load the data
# -------------
# ``load_airfrans(task="full")`` reads the official split from
# ``manifest.json``: 800 training and 200 test cases. Each case is a dict with
# ``x`` ``(N, 7)`` = ``[x, y, signed distance, direction to the nearest wall
# point (2), inflow velocity (2)]``, ``y`` ``(N, 4)`` = ``[Ux, Uy, p, nut]``,
# both normalised with the mean and std of the training nodes (``coef_norm``),
# ``pos`` ``(N, 2)`` and ``surf``, a mask of the nodes within 1e-5 of an
# airfoil point. That mask also catches the first off-wall mesh layers (about
# 5 times as many nodes as the wall itself); the benchmark trains on it, as
# here, and scores on the exact wall nodes (``wall_mask``). For a small run,
# ``write_small_manifest`` writes a manifest with the first cases of each
# split, and the loader reads only those.


def write_small_manifest(data_dir: Path, out: Path, n_train: int, n_test: int) -> Path:
    """A ``manifest.json`` with the first ``n_train`` / ``n_test`` full-task cases."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    small = dict(
        full_train=manifest["full_train"][:n_train],
        full_test=manifest["full_test"][:n_test],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(small))
    return out


def load_data(data_dir: Path, cache_dir: Path, manifest: Path | None) -> dict:
    """The full-task split and its normalisation; prints what was read."""
    data = load_airfrans(
        data_dir, cache_dir=cache_dir, task="full", manifest_path=manifest
    )
    n_nodes = [len(s["x"]) for s in data["train"] + data["test"]]
    print(
        f"cases: {len(data['train'])} train / {len(data['test'])} test | "
        f"nodes per case: min {min(n_nodes)}, max {max(n_nodes)}"
    )
    return data


# %%
# Build the model
# ---------------
# ``GINO`` takes the node features ``x`` at the node positions, lifts them to
# the latent grid with an input graph operator (each latent node aggregates
# the ``in_k`` nearest nodes within ``in_radius``), runs a 2D FNO on the
# grid, and evaluates an output graph operator at every queried node (the
# ``out_k`` nearest latent nodes within ``out_radius``). Two options of the
# benchmark run: the distance from every latent node to the airfoil as an
# extra latent channel (``latent_feature_channels=1``), and each node's own 7
# features concatenated to the output-GNO result before the projection
# (``out_query_feature_channels=7``), so that the decoder knows where the
# wall is.


def build_model(size: dict) -> GINO:
    """The benchmark's GINO at the size ``size``."""
    return GINO(
        in_channels=7,
        out_channels=4,
        gno_coord_dim=2,
        fno_in_channels=7,
        latent_feature_channels=1,
        gno_use_open3d=False,
        in_gno_radius=size["in_radius"],
        out_gno_radius=size["out_radius"],
        fno_n_modes=tuple(size["fno_modes"]),
        fno_hidden_channels=size["fno_hidden"],
        fno_n_layers=size["fno_layers"],
        fno_lifting_channel_ratio=size["fno_lifting_ratio"],
        out_gno_channel_mlp_hidden_layers=tuple(size["out_gno_mlp"]),
        out_query_feature_channels=7,
    )


def latent_grid(res) -> np.ndarray:
    """``(res_x, res_y, 2)`` latent node positions over ``BOX``."""
    gx = np.linspace(*BOX[0], res[0], dtype=np.float32)
    gy = np.linspace(*BOX[1], res[1], dtype=np.float32)
    return np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)


def latent_sdf(sample: dict, lq: np.ndarray) -> np.ndarray:
    """Distance from every latent node to the nearest airfoil-surface node."""
    d, _ = cKDTree(sample["pos"][sample["surf"]]).query(lq.reshape(-1, 2))
    return d.reshape(*lq.shape[:-1], 1).astype(np.float32)


# %%
# Neighbour graphs
# ----------------
# The node subset changes at every step, so both graphs are rebuilt inside
# the jitted step: for every query, the ``k`` nearest sources by brute force
# (``lax.top_k`` over the squared distances, in chunks of queries), kept if
# within the radius. The result has a fixed shape, so the step compiles once.
# Output queries outside the latent box (about 0.9 % of the nodes) are clamped
# to the box.


def radius_neighbors(sources, queries, radius, k, chunk=512) -> dict:
    """The ``k`` nearest sources within ``radius`` of each query, as GINO's
    neighbour layout; empty slots get segment id ``n_queries``."""
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


def make_forward(model: GINO, lq: jnp.ndarray, size: dict):
    """``forward(params, x, pos, lat) -> (N, 4)`` for one node subset."""
    lq_flat = lq.reshape(-1, 2)
    lo = jnp.array([BOX[0][0], BOX[1][0]], jnp.float32)
    hi = jnp.array([BOX[0][1], BOX[1][1]], jnp.float32)

    def forward(params, x, pos, lat):
        out_q = jnp.clip(pos, lo, hi)
        nbrs_in = radius_neighbors(pos, lq_flat, size["in_radius"], size["in_k"])
        nbrs_out = radius_neighbors(lq_flat, out_q, size["out_radius"], size["out_k"])
        out = model.apply(
            params,
            input_geom=pos[None],
            latent_queries=lq[None],
            output_queries=out_q[None],
            x=x[None],
            latent_features=lat[None],
            neighbors_in=nbrs_in,
            neighbors_out=nbrs_out,
            output_query_features=x[None],
        )
        return out[0]

    return forward


def init_params(model: GINO, lq: jnp.ndarray, size: dict, seed: int, n: int):
    """Initialise on ``n`` random points, as the benchmark, and count parameters."""
    key = jax.random.PRNGKey(seed)
    pos = jax.random.uniform(key, (n, 2), minval=-1.0, maxval=1.0)
    x = jnp.zeros((n, 7), jnp.float32)
    params = model.init(
        key,
        input_geom=pos[None],
        latent_queries=lq[None],
        output_queries=pos[None],
        x=x[None],
        latent_features=jnp.zeros((*lq.shape[:-1], 1), jnp.float32)[None],
        neighbors_in=radius_neighbors(
            pos, lq.reshape(-1, 2), size["in_radius"], size["in_k"]
        ),
        neighbors_out=radius_neighbors(
            lq.reshape(-1, 2), pos, size["out_radius"], size["out_k"]
        ),
        output_query_features=x[None],
    )
    print(f"GINO | parameters: {count_model_params(params):,}")
    return params


# %%
# Loss
# ----
# The official ``MSE_weighted``: the MSE of each channel on the non-wall
# ("volume") nodes and on the surface nodes, each averaged over the 4
# channels, summed with weight 1, on normalised fields.


def region_losses(out, y, surf, weight):
    """Volume + ``weight`` x surface MSE, and the two parts."""
    d2 = (out - y) ** 2
    s = surf.astype(out.dtype)[:, None]
    l_surf = jnp.mean(jnp.sum(d2 * s, 0) / (jnp.sum(s) + 1e-8))
    l_vol = jnp.mean(jnp.sum(d2 * (1 - s), 0) / (jnp.sum(1 - s) + 1e-8))
    return l_vol + weight * l_surf, (l_vol, l_surf)


# %%
# Optimiser for complex weights
# -----------------------------
# GINO's FNO has complex spectral weights. For a real loss and a complex
# weight, ``jax.grad`` returns the conjugate of the steepest-ascent direction
# that optax transforms assume, so plain Adam would move the imaginary parts
# the wrong way. ``scale_by_adam_cx`` conjugates complex gradients first and
# keeps the second moment real; real weights are updated exactly as by Adam.
# The learning rate follows ``optax.cosine_onecycle_schedule`` over
# ``(n_train + 1) * epochs`` steps. There is no weight decay, clipping or EMA
# in this run.


def make_optimizer(protocol: dict, total_steps: int):
    """Complex-aware Adam on a cosine one-cycle schedule."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=total_steps,
        peak_value=protocol["learning_rate"],
        pct_start=protocol["pct_start"],
        div_factor=protocol["div_factor"],
        final_div_factor=protocol["final_div_factor"],
    )
    tx = optax.chain(
        scale_by_adam_cx(b1=0.9, b2=0.999, eps=1e-8),
        optax.scale_by_learning_rate(schedule),
    )
    return tx, schedule


# %%
# Train
# -----
# One case per step, in a new random order every epoch, and a fresh random
# subset of ``n_points`` nodes of that case at every step. Every
# ``test_every`` epochs the test cases are scored on one random subset each,
# for the loss curve only: nothing is selected by it, and the weights after
# the last epoch are the result. The first step compiles the forward pass and
# both neighbour searches, so the first epoch is slower than the others.


def train(forward, params, train_set, test_set, lats, epochs, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    p = protocol
    tx, _ = make_optimizer(p, (len(train_set) + 1) * epochs)
    opt_state = tx.init(params)

    def loss_fn(params, x, y, surf, pos, lat):
        return region_losses(forward(params, x, pos, lat), y, surf, p["surf_weight"])

    @jax.jit
    def train_step(params, opt_state, x, y, surf, pos, lat):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, y, surf, pos, lat
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    eval_step = jax.jit(loss_fn)
    rng = np.random.default_rng(0)

    def batch(s):
        n = min(p["n_points"], len(s["x"]))
        idx = rng.choice(len(s["x"]), size=n, replace=False)
        arrays = (jnp.asarray(s[k][idx]) for k in ("x", "y", "surf", "pos"))
        return (*arrays, lats[s["case_id"]])

    history = []
    for epoch in range(epochs):
        t0 = perf_counter()
        stats = []
        for i in rng.permutation(len(train_set)):
            params, opt_state, loss, (lv, ls) = train_step(
                params, opt_state, *batch(train_set[i])
            )
            stats.append((loss, lv, ls))
        tr = np.asarray(jnp.stack([jnp.stack(s) for s in stats])).mean(0)
        row = dict(epoch=epoch, train_vol=float(tr[1]), train_surf=float(tr[2]))
        if epoch % p["test_every"] == 0 or epoch == epochs - 1:
            ev = [eval_step(params, *batch(s))[1] for s in test_set]
            row["test_vol"] = float(np.mean([float(v) for v, _ in ev]))
            row["test_surf"] = float(np.mean([float(s) for _, s in ev]))
        row["secs"] = perf_counter() - t0
        history.append(row)
        msg = f"epoch {epoch}: train vol={tr[1]:.5f} surf={tr[2]:.5f}"
        if "test_vol" in row:
            msg += f" | test vol={row['test_vol']:.5f} surf={row['test_surf']:.5f}"
        print(f"{msg} | epoch_time={row['secs']:.1f}s")
    return params, history


# %%
# Evaluate with the official metrics
# ----------------------------------
# As in the benchmark's GINO evaluation:
#
# * wall nodes: the mesh nodes that coincide with a point of
#   ``<case>_aerofoil.vtp``, where the no-slip velocity is exactly 0;
# * inference: random ``n_points`` subsets until every node has been hit,
#   each node's predictions averaged;
# * wall values: ``Ux``, ``Uy`` and ``nut`` set to (the normalised value of) 0
#   on the wall nodes;
# * Volume and Surf: the MSE of each normalised channel on the non-wall and the
#   surface nodes, averaged over the 4 channels;
# * C_D and C_L: pressure and wall shear stress (from the VTK gradient of the
#   velocity) integrated over the airfoil, rotated by the angle of attack.


def predict_full_mesh(forward_jit, params, s, lat, n_points, rng) -> np.ndarray:
    """Random ``n_points`` passes until every node is covered, averaged."""
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


def read_vtk(path: Path):
    """A ``.vtu`` (volume) or ``.vtp`` (surface) file as a VTK data set."""
    reader = (
        vtk.vtkXMLUnstructuredGridReader()
        if str(path).endswith(".vtu")
        else vtk.vtkXMLPolyDataReader()
    )
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def wall_mask(sample: dict, data_dir: Path) -> np.ndarray:
    """The mesh nodes that coincide with a point of ``<case>_aerofoil.vtp``."""
    cid = sample["case_id"]
    foil = read_vtk(Path(data_dir) / cid / f"{cid}_aerofoil.vtp")
    d, _ = cKDTree(vtk_to_numpy(foil.GetPoints().GetData())[:, :2]).query(sample["pos"])
    return d < WALL_TOL


def lift_drag(internal, aerofoil, U, p, surf, u_inf, angle_deg) -> np.ndarray:
    """``(C_D, C_L)`` from pressure and wall shear stress on the airfoil cells."""
    mesh_pts = vtk_to_numpy(internal.GetPoints().GetData())[surf, :2]
    foil_pts = vtk_to_numpy(aerofoil.GetPoints().GetData())[:, :2]
    _, order = cKDTree(mesh_pts).query(foil_pts)
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
    jac = g.reshape(-1, 3, 3)[surf, :2, :2][order]
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
    force = (wss_c * length[:, None]).sum(0) - (
        -p_c[:, None] * n_c * length[:, None]
    ).sum(0)
    a = np.deg2rad(angle_deg)
    basis = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return 2 * (basis @ force) / u_inf**2


def evaluate(forward_jit, params, test_set, lats, coef_norm, data_dir, n_points):
    """Official scores of every test case and their means; predictions."""
    _, _, mean_out, std_out = coef_norm
    zero_norm = (0.0 - mean_out) / (std_out + 1e-8)
    rng = np.random.default_rng(0)
    rows, preds = [], []
    for s in test_set:
        cid = s["case_id"]
        pred = predict_full_mesh(forward_jit, params, s, lats[cid], n_points, rng)
        surf = wall_mask(s, data_dir)
        pred[surf, 0:2] = zero_norm[0:2]
        pred[surf, 3] = zero_norm[3]
        d2 = (pred - s["y"]) ** 2
        internal = read_vtk(Path(data_dir) / cid / f"{cid}_internal.vtu")
        aerofoil = read_vtk(Path(data_dir) / cid / f"{cid}_aerofoil.vtp")
        u_inf, angle = float(cid.split("_")[2]), float(cid.split("_")[3])
        y_true = s["y"] * (std_out + 1e-8) + mean_out
        y_pred = pred * (std_out + 1e-8) + mean_out
        cl_t = lift_drag(
            internal, aerofoil, y_true[:, :2], y_true[:, 2], surf, u_inf, angle
        )[1]
        cl_p = lift_drag(
            internal, aerofoil, y_pred[:, :2], y_pred[:, 2], surf, u_inf, angle
        )[1]
        rows.append(dict(case=cid, volume=float(d2[~surf].mean()),
                         surf=float(d2[surf].mean()), cl_true=cl_t, cl_pred=cl_p,
                         relerr_cl=abs((cl_t - cl_p) / cl_t)))  # fmt: skip
        preds.append(y_pred)
    g = lambda key: np.array([r[key] for r in rows])  # noqa: E731
    summary = dict(
        volume=float(g("volume").mean()),
        surf=float(g("surf").mean()),
        relerr_cl=float(g("relerr_cl").mean()),
        spearman_cl=float(spearmanr(g("cl_true"), g("cl_pred"))[0]),
    )
    return summary, rows, preds


# %%
# Plot
# ----
# The four standard panels around the airfoil of one test case: an input
# (the signed distance to the airfoil, a node feature), the true and the
# predicted pressure coefficient ``Cp = p / (U_inf^2 / 2)`` on one colour
# scale, and the absolute error. Below them, ``-Cp`` on the wall nodes.
# The triangles are the quads of the volume mesh split in two, so nothing is
# drawn across the airfoil. A second figure shows the losses per epoch.

WINDOW = (-0.5, 1.5, -0.45, 0.45)  # the view around the airfoil, in chords


def plot_prediction(data_dir, s, y_true, y_pred, sdf, path: Path, title: str) -> None:
    """Save input, true Cp, predicted Cp, absolute error and surface -Cp."""
    cid = s["case_id"]
    internal = read_vtk(Path(data_dir) / cid / f"{cid}_internal.vtu")
    quads = vtk_to_numpy(internal.GetCells().GetConnectivityArray()).reshape(-1, 4)
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    P = s["pos"]
    c = P[tri].mean(1)
    keep = (c[:, 0] > WINDOW[0] - 0.1) & (c[:, 0] < WINDOW[1] + 0.1)
    keep &= np.abs(c[:, 1]) < WINDOW[3] + 0.1
    T = Triangulation(P[:, 0], P[:, 1], tri[keep])
    q = 0.5 * float(cid.split("_")[2]) ** 2
    truth, model = y_true[:, 2] / q, y_pred[:, 2] / q
    cp_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0)
    panels = [
        ("Input: signed distance", sdf, "viridis", {}),
        ("Truth: Cp", np.clip(truth, -3, 1), "RdBu_r", dict(norm=cp_norm)),
        ("Prediction: Cp", np.clip(model, -3, 1), "RdBu_r", dict(norm=cp_norm)),
        ("Absolute error: Cp", np.abs(model - truth), "magma", {}),
    ]
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.3, wspace=0.15)
    for j, (name, values, cmap, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.tripcolor(
            T, values, shading="gouraud", cmap=cmap, rasterized=True, **kw
        )
        ax.set_xlim(WINDOW[0], WINDOW[1])
        ax.set_ylim(WINDOW[2], WINDOW[3])
        ax.set_aspect("equal")
        ax.set_facecolor("0.3")  # the airfoil, which has no mesh nodes inside
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax = fig.add_subplot(gs[1, :])
    wall = wall_mask(s, data_dir)
    order = np.argsort(P[wall, 0])
    xs = P[wall, 0][order]
    ax.plot(xs, -truth[wall][order], ".", ms=3, color="0.15", label="truth")
    ax.plot(xs, -model[wall][order], ".", ms=2, color="#9467bd", label="prediction")
    ax.set_xlabel("x / c")
    ax.set_ylabel("-Cp on the airfoil surface")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and test losses per epoch."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_vol"] for h in history], marker="o", label="train, volume")
    ax.plot(ep, [h["train_surf"] for h in history], marker="o", label="train, surface")
    te = [h for h in history if "test_vol" in h]
    ax.plot(
        [h["epoch"] for h in te],
        [h["test_vol"] for h in te],
        "s--",
        label="test, volume",
    )
    ax.plot(
        [h["epoch"] for h in te],
        [h["test_surf"] for h in te],
        "s--",
        label="test, surface",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalised fields)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes the first few training and test cases, uses the small
# GINO and smaller node subsets, and trains for a few epochs. Without it the
# script trains the benchmark-size GINO on the 800 / 200 split for 500 epochs.
# That mode is meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("AIRFRANS_DATA_DIR"),
        help="AirfRANS Dataset folder with manifest.json (or set $AIRFRANS_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few cases, small model")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/airfrans_gino_tutorial")
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="default: <out-dir>/cache"
    )
    ap.add_argument("--n-train", type=int, default=8, help="cases in smoke mode")
    ap.add_argument("--n-test", type=int, default=2, help="cases in smoke mode")
    ap.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="nodes per case per step (default: 32,000, smoke 8,000)",
    )
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $AIRFRANS_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    data_dir = Path(args.data_dir).expanduser()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or out_dir / "cache").expanduser()
    manifest = None
    if args.smoke:
        manifest = write_small_manifest(
            data_dir, out_dir / "manifest_small.json", args.n_train, args.n_test
        )
    epochs = args.epochs or (3 if args.smoke else PROTOCOL["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    n_points = args.n_points or (8_000 if args.smoke else PROTOCOL["n_points"])
    protocol = dict(PROTOCOL, n_points=n_points)
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    data = load_data(data_dir, cache_dir, manifest)
    lq_np = latent_grid(size["latent_res"])
    lats = {
        s["case_id"]: jnp.asarray(latent_sdf(s, lq_np))
        for s in data["train"] + data["test"]
    }
    print(f"data loading: {perf_counter() - t0:.1f}s")

    model = build_model(size)
    lq = jnp.asarray(lq_np)
    forward = make_forward(model, lq, size)
    params = init_params(model, lq, size, args.seed, n_points)
    params, history = train(
        forward, params, data["train"], data["test"], lats, epochs, protocol
    )

    t0 = perf_counter()
    forward_jit = jax.jit(forward)
    metrics, rows, preds = evaluate(
        forward_jit, params, data["test"], lats, data["coef_norm"], data_dir, n_points
    )
    print(f"evaluation: {perf_counter() - t0:.1f}s")
    print(
        f"Volume {metrics['volume']:.5f} | Surf {metrics['surf']:.5f} | "
        f"C_L rel. error {metrics['relerr_cl']:.4f} | "
        f"Spearman rho_L {metrics['spearman_cl']:.4f}"
    )

    vol = np.array([r["volume"] for r in rows])
    k = int(np.argsort(vol)[len(vol) // 2])
    s = data["test"][k]
    mean_in, std_in, mean_out, std_out = data["coef_norm"]
    plot_prediction(
        data_dir,
        s,
        s["y"] * (std_out + 1e-8) + mean_out,
        preds[k],
        s["x"][:, 2] * (std_in[2] + 1e-8) + mean_in[2],
        out_dir / "prediction.png",
        f"AirfRANS, GINO: test case with the median Volume error ({vol[k]:.2e}), "
        f"C_L true {rows[k]['cl_true']:.3f} / predicted {rows[k]['cl_pred']:.3f}",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
