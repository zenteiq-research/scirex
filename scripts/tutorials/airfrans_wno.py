# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License (the data
# features, the training recipe and the metrics); see THIRD_PARTY_NOTICES.md.

"""
Predicting the flow around an airfoil with WNO (AirfRANS)
=========================================================

**Problem.** Given the CFD mesh of one airfoil case (about 180,000 nodes) and
its inflow (speed and angle of attack), predict the steady velocity
``(Ux, Uy)``, pressure ``p`` and turbulent viscosity ``nut`` at every node.
The 1,000 cases of AirfRANS (Bonnet et al. 2022) are steady 2D RANS
simulations with the k-omega SST model, each on its own mesh, so the data are
irregular point clouds. The Wavelet Neural Operator (WNO, Tripura &
Chakraborty 2023) works on a regular grid, so it is wrapped in a grid
encoder (the airfoil's signed distance on a 128 x 256 grid) and a node
decoder (the latent grid field sampled at every node).

**What you will learn.**

1. How to read an AirfRANS case from its ``.vtu`` / ``.vtp`` files and build
   the official Transolver node features, plus the two added ones.
2. How to scale the targets by the inflow and normalise them with float64
   statistics of the training cases.
3. How to run a grid operator, ``scirex.models.WNO``, on an irregular mesh:
   rasterise the geometry onto a grid, run WNO, and sample the latent field
   back at the nodes; and how to count its parameters.
4. How to train on random 32,000-node subsets with Adam, ``OneCycleLR``
   (learning rate and beta1), gradient clipping and an EMA of the weights.
5. How to score with the official metrics: Volume and Surf (normalised MSE),
   the lift coefficient C_L from integrated pressure and wall shear stress,
   and Spearman's rho of C_L; and how to plot the pressure coefficient.

**Caveat.** The ``--smoke`` mode trains a tiny WNO on a few cases for a few
epochs. It checks that the pipeline works and gives poor predictions. It is
not a benchmark number. For the benchmark protocol (720 training cases, 500
epochs, 3,964,728 parameters) use ``scripts/benchmarks/airfrans/train.py``;
see the "Reproduce at full scale" section of the tutorial page.

Usage, from the repository root (needs the ``benchmark`` extra for ``vtk``)::

    python scripts/tutorials/airfrans_wno.py \\
        --data-dir ~/data/AirfRANS/Dataset --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.models.WNO`` and
# ``scirex.utils.count_model_params``. The meshes are read with ``vtk``; the
# loop itself is plain JAX and optax.

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import vtk
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation
from scipy.stats import spearmanr
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import scirex
from scirex.models import WNO
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Float32 is intended, as in the benchmark, also if float64 was enabled
# earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0
NU = 1.56e-5  # kinematic viscosity of the AirfRANS simulations
# The box of the WNO grid, Transolver's reference box: x in [-2, 4],
# y in [-1.5, 1.5] chords.
GRID_RANGE = ((-2.0, 4.0), (-1.5, 1.5))
# Airfoil outlines have 843 to 1,217 segments; padding them to one size keeps
# a single compiled geometry function.
MAX_SEGMENTS = 1280

# Training settings of the benchmark, the same values as PROTOCOL in
# scripts/benchmarks/airfrans/_config.py: the official Transolver AirfRANS
# recipe plus EMA, gradient clipping, FP32 matmuls and 500 epochs.
PROTOCOL = dict(
    n_val=80,
    subsample=32_000,
    epochs=500,
    lr=1e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    base_momentum=0.85,
    max_momentum=0.95,
    surf_weight=1.0,
    val_every=10,
    val_repeats=20,
    ema_decay=0.9995,
    grad_clip_global_norm=10.0,
    matmul_precision="highest",
)

# Model sizes. "full" is the benchmark's WNO (3,964,728 parameters, 1.8 %
# more than the paper's Transolver). "smoke" is a tiny WNO on a coarser grid
# that trains on a CPU.
SIZES = dict(
    full=dict(
        H=128,
        W=256,
        hidden=28,
        n_layers=4,
        level=4,
        wavelet="db4",
        latent=64,
        dec_hidden=256,
        dec_layers=3,
    ),
    smoke=dict(
        H=32,
        W=64,
        hidden=8,
        n_layers=2,
        level=2,
        wavelet="db4",
        latent=16,
        dec_hidden=64,
        dec_layers=2,
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
# Read the cases
# --------------
# ``manifest.json`` lists the case ids of the official "full" task. As in the
# official ``main.py``, the last 80 of ``full_train`` are the validation cases
# and ``full_test`` (200 cases) is the test set. A case folder holds
# ``<case>_internal.vtu`` (the volume mesh with ``U``, ``p``, ``nut`` and
# ``implicit_distance``) and ``<case>_aerofoil.vtp`` (the airfoil surface with
# its normals). The inflow speed and angle are in the case id.


def read_split(data_dir: Path, n_val: int = 80) -> dict[str, list[str]]:
    """The official split: ``full_train[:-n_val]`` / ``[-n_val:]`` / ``full_test``."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    train = manifest["full_train"]
    return dict(train=train[:-n_val], val=train[-n_val:], test=manifest["full_test"])


def read_vtk(path: Path):
    """A ``.vtu`` (volume) or ``.vtp`` (surface) file as a VTK data set."""
    reader = (
        vtk.vtkXMLPolyDataReader()
        if str(path).endswith(".vtp")
        else vtk.vtkXMLUnstructuredGridReader()
    )
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()


def read_case(data_dir: Path, case_id: str) -> dict[str, np.ndarray]:
    """The official node features of one case, as the official ``dataset.py``.

    ``x7 = [x, y, U_inf cos a, U_inf sin a, -implicit_distance, n_x, n_y]``,
    ``y = [Ux, Uy, p, nut]``. Wall nodes are those with ``Ux == 0``; their
    normal is minus the airfoil normal, and every other node's is 0. The
    airfoil outline is kept as line segments ``(S, 2, 2)`` for the grid.
    """
    folder = Path(data_dir) / case_id
    internal = read_vtk(folder / f"{case_id}_internal.vtu")
    airfoil = read_vtk(folder / f"{case_id}_aerofoil.vtp")
    pts = vtk_to_numpy(internal.GetPoints().GetData())[:, :2].astype(np.float32)
    fields = internal.GetPointData()
    U = vtk_to_numpy(fields.GetArray("U"))
    p = vtk_to_numpy(fields.GetArray("p"))
    nut = vtk_to_numpy(fields.GetArray("nut"))
    sdf = vtk_to_numpy(fields.GetArray("implicit_distance"))
    surf = U[:, 0] == 0
    u_inf = float(case_id.split("_")[2])
    alpha = np.deg2rad(float(case_id.split("_")[3]))
    inflow = np.broadcast_to(
        np.array([np.cos(alpha), np.sin(alpha)]) * u_inf, (len(pts), 2)
    )
    normal = np.zeros((len(pts), 2))
    apts = vtk_to_numpy(airfoil.GetPoints().GetData())[:, :2].astype(np.float32)
    anormal = -vtk_to_numpy(airfoil.GetPointData().GetArray("Normals"))[:, :2]
    where = {tuple(q): i for i, q in enumerate(apts.tolist())}
    wall = np.flatnonzero(surf)
    normal[wall] = anormal[[where[tuple(q)] for q in pts[wall].tolist()]]
    x7 = np.concatenate([pts, inflow, -sdf[:, None], normal], axis=1).astype(np.float32)
    y = np.column_stack([U[:, 0], U[:, 1], p, nut]).astype(np.float32)
    lines = vtk_to_numpy(airfoil.GetLines().GetData()).reshape(-1, 3)[:, 1:]
    segments = vtk_to_numpy(airfoil.GetPoints().GetData())[:, :2][lines]
    return dict(x7=x7, y=y, pos=pts, surf=surf, segments=segments.astype(np.float32))


def load_case(data_dir: Path, cache_dir: Path, case_id: str) -> dict[str, np.ndarray]:
    """:func:`read_case`, cached as ``<cache_dir>/<case_id>.npz``."""
    path = Path(cache_dir) / f"{case_id}.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    case = read_case(data_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **case)
    return case


# %%
# Features, target scaling and normalisation
# ------------------------------------------
# Two features are added to the official seven: ``log10(|wall distance| +
# 1e-6)``, which separates the thin near-wall layers that a linear distance
# cannot, and ``log10(Re)``. The targets are divided by the inflow:
# ``(Ux, Uy) / U_inf``, ``p / U_inf^2``, ``nut / U_inf``. Inputs and scaled
# targets are then standardised with the mean and population std over every
# node of the training cases, accumulated in float64.


def inflow_speed(x7: np.ndarray) -> float:
    return float(np.linalg.norm(x7[0, 2:4]))


def features(x7: np.ndarray) -> np.ndarray:
    """The 7 official features plus log10 wall distance and log10 Re."""
    log_d = np.log10(np.abs(x7[:, 4:5]) + 1e-6)
    log_re = np.full((len(x7), 1), np.log10(inflow_speed(x7) / NU))
    return np.concatenate([x7, log_d, log_re], axis=1).astype(np.float32)


def target_scale(u_inf: float) -> np.ndarray:
    """Per-channel divisor of the targets: U_inf, U_inf, U_inf^2, U_inf."""
    return np.array([u_inf, u_inf, u_inf**2, u_inf], np.float32)


def fit_normalisation(cases: list[dict]) -> dict[str, np.ndarray]:
    """Float64 mean and population std of the features and scaled targets."""
    n, sx, sxx, sy, syy = 0, 0.0, 0.0, 0.0, 0.0
    for c in cases:
        x = features(c["x7"]).astype(np.float64)
        y = (c["y"] / target_scale(inflow_speed(c["x7"]))).astype(np.float64)
        n += len(x)
        sx, sxx = sx + x.sum(0), sxx + (x**2).sum(0)
        sy, syy = sy + y.sum(0), syy + (y**2).sum(0)
    mx, my = sx / n, sy / n
    return dict(
        x_mean=mx.astype(np.float32),
        x_std=np.sqrt(np.maximum(sxx / n - mx**2, 0.0)).astype(np.float32),
        y_mean=my.astype(np.float32),
        y_std=np.sqrt(np.maximum(syy / n - my**2, 0.0)).astype(np.float32),
    )


def prepare_case(case: dict, norm: dict) -> dict[str, np.ndarray]:
    """Normalised inputs and targets of one case, plus what scoring needs."""
    scale = target_scale(inflow_speed(case["x7"]))
    x = (features(case["x7"]) - norm["x_mean"]) / (norm["x_std"] + 1e-8)
    y = (case["y"] / scale - norm["y_mean"]) / (norm["y_std"] + 1e-8)
    return dict(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        pos=case["pos"],
        surf=case["surf"],
        scale=scale,
    )


def to_physical(pred_norm: np.ndarray, scale: np.ndarray, norm: dict) -> np.ndarray:
    """Model output (normalised scaled targets) to ``(Ux, Uy, p, nut)``."""
    return (pred_norm * (norm["y_std"] + 1e-8) + norm["y_mean"]) * scale


# %%
# Build the model
# ---------------
# WNO needs a regular grid, the mesh is irregular. The benchmark's
# ``WNOGridOperator`` bridges the two:
#
# * encoder: an ``H x W`` grid over ``GRID_RANGE`` carries the case's geometry
#   (``tanh(signed distance / 0.5)`` to the airfoil outline and an inside
#   mask) and the normalised inflow; WNO appends the grid coordinates;
# * WNO: lifting, ``n_layers`` wavelet blocks, projection to ``latent``
#   channels, in float32;
# * decoder: for every queried node, the latent field sampled bilinearly at
#   the node (clamped to the box), concatenated with the node's 9 features,
#   through an MLP (exact GELU) to the 4 outputs.
#
# The geometry grid depends only on the case, not on which nodes are sampled,
# so it is computed once per case (``case_geometry``).


def grid_points(H: int, W: int) -> jnp.ndarray:
    """``(H * W, 2)`` points of the ``H x W`` grid over ``GRID_RANGE``."""
    xs = jnp.linspace(GRID_RANGE[0][0], GRID_RANGE[0][1], W)
    ys = jnp.linspace(GRID_RANGE[1][0], GRID_RANGE[1][1], H)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([gx, gy], -1).reshape(-1, 2)


def signed_distance(points, seg, n_seg):
    """Distance of ``points`` to the outline ``seg`` (``n_seg`` real segments),
    negative inside; inside means an odd number of crossings of a ray in +x."""
    a, b = seg[:, 0], seg[:, 1]
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = jnp.clip(
        jnp.sum(ap * ab[None], -1) / jnp.maximum(jnp.sum(ab * ab, -1), 1e-12)[None],
        0.0,
        1.0,
    )
    d = jnp.sqrt(jnp.min(jnp.sum((ap - t[..., None] * ab[None]) ** 2, -1), axis=1))
    py, px = points[:, 1:2], points[:, 0:1]
    ay, by, ax, bx = a[None, :, 1], b[None, :, 1], a[None, :, 0], b[None, :, 0]
    straddle = (ay > py) != (by > py)
    xcross = ax + (py - ay) * (bx - ax) / jnp.where(by == ay, 1e-12, by - ay)
    real = jnp.arange(seg.shape[0])[None] < n_seg
    inside = (jnp.sum(straddle & (px < xcross) & real, axis=1) % 2) == 1
    return jnp.where(inside, -d, d), inside


@functools.partial(jax.jit, static_argnums=(2, 3))
def grid_geometry(seg, n_seg, H: int, W: int) -> jnp.ndarray:
    """``(H, W, 2)``: tanh(signed distance / 0.5) and the inside mask."""
    sdf, inside = signed_distance(grid_points(H, W), seg, n_seg)
    geo = jnp.stack([jnp.tanh(sdf / 0.5), inside.astype(jnp.float32)], -1)
    return geo.reshape(H, W, 2)


def case_geometry(segments: np.ndarray, size: dict) -> jnp.ndarray:
    """The geometry grid of one case, from its outline padded to MAX_SEGMENTS."""
    seg = np.asarray(segments, np.float32)
    if len(seg) > MAX_SEGMENTS:
        raise ValueError(f"{len(seg)} outline segments > MAX_SEGMENTS={MAX_SEGMENTS}")
    padded = np.concatenate([seg, np.repeat(seg[-1:], MAX_SEGMENTS - len(seg), 0)])
    return grid_geometry(
        jnp.asarray(padded), jnp.asarray(len(seg)), size["H"], size["W"]
    )


def bilinear(grid: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """Sample ``grid`` ``(H, W, C)`` over ``GRID_RANGE`` at ``pos`` ``(N, 2)``;
    positions outside the box are clamped to its edge."""
    H, W, _ = grid.shape
    (x0b, x1b), (y0b, y1b) = GRID_RANGE
    fx = (jnp.clip(pos[:, 0], x0b, x1b) - x0b) / (x1b - x0b) * (W - 1)
    fy = (jnp.clip(pos[:, 1], y0b, y1b) - y0b) / (y1b - y0b) * (H - 1)
    x0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, W - 2)
    y0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, H - 2)
    wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
    return (1 - wy) * ((1 - wx) * grid[y0, x0] + wx * grid[y0, x0 + 1]) + wy * (
        (1 - wx) * grid[y0 + 1, x0] + wx * grid[y0 + 1, x0 + 1]
    )


class WNOGridOperator(nn.Module):
    """``scirex.models.WNO`` between a grid encoder and a node decoder."""

    H: int = 128
    W: int = 256
    hidden: int = 28
    n_layers: int = 4
    level: int = 4
    wavelet: str = "db4"
    latent: int = 64
    dec_hidden: int = 256
    dec_layers: int = 3

    @nn.compact
    def __call__(self, x, pos, geo):
        inflow = x[0, 2:4]  # the normalised inflow, the same at every node
        g = jnp.concatenate([geo, jnp.broadcast_to(inflow, (self.H, self.W, 2))], -1)
        g = jax.lax.stop_gradient(g)[None]
        lat = WNO(
            in_channels=4,
            out_channels=self.latent,
            hidden_channels=self.hidden,
            n_layers=self.n_layers,
            level=self.level,
            size=(self.H, self.W),
            wavelet=self.wavelet,
            use_grid=True,
            grid_boundaries=(GRID_RANGE[1], GRID_RANGE[0]),
            compute_dtype=jnp.float32,
            param_dtype=jnp.float32,
            projection_hidden_channels=128,
        )(g)[0]
        h = jnp.concatenate([bilinear(lat, pos), x], -1)
        for _ in range(self.dec_layers - 1):
            h = nn.gelu(nn.Dense(self.dec_hidden)(h), approximate=False)
        return nn.Dense(4)(h)


def build_model(size: dict) -> WNOGridOperator:
    """The benchmark's WNO grid operator at the size ``size``."""
    return WNOGridOperator(**size)


def init_params(model, seed: int, x: np.ndarray, pos: np.ndarray, geo):
    """Initialise on one node subset and count the parameters."""
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(x), jnp.asarray(pos), geo)
    print(f"WNO grid operator | parameters: {count_model_params(params):,}")
    return params


def apply_model(model, params, x, pos, geo):
    """``(N, 4)`` normalised prediction for the nodes ``x``, ``pos``."""
    return model.apply(params, x, pos, geo)


# %%
# Loss
# ----
# The official ``MSE_weighted``: the MSE of each channel on the non-wall
# ("volume") nodes and on the wall nodes, each averaged over the 4 channels,
# summed with weight 1.


def region_losses(out, y, surf):
    """Volume and wall MSE, each the mean over the 4 channels."""
    se = (out - y) ** 2
    w = surf[:, None].astype(jnp.float32)
    loss_surf = jnp.mean(jnp.sum(se * w, 0) / jnp.maximum(w.sum(), 1.0))
    loss_vol = jnp.mean(jnp.sum(se * (1 - w), 0) / jnp.maximum((1 - w).sum(), 1.0))
    return loss_vol, loss_surf


# %%
# Optimiser and schedule
# ----------------------
# Adam with PyTorch's ``OneCycleLR``: two cosine phases for the learning rate
# (up to 1e-3 over the first 30 % of the steps, then down to 1e-3 / 25 / 1e4)
# and, in the opposite direction, for Adam's beta1 (0.95 -> 0.85 -> 0.95).
# ``optax.inject_hyperparams`` lets the step set both. WNO's wavelet weights
# are real, so plain Adam is right; ``hyperparam_dtype=jnp.float32`` keeps the
# injected values real even next to complex weights.


def onecycle(total_steps: int, p: dict):
    """``OneCycleLR`` as two functions of the 0-based step: lr and beta1."""
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


def make_optimizer():
    """Adam whose learning rate and beta1 are set at every step."""
    return optax.inject_hyperparams(optax.adam, hyperparam_dtype=jnp.float32)(
        learning_rate=1e-3, b1=0.95, b2=0.999, eps=1e-8
    )


# %%
# Train
# -----
# One case per step, in a new random order every epoch, and a fresh random
# subset of 32,000 nodes of that case at every step; the case's geometry grid
# goes with it. The gradient is clipped
# at global norm 10 (it only cuts spikes), and an exponential moving average
# (EMA) of the weights, decay 0.9995, is the model that is evaluated. Every
# ``val_every`` epochs the EMA weights are scored on the validation cases, on
# ``val_repeats`` random subsets each; this is logged only.


def train(model, params, train_cases, val_cases, epochs, seed, protocol=PROTOCOL):
    """Train for ``epochs`` epochs. Return the EMA weights and the history."""
    p = protocol
    rng = np.random.default_rng(seed)

    def sample(d):
        idx = rng.choice(len(d["x"]), min(p["subsample"], len(d["x"])), replace=False)
        x, pos, y, surf = (jnp.asarray(d[k][idx]) for k in ("x", "pos", "y", "surf"))
        return x, pos, d["geo"], y, surf

    lr_f, b1_f = onecycle((len(train_cases) + 1) * epochs, p)
    tx = make_optimizer()
    opt = tx.init(params)
    ema = params
    decay, clip, surf_w = p["ema_decay"], p["grad_clip_global_norm"], p["surf_weight"]

    def loss_fn(params, x, pos, geo, y, surf):
        lv, ls = region_losses(apply_model(model, params, x, pos, geo), y, surf)
        return lv + surf_w * ls, (lv, ls)

    @jax.jit
    def step(params, ema, opt, x, pos, geo, y, surf, k):
        (_, (lv, ls)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x, pos, geo, y, surf
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
        return params, ema, opt, lv, ls

    evaluate_loss = jax.jit(lambda params, *b: loss_fn(params, *b)[1])

    history, k = [], 0
    for epoch in range(epochs):
        t0 = perf_counter()
        lvs, lss = [], []
        for i in rng.permutation(len(train_cases)):
            params, ema, opt, lv, ls = step(
                params, ema, opt, *sample(train_cases[i]), jnp.asarray(k, jnp.int32)
            )
            k += 1
            lvs.append(lv)
            lss.append(ls)
        row = dict(
            epoch=epoch,
            train_vol=float(jnp.mean(jnp.stack(lvs))),
            train_surf=float(jnp.mean(jnp.stack(lss))),
            secs=perf_counter() - t0,
        )
        if epoch % p["val_every"] == p["val_every"] - 1 or epoch in (0, epochs - 1):
            vals = [
                [float(v) for v in evaluate_loss(ema, *sample(d))]
                for _ in range(p["val_repeats"])
                for d in val_cases
            ]
            row["val_vol"], row["val_surf"] = (float(v) for v in np.mean(vals, 0))
        history.append(row)
        msg = (
            f"epoch {epoch}: train vol={row['train_vol']:.5f} "
            f"surf={row['train_surf']:.5f}"
        )
        if "val_vol" in row:
            msg += f" | val (EMA) vol={row['val_vol']:.5f} surf={row['val_surf']:.5f}"
        print(f"{msg} | epoch_time={row['secs']:.1f}s")
    return ema, history


# %%
# Evaluate with the official metrics
# ----------------------------------
# The official ``utils/metrics.py``, re-implemented:
#
# * inference: random 32,000-node subsets are predicted until every node has
#   been hit, and each node's predictions are averaged;
# * wall values: ``Ux``, ``Uy`` and ``nut`` are set to their exact value 0 on
#   the wall nodes;
# * Volume and Surf: the MSE of each channel in the official normalisation
#   (mean and std of the raw targets over the training cases), on the non-wall
#   and the wall nodes, averaged over the 4 channels;
# * C_D and C_L: pressure and wall shear stress (from the VTK gradient of
#   ``U``) integrated over the airfoil, rotated by the angle of attack.


def official_output_stats(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the raw targets, as the official ``dataset.py``
    (a running float64 update, the variance around the float32 mean)."""
    n, mean = 0, 0.0
    for c in cases:
        k = len(c["y"])
        mean = mean + (c["y"].sum(0, dtype=np.float64) - k * mean) / (n + k)
        n += k
    mean = np.asarray(mean).astype(np.float32)
    n, var = 0, 0.0
    for c in cases:
        k = len(c["y"])
        var = var + (((c["y"] - mean) ** 2).sum(0, dtype=np.float64) - k * var) / (
            n + k
        )
        n += k
    return mean, np.sqrt(var).astype(np.float32)


def cover_predict(predict, n_nodes: int, subsample: int, rng) -> np.ndarray:
    """Average of random-subset predictions until every node is covered."""
    acc, hits = None, np.zeros(n_nodes)
    while (hits == 0).any():
        idx = rng.choice(n_nodes, min(subsample, n_nodes), replace=False)
        out = np.asarray(predict(idx), dtype=np.float64)
        if acc is None:
            acc = np.zeros((n_nodes, out.shape[1]))
        acc[idx] += out
        hits[idx] += 1
    return acc / hits[:, None]


def field_scores(pred, truth, surf, out_mean, out_std) -> dict:
    """Per-channel MSE in the official normalisation on volume / wall nodes."""
    z = lambda v: (v - out_mean) / (out_std + 1e-8)  # noqa: E731
    se = (z(pred) - z(truth)) ** 2
    vol, wall = se[~surf].mean(0), se[surf].mean(0)
    return dict(volume=float(vol.mean()), surf=float(wall.mean()))


def _set_array(ds, name: str, values: np.ndarray) -> None:
    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
    arr.SetName(name)
    ds.GetPointData().RemoveArray(name)
    ds.GetPointData().AddArray(arr)


def force_coefficients(data_dir: Path, case_id: str, fields=None) -> np.ndarray:
    """``(C_D, C_L)`` of the true fields (``None``) or of ``(Ux, Uy, p, nut)``,
    as the official ``Compute_coefficients``."""
    folder = Path(data_dir) / case_id
    internal = read_vtk(folder / f"{case_id}_internal.vtu")
    airfoil = read_vtk(folder / f"{case_id}_aerofoil.vtp")
    U = vtk_to_numpy(internal.GetPointData().GetArray("U")).copy()
    pts = vtk_to_numpy(internal.GetPoints().GetData())[:, :2]
    wall = np.flatnonzero(U[:, 0] == 0)
    where = {tuple(q): i for i, q in enumerate(pts[wall].tolist())}
    apts = vtk_to_numpy(airfoil.GetPoints().GetData())[:, :2]
    order = np.array([wall[where[tuple(q)]] for q in apts.tolist()])
    if fields is not None:
        U[:, :2] = fields[:, :2]
        _set_array(internal, "U", U)
        _set_array(internal, "p", fields[:, 2])
    grad = vtk.vtkGradientFilter()
    grad.SetInputData(internal)
    grad.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U"
    )
    grad.SetResultArrayName("grad")
    grad.SetFasterApproximation(False)
    grad.Update()
    jac = vtk_to_numpy(grad.GetOutput().GetPointData().GetArray("grad"))
    jac = jac.reshape(-1, 3, 3)[order, :2, :2]
    S = 0.5 * (jac + jac.transpose(0, 2, 1))
    S = S - S.trace(axis1=1, axis2=2).reshape(-1, 1, 1) * np.eye(2)[None] / 3
    normals = -vtk_to_numpy(airfoil.GetPointData().GetArray("Normals"))[:, :2]
    wss = (2 * NU * S * normals.reshape(-1, 1, 2)).sum(axis=2)
    _set_array(airfoil, "wallShearStress", wss)
    _set_array(airfoil, "p", vtk_to_numpy(internal.GetPointData().GetArray("p"))[order])
    ptc = vtk.vtkPointDataToCellData()
    ptc.SetInputData(airfoil)
    ptc.SetPassPointData(True)
    ptc.Update()
    cells = ptc.GetOutput().GetCellData()
    length = vtk_to_numpy(cells.GetArray("Length")).reshape(-1, 1)
    p_c = vtk_to_numpy(cells.GetArray("p"))[:, None]
    n_c = vtk_to_numpy(cells.GetArray("Normals"))[:, :2]
    wss_c = vtk_to_numpy(cells.GetArray("wallShearStress"))
    force = (wss_c * length).sum(0) - (-p_c * n_c * length).sum(0)
    a = np.deg2rad(float(case_id.split("_")[3]))
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return 2 * (rot @ force) / float(case_id.split("_")[2]) ** 2


def evaluate(
    model, params, cases, case_ids, norm, out_stats, data_dir, size, protocol=PROTOCOL
):
    """Official scores of every case and their means; physical predictions."""
    predict = jax.jit(lambda q, x, pos, geo: apply_model(model, q, x, pos, geo))
    rng = np.random.default_rng(0)
    rows, preds = [], []
    for raw, cid in zip(cases, case_ids, strict=True):
        d = prepare_case(raw, norm)
        geo = case_geometry(raw["segments"], size)
        pred_n = cover_predict(
            lambda idx, d=d, geo=geo: predict(
                params, jnp.asarray(d["x"][idx]), jnp.asarray(d["pos"][idx]), geo
            ),
            len(d["x"]),
            protocol["subsample"],
            rng,
        )
        pred = to_physical(pred_n, d["scale"], norm).astype(np.float64)
        pred[raw["surf"], 0] = pred[raw["surf"], 1] = pred[raw["surf"], 3] = 0.0
        scores = field_scores(pred, raw["y"], raw["surf"], *out_stats)
        cl_t = force_coefficients(data_dir, cid)[1]
        cl_p = force_coefficients(data_dir, cid, pred)[1]
        rows.append(dict(case=cid, **scores, cl_true=cl_t, cl_pred=cl_p,
                         relerr_cl=abs((cl_t - cl_p) / cl_t)))  # fmt: skip
        preds.append(pred.astype(np.float32))
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
# (the log10 wall distance, the added feature), the true and the predicted
# pressure coefficient ``Cp = p / (U_inf^2 / 2)`` on one colour scale, and
# the absolute error. Below them, ``-Cp`` on the airfoil surface, where the
# lift comes from. The mesh triangles are the quads of the volume mesh split
# in two, so nothing is drawn across the airfoil. A second figure shows the
# losses per epoch.

WINDOW = (-0.5, 1.5, -0.45, 0.45)  # the view around the airfoil, in chords


def plot_prediction(data_dir, case_id, raw, pred, path: Path, title: str) -> None:
    """Save input, true Cp, predicted Cp, absolute error and surface -Cp."""
    internal = read_vtk(Path(data_dir) / case_id / f"{case_id}_internal.vtu")
    quads = vtk_to_numpy(internal.GetCells().GetConnectivityArray()).reshape(-1, 4)
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    P = raw["pos"]
    c = P[tri].mean(1)
    keep = (c[:, 0] > WINDOW[0] - 0.1) & (c[:, 0] < WINDOW[1] + 0.1)
    keep &= np.abs(c[:, 1]) < WINDOW[3] + 0.1
    T = Triangulation(P[:, 0], P[:, 1], tri[keep])
    q = 0.5 * float(case_id.split("_")[2]) ** 2
    truth, model = raw["y"][:, 2] / q, pred[:, 2] / q
    cp_norm = TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=1.0)
    panels = [
        ("Input: log10 wall distance", np.log10(np.abs(raw["x7"][:, 4]) + 1e-6),
         "viridis", {}),
        ("Truth: Cp", np.clip(truth, -3, 1), "RdBu_r", dict(norm=cp_norm)),
        ("Prediction: Cp", np.clip(model, -3, 1), "RdBu_r", dict(norm=cp_norm)),
        ("Absolute error: Cp", np.abs(model - truth), "magma", {}),
    ]  # fmt: skip
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
    wall = raw["surf"]
    order = np.argsort(P[wall, 0])
    ax.plot(
        P[wall, 0][order], -truth[wall][order], ".", ms=3, color="0.15", label="truth"
    )
    ax.plot(
        P[wall, 0][order],
        -model[wall][order],
        ".",
        ms=2,
        color="#9467bd",
        label="prediction",
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("-Cp on the airfoil surface")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training and validation (EMA) losses per epoch."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_vol"] for h in history], marker="o", label="train, volume")
    ax.plot(ep, [h["train_surf"] for h in history], marker="o", label="train, surface")
    val = [h for h in history if "val_vol" in h]
    ax.plot(
        [h["epoch"] for h in val],
        [h["val_vol"] for h in val],
        "s--",
        label="validation (EMA), volume",
    )
    ax.plot(
        [h["epoch"] for h in val],
        [h["val_surf"] for h in val],
        "s--",
        label="validation (EMA), surface",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (normalised targets)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes the first few training, validation and test cases, uses
# the tiny WNO and trains for a few epochs. Without it the script trains the
# benchmark-size WNO on the 720 / 80 / 200 split for 500 epochs. That mode is
# meant for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("AIRFRANS_DATA_DIR"),
        help="AirfRANS Dataset folder with manifest.json (or set $AIRFRANS_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few cases, tiny model")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/airfrans_wno_tutorial")
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="default: <out-dir>/cache"
    )
    ap.add_argument("--n-train", type=int, default=8, help="cases in smoke mode")
    ap.add_argument("--n-val", type=int, default=2, help="cases in smoke mode")
    ap.add_argument("--n-test", type=int, default=4, help="cases in smoke mode")
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="EMA decay of the evaluated weights (default: 0.9995, smoke 0.99)",
    )
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $AIRFRANS_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    jax.config.update("jax_default_matmul_precision", PROTOCOL["matmul_precision"])
    data_dir = Path(args.data_dir).expanduser()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.cache_dir or out_dir / "cache").expanduser()
    split = read_split(data_dir, PROTOCOL["n_val"])
    if args.smoke:
        split = dict(
            train=split["train"][: args.n_train],
            val=split["val"][: args.n_val],
            test=split["test"][: args.n_test],
        )
    epochs = args.epochs or (3 if args.smoke else PROTOCOL["epochs"])
    size = SIZES["smoke" if args.smoke else "full"]
    # An EMA with decay 0.9995 averages over about 2,000 steps; a smoke run
    # has far fewer, so it uses 0.99 (about 100 steps).
    ema_decay = args.ema_decay or (0.99 if args.smoke else PROTOCOL["ema_decay"])
    protocol = dict(PROTOCOL, ema_decay=ema_decay)
    if not args.smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    raw = {
        k: [load_case(data_dir, cache_dir, c) for c in ids] for k, ids in split.items()
    }
    n_nodes = [len(c["y"]) for v in raw.values() for c in v]
    print(
        f"cases: {len(split['train'])} train / {len(split['val'])} val / "
        f"{len(split['test'])} test | "
        f"nodes per case: min {min(n_nodes)}, max {max(n_nodes)}"
    )
    norm = fit_normalisation(raw["train"])
    out_stats = official_output_stats(raw["train"])
    train_cases = [prepare_case(c, norm) for c in raw["train"]]
    val_cases = [prepare_case(c, norm) for c in raw["val"]]
    for d, c in zip(train_cases + val_cases, raw["train"] + raw["val"], strict=True):
        d["geo"] = case_geometry(c["segments"], size)
    print(f"data loading: {perf_counter() - t0:.1f}s")

    model = build_model(size)
    n0 = min(PROTOCOL["subsample"], len(train_cases[0]["x"]))
    d0 = train_cases[0]
    params = init_params(model, args.seed, d0["x"][:n0], d0["pos"][:n0], d0["geo"])
    params, history = train(
        model, params, train_cases, val_cases, epochs, args.seed, protocol
    )

    t0 = perf_counter()
    metrics, rows, preds = evaluate(
        model,
        params,
        raw["test"],
        split["test"],
        norm,
        out_stats,
        data_dir,
        size,
        protocol,
    )
    print(f"evaluation: {perf_counter() - t0:.1f}s")
    print(
        f"Volume {metrics['volume']:.5f} | Surf {metrics['surf']:.5f} | "
        f"C_L rel. error {metrics['relerr_cl']:.4f} | "
        f"Spearman rho_L {metrics['spearman_cl']:.4f}"
    )

    vol = np.array([r["volume"] for r in rows])
    k = int(np.argsort(vol)[len(vol) // 2])
    plot_prediction(
        data_dir,
        split["test"][k],
        raw["test"][k],
        preds[k],
        out_dir / "prediction.png",
        f"AirfRANS, WNO: test case with the median Volume error ({vol[k]:.2e}), "
        f"C_L true {rows[k]['cl_true']:.3f} / predicted {rows[k]['cl_pred']:.3f}",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
