# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Solving Darcy flow without labels: physics-only FNO
===================================================

**Problem.** Given a piecewise-constant permeability field ``a(x)`` with values
in {3, 12} on the unit square, predict the pressure ``u`` that solves
``-div(a grad u) = 1`` with ``u = 0`` on the boundary. The data are Li et
al.'s released Darcy fields, strided to 85x85. Unlike the data-driven
benchmark, the network never sees a training solution: it is trained to
minimise the finite-element Ritz energy of the PDE, whose minimiser is the
solution. The released solutions are only used to score the test set.

**What you will learn.**

1. How to turn the PDE into a loss: the Q4 Ritz energy
   ``J(u) = 1/2 int a |grad u|^2 - int f u`` on bilinear elements, built with
   ``scirex.losses.Q4Galerkin``.
2. How to give each element one coefficient value (the majority of the
   native 421x421 nodes it covers) and why that choice matters.
3. How to impose ``u = 0`` exactly and set the output scale without labels,
   from one finite-element solve of the torsion problem.
4. How to train a ``scirex.models.FNO`` on that loss with the complex-aware
   ``scirex.training.adamw_cx``, and why the loss is scaled.
5. How to score the prediction against the released solutions, which are
   sampled half a native cell away from the mesh nodes.

**Caveat.** The ``--smoke`` mode trains a tiny FNO on a few coefficient
fields for a few hundred steps. It checks that the pipeline works and is
not a benchmark number. For the benchmark protocol (1000 coefficient fields,
100,000 steps, 2,836,281 parameters) use
``scripts/benchmarks/darcy_physics/train_fno.py``; see the "Reproduce at full
scale" section of the tutorial page.

Usage, from the repository root::

    python scripts/tutorials/darcyphysics_fno.py --data-dir ~/data/darcy --smoke
"""

# %%
# Setup
# -----
# Only the public ScireX interface is used: ``scirex.losses.Q4Galerkin``,
# ``scirex.models.FNO``, ``scirex.training.adamw_cx`` and
# ``scirex.utils.count_model_params``. The data are read and the reference
# finite-element solve is done with SciPy; the loop is plain JAX and optax.

from __future__ import annotations

import argparse
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
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import scirex
from scirex.losses import Q4Galerkin
from scirex.models import FNO
from scirex.training import adamw_cx
from scirex.utils import count_model_params

matplotlib.use("Agg")  # write figures to files, no display needed
# Keep float32, also if float64 was enabled earlier in the process.
jax.config.update("jax_enable_x64", False)

SEED = 0

# The data split of the Darcy benchmark (scripts/benchmarks/darcy/_config.py).
PROTOCOL = dict(
    train_file="piececonst_r421_N1024_smooth1.mat",
    test_file="piececonst_r421_N1024_smooth2.mat",
    n_train=1000,
    n_test=200,
    subsample_rate=5,  # 421 -> 85 nodes per side
)

# The physics-only FNO recipe (scripts/benchmarks/darcy_physics/train_fno.py):
# 2000 epochs of 50 steps = 100,000 steps, lr halved every 400 epochs.
RECIPE = dict(
    epochs=2000,
    batch_size=20,
    learning_rate=1e-3,
    lr_decay_epochs=400,
    lr_decay_rate=0.5,
    weight_decay=0.0,
    adam_eps=1e-8,
)

# Model sizes. "full" is the benchmark's FNO (2,836,281 parameters, matched to
# Transolver's 2,826,945). "smoke" is a tiny FNO that trains on a CPU.
SIZES = dict(
    full=dict(hidden_channels=40, n_modes=(20, 20), n_layers=4),
    smoke=dict(hidden_channels=12, n_modes=(12, 12), n_layers=2),
)

# Finite-element contractions at full float32 precision (on a GPU, JAX's
# default TF32 matmuls perturb the Q4 Jacobians by up to 5%).
FE_PRECISION = "highest"


def print_environment(seed: int) -> None:
    """Print the versions, the devices and the seed, so a run can be repeated."""
    print(f"python {platform.python_version()} | numpy {np.__version__}")
    print(f"scirex {scirex.__version__}")
    print(
        f"jax {jax.__version__} | flax {flax.__version__} | optax {optax.__version__}"
    )
    print(f"devices: {jax.devices()}")
    print(f"SEED = {seed}")


# %%
# Load the data
# -------------
# Each ``.mat`` file holds ``coeff`` (the permeability, {3, 12}) and ``sol``
# (the pressure) on the native 421x421 grid. The network sees the
# coefficient at every 5th node (85x85). The loss needs one coefficient value
# per finite element, taken from all the native nodes the element covers, so
# the native training coefficients are reduced to one value per element
# here. The training solutions are not read at all.


def majority_per_cell(a_native: np.ndarray, resolution: int) -> np.ndarray:
    """``(n, N, N)`` native node values -> ``(n, resolution, resolution)``:
    the more frequent of the two values over the nodes each cell covers."""
    rate = (a_native.shape[1] - 1) // resolution
    blocks = np.lib.stride_tricks.sliding_window_view(
        a_native, (rate + 1, rate + 1), axis=(1, 2)
    )[:, ::rate, ::rate]
    lo, hi = float(a_native.min()), float(a_native.max())
    return np.where(blocks.mean(axis=(-2, -1)) >= 0.5 * (lo + hi), hi, lo)


def load_darcy(data_dir: Path, n_train: int, n_test: int, rate: int = 5) -> dict:
    """Training coefficients (nodes and cells) and test coefficients and solutions."""
    out = {}
    for name, file, n in (
        ("train", PROTOCOL["train_file"], n_train),
        ("test", PROTOCOL["test_file"], n_test),
    ):
        path = Path(data_dir) / file
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        keys = ("coeff",) if name == "train" else ("coeff", "sol")
        raw = loadmat(path.as_posix(), variable_names=keys)
        if raw["coeff"].shape[0] < n:
            raise ValueError(f"{path} has only {raw['coeff'].shape[0]} samples")
        a_native = raw["coeff"][:n].astype(np.float32)
        out[f"a_{name}"] = a_native[:, ::rate, ::rate]
        if name == "train":
            resolution = out["a_train"].shape[1] - 1
            out["cells_train"] = majority_per_cell(a_native, resolution)
            out["a_range"] = (float(a_native.min()), float(a_native.max()))
        else:
            out["u_test"] = raw["sol"][:n, ::rate, ::rate].astype(np.float32)
    print(
        f"coefficient fields: {n_train} train (no solutions) / {n_test} test | "
        f"nodes {out['a_train'].shape[1:]} | cells {out['cells_train'].shape[1:]} | "
        f"values {out['a_range']}"
    )
    return out


# %%
# The physics loss
# ----------------
# The mesh has one bilinear (Q4) element per cell of the 85x85 node grid,
# 84x84 elements. ``Q4Galerkin`` precomputes the shape-function gradients and
# the Jacobians at 2x2 Gauss points. The Ritz energy of the PDE is
#
#     J(u) = 1/2 int a |grad u|^2 dx - int f u dx,   f = 1,
#
# and its minimiser over the Q4 functions that vanish on the boundary is the
# finite-element solution. ``grad_energy`` gives the first integral; the
# second is the integral of ``u``, evaluated at the same Gauss points. The
# coefficient enters through ``kappa``: one value per element.


def build_q4_mesh(resolution: int):
    """Nodes, CCW quads and boundary mask of the uniform grid on [0, 1]^2."""
    xs = np.linspace(0.0, 1.0, resolution + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    i, j = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing="ij")
    n0 = (i * (resolution + 1) + j).ravel()
    quad = np.stack([n0, n0 + resolution + 1, n0 + resolution + 2, n0 + 1], axis=1)
    boundary = (xy == 0.0).any(axis=1) | (xy == 1.0).any(axis=1)
    return xy, quad.astype(np.int32), boundary


def shape_at_gauss() -> np.ndarray:
    """Bilinear shape functions of the 4 element nodes at the 2x2 Gauss
    points, ``(Q, 4)``, in ``Q4Galerkin``'s point and node order."""
    g = 1.0 / np.sqrt(3.0)
    gauss = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    return (
        0.25
        * (1 + gauss[:, None, 0] * corners[None, :, 0])
        * (1 + gauss[:, None, 1] * corners[None, :, 1])
    )


def make_ritz_energy(galerkin: Q4Galerkin):
    """Return ``energy(u_batch, cells_batch)``: the mean Ritz energy of a batch."""
    shape = jnp.asarray(shape_at_gauss(), dtype=jnp.float32)
    _, _, det_j = galerkin.physical_grad_and_jacobian()
    weight = jnp.abs(det_j)  # (Q, T): Gauss weights are 1

    def kappa(cells, point):
        resolution = cells.shape[0]
        i = jnp.clip(
            jnp.floor(point[0] * resolution).astype(jnp.int32), 0, resolution - 1
        )
        j = jnp.clip(
            jnp.floor(point[1] * resolution).astype(jnp.int32), 0, resolution - 1
        )
        return cells[i, j]

    def energy_one(u, cells):
        stiffness = 0.5 * galerkin.grad_energy(u, kappa, cells)
        u_at_gauss = jnp.einsum("qp,tp->qt", shape, u[galerkin.quad])
        return stiffness - jnp.sum(weight * u_at_gauss)

    def energy(u_batch, cells_batch):
        return jnp.mean(jax.vmap(energy_one)(u_batch, cells_batch))

    return energy


# %%
# Boundary condition and output scale, without labels
# ---------------------------------------------------
# The prediction is ``u = (c N(a) + m) sin(pi x) sin(pi y)``: the sine factor
# is zero on the boundary, so ``u = 0`` holds exactly and the loss needs no
# boundary term. ``c`` and ``m`` replace the solution statistics a
# data-driven model normalises with. They come from one Q4 solve of the
# torsion problem ``-Lap(phi) = 1`` (``phi = 0`` on the boundary), divided by
# ``sqrt(a_lo a_hi) = 6``, the effective coefficient of a 50/50 two-phase
# medium in 2D: ``m`` and ``c`` are the mean and standard deviation of
# ``phi / 6`` over the nodes. No solution of the data is used.


def torsion_output_scale(
    galerkin: Q4Galerkin, boundary, a_range
) -> tuple[float, float]:
    """``(c, m)``: std and mean of the Q4 torsion function over ``sqrt(a_lo a_hi)``."""
    dn_dx, dn_dy, det_j = (
        np.asarray(v, np.float64) for v in galerkin.physical_grad_and_jacobian()
    )
    w = np.abs(det_j)[..., None, None]  # (Q, T, 1, 1)
    local_k = np.sum(
        w
        * (
            dn_dx[..., :, None] * dn_dx[..., None, :]
            + dn_dy[..., :, None] * dn_dy[..., None, :]
        ),
        axis=0,
    )  # (T, 4, 4)
    local_f = np.einsum("qt,qp->tp", np.abs(det_j), shape_at_gauss())  # int of N_i
    quad = np.asarray(galerkin.quad)
    n = galerkin.n_nodes
    rows, cols = np.repeat(quad, 4, axis=1).ravel(), np.tile(quad, (1, 4)).ravel()
    stiffness = coo_matrix((local_k.ravel(), (rows, cols)), shape=(n, n)).tolil()
    load = np.zeros(n)
    np.add.at(load, quad.ravel(), local_f.ravel())
    stiffness[boundary, :] = 0.0
    stiffness[boundary, boundary] = 1.0
    load[boundary] = 0.0
    phi = spsolve(stiffness.tocsr(), load) / np.sqrt(a_range[0] * a_range[1])
    return float(phi.std()), float(phi.mean())


def sin_boundary_factor(xy: np.ndarray) -> np.ndarray:
    """``sin(pi x) sin(pi y)``: zero on the boundary, one at the centre."""
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])


# %%
# Scoring points
# --------------
# The network predicts at the mesh nodes ``j / 84``, where the loss acts. Li
# et al.'s generator stores its solutions at the native cell centres, half a
# native cell inside those nodes. A fixed linear interpolation along each
# axis moves the prediction there before it is compared with the released
# solution, ``U -> M U M^T``.


def cell_centre_interpolation(n_side: int, rate: int) -> np.ndarray:
    """``(n_side, n_side)`` 1D interpolation from the nodes to the sample points."""
    n_native = rate * (n_side - 1) + 1
    nodes = np.linspace(0.0, 1.0, n_side)
    points = (rate * np.arange(n_side) + 0.5) / n_native
    eye = np.eye(n_side)
    return np.stack(
        [np.interp(points, nodes, eye[:, j]) for j in range(n_side)], axis=1
    )


# %%
# Normalise the input
# -------------------
# As in the data-driven FNO, the coefficient gets a per-pixel unit Gaussian,
# fitted on the training coefficients only. There is no output normaliser:
# ``c`` and ``m`` above play that role.


class PixelNormalizer:
    """``(v - mean) / (std + eps)`` with a mean and a std per grid point."""

    def __init__(self, train: np.ndarray, eps: float = 1e-5) -> None:
        self.mean = train.mean(axis=0, keepdims=True)
        self.std = train.std(axis=0, keepdims=True)
        self.eps = eps

    def encode(self, v):
        return (v - self.mean) / (self.std + self.eps)


# %%
# Build the model
# ---------------
# ``FNO`` is channels-first, so the coefficient is ``(batch, 1, 85, 85)``. The
# defaults append the grid coordinates as two more input channels and keep a
# soft-gated channel MLP. ``count_model_params`` counts a complex spectral
# weight as two real numbers.


def build_model(size: dict, sample: np.ndarray, seed: int):
    """Build the FNO, initialise it on one sample and print the parameter count."""
    model = FNO(in_channels=1, out_channels=1, **size)
    key = jax.random.split(jax.random.PRNGKey(seed))[1]
    params = model.init(key, jnp.asarray(sample)[None])
    print(f"FNO {size} | parameters: {count_model_params(params):,}")
    return model, params


# %%
# Loss
# ----
# ``predict_nodes`` turns the network output into a nodal solution in
# physical units that is zero on the boundary. The loss is the Ritz energy of
# that solution, with the finite-element contractions at full precision.


def make_predict_nodes(model, c: float, m: float, boundary_factor):
    """Encoded ``(B, 1, H, W)`` coefficient -> ``(B, H * W)`` nodal pressure."""

    def predict_nodes(params, a_enc):
        out = model.apply(params, a_enc)[:, 0].reshape(a_enc.shape[0], -1)
        return (c * out + m) * boundary_factor

    return predict_nodes


def make_loss(predict_nodes, energy):
    """Return ``loss_fn(params, a_enc, cells) -> Ritz energy``."""

    def loss_fn(params, a_enc, cells):
        u = predict_nodes(params, a_enc)
        with jax.default_matmul_precision(FE_PRECISION):
            return energy(u, cells)

    return loss_fn


# %%
# Optimiser
# ---------
# The FNO's spectral weights are complex, and ``jax.grad`` returns the
# conjugate of their descent direction, so the complex-aware ``adamw_cx`` is
# used. The learning rate is halved every ``lr_decay_epochs`` epochs.
# ``inject_hyperparams`` would otherwise give the learning rate, the betas,
# ``eps`` and the decay the params' highest dtype, complex64, so it is told
# to keep them float32.


def make_optimizer(recipe: dict, steps_per_epoch: int):
    """``adamw_cx`` with a staircase learning-rate decay."""
    schedule = optax.exponential_decay(
        init_value=recipe["learning_rate"],
        transition_steps=max(steps_per_epoch * recipe["lr_decay_epochs"], 1),
        decay_rate=recipe["lr_decay_rate"],
        staircase=True,
    )
    return optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
        learning_rate=schedule,
        weight_decay=recipe["weight_decay"],
        eps=recipe["adam_eps"],
    )


# %%
# Train
# -----
# Batches of 20 coefficient fields in a new random order every epoch. The
# Ritz energy is small (about -0.003) and at initialisation most of its
# gradients are below Adam's ``eps = 1e-8``, where Adam's steps vanish. The
# loss is therefore multiplied by the number of interior nodes (6889 at
# 85x85, about 1/h^2) before differentiation; the minimiser is unchanged.
# Every ``eval_every`` steps the mean training energy and the test error are
# logged. No step is selected by the test error: the final weights are the
# result.


def train(
    model,
    params,
    key,
    data,
    predict_nodes,
    loss_fn,
    scorer,
    epochs,
    loss_scale,
    eval_every,
    recipe=RECIPE,
):
    """Train for ``epochs`` epochs. Return the final parameters and the history."""
    a_train, cells_train = data["a_train_enc"], data["cells_train"]
    bs = recipe["batch_size"]
    n_train = len(a_train)
    steps_per_epoch = max(n_train // bs, 1)
    tx = make_optimizer(recipe, steps_per_epoch)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, a_enc, cells):
        loss, grads = jax.value_and_grad(
            lambda p: loss_scale * loss_fn(p, a_enc, cells)
        )(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss / loss_scale

    history, window, step, t0 = [], [], 0, perf_counter()
    total = steps_per_epoch * epochs
    for _ in range(epochs):
        key, sub = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sub, n_train))
        for b in range(steps_per_epoch):
            idx = perm[b * bs : (b + 1) * bs]
            params, opt_state, loss = train_step(
                params,
                opt_state,
                jnp.asarray(a_train[idx]),
                jnp.asarray(cells_train[idx]),
            )
            window.append(float(loss))
            step += 1
            if step % eval_every and step != total:
                continue
            test = scorer(params)["test_rel_l2"]
            history.append(
                dict(
                    step=step,
                    energy=float(np.mean(window)),
                    test_rel_l2=test,
                    secs=perf_counter() - t0,
                )
            )
            window = []
            print(
                f"step {step}/{total}: energy={history[-1]['energy']:.4e} "
                f"test_rel_l2={test:.4f} elapsed={history[-1]['secs']:.1f}s"
            )
    return params, history


# %%
# Evaluate
# --------
# The prediction is moved to the released sample points and compared with
# the released test solutions: the relative L2 error of every test field,
# averaged. The error at the nodes, without the map, is reported too. Even an
# exact Q4 solve is 0.31% from the released solutions (a finite-difference
# solve on the native grid): that is the floor of this formulation.


def make_scorer(predict_nodes, data, to_points, batch_size: int = 20):
    """Return ``scorer(params) -> dict`` of test errors and predictions."""
    a_test, u_test = data["a_test_enc"], data["u_test"]
    n_side = u_test.shape[1]

    @jax.jit
    def predict(params, a_enc):
        u = predict_nodes(params, a_enc).reshape(-1, n_side, n_side)
        mapped = jnp.einsum(
            "ij,bjk,lk->bil", to_points, u, to_points, precision="highest"
        )
        return mapped, u

    def scorer(params):
        mapped, nodes = (
            np.concatenate(parts)
            for parts in zip(
                *(
                    predict(params, jnp.asarray(a_test[i : i + batch_size]))
                    for i in range(0, len(a_test), batch_size)
                )
            )
        )
        flat = u_test.reshape(len(u_test), -1)

        def errors(pred):
            pred = pred.reshape(len(pred), -1)
            return np.linalg.norm(pred - flat, axis=1) / np.linalg.norm(flat, axis=1)

        e = errors(mapped)
        return dict(
            test_rel_l2=float(e.mean()),
            test_rel_l2_nodes=float(errors(nodes).mean()),
            errors=e,
            predictions=mapped,
        )

    return scorer


# %%
# Plot
# ----
# The standard figure: the input coefficient, the released solution, the
# prediction on the same colour scale and the absolute error, for one test
# field. A second figure shows the training energy and the test error
# against the optimiser step.


def plot_prediction(a, truth, pred, path: Path, title: str) -> None:
    """Save coefficient, true pressure, predicted pressure and absolute error."""
    lim = (float(truth.min()), float(truth.max()))
    panels = [
        ("Input: permeability a", a, "viridis", None),
        ("Truth: pressure u", truth, "jet", lim),
        ("Prediction: pressure u", pred, "jet", lim),
        ("Absolute error", np.abs(pred - truth), "magma", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    for ax, (name, values, cmap, vlim) in zip(axes, panels):
        vmin, vmax = vlim if vlim else (None, None)
        im = ax.imshow(
            values.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    """Save the training Ritz energy and the test error per logged step."""
    steps = [h["step"] for h in history]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax0.plot(steps, [h["energy"] for h in history], marker="o")
    ax0.set_xlabel("optimiser step")
    ax0.set_ylabel("Ritz energy J (training, mean over the window)")
    ax0.grid(True, alpha=0.3)
    ax1.plot(steps, [100 * h["test_rel_l2"] for h in history], marker="o", color="C3")
    ax1.set_xlabel("optimiser step")
    ax1.set_ylabel("test relative L2 [%]")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")
    fig.savefig(path, dpi=120)
    plt.close(fig)


# %%
# Put it together
# ---------------
# ``--smoke`` takes a few coefficient fields, uses the tiny FNO and trains
# for a few epochs. Without it the script trains the benchmark-size FNO on
# the 1000 released coefficient fields for 100,000 steps. That mode is meant
# for a GPU.


def main(argv: list[str] | None = None) -> dict:
    """Run the tutorial end to end and return the final metrics."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("DARCY_DATA_DIR"),
        help="folder with piececonst_r421_N1024_smooth{1,2}.mat (or $DARCY_DATA_DIR)",
    )
    ap.add_argument("--smoke", action="store_true", help="few fields, tiny FNO")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/darcyphysics_fno_tutorial")
    )
    ap.add_argument("--n-train", type=int, default=100, help="fields in smoke mode")
    ap.add_argument("--n-test", type=int, default=20, help="fields in smoke mode")
    ap.add_argument("--eval-every", type=int, default=None, help="steps between logs")
    args = ap.parse_args(argv)
    if args.data_dir is None:
        ap.error("pass --data-dir or set $DARCY_DATA_DIR")

    t_start = perf_counter()
    print_environment(args.seed)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.smoke
    n_train, n_test = (
        (args.n_train, args.n_test)
        if smoke
        else (PROTOCOL["n_train"], PROTOCOL["n_test"])
    )
    epochs = args.epochs or (10 if smoke else RECIPE["epochs"])
    eval_every = args.eval_every or (10 if smoke else 1000)
    size = SIZES["smoke" if smoke else "full"]
    if not smoke and jax.devices()[0].platform == "cpu":
        print("warning: the full protocol on a CPU is very slow; use --smoke")

    t0 = perf_counter()
    raw = load_darcy(
        Path(args.data_dir).expanduser(), n_train, n_test, PROTOCOL["subsample_rate"]
    )
    print(f"data loading: {perf_counter() - t0:.1f}s")
    n_side = raw["a_train"].shape[1]
    xy, quad, boundary = build_q4_mesh(n_side - 1)
    with jax.default_matmul_precision(FE_PRECISION):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
    c, m = torsion_output_scale(galerkin, boundary, raw["a_range"])
    print(f"label-free output scale c={c:.5f} offset m={m:.5f}")
    energy = make_ritz_energy(galerkin)

    a_norm = PixelNormalizer(raw["a_train"][:, None])
    data = dict(
        a_train_enc=a_norm.encode(raw["a_train"][:, None]).astype(np.float32),
        a_test_enc=a_norm.encode(raw["a_test"][:, None]).astype(np.float32),
        cells_train=raw["cells_train"].astype(np.float32),
        u_test=raw["u_test"],
    )
    model, params = build_model(size, data["a_train_enc"][0], args.seed)
    predict_nodes = make_predict_nodes(
        model, c, m, jnp.asarray(sin_boundary_factor(xy), dtype=jnp.float32)
    )
    to_points = jnp.asarray(
        cell_centre_interpolation(n_side, PROTOCOL["subsample_rate"]), dtype=jnp.float32
    )
    scorer = make_scorer(predict_nodes, data, to_points)
    loss_scale = float((~boundary).sum())
    print(f"loss scale (interior nodes): {loss_scale:.0f}")

    key = jax.random.split(jax.random.PRNGKey(args.seed))[0]
    params, history = train(
        model,
        params,
        key,
        data,
        predict_nodes,
        make_loss(predict_nodes, energy),
        scorer,
        epochs,
        loss_scale,
        eval_every,
    )
    metrics = scorer(params)
    print(
        f"test relative L2: {metrics['test_rel_l2']:.4f} at the released points | "
        f"{metrics['test_rel_l2_nodes']:.4f} at the nodes"
    )

    k = int(np.argsort(metrics["errors"])[len(metrics["errors"]) // 2])
    plot_prediction(
        raw["a_test"][k],
        raw["u_test"][k],
        metrics["predictions"][k],
        out_dir / "prediction.png",
        f"Darcy, physics-only FNO: test field with the median error "
        f"({metrics['errors'][k]:.2%})",
    )
    plot_loss(history, out_dir / "loss.png")
    print(f"figures in {out_dir} | total time {perf_counter() - t_start:.1f}s")
    return {k: v for k, v in metrics.items() if k not in ("errors", "predictions")}


if __name__ == "__main__":
    main()
