# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Model-independent pieces of physics-only (Ritz-energy) training on the
Darcy benchmark, shared by every model's run and by the checks
(``check_floor.py``, ``check_sampler.py``).

The Q4 Ritz loss takes one constant coefficient value per *cell* of a
``resolution x resolution`` mesh, while the released data stores ``a`` at the
*nodes* of a 421 x 421 grid. How each cell's value is picked changes the
discrete PDE the loss solves:

- ``"corner"``: the node at the cell's lower-left corner -- what taking every
  ``r``-th node and dropping the last row/column gives;
- ``"majority"``: the more frequent of the two coefficient values over all
  ``(r+1)^2`` nodes the cell covers (ties go to the higher value);
- ``"mean"``: the arithmetic mean over those nodes;

for a cell spanning native nodes ``[r*i, r*i + r]`` in each direction
(``r = (n_native - 1) / resolution``).

Also here: the fixed map from the Q4 nodes to the points where the released
solutions are sampled (:func:`cell_centre_interpolation`), the exact boundary
factor, the label-free output scale from the torsion function
(:func:`torsion_output_scale`), the precision of the finite-element
contractions (:data:`FE_PRECISION`), the fixed, seed-independent pool of
extra coefficient fields drawn from the data's own law
(:func:`sample_extra_coefficients`), and the training-curve figure
(:func:`plot_training_curves`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from scirex.losses import Q4Galerkin
from scirex.losses.darcy_piecewise_coefficient import (
    assemble_and_solve_q4,
    sample_piecewise_coefficient,
)

CellMode = Literal["corner", "majority", "mean"]
CELL_MODES: tuple[CellMode, ...] = ("corner", "majority", "mean")


def node_to_cell_coefficient(
    a_node: np.ndarray, resolution: int, mode: CellMode
) -> np.ndarray:
    """``a_node`` shape ``(N, n, n)`` on a native node grid with
    ``n - 1`` divisible by ``resolution`` -> ``(N, resolution, resolution)``
    per-cell coefficient."""
    if a_node.ndim != 3 or a_node.shape[1] != a_node.shape[2]:
        raise ValueError(f"expected (N, n, n) node fields, got {a_node.shape}")
    n_native = a_node.shape[1]
    if (n_native - 1) % resolution:
        raise ValueError(
            f"native grid of {n_native} nodes can't be split into {resolution} "
            "equal cells"
        )
    rate = (n_native - 1) // resolution

    if mode == "corner":
        return a_node[:, : n_native - 1 : rate, : n_native - 1 : rate]

    # (N, resolution, resolution, rate+1, rate+1): every node each cell covers.
    blocks = np.lib.stride_tricks.sliding_window_view(
        a_node, (rate + 1, rate + 1), axis=(1, 2)
    )[:, ::rate, ::rate]
    if mode == "mean":
        return blocks.mean(axis=(-2, -1))
    if mode == "majority":
        lo, hi = float(a_node.min()), float(a_node.max())
        return np.where(blocks.mean(axis=(-2, -1)) >= 0.5 * (lo + hi), hi, lo)
    raise ValueError(f"unknown mode {mode!r}; expected one of {CELL_MODES}")


def cell_centre_interpolation(n_side: int, subsample_rate: int) -> np.ndarray:
    """``(n_side, n_side)`` matrix of 1-D linear interpolation from the Q4
    nodes ``x_j = j / (n_side - 1)`` to the points where Li et al.'s solutions
    are sampled.

    Their generator solves on the nodes of ``[0, 1]`` and spline-interpolates
    the result to the cell centres ``(i + 1/2) / n_native`` of the native grid,
    so the strided benchmark points are ``(subsample_rate * j + 1/2) /
    n_native``, half a native cell inside the nodes. Applied along both axes,
    ``U -> M U M^T``.
    """
    n_native = subsample_rate * (n_side - 1) + 1
    nodes = np.linspace(0.0, 1.0, n_side)
    points = (subsample_rate * np.arange(n_side) + 0.5) / n_native
    eye = np.eye(n_side)
    return np.stack(
        [np.interp(points, nodes, eye[:, j]) for j in range(n_side)], axis=1
    )


FE_PRECISION = "highest"
# Fixed pool of extra coefficient fields (--extra-coefficients): independent
# of --seed, so every set size uses a prefix of the same pool.
EXTRA_POOL_KEY = 10_000
EXTRA_POOL_SIZE = 100_000


def split_coefficient(
    a_native: np.ndarray, subsample_rate: int, cell_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Native node coefficient ``(N, n, n)`` -> ``(model_input, cells)``:
    the strided ``(N, s, s)`` node field the model sees (identical to the
    data-driven script's input) and the ``(N, s-1, s-1)`` per-cell field
    the Q4 loss uses."""
    model_input = a_native[:, ::subsample_rate, ::subsample_rate]
    resolution = model_input.shape[1] - 1
    cells = node_to_cell_coefficient(a_native, resolution, cell_mode)
    return model_input, cells


def extra_coefficient_keys(n: int) -> jax.Array:
    """Keys of the first ``n`` fields of the fixed extra-coefficient pool."""
    if n > EXTRA_POOL_SIZE:
        raise ValueError(f"at most {EXTRA_POOL_SIZE} extra coefficients, got {n}")
    return jax.random.split(jax.random.PRNGKey(EXTRA_POOL_KEY), EXTRA_POOL_SIZE)[:n]


def sample_extra_coefficients(
    n: int, n_native: int, subsample_rate: int, cell_mode: str, chunk: int = 250
) -> tuple[np.ndarray, np.ndarray]:
    """The first ``n`` pool fields as ``(model_input, cells)``, the same
    layout as :func:`split_coefficient` gives for the released fields.
    Sampled in chunks so the native fields never all sit in memory."""
    keys = extra_coefficient_keys(n)
    model_inputs, cells = [], []
    for c0 in range(0, n, chunk):
        native = np.stack(
            [sample_piecewise_coefficient(k, n_native) for k in keys[c0 : c0 + chunk]]
        )
        a_c, cells_c = split_coefficient(native, subsample_rate, cell_mode)
        model_inputs.append(a_c.astype(np.float32))
        cells.append(cells_c.astype(np.float32))
    return np.concatenate(model_inputs), np.concatenate(cells)


def sin_boundary_factor(xy: np.ndarray) -> np.ndarray:
    """``sin(pi x) sin(pi y)`` at each node: zero on the boundary of the unit
    square, one at its centre."""
    return np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])


def plot_training_curves(
    steps: list[int],
    energy: list[float],
    test_err: list[float],
    out_path: Path,
    title: str,
) -> None:
    """Training loss and test error against optimizer steps, side by side.

    The training loss is the Ritz energy itself: negative and decreasing
    towards its minimum (the energy of the Q4 solution), so it is drawn on a
    linear axis. The shared log-scale helper would have to plot ``|J|``,
    which grows as ``J`` decreases."""
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax0.plot(steps, energy, color="#256abf", lw=1.5, marker="o", ms=3)
    ax0.set_xlabel("optimizer step")
    ax0.set_ylabel("Ritz energy J (mean over recent training batches)")
    ax0.set_title("train loss: Q4 Ritz energy (decreases to its minimum)")
    ax0.grid(alpha=0.3)
    ax1.plot(
        steps, [100 * e for e in test_err], color="crimson", lw=1.5, marker="o", ms=3
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("test rel-L2 [%]")
    ax1.set_title("test error vs released solutions")
    ax1.grid(alpha=0.3, which="both")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def torsion_output_scale(
    galerkin: Q4Galerkin, interior_mask: np.ndarray, a_lo: float, a_hi: float
) -> tuple[float, float]:
    """Label-free output scale and offset ``(c, m)``: standard deviation and
    mean over the nodes of ``phi / sqrt(a_lo * a_hi)``, with ``phi`` the Q4
    solution of ``-Lap(phi) = 1``, ``phi = 0`` on the boundary."""
    resolution = int(round(np.sqrt(galerkin.quad.shape[0])))
    phi = assemble_and_solve_q4(
        galerkin, jnp.ones((resolution, resolution), dtype=jnp.float32), interior_mask
    )
    u_ref = np.asarray(phi, dtype=np.float64) / np.sqrt(a_lo * a_hi)
    return float(u_ref.std()), float(u_ref.mean())
