# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared pieces of the tests of the four physics-only Darcy tutorial scripts.

Every script writes out the physics of the benchmark (element coefficient,
Q4 mesh, Ritz energy, torsion output scale, scoring map) with public API
only. :func:`check_physics_matches_benchmark` checks each piece against the
benchmark's own code in ``scripts/benchmarks/darcy_physics/_ritz.py`` and
``scirex.losses.darcy_piecewise_coefficient``.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tests import tolerances
from tests.public_api_helpers import REPO

TRAIN_FILE = "piececonst_r421_N1024_smooth1.mat"
TEST_FILE = "piececonst_r421_N1024_smooth2.mat"


def load_script(model: str):
    """Import ``scripts/tutorials/darcyphysics_<model>.py`` as a module."""
    path = REPO / "scripts" / "tutorials" / f"darcyphysics_{model}.py"
    spec = importlib.util.spec_from_file_location(
        f"darcyphysics_{model}_tutorial", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_darcy(folder: Path, n: int = 12, native: int = 106) -> Path:
    """Write both ``.mat`` files with the real layout: ``coeff`` in {3, 12}
    and ``sol``, ``(n, native, native)``. With the benchmark's stride of 5,
    ``native=106`` gives a 22x22 node grid (21x21 elements)."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    s = np.linspace(0.0, 1.0, native)
    bump = np.outer(np.sin(np.pi * s), np.sin(np.pi * s))
    for name in (TRAIN_FILE, TEST_FILE):
        coeff = np.where(rng.normal(size=(n, native, native)) > 0, 12, 3)
        sol = 0.01 * bump[None] * (1.0 + 0.1 * rng.normal(size=(n, 1, 1)))
        savemat(folder / name, dict(coeff=coeff.astype(np.uint8), sol=sol))
    return folder


def random_native(n: int = 3, native: int = 106, seed: int = 0) -> np.ndarray:
    """Native ``(n, native, native)`` coefficient fields in {3, 12}."""
    rng = np.random.default_rng(seed)
    return np.where(rng.normal(size=(n, native, native)) > 0, 12.0, 3.0).astype(
        np.float32
    )


def check_physics_matches_benchmark(tutorial) -> None:
    """Element coefficient, mesh, energy, output scale and scoring map of the
    tutorial equal the benchmark's."""
    from scirex.losses import Q4Galerkin
    from scirex.losses.darcy_piecewise_coefficient import (
        BatchedEnergyFunctionalLossQ4,
        build_grid_aligned_q4_mesh,
        kappa_fn,
        source_fn,
    )
    from scripts.benchmarks.darcy_physics import _ritz

    a_native = random_native()
    resolution = 21
    np.testing.assert_array_equal(
        tutorial.majority_per_cell(a_native, resolution),
        _ritz.node_to_cell_coefficient(a_native, resolution, "majority"),
    )

    xy, quad, boundary = tutorial.build_q4_mesh(resolution)
    xy_b, quad_b, boundary_b = build_grid_aligned_q4_mesh(resolution)
    np.testing.assert_array_equal(xy, xy_b)
    np.testing.assert_array_equal(quad, quad_b)
    np.testing.assert_array_equal(boundary, boundary_b)

    with jax.default_matmul_precision("highest"):
        galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
        c, m = tutorial.torsion_output_scale(galerkin, boundary, (3.0, 12.0))
        c_b, m_b = _ritz.torsion_output_scale(galerkin, ~boundary, 3.0, 12.0)
        np.testing.assert_allclose([c, m], [c_b, m_b], rtol=tolerances.rtol(1e-5))

        cells = jnp.asarray(tutorial.majority_per_cell(a_native, resolution))
        rng = np.random.default_rng(1)
        u = jnp.asarray(
            0.01 * rng.random((3, xy.shape[0])) * (~boundary), dtype=jnp.float32
        )
        ours = tutorial.make_ritz_energy(galerkin)(u, cells)
        theirs = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)(u, cells)
        np.testing.assert_allclose(
            float(ours), float(theirs), rtol=tolerances.rtol(1e-5)
        )

    np.testing.assert_array_equal(
        tutorial.cell_centre_interpolation(22, 5),
        _ritz.cell_centre_interpolation(22, 5),
    )
    np.testing.assert_array_equal(
        tutorial.sin_boundary_factor(xy), _ritz.sin_boundary_factor(xy)
    )
