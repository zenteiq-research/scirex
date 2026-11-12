# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Manufactured-solution correctness test on a genuinely irregular (non-convex)
domain: an L-shape, built as ``[0,2]x[0,2]`` minus the notch ``[1,2]x[1,2]``.

This is the test that was missing before: every other test involving
:mod:`~scirex.experimental.losses.domain_membership` either checks pure geometry
(a circle's known signed distance) or an integral (a masked area matching a
known analytic area), and every test of the AD differentiators checks a
known analytic derivative but on a plain region with no domain shape at
all. Nothing previously combined "genuinely irregular domain shape" with
"derivative checked against a known exact solution" in one test.

An L-shape is deliberately chosen over a circle: it has a reentrant
(concave, 270 degree) corner at ``(1, 1)``, exactly the case
``domain_membership``'s own module docstring flags as the nearest-
neighbor signed-distance approximation's known weak point ("can be wrong
deep inside concave pockets"). This test exercises points on both sides of
that corner, not just points far from any boundary feature.

The manufactured solution is a plain polynomial, ``u(x, y) = x^2 + y^2``,
not a trained model -- it has a *known, exact* first derivative
(``2x``, ``2y``) and second derivative (constant Laplacian ``4``), so any
deviation from those values is unambiguously a bug, not model/training
noise. Forcing is chosen as ``f = -4`` so ``u`` exactly satisfies
``-Laplacian(u) - f = 0`` everywhere -- the strongest correctness check
available: not just "derivatives look right" but "the whole
quadrature + masking + AD + PDE-term-assembly pipeline reports the PDE as
exactly satisfied for a solution that truly does satisfy it," and reports
it as clearly violated for one that doesn't.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.losses.ad_pde_terms import DiffusionTermAD, SourceTermAD
from scirex.experimental.losses.domain_membership import (
    is_inside,
    mask_quadrature_weights,
)
from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.experimental.losses.quadrature import (
    gauss_legendre_points_weights,
    make_patches,
    patch_quadrature_coords,
)
from scirex.losses.equation_losses import PDEResidualLoss
from tests import tolerances

_BOUNDS = [(0.0, 2.0), (0.0, 2.0)]
_TRUE_L_AREA = 3.0  # 2x2 bounding box (area 4) minus the 1x1 notch (area 1)

# CCW boundary of the L-shape (bounding box minus the [1,2]x[1,2] notch),
# one (start, end, outward_normal) entry per straight edge -- a polygon's
# normal really is piecewise constant per edge, not an approximation.
_EDGES = [
    ((0.0, 0.0), (2.0, 0.0), (0.0, -1.0)),
    ((2.0, 0.0), (2.0, 1.0), (1.0, 0.0)),
    ((2.0, 1.0), (1.0, 1.0), (0.0, 1.0)),  # the notch's horizontal edge
    ((1.0, 1.0), (1.0, 2.0), (1.0, 0.0)),  # the notch's vertical edge
    ((1.0, 2.0), (0.0, 2.0), (0.0, 1.0)),
    ((0.0, 2.0), (0.0, 0.0), (-1.0, 0.0)),
]


def _lshape_boundary(n_per_edge: int = 80) -> tuple[jnp.ndarray, jnp.ndarray]:
    points, normals = [], []
    for (x0, y0), (x1, y1), normal in _EDGES:
        t = np.linspace(0.0, 1.0, n_per_edge, endpoint=False)
        points.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=-1))
        normals.append(np.tile(normal, (n_per_edge, 1)))
    return (
        jnp.asarray(np.concatenate(points, axis=0), dtype=jnp.float32),
        jnp.asarray(np.concatenate(normals, axis=0), dtype=jnp.float32),
    )


def _u(coords: jnp.ndarray) -> jnp.ndarray:
    """Manufactured solution u(x, y) = x^2 + y^2. Exact: du/dx=2x, du/dy=2y,
    Laplacian(u) = 4 everywhere."""
    x, y = coords[:, 0], coords[:, 1]
    return x**2 + y**2


def _u_wrong(coords: jnp.ndarray) -> jnp.ndarray:
    """A field that does NOT satisfy the same PDE -- for the negative
    control confirming the residual check isn't vacuously zero."""
    x, y = coords[:, 0], coords[:, 1]
    return jnp.sin(x) * jnp.cos(y)


def _assemble_quadrature(
    n_patches_per_axis: list[int], n_quad: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    patches = make_patches(_BOUNDS, n_patches_per_axis)
    ref_points, ref_weights = gauss_legendre_points_weights(n_quad, dim=2)
    coords_list, weights_list = [], []
    for patch in patches:
        c, w = patch_quadrature_coords(patch, ref_points, ref_weights)
        coords_list.append(c)
        weights_list.append(w)
    return jnp.concatenate(coords_list, axis=0), jnp.concatenate(weights_list, axis=0)


def test_lshape_membership_correct_near_reentrant_corner() -> None:
    """Stress-tests exactly the case the module docstring flags as a known
    weak point: points immediately on either side of the concave corner."""
    boundary_points, boundary_normals = _lshape_boundary()
    points = jnp.asarray(
        [
            [0.99, 0.99],  # just inside the reentrant corner
            [1.01, 1.01],  # just inside the notch, past the corner
            [0.5, 0.5],  # far from any boundary feature, clearly inside
            [1.5, 1.5],  # deep in the notch, clearly outside
        ]
    )
    inside = is_inside(points, boundary_points, boundary_normals)
    assert bool(inside[0])
    assert not bool(inside[1])
    assert bool(inside[2])
    assert not bool(inside[3])


def test_masked_area_matches_true_lshape_area() -> None:
    boundary_points, boundary_normals = _lshape_boundary()
    coords, weights = _assemble_quadrature(n_patches_per_axis=[6, 6], n_quad=3)
    masked = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude="outside"
    )
    assert float(jnp.sum(masked)) == pytest.approx(_TRUE_L_AREA, abs=1e-4)


def test_derivatives_exactly_match_analytic_solution_on_lshape() -> None:
    """AD has no discretization error against a polynomial -- these should
    match to floating point precision, not just approximately."""
    coords, _ = _assemble_quadrature(n_patches_per_axis=[6, 6], n_quad=3)
    exact = ExactPointwiseADDiff()

    du_dx = exact.derivative(_u, coords, dim=0, order=1)
    du_dy = exact.derivative(_u, coords, dim=1, order=1)
    x, y = coords[:, 0], coords[:, 1]
    assert jnp.allclose(du_dx, 2 * x, atol=tolerances.atol(1e-5))
    assert jnp.allclose(du_dy, 2 * y, atol=tolerances.atol(1e-5))

    laplacian = exact.derivative(_u, coords, dim=0, order=2) + exact.derivative(
        _u, coords, dim=1, order=2
    )
    assert jnp.allclose(laplacian, 4.0, atol=tolerances.atol(1e-5))


def test_pde_residual_vanishes_for_exact_solution_on_lshape_masked_domain() -> None:
    """The strongest check: the full quadrature + masking + AD + term
    assembly pipeline reports the PDE as satisfied (residual ~0) for a
    solution that truly satisfies it, restricted to the L-shaped domain."""
    boundary_points, boundary_normals = _lshape_boundary()
    coords, weights = _assemble_quadrature(n_patches_per_axis=[6, 6], n_quad=3)
    masked_weights = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude="outside"
    )

    exact = ExactPointwiseADDiff()
    diffusion = DiffusionTermAD(dims=[0, 1], coeff=1.0, differentiator=exact)
    pde = PDEResidualLoss(
        terms={"diffusion": (-1.0, diffusion), "source": (-1.0, SourceTermAD(-4.0))},
        reduction="mean",
    )

    residual = pde.residual({"u_fn": _u, "coords": coords})
    assert jnp.allclose(residual, 0.0, atol=1e-4)

    weighted_loss = jnp.sqrt(
        jnp.sum(masked_weights * residual**2) / jnp.sum(masked_weights)
    )
    assert float(weighted_loss) < 1e-6


def test_pde_residual_is_nonzero_for_a_wrong_solution_on_lshape() -> None:
    """Negative control: confirms the check above isn't vacuously zero for
    any input."""
    boundary_points, boundary_normals = _lshape_boundary()
    coords, weights = _assemble_quadrature(n_patches_per_axis=[6, 6], n_quad=3)
    masked_weights = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude="outside"
    )

    exact = ExactPointwiseADDiff()
    diffusion = DiffusionTermAD(dims=[0, 1], coeff=1.0, differentiator=exact)
    pde = PDEResidualLoss(
        terms={"diffusion": (-1.0, diffusion), "source": (-1.0, SourceTermAD(-4.0))},
        reduction="mean",
    )

    residual = pde.residual({"u_fn": _u_wrong, "coords": coords})
    weighted_loss = jnp.sqrt(
        jnp.sum(masked_weights * residual**2) / jnp.sum(masked_weights)
    )
    assert float(weighted_loss) > 0.5


def test_patchquadratureaddiff_matches_analytic_solution_on_lshape() -> None:
    """Same manufactured-solution check, now through PatchQuadratureADDiff
    (the actual mechanism used for a globally-mixing model), confirming the
    per-patch composition machinery introduces no error against a smooth
    analytic field, and its own weights already match the L-shape's area
    once masked."""
    boundary_points, boundary_normals = _lshape_boundary()
    pq = PatchQuadratureADDiff(bounds=_BOUNDS, n_patches_per_axis=[6, 6], n=3)
    masked_weights = mask_quadrature_weights(
        pq.coords, pq.weights, boundary_points, boundary_normals, exclude="outside"
    )

    diffusion = DiffusionTermAD(dims=[0, 1], coeff=1.0, differentiator=pq)
    pde = PDEResidualLoss(
        terms={"diffusion": (-1.0, diffusion), "source": (-1.0, SourceTermAD(-4.0))},
        reduction="mean",
    )

    residual = pde.residual({"u_fn": _u, "coords": pq.coords})
    assert jnp.allclose(residual, 0.0, atol=1e-4)
    assert float(jnp.sum(masked_weights)) == pytest.approx(_TRUE_L_AREA, abs=1e-4)
