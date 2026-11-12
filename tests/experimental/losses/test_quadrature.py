# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the quadrature/patch geometry primitives (JAX): Patch,
gauss_legendre_points_weights, make_patches, patch_quadrature_coords.

* ``test_gausslegendre_*`` -- node/weight correctness against known 1D
  values, and polynomial-exactness of the tensor-product rule.
* ``test_patch_*`` / ``test_makepatches_*`` -- patch geometry (Jacobian,
  validation) and that a partition tiles the domain exactly.
* ``test_patchquadraturecoords_*`` -- reference-to-physical mapping.
* ``test_assembly_*`` -- the end-to-end claim this module exists for: a
  domain integral assembled from many small patches, each with its own few
  quadrature points, exactly matches the analytic integral for a smooth
  (polynomial) integrand, regardless of how many patches it's split into.
"""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.losses.quadrature import (
    Patch,
    gauss_legendre_points_weights,
    make_patches,
    patch_quadrature_coords,
)
from tests import tolerances

# ---------------------------------------------------------------------------
# gauss_legendre_points_weights
# ---------------------------------------------------------------------------


def test_gausslegendre_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n >= 1"):
        gauss_legendre_points_weights(n=0, dim=1)


def test_gausslegendre_rejects_dim_below_one() -> None:
    with pytest.raises(ValueError, match="dim >= 1"):
        gauss_legendre_points_weights(n=2, dim=0)


def test_gausslegendre_1d_n1_is_midpoint_rule() -> None:
    points, weights = gauss_legendre_points_weights(n=1, dim=1)
    assert jnp.allclose(points, jnp.array([[0.0]]))
    assert jnp.allclose(weights, jnp.array([2.0]))


def test_gausslegendre_1d_n2_matches_known_nodes_weights() -> None:
    points, weights = gauss_legendre_points_weights(n=2, dim=1)
    expected_nodes = jnp.array([-1.0, 1.0]) / jnp.sqrt(3.0)
    assert jnp.allclose(
        jnp.sort(points[:, 0]), jnp.sort(expected_nodes), atol=tolerances.atol(1e-6)
    )
    assert jnp.allclose(weights, jnp.array([1.0, 1.0]))


def test_gausslegendre_2d_shapes_and_weight_sum() -> None:
    n, dim = 3, 2
    points, weights = gauss_legendre_points_weights(n=n, dim=dim)
    assert points.shape == (n**dim, dim)
    assert weights.shape == (n**dim,)
    # reference cube [-1, 1]^2 has measure 4
    assert jnp.allclose(jnp.sum(weights), 4.0, atol=tolerances.atol(1e-5))


def test_gausslegendre_3d_shapes() -> None:
    n, dim = 2, 3
    points, weights = gauss_legendre_points_weights(n=n, dim=dim)
    assert points.shape == (n**dim, dim)
    assert weights.shape == (n**dim,)
    assert jnp.allclose(jnp.sum(weights), 8.0, atol=tolerances.atol(1e-5))


def test_gausslegendre_exact_for_polynomial_within_degree() -> None:
    """n=3 per axis is exact through degree 5 per axis; check against the
    analytic integral of x^4*y^4 (well within degree) over [-1, 1]^2."""
    points, weights = gauss_legendre_points_weights(n=3, dim=2)
    integrand = points[:, 0] ** 4 * points[:, 1] ** 4
    quad_result = jnp.sum(weights * integrand)

    # analytic: integral_{-1}^{1} x^4 dx = 2/5, squared for the product
    expected = (2.0 / 5.0) ** 2
    assert jnp.allclose(quad_result, expected, atol=tolerances.atol(1e-5))


def test_gausslegendre_inexact_beyond_degree() -> None:
    """n=2 per axis is only exact through degree 3 per axis -- a degree-4
    monomial should NOT be integrated exactly, confirming the test above
    isn't accidentally exact for trivial reasons."""
    points, weights = gauss_legendre_points_weights(n=2, dim=1)
    integrand = points[:, 0] ** 4
    quad_result = jnp.sum(weights * integrand)
    expected = 2.0 / 5.0
    assert not jnp.allclose(quad_result, expected, atol=tolerances.atol(1e-5))


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------


def test_patch_jacobian_2d() -> None:
    patch = Patch(center=(0.5, 0.5), half_width=(0.25, 0.1))
    assert patch.jacobian == pytest.approx(0.25 * 0.1)


def test_patch_dim_property() -> None:
    patch = Patch(center=(0.0, 0.0, 0.0), half_width=(1.0, 1.0, 1.0))
    assert patch.dim == 3


def test_patch_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        Patch(center=(0.0, 0.0), half_width=(1.0,))


def test_patch_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        Patch(center=(), half_width=())


def test_patch_rejects_nonpositive_half_width() -> None:
    with pytest.raises(ValueError, match="half_width > 0"):
        Patch(center=(0.0,), half_width=(0.0,))
    with pytest.raises(ValueError, match="half_width > 0"):
        Patch(center=(0.0,), half_width=(-1.0,))


# ---------------------------------------------------------------------------
# make_patches
# ---------------------------------------------------------------------------


def test_makepatches_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="one n_patches_per_axis entry"):
        make_patches(bounds=[(0.0, 1.0), (0.0, 1.0)], n_patches_per_axis=[2])


def test_makepatches_rejects_empty_bounds() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        make_patches(bounds=[], n_patches_per_axis=[])


def test_makepatches_rejects_hi_not_greater_than_lo() -> None:
    with pytest.raises(ValueError, match="hi > lo"):
        make_patches(bounds=[(1.0, 1.0)], n_patches_per_axis=[2])
    with pytest.raises(ValueError, match="hi > lo"):
        make_patches(bounds=[(1.0, 0.0)], n_patches_per_axis=[2])


def test_makepatches_rejects_n_below_one() -> None:
    with pytest.raises(ValueError, match="n_patches_per_axis >= 1"):
        make_patches(bounds=[(0.0, 1.0)], n_patches_per_axis=[0])


def test_makepatches_single_patch_covers_whole_domain() -> None:
    patches = make_patches(bounds=[(0.0, 1.0), (2.0, 4.0)], n_patches_per_axis=[1, 1])
    assert len(patches) == 1
    (patch,) = patches
    assert patch.center == pytest.approx((0.5, 3.0))
    assert patch.half_width == pytest.approx((0.5, 1.0))


def test_makepatches_count_matches_product_of_axis_counts() -> None:
    patches = make_patches(
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], n_patches_per_axis=[2, 3, 4]
    )
    assert len(patches) == 2 * 3 * 4


def test_makepatches_tiles_domain_with_no_gap_or_overlap() -> None:
    """Every patch's own volume, summed, equals the domain's volume -- the
    partition-additivity property the whole assembly mechanism depends on."""
    bounds = [(0.0, 2.0), (-1.0, 1.0)]
    patches = make_patches(bounds=bounds, n_patches_per_axis=[4, 3])

    domain_volume = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
    patch_volume_sum = sum(2.0**p.dim * p.jacobian for p in patches)
    assert patch_volume_sum == pytest.approx(domain_volume)

    # and every patch center actually falls inside the domain bounds
    for p in patches:
        for c, (lo, hi) in zip(p.center, bounds, strict=True):
            assert lo < c < hi


# ---------------------------------------------------------------------------
# patch_quadrature_coords
# ---------------------------------------------------------------------------


def test_patchquadraturecoords_maps_reference_cube_correctly() -> None:
    patch = Patch(center=(1.0, -2.0), half_width=(0.5, 2.0))
    ref_points, ref_weights = gauss_legendre_points_weights(n=2, dim=2)

    coords, weights = patch_quadrature_coords(patch, ref_points, ref_weights)

    assert jnp.all(coords[:, 0] > patch.center[0] - patch.half_width[0])
    assert jnp.all(coords[:, 0] < patch.center[0] + patch.half_width[0])
    assert jnp.all(coords[:, 1] > patch.center[1] - patch.half_width[1])
    assert jnp.all(coords[:, 1] < patch.center[1] + patch.half_width[1])

    assert jnp.allclose(weights, ref_weights * patch.jacobian)


def test_patchquadraturecoords_weight_sum_equals_patch_volume() -> None:
    patch = Patch(center=(0.0, 0.0), half_width=(0.3, 0.7))
    ref_points, ref_weights = gauss_legendre_points_weights(n=3, dim=2)
    _, weights = patch_quadrature_coords(patch, ref_points, ref_weights)

    patch_volume = 2.0**patch.dim * patch.jacobian
    assert jnp.sum(weights) == pytest.approx(patch_volume, abs=1e-5)


def test_patchquadraturecoords_rejects_dim_mismatch() -> None:
    patch = Patch(center=(0.0, 0.0), half_width=(1.0, 1.0))
    ref_points, ref_weights = gauss_legendre_points_weights(n=2, dim=3)
    with pytest.raises(ValueError, match=r"patch\.dim"):
        patch_quadrature_coords(patch, ref_points, ref_weights)


def test_patchquadraturecoords_rejects_weight_shape_mismatch() -> None:
    patch = Patch(center=(0.0, 0.0), half_width=(1.0, 1.0))
    ref_points, _ = gauss_legendre_points_weights(n=2, dim=2)
    bad_weights = jnp.ones((ref_points.shape[0] + 1,))
    with pytest.raises(ValueError, match="ref_weights"):
        patch_quadrature_coords(patch, ref_points, bad_weights)


# ---------------------------------------------------------------------------
# End-to-end assembly: split + quadrature + sum == analytic integral
# ---------------------------------------------------------------------------


def _assemble_integral(
    bounds: list[tuple[float, float]],
    n_patches_per_axis: list[int],
    n_quad: int,
    fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> float:
    patches = make_patches(bounds, n_patches_per_axis)
    ref_points, ref_weights = gauss_legendre_points_weights(n=n_quad, dim=len(bounds))
    total = 0.0
    for patch in patches:
        coords, weights = patch_quadrature_coords(patch, ref_points, ref_weights)
        total += float(jnp.sum(weights * fn(coords)))
    return total


def test_assembly_matches_analytic_integral_regardless_of_patch_count() -> None:
    """integral over [0,1]^2 of (6*x^2*y^2) = 1*1 = 1 (since integral x^2 dx
    over [0,1] = 1/3, product of two such factors * 6 -> 6*(1/3)*(1/3)=2/3).
    Recomputed explicitly below rather than asserted from memory."""

    def fn(coords: jnp.ndarray) -> jnp.ndarray:
        x, y = coords[:, 0], coords[:, 1]
        return 6.0 * x**2 * y**2

    expected = 6.0 * (1.0 / 3.0) * (1.0 / 3.0)

    for n_patches_per_axis in ([1, 1], [2, 2], [3, 5]):
        result = _assemble_integral(
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            n_patches_per_axis=n_patches_per_axis,
            n_quad=3,
            fn=fn,
        )
        assert result == pytest.approx(expected, abs=1e-5), n_patches_per_axis


def test_assembly_matches_scipy_reference_on_nonsquare_domain() -> None:
    def fn(coords: jnp.ndarray) -> jnp.ndarray:
        x, y = coords[:, 0], coords[:, 1]
        return jnp.sin(x) * x**2 + y**4

    bounds = [(-1.0, 2.0), (0.0, 3.0)]
    result = _assemble_integral(
        bounds=bounds, n_patches_per_axis=[4, 3], n_quad=4, fn=fn
    )

    # High-resolution manual trapezoidal rule as an independent reference,
    # since sin(x)*x^2 is not polynomial (Gauss-Legendre is only exact for
    # polynomials) -- use a fine enough grid that trapezoid error is
    # negligible relative to the tolerance below. Written by hand instead
    # of via a library trapezoid function to avoid a numpy-version-specific
    # API (np.trapz vs np.trapezoid).
    def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        return np.sum((y[..., :-1] + y[..., 1:]) / 2.0 * dx, axis=-1)

    xs = np.linspace(bounds[0][0], bounds[0][1], 4001)
    ys = np.linspace(bounds[1][0], bounds[1][1], 4001)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    integrand = np.sin(xx) * xx**2 + yy**4
    reference = _trapz(_trapz(integrand, ys), xs)

    assert result == pytest.approx(reference, rel=1e-4)
