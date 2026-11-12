# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the domain-membership primitives (JAX):
``signed_distance``/``is_inside``/``mask_quadrature_weights``.

Deliberately synthetic and dataset-agnostic: a circle boundary (analytic
ground truth, ``sdf(query) = |query - center| - radius``), never a real
mesh/dataset loader -- this module has no dependency on any specific
dataset, and these tests prove that by construction (nothing here could
import a dataset loader even if it wanted to).

* ``test_signed_distance_matches_analytic_circle_sdf`` /
  ``test_is_inside_matches_analytic_circle`` -- correctness against the
  closed-form circle SDF.
* ``test_signed_distance_under_jit_raises_tracer_error`` -- pins the
  documented ``jax.jit`` incompatibility (this module runs outside it,
  see its own docstring) rather than leaving it only described in prose.
* ``test_mask_quadrature_weights_exclude_*`` -- the actual mechanism this
  module exists for: zeroing quadrature weights on the excluded side of the
  boundary, checked for *both* ``exclude`` directions -- there is no
  "obviously correct" default (see ``mask_quadrature_weights``'s docstring),
  so both conventions get their own correctness test rather than only the
  default.
* ``test_masked_integral_converges_to_*`` -- the end-to-end claim, for both
  conventions: assembling patch quadrature over a bounding box, masked by
  this module, approximates the true (non-box) domain's area, and the
  approximation improves as patches shrink -- the same
  domain-splitting-additivity argument :mod:`~scirex.experimental.losses.quadrature`
  relies on, extended to an irregular domain shape via the zeroth-order
  masking approximation documented in ``mask_quadrature_weights``.
* ``test_cull_excluded_patches_*`` -- a patch entirely inside the excluded
  region is dropped (for both ``exclude`` conventions), a straddling or
  fully-valid patch always survives, and a bad ``exclude`` value is
  rejected the same way ``mask_quadrature_weights`` rejects one.
* ``test_culling_preserves_masked_integral`` -- the correctness guarantee
  ``cull_excluded_patches`` exists to preserve: dropping fully-excluded
  patches must not change the masked integral at all, only how many points
  it took to compute it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.losses.domain_membership import (
    cull_excluded_patches,
    is_inside,
    mask_quadrature_weights,
    signed_distance,
)
from scirex.experimental.losses.quadrature import (
    Patch,
    gauss_legendre_points_weights,
    make_patches,
    patch_quadrature_coords,
)

_CENTER = (0.5, 0.5)
_RADIUS = 0.3


def _circle_boundary(n_boundary: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dense point cloud on a circle, with its (analytically known) outward
    unit normals -- a synthetic stand-in for any real boundary point cloud."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    points = np.stack(
        [_CENTER[0] + _RADIUS * np.cos(theta), _CENTER[1] + _RADIUS * np.sin(theta)],
        axis=-1,
    )
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return jnp.asarray(points), jnp.asarray(normals)


def _analytic_circle_sdf(query: jnp.ndarray) -> jnp.ndarray:
    center = jnp.asarray(_CENTER)
    return jnp.linalg.norm(query - center, axis=-1) - _RADIUS


# ---------------------------------------------------------------------------
# signed_distance / is_inside
# ---------------------------------------------------------------------------


def test_signed_distance_matches_analytic_circle_sdf() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    key_query = np.random.default_rng(0)
    query = jnp.asarray(key_query.uniform(0.0, 1.0, size=(50, 2)))

    got = signed_distance(query, boundary_points, boundary_normals)
    expected = _analytic_circle_sdf(query)

    # dense (400-point) boundary sampling -> nearest-neighbor SDF approx is
    # close to exact for a smooth convex boundary like a circle
    assert jnp.allclose(got, expected, atol=5e-3)


def test_is_inside_matches_analytic_circle() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)

    clearly_inside = jnp.asarray([_CENTER, (0.5, 0.35), (0.6, 0.55)])
    clearly_outside = jnp.asarray([(0.0, 0.0), (1.0, 1.0), (0.9, 0.5)])

    inside_result = is_inside(clearly_inside, boundary_points, boundary_normals)
    outside_result = is_inside(clearly_outside, boundary_points, boundary_normals)
    assert bool(jnp.all(inside_result))
    assert not bool(jnp.any(outside_result))


def test_signed_distance_under_jit_raises_tracer_error() -> None:
    """Pins the documented failure mode (see the module docstring's 'Runs
    outside jax.jit' section): a traced query_points array can't be
    converted with np.asarray, so this must fail loudly under jit rather
    than silently doing something else. Not a fix, just confirming the
    documented limitation is real and behaves as described."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=20)

    @jax.jit
    def traced_call(query: jnp.ndarray) -> jnp.ndarray:
        return signed_distance(query, boundary_points, boundary_normals)

    with pytest.raises(jax.errors.TracerArrayConversionError):
        traced_call(jnp.zeros((5, 2)))


def test_signed_distance_rejects_mismatched_normal_count() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=20)
    with pytest.raises(ValueError, match="one normal per boundary point"):
        signed_distance(jnp.zeros((5, 2)), boundary_points, boundary_normals[:-1])


def test_signed_distance_rejects_dim_mismatch() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=20)
    with pytest.raises(ValueError, match="query_points has dim"):
        signed_distance(jnp.zeros((5, 3)), boundary_points, boundary_normals)


# ---------------------------------------------------------------------------
# mask_quadrature_weights
# ---------------------------------------------------------------------------


def test_mask_quadrature_weights_exclude_inside_zeros_the_circle_interior() -> None:
    """The default -- matches the external-flow convention (ShapeNetCar,
    AirFoil): the circle's interior is an excluded solid, the surrounding
    area is the domain."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    coords = jnp.asarray([_CENTER, (0.0, 0.0), (0.55, 0.55), (1.0, 1.0)])
    weights = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    masked = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude="inside"
    )

    assert masked[0] == 0.0  # center: inside the circle -> excluded
    assert masked[1] == 2.0  # far corner: outside the circle -> kept
    assert masked[2] == 0.0  # inside the circle -> excluded
    assert masked[3] == 4.0  # far corner: outside the circle -> kept


def test_mask_quadrature_weights_exclude_outside_zeros_all_but_the_circle() -> None:
    """The opposite convention: the circle interior IS the domain (e.g. flow
    confined inside an enclosed boundary), everything outside is excluded."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    coords = jnp.asarray([_CENTER, (0.0, 0.0), (0.55, 0.55), (1.0, 1.0)])
    weights = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    masked = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude="outside"
    )

    assert masked[0] == 1.0  # center: inside the circle -> kept
    assert masked[1] == 0.0  # far corner: outside the circle -> excluded
    assert masked[2] == 3.0  # inside the circle -> kept
    assert masked[3] == 0.0  # far corner: outside the circle -> excluded


def test_mask_quadrature_weights_rejects_bad_exclude() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=20)
    with pytest.raises(ValueError, match="exclude must be"):
        mask_quadrature_weights(
            jnp.zeros((3, 2)),
            jnp.ones((3,)),
            boundary_points,
            boundary_normals,
            exclude="bogus",
        )


# ---------------------------------------------------------------------------
# end-to-end: masked patch-quadrature integral approximates the circle's area
# ---------------------------------------------------------------------------


def _masked_area(
    n_patches_per_axis: list[int], n_quad: int, n_boundary: int, exclude: str
) -> float:
    boundary_points, boundary_normals = _circle_boundary(n_boundary)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    patches = make_patches(bounds, n_patches_per_axis)
    ref_points, ref_weights = gauss_legendre_points_weights(n_quad, dim=2)

    coords_list, weights_list = [], []
    for patch in patches:
        coords, weights = patch_quadrature_coords(patch, ref_points, ref_weights)
        coords_list.append(coords)
        weights_list.append(weights)
    coords = jnp.concatenate(coords_list, axis=0)
    weights = jnp.concatenate(weights_list, axis=0)

    masked = mask_quadrature_weights(
        coords, weights, boundary_points, boundary_normals, exclude=exclude
    )
    return float(jnp.sum(masked))


def test_masked_integral_converges_to_circle_area_with_exclude_outside() -> None:
    """exclude="outside": the circle interior IS the domain."""
    true_area = np.pi * _RADIUS**2

    coarse = _masked_area([4, 4], n_quad=3, n_boundary=400, exclude="outside")
    fine = _masked_area([16, 16], n_quad=3, n_boundary=400, exclude="outside")
    err_coarse = abs(coarse - true_area)
    err_fine = abs(fine - true_area)

    assert err_coarse < 0.05
    # refining patches shrinks the boundary-masking approximation error, the
    # same way refining a grid shrinks ordinary quadrature error
    assert err_fine < err_coarse


def test_masked_integral_converges_to_square_minus_circle_with_exclude_inside() -> None:
    """exclude="inside" (the default): the circle interior is an excluded
    solid, matching the external-flow convention (ShapeNetCar/AirFoil)."""
    true_area = 1.0 - np.pi * _RADIUS**2  # unit square minus the circle

    coarse = _masked_area([4, 4], n_quad=3, n_boundary=400, exclude="inside")
    fine = _masked_area([16, 16], n_quad=3, n_boundary=400, exclude="inside")
    err_coarse = abs(coarse - true_area)
    err_fine = abs(fine - true_area)

    assert err_coarse < 0.05
    assert err_fine < err_coarse


# ---------------------------------------------------------------------------
# cull_excluded_patches
# ---------------------------------------------------------------------------


def test_cull_excluded_patches_drops_fully_inside_excluded_patch() -> None:
    """A patch small enough to sit entirely inside the circle -- with
    exclude="inside" (the default), every one of its points would be masked
    to zero anyway, so it should be dropped."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    ref_points, ref_weights = gauss_legendre_points_weights(3, dim=2)

    fully_inside = Patch(center=_CENTER, half_width=(0.02, 0.02))
    straddling = Patch(center=(0.5 + _RADIUS, 0.5), half_width=(0.05, 0.05))
    fully_outside = Patch(center=(0.9, 0.9), half_width=(0.02, 0.02))

    kept = cull_excluded_patches(
        [fully_inside, straddling, fully_outside],
        ref_points,
        ref_weights,
        boundary_points,
        boundary_normals,
        exclude="inside",
    )

    assert fully_inside not in kept
    assert straddling in kept
    assert fully_outside in kept


def test_cull_excluded_patches_respects_exclude_outside() -> None:
    """Opposite convention: everything outside the circle is excluded, so a
    patch fully outside should be the one dropped, not the one fully inside."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    ref_points, ref_weights = gauss_legendre_points_weights(3, dim=2)

    fully_inside = Patch(center=_CENTER, half_width=(0.02, 0.02))
    fully_outside = Patch(center=(0.9, 0.9), half_width=(0.02, 0.02))

    kept = cull_excluded_patches(
        [fully_inside, fully_outside],
        ref_points,
        ref_weights,
        boundary_points,
        boundary_normals,
        exclude="outside",
    )

    assert fully_inside in kept
    assert fully_outside not in kept


def test_cull_excluded_patches_rejects_bad_exclude() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=20)
    ref_points, ref_weights = gauss_legendre_points_weights(3, dim=2)
    patch = Patch(center=(0.5, 0.5), half_width=(0.1, 0.1))

    with pytest.raises(ValueError, match="exclude must be"):
        cull_excluded_patches(
            [patch], ref_points, ref_weights, boundary_points, boundary_normals, "bogus"
        )


def test_culling_preserves_masked_integral() -> None:
    """The correctness guarantee: dropping fully-excluded patches must not
    change the masked integral at all, since their contribution was already
    zero -- only the total number of evaluated points should shrink."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=400)
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    ref_points, ref_weights = gauss_legendre_points_weights(3, dim=2)
    all_patches = make_patches(bounds, [8, 8])

    def _coords_weights(patches: list[Patch]) -> tuple[jnp.ndarray, jnp.ndarray]:
        per_patch = [
            patch_quadrature_coords(p, ref_points, ref_weights) for p in patches
        ]
        return (
            jnp.concatenate([c for c, _ in per_patch], axis=0),
            jnp.concatenate([w for _, w in per_patch], axis=0),
        )

    coords_all, weights_all = _coords_weights(all_patches)
    masked_all = mask_quadrature_weights(
        coords_all, weights_all, boundary_points, boundary_normals, exclude="inside"
    )

    culled_patches = cull_excluded_patches(
        all_patches,
        ref_points,
        ref_weights,
        boundary_points,
        boundary_normals,
        "inside",
    )
    assert len(culled_patches) < len(all_patches)

    coords_culled, weights_culled = _coords_weights(culled_patches)
    masked_culled = mask_quadrature_weights(
        coords_culled,
        weights_culled,
        boundary_points,
        boundary_normals,
        exclude="inside",
    )

    assert coords_culled.shape[0] < coords_all.shape[0]
    assert jnp.allclose(jnp.sum(masked_all), jnp.sum(masked_culled))
