# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Gauss-Legendre quadrature and rectangular-patch geometry.

Pure numerics -- no JAX autodiff, no model dependency. This is the geometry
layer for the patch-based AD physics-loss mechanism (see
`exact_ad_differentiation` and the forthcoming
`patch_quadrature_ad` module for the AD side this composes with):
given a rectangular domain and how many patches to cut each axis into, this
module produces

* the patches themselves -- non-overlapping, axis-aligned cells that
  exactly partition the domain (`Patch`, `make_patches`), and
* for each patch, the physical-space coordinates and quadrature weights of
  a tensor-product Gauss-Legendre rule mapped from the reference cube
  `[-1, 1]^dim` onto that patch (`gauss_legendre_points_weights`,
  `patch_quadrature_coords`).

Why this makes the split exact
--------------------------------
Splitting a domain integral into per-patch integrals is exact on its own --
`integral(Omega) = sum_e integral(Omega_e)` for any partition, no
approximation from the split itself. The only approximation is *within*
each patch, from using a finite quadrature rule instead of the true
integral -- and an `n`-point-per-axis Gauss-Legendre rule is already
exact for any polynomial integrand up to degree `2n - 1` per axis, so for
smooth (near-polynomial) integrands, a handful of points per patch already
gives an accurate patch contribution. Assembling many patches this way
therefore loses nothing beyond each patch's own quadrature error -- adding
more, smaller patches does not introduce new error, it only ever reduces
each patch's share of the integrand's variation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Patch:
    """One axis-aligned rectangular cell of a domain partition.

    Parameters
    ----------
    center : tuple[float, ...]
        physical-space center of the patch, one value per dimension
    half_width : tuple[float, ...]
        half the patch's side length along each dimension -- the `h/2`
        in the reference-element map `x = center + half_width * ref_point`
    """

    center: tuple[float, ...]
    half_width: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.center) != len(self.half_width):
            raise ValueError(
                "Patch requires center and half_width to have the same "
                f"length, got {len(self.center)} and {len(self.half_width)}"
            )
        if len(self.center) == 0:
            raise ValueError("Patch requires at least one dimension")
        if any(h <= 0 for h in self.half_width):
            raise ValueError(
                f"Patch requires every half_width > 0, got {self.half_width}"
            )

    @property
    def dim(self) -> int:
        """int: number of spatial dimensions."""
        return len(self.center)

    @property
    def jacobian(self) -> float:
        """float: `prod(half_width)` -- the constant Jacobian of the affine
        reference-element map from `[-1, 1]^dim` (measure `2**dim`) onto
        this patch (measure `prod(2 * half_width) = 2**dim * jacobian`)."""
        j = 1.0
        for h in self.half_width:
            j *= h
        return j


def gauss_legendre_points_weights(n: int, dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tensor-product Gauss-Legendre rule on the reference cube `[-1, 1]^dim`.

    Parameters
    ----------
    n : int
        quadrature points per axis, >= 1. Exact for polynomials up to
        degree `2n - 1` per axis (e.g. `n=3` is exact through degree 5).
    dim : int
        number of dimensions, >= 1

    Returns
    -------
    points : jnp.ndarray, shape (n**dim, dim)
        reference-cube coordinates of every quadrature point
    weights : jnp.ndarray, shape (n**dim,)
        the product weight for each point; sums to `2**dim` (the measure
        of `[-1, 1]^dim`) -- not yet scaled by any patch's Jacobian, see
        `patch_quadrature_coords` for that
    """
    if n < 1:
        raise ValueError(f"gauss_legendre_points_weights requires n >= 1, got {n}")
    if dim < 1:
        raise ValueError(f"gauss_legendre_points_weights requires dim >= 1, got {dim}")

    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(n)

    point_grids = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
    points = np.stack([g.reshape(-1) for g in point_grids], axis=-1)

    weight_grids = np.meshgrid(*([weights_1d] * dim), indexing="ij")
    weights = np.ones(n**dim)
    for wg in weight_grids:
        weights = weights * wg.reshape(-1)

    return (
        jnp.asarray(points, dtype=jnp.float32),
        jnp.asarray(weights, dtype=jnp.float32),
    )


def make_patches(
    bounds: Sequence[tuple[float, float]], n_patches_per_axis: Sequence[int]
) -> list[Patch]:
    """Partition an axis-aligned rectangular domain into equal-size patches.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        `(lo, hi)` per dimension, e.g. `[(0.0, 1.0), (0.0, 1.0)]` for
        the unit square
    n_patches_per_axis : Sequence[int]
        how many equal-width patches to cut each axis into, >= 1 each --
        one entry per `bounds` dimension

    Returns
    -------
    list[Patch]
        one `Patch` per cell of the resulting grid, in the flattened
        order of `np.meshgrid(..., indexing="ij")` over the per-axis
        patch indices. Every patch has the same `half_width` (equal-size
        partition); together they exactly tile `bounds` with no gaps or
        overlap.
    """
    if len(bounds) != len(n_patches_per_axis):
        raise ValueError(
            "make_patches requires one n_patches_per_axis entry per bounds "
            f"dimension, got {len(bounds)} bounds and "
            f"{len(n_patches_per_axis)} n_patches_per_axis"
        )
    if len(bounds) == 0:
        raise ValueError("make_patches requires at least one dimension")
    for d, ((lo, hi), n) in enumerate(zip(bounds, n_patches_per_axis, strict=True)):
        if hi <= lo:
            raise ValueError(
                f"make_patches requires hi > lo for axis {d}, got ({lo}, {hi})"
            )
        if n < 1:
            raise ValueError(
                f"make_patches requires n_patches_per_axis >= 1 for axis {d}, got {n}"
            )

    per_axis_centers = []
    per_axis_half_width = []
    for (lo, hi), n in zip(bounds, n_patches_per_axis, strict=True):
        width = (hi - lo) / n
        half_width = width / 2.0
        per_axis_centers.append(lo + half_width + width * np.arange(n))
        per_axis_half_width.append(half_width)

    center_grids = np.meshgrid(*per_axis_centers, indexing="ij")
    flat_centers = np.stack([g.reshape(-1) for g in center_grids], axis=-1)

    return [
        Patch(center=tuple(c.tolist()), half_width=tuple(per_axis_half_width))
        for c in flat_centers
    ]


def patch_quadrature_coords(
    patch: Patch, ref_points: jnp.ndarray, ref_weights: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map a reference-cube quadrature rule onto one physical patch.

    Reference-element map: `x = center + half_width * ref_point` (per
    dimension); the Jacobian is the constant `Patch.jacobian`, since
    the map is affine.

    Parameters
    ----------
    patch : Patch
        the physical patch to map onto
    ref_points : jnp.ndarray, shape (n_quad, dim)
        reference-cube quadrature points, e.g. from
        `gauss_legendre_points_weights`
    ref_weights : jnp.ndarray, shape (n_quad,)
        reference-cube quadrature weights (unscaled), matching `ref_points`

    Returns
    -------
    coords : jnp.ndarray, shape (n_quad, dim)
        physical-space quadrature coordinates inside `patch`
    weights : jnp.ndarray, shape (n_quad,)
        `ref_weights * patch.jacobian` -- the weight to use directly when
        summing a per-point integrand value into this patch's contribution
        to the domain integral
    """
    if ref_points.ndim != 2 or ref_points.shape[-1] != patch.dim:
        raise ValueError(
            f"patch_quadrature_coords: ref_points has shape {ref_points.shape}, "
            f"expected (n_quad, {patch.dim}) to match patch.dim={patch.dim}"
        )
    if ref_weights.shape != (ref_points.shape[0],):
        raise ValueError(
            f"patch_quadrature_coords: ref_weights has shape {ref_weights.shape}, "
            f"expected ({ref_points.shape[0]},) to match ref_points"
        )

    center = jnp.asarray(patch.center, dtype=ref_points.dtype)
    half_width = jnp.asarray(patch.half_width, dtype=ref_points.dtype)
    coords = center + half_width * ref_points
    weights = ref_weights * patch.jacobian
    return coords, weights
