# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Domain-membership testing for irregular-geometry patches.

Companion to `quadrature`: that module's
`make_patches` tiles a rectangular *bounding box* with no notion of
whether the true domain has a hole or an excluded region inside that box
(e.g. the solid interior of a car or an airfoil, in a flow-around-an-object
problem). This module answers the one question needed to extend the patch
mechanism to that case: which side of a boundary does a given point fall
on? (Which side is "the domain" is a separate, explicit choice -- see
`mask_quadrature_weights`'s `exclude` parameter -- not something
this module assumes for you.)

Dataset-agnostic by construction
-----------------------------------
This module knows nothing about any specific dataset, mesh file format, or
model. It takes a domain boundary as a plain point cloud plus its outward
unit normals -- `(boundary_points, boundary_normals)`, both
`(n_boundary, dim)` arrays -- and nothing else. Where that point cloud and
its normals come from is entirely the caller's problem: a 2D airfoil
boundary curve, a 3D car surface mesh, or (as in this module's own tests) a
synthetic circle with an analytic normal at every point. Any dataset that
can produce boundary points + normals in this generic form can use this
module; none of its logic is specific to any one of them.

The membership test
----------------------
`signed_distance` approximates a true signed distance function from a
boundary point cloud alone (no surface mesh connectivity/faces needed):
for each query point, find its nearest boundary point (`scipy.spatial.cKDTree`,
already a hard dependency elsewhere in this repo -- see
`neighbor_search`'s native fallback -- so this
avoids adding a new one), then sign the distance to that nearest point by
the dot product with the boundary's normal there: positive on the side the
normal points *towards*, negative on the side it points *away* from, zero
on the boundary. This is purely geometric -- it does not by itself say
which side is "the domain"; see `mask_quadrature_weights`. This is
the standard
nearest-neighbor approximation to a true SDF -- exact only where the
nearest boundary point is also the true closest-point projection (always
true for a sphere/circle; approximately true near a smooth boundary with
a dense-enough point cloud; can be wrong deep inside concave pockets where
the nearest sample point isn't the true closest surface point -- a known,
accepted limitation of this approximation, not attempted to be fixed here).

Runs outside jax.jit
-----------------------
Every function here (pure numpy/scipy, runs outside jax.jit, same
convention as `neighbor_search`'s `build_neighbor_graph`)
builds a `scipy.spatial.cKDTree` from its inputs, which requires a
concrete numpy array -- `np.asarray` on a traced value raises
`TracerArrayConversionError`. Call these directly on concrete arrays (the
intended use: mask a fixed sample's quadrature weights once, e.g. when
building the dataset, and reuse the result), not from inside
`jax.jit`/`jax.grad`/`jax.vmap`.

Dropping fully-excluded patches before they reach the model
-----------------------------------------------------------------
`mask_quadrature_weights` zeroes a point's weight only *after* it has
already been evaluated through the model -- a patch (see
`quadrature`) sitting entirely inside the
excluded region still costs a full forward pass before its weight is
zeroed. `cull_excluded_patches` asks the same membership question one
step earlier, per patch instead of per point, so a patch that would end up
contributing nothing is never evaluated at all: this is the one function
here that knows about `Patch`,
everything else in this module stays point-cloud-only.
"""

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .quadrature import Patch, patch_quadrature_coords


def signed_distance(
    query_points: jnp.ndarray,
    boundary_points: jnp.ndarray,
    boundary_normals: jnp.ndarray,
) -> jnp.ndarray:
    """Nearest-neighbor-approximate signed distance from each query point to
    a boundary point cloud.

    !!! note
        Runs outside `jax.jit` -- see the module docstring. Builds a
        `scipy.spatial.cKDTree` internally, which needs concrete numpy
        arrays; a traced `query_points` raises `TracerArrayConversionError`
        rather than a clear error pointing at this.

    Parameters
    ----------
    query_points : jnp.ndarray, shape (n_query, dim)
        points to test, in the same coordinate space as `boundary_points`
    boundary_points : jnp.ndarray, shape (n_boundary, dim)
        a point cloud sampled on the domain's boundary -- no connectivity/
        faces required, just points
    boundary_normals : jnp.ndarray, shape (n_boundary, dim)
        unit normal at each `boundary_points` row (unit length is assumed,
        not checked or renormalized here -- the caller owns that); which
        direction these point is a convention the caller chooses and must
        track -- see `mask_quadrature_weights`

    Returns
    -------
    jnp.ndarray, shape (n_query,)
        signed distance to the nearest boundary point: positive on the side
        the normal there points *towards*, negative on the side it points
        *away* from, (approximately) zero on the boundary
    """
    if boundary_points.shape[0] != boundary_normals.shape[0]:
        raise ValueError(
            "signed_distance requires one normal per boundary point, got "
            f"{boundary_points.shape[0]} boundary_points and "
            f"{boundary_normals.shape[0]} boundary_normals"
        )
    if query_points.shape[-1] != boundary_points.shape[-1]:
        raise ValueError(
            f"signed_distance: query_points has dim {query_points.shape[-1]}, "
            f"expected {boundary_points.shape[-1]} to match boundary_points"
        )

    query_np = np.asarray(query_points)
    boundary_np = np.asarray(boundary_points)
    normals_np = np.asarray(boundary_normals)

    tree = cKDTree(boundary_np)
    unsigned_dist, nearest_idx = tree.query(query_np, k=1)

    nearest_point = boundary_np[nearest_idx]
    nearest_normal = normals_np[nearest_idx]
    sign = np.sign(np.sum((query_np - nearest_point) * nearest_normal, axis=-1))
    # a query landing exactly on a boundary point has sign 0 from np.sign,
    # which is correct (distance 0 either way), but guard against 0 * 0
    # ambiguity being mistaken for "inside" by callers checking `<= 0`
    signed = sign * unsigned_dist

    return jnp.asarray(signed, dtype=query_points.dtype)


def is_inside(
    query_points: jnp.ndarray,
    boundary_points: jnp.ndarray,
    boundary_normals: jnp.ndarray,
) -> jnp.ndarray:
    """Purely geometric boolean test: `signed_distance(...) < 0`, i.e. on
    the side of the boundary the normals point *away* from.

    This says nothing about which side is "the domain" -- that depends on
    convention (a boundary enclosing a valid internal region, e.g. flow
    inside a pipe, vs. a boundary enclosing an excluded solid, e.g. flow
    around a car or airfoil, where the *outside* is the domain). See
    `mask_quadrature_weights`'s `exclude` parameter for where that
    choice is actually made explicit.

    Parameters and boundary/normal conventions match `signed_distance`.

    Returns
    -------
    jnp.ndarray, shape (n_query,), dtype bool
        True on the side the normals point away from
    """
    return signed_distance(query_points, boundary_points, boundary_normals) < 0.0


def mask_quadrature_weights(
    coords: jnp.ndarray,
    weights: jnp.ndarray,
    boundary_points: jnp.ndarray,
    boundary_normals: jnp.ndarray,
    exclude: str = "inside",
) -> jnp.ndarray:
    """Zero out the quadrature weight of every point on the excluded side of
    the boundary -- the pragmatic boundary-patch approximation this module
    uses instead of true cut-cell/embedded-boundary quadrature.

    Which side is excluded is not guessable from geometry alone and must be
    given explicitly via `exclude`:

    * `"inside"` (default) -- excludes the side the normals point *away*
      from. Matches the common external-flow convention (ShapeNetCar,
      AirFoil, ...): the boundary is a solid object's surface with
      outward-pointing normals, the solid's interior is excluded, and the
      surrounding fluid (outside the object) is the domain being
      integrated over.
    * `"outside"` -- excludes the side the normals point *towards*.
      Matches the opposite (internal-flow / enclosed-domain) convention:
      the boundary encloses the valid domain, and normals point out of it.

    Passing the wrong one silently keeps the excluded region and drops the
    real domain instead -- there is no way to detect this from `coords`/
    `weights`/`boundary_points`/`boundary_normals` alone, so this is a
    modeling choice the caller must make deliberately, not a default to
    trust blindly.

    Meant to post-process
    `weights`
    (or any other `(coords, weights)` pair from
    `quadrature`): a patch straddling the true
    boundary still contributes its non-excluded points' full weight, but any
    of its points landing in the excluded region contribute nothing, rather
    than overcounting area outside the true domain. This is a zeroth-order
    approximation -- the true cut-cell contribution is smaller than what any
    single non-excluded quadrature point's weight represents, so this
    systematically retains slightly more area than the true domain near the
    boundary. Refining `n_patches_per_axis` (smaller patches) shrinks this
    error, the same way it shrinks ordinary quadrature error.

    Parameters
    ----------
    coords : jnp.ndarray, shape (n_points, dim)
        quadrature coordinates to test, e.g. `PatchQuadratureADDiff.coords`
    weights : jnp.ndarray, shape (n_points,)
        the matching unmasked quadrature weights, e.g.
        `PatchQuadratureADDiff.weights`
    boundary_points, boundary_normals : jnp.ndarray
        as in `signed_distance`
    exclude : str, optional
        `"inside"` or `"outside"` -- which side of the boundary to zero
        out, by default `"inside"`. See above.

    Returns
    -------
    jnp.ndarray, shape (n_points,)
        `weights`, with every excluded-region entry set to 0.0
    """
    if exclude not in ("inside", "outside"):
        raise ValueError(f'exclude must be "inside" or "outside", got {exclude!r}')

    inside = is_inside(coords, boundary_points, boundary_normals)
    keep = ~inside if exclude == "inside" else inside
    return jnp.where(keep, weights, 0.0)


def cull_excluded_patches(
    patches: Sequence[Patch],
    ref_points: jnp.ndarray,
    ref_weights: jnp.ndarray,
    boundary_points: jnp.ndarray,
    boundary_normals: jnp.ndarray,
    exclude: str = "inside",
) -> list[Patch]:
    """Drop every patch whose quadrature points are entirely on the excluded
    side of the boundary -- the same decision `mask_quadrature_weights`
    makes per point, made once per patch instead, before that patch's
    coordinates are ever built into a model input.

    A patch is dropped only if *every one* of its own quadrature points
    would be masked to zero by `mask_quadrature_weights` anyway, so
    dropping it changes nothing about the masked integral. A patch
    straddling the true boundary keeps at least one non-excluded point, so
    it is always kept, unchanged -- this does not attempt cut-cell clipping,
    it only removes patches that would have contributed nothing. Reusing
    each patch's own quadrature points (rather than, say, its corners) for
    this test keeps the cull decision exactly consistent with what masking
    would have done to the same points, no separate geometric approximation
    is introduced.

    Whether this actually saves compute depends on where it runs: a boolean
    computed inside a `jax.jit`-traced training step cannot skip a
    forward pass, XLA still executes whatever shape the graph was traced
    with. Call this once per sample, outside `jax.jit` (see the module
    docstring's "Runs outside jax.jit" section -- the same applies here, via
    `is_inside`), and pass its output as the `patches` argument of
    `PatchQuadratureADDiff`
    -- the excluded region is static per sample and never depends on model
    parameters, so this is a decide-once, cache-the-result computation, the
    same convention already used for a sample's boundary/SDF/normals.

    Parameters
    ----------
    patches : Sequence[Patch]
        candidate patches, e.g. from
        `make_patches`
    ref_points, ref_weights : jnp.ndarray
        reference-cube quadrature rule, e.g. from
        `gauss_legendre_points_weights`
        -- the same rule that will later build each surviving patch's actual
        quadrature coordinates, so the cull decision matches what masking
        would have done to those exact points
    boundary_points, boundary_normals : jnp.ndarray
        as in `signed_distance`
    exclude : str, optional
        `"inside"` or `"outside"`, same convention and default as
        `mask_quadrature_weights` -- must match whatever value is used
        for masking, or the cull decision and the mask will disagree about
        which side is the domain

    Returns
    -------
    list[Patch]
        the subset of `patches` with at least one non-excluded quadrature
        point, in the same relative order as `patches`
    """
    if exclude not in ("inside", "outside"):
        raise ValueError(f'exclude must be "inside" or "outside", got {exclude!r}')

    kept = []
    for patch in patches:
        coords, _ = patch_quadrature_coords(patch, ref_points, ref_weights)
        inside = is_inside(coords, boundary_points, boundary_normals)
        keep_mask = ~inside if exclude == "inside" else inside
        if bool(jnp.any(keep_mask)):
            kept.append(patch)
    return kept
