# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Boundary point cloud + normals for one ShapeNetCar sample, in the
generic `(boundary_points, boundary_normals)` form
`mask_quadrature_weights`
expects.

This is the one dataset-specific adapter `domain_membership`
was deliberately built without: everything in that module works from a
plain boundary point cloud + normals and has no idea ShapeNetCar exists.
This file is the small piece of glue that produces those two arrays from
one ShapeNetCar sample -- nothing here is reusable by any other dataset,
and none of `domain_membership`/`quadrature`/
`patch_quadrature_ad` needed to change to support it.

Why this re-reads the raw VTK file instead of slicing a loaded sample
------------------------------------------------------------------------
`load_shapenet_car`
already returns a per-sample `pos`/`surf` pair that looks like it should
be enough (`sample["pos"][sample["surf"]]` does give the boundary point
cloud correctly, in the right order -- see the point-order note in
`shapenet_car_drag`, which relies on the
exact same `pos = concatenate([pos_ext, pos_surf])` fact). The normals
are the problem: they live in `sample["x"]`'s columns `4:7`, and
`load_shapenet_car` normalizes `x` to zero mean / unit std for training
(see its own docstring's `coef_norm`) -- a normalized "normal" is no
longer unit length, which
`signed_distance` requires.
Recomputing normals from the raw mesh instead of trying to denormalize
them is simpler and avoids depending on `coef_norm` at all.
This mirrors the exact same pattern
`shapenet_car_drag` already uses for the
same underlying reason (needing raw surface geometry a normalized training
sample doesn't preserve): re-read the raw `quadpress_smpl.vtk` via the
sample's own `sample_rel_path`.

Outward orientation depends on the mesh being watertight
------------------------------------------------------------
`_get_normal` relies
on VTK's `vtkPolyDataNormals(SetAutoOrientNormals=1)`, which VTK's own
documentation says only guarantees a globally correct sign for a *closed*
(watertight) surface -- on an open surface it can still return a
consistent-looking answer with no error, just not reliably the right one.
"Inside"/"outside" are only well-defined concepts for a closed boundary in
the first place, so this isn't a VTK quirk to work around, it's a real
precondition. `shapenet_car_boundary` checks for this explicitly (see
`_is_watertight`) and raises rather than silently handing back a
possibly-mis-oriented normal to
`mask_quadrature_weights`,
which would otherwise mask the wrong side with no error at all.

Requires `vtk` (already a hard dependency of
`shapenet_car`).

Culling, not (yet) persistent caching
------------------------------------------
`shapenet_car_culled_patches` wires this module's boundary together
with `cull_excluded_patches`
for one sample, dropping patches entirely inside the car's solid interior
before they ever cost a forward pass. It recomputes the culled list on
every call, same as `shapenet_car_boundary` itself recomputes the
boundary on every call -- there is no on-disk cache for either yet, unlike
`load_shapenet_car`'s
`.npz` cache for the main training sample. Both are one-time-per-sample
costs today (called once when building a
`PatchQuadratureADDiff`,
not from inside a training step), so this is a real cost, not a per-step
one, but it is not yet cached to disk the way the rest of a sample is --
adding that is a separate follow-up, not done here.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from scirex.experimental.losses.domain_membership import cull_excluded_patches
from scirex.experimental.losses.quadrature import (
    Patch,
    gauss_legendre_points_weights,
    make_patches,
)

from .shapenet_car import (
    _get_normal,
    _load_unstructured_grid_data,
    _unstructured_grid_data_to_poly_data,
)


def _is_watertight(unstructured_grid_data: Any) -> bool:
    """True if the surface has no boundary or non-manifold edges -- the
    precondition VTK's `SetAutoOrientNormals` actually needs for a
    reliable global sign (see the module docstring). A mesh that fails this
    still produces *some* normals, just not dependably oriented ones.

    A boundary edge (belongs to only one polygon) means the surface has an
    opening, not a closed shell. A non-manifold edge (belongs to more than
    two polygons) means the surface isn't a proper 2-manifold at all. Either
    one breaks the watertight precondition.
    """
    import vtk

    poly_data, _ = _unstructured_grid_data_to_poly_data(unstructured_grid_data)
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(poly_data)
    feature_edges.BoundaryEdgesOn()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()
    return bool(feature_edges.GetOutput().GetNumberOfCells() == 0)


def shapenet_car_boundary(
    data_dir: str | Path, sample: dict[str, Any]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Boundary point cloud + outward unit normals for one ShapeNetCar sample.

    Parameters
    ----------
    data_dir : path
        the same `data_dir` root passed to
        `load_shapenet_car`
    sample : dict
        one sample dict from `load_shapenet_car`'s `"train"`/`"val"`
        list -- only its `"sample_rel_path"` key is used

    Returns
    -------
    boundary_points : jnp.ndarray, shape (n_surface_points, 3)
    boundary_normals : jnp.ndarray, shape (n_surface_points, 3)
        outward-pointing unit normals (`vtkPolyDataNormals` with
        auto-orientation, as computed by
        `_get_normal`)

    Both are ready to pass directly as
    `mask_quadrature_weights`'s
    `boundary_points`/`boundary_normals`, with `exclude="inside"` (the
    car's solid interior is excluded, the surrounding fluid is the domain --
    `mask_quadrature_weights`'s default).

    Raises
    ------
    KeyError
        if `sample` is missing `"sample_rel_path"`
    FileNotFoundError
        if the sample's `quadpress_smpl.vtk` doesn't exist at the expected
        path -- a stale cache, a moved `data_dir`, or a partially cleaned
        dataset
    ValueError
        if the file exists but reads back empty (0 points -- a truncated or
        corrupted file, see `_load_unstructured_grid_data`'s known
        failure mode), or if the mesh isn't watertight -- see the module
        docstring
    """
    from vtk.util.numpy_support import vtk_to_numpy

    if "sample_rel_path" not in sample:
        raise KeyError(
            "shapenet_car_boundary requires sample['sample_rel_path'], which "
            "load_shapenet_car attaches to every sample dict it returns -- "
            "got a dict with keys "
            f"{sorted(sample.keys())}"
        )

    vtk_path = Path(data_dir) / sample["sample_rel_path"] / "quadpress_smpl.vtk"
    if not vtk_path.exists():
        raise FileNotFoundError(
            f"shapenet_car_boundary could not find {vtk_path} -- check that "
            "data_dir points at the right root and that this sample's cache "
            "hasn't been moved or partially cleaned."
        )

    ug_press = _load_unstructured_grid_data(vtk_path)
    if ug_press.GetNumberOfPoints() == 0:
        raise ValueError(
            f"shapenet_car_boundary read {vtk_path} but got an empty mesh "
            "(0 points) -- vtkUnstructuredGridReader returns an empty grid "
            "instead of raising when a file can't actually be parsed, so "
            "this usually means the file exists but is truncated or "
            "corrupted, not that it's missing (see FileNotFoundError above "
            "for that case)."
        )

    if not _is_watertight(ug_press):
        raise ValueError(
            f"shapenet_car_boundary: the mesh at {vtk_path} is not "
            "watertight (it has boundary or non-manifold edges), so "
            "vtkPolyDataNormals's outward-orientation guarantee does not "
            "hold for it -- see the module docstring's 'Outward orientation "
            "depends on the mesh being watertight' section. Using this "
            "mesh's normals with mask_quadrature_weights could silently "
            "mask the wrong side."
        )

    boundary_points = vtk_to_numpy(ug_press.GetPoints().GetData())
    boundary_normals = _get_normal(ug_press)

    return (
        jnp.asarray(boundary_points, dtype=jnp.float32),
        jnp.asarray(boundary_normals, dtype=jnp.float32),
    )


def shapenet_car_culled_patches(
    data_dir: str | Path,
    sample: dict[str, Any],
    bounds: Sequence[tuple[float, float]],
    n_patches_per_axis: Sequence[int],
    n: int,
) -> list[Patch]:
    """Patches from a uniform bounding-box grid that survive culling against
    one ShapeNetCar sample's real boundary.

    Ties `shapenet_car_boundary` together with
    `make_patches` and
    `cull_excluded_patches`
    for one sample: a patch entirely inside the car's solid interior is
    dropped here, before it ever reaches a
    `PatchQuadratureADDiff`
    (pass this function's return value as that class's `patches`
    argument), so it never costs a forward pass. Uses `exclude="inside"`,
    the same external-flow convention `shapenet_car_boundary`'s own
    docstring already documents for this dataset (the car's interior is
    excluded, the surrounding fluid is the domain).

    Parameters
    ----------
    data_dir, sample : as in `shapenet_car_boundary`
    bounds, n_patches_per_axis, n : as in
        `PatchQuadratureADDiff`
        -- must match whatever is later used to construct that class, or the
        culled patches won't correspond to its uniform grid

    Returns
    -------
    list[Patch]
        the subset of the full uniform grid with at least one point outside
        the car's solid interior, in
        `make_patches`'s flattened
        grid order
    """
    boundary_points, boundary_normals = shapenet_car_boundary(data_dir, sample)
    dim = len(bounds)
    all_patches = make_patches(bounds, n_patches_per_axis)
    ref_points, ref_weights = gauss_legendre_points_weights(n, dim)
    return cull_excluded_patches(
        all_patches,
        ref_points,
        ref_weights,
        boundary_points,
        boundary_normals,
        exclude="inside",
    )
