# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""T3 (triangle) mesh generation via the `gmsh` Python API.

This module -- unlike everything else under `src/scirex/` -- is meant to
run on a developer's own machine, not on the (typically headless, minimal)
GPU training server: `gmsh` is a full mesh-generation engine, a much
heavier dependency than the `meshio` file-format reader
`scirex.experimental.geometry.mesh_io` uses to load its output. The intended
workflow is:

1. On a machine with `gmsh` installed, generate a mesh once with
   `scripts/mesh/generate_gmsh_mesh.py` (or call
   `generate_domain_mesh` directly), producing a `.npz` file via
   `save_mesh`.
2. Sync that `.npz` file to the training server (`scp`/`rsync`).
3. Point a Galerkin PINO benchmark script's `--mesh-path` at it --
   `resolve_mesh` then loads it with
   plain `numpy`, no `gmsh`/`meshio` needed on the server at all.

Three domains are supported so far (`square`, `circular`/disk,
`lshape`) -- adding a fourth is one more `generate_*` function plus a
branch in `generate_domain_mesh`'s dispatch, not a redesign.

Element order
-------------
`order=1` (the default) gives a T3 (linear, 3-node) triangle mesh, gmsh
element type 2. `order=2` elevates it to a T6 (quadratic, 6-node)
triangle mesh via `gmsh.model.mesh.setOrder(2)` -- element type 9, node
ordering `[corner1, corner2, corner3, mid(1,2), mid(2,3), mid(3,1)]`
(gmsh's own convention, matched exactly by
`T6Galerkin`). On a curved
boundary (`circular`), order-2 elevation is curvilinear -- the new
mid-edge boundary nodes sit on the true circle, not at the straight
chord's midpoint -- which is what makes T6 meshes a strictly better
approximation of curved geometry, not just a higher-order field on the
same polygon.
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

_RAISE_NO_GMSH = (
    "gmsh is not installed in this environment. Mesh generation is meant to "
    "run on a developer's machine (`pip install gmsh`), not the training "
    "server -- generate the mesh here, then sync the resulting .npz to the "
    "server and pass it via --mesh-path (see this module's docstring)."
)


@contextlib.contextmanager
def _gmsh_session() -> Iterator[Any]:
    """Import, initialize, and (on exit) finalize `gmsh` -- the
    import-guard/init/finalize ceremony every domain generator below needs
    identically, factored out so a 4th domain (an easy addition per this
    module's own docstring) doesn't have to repeat it a 4th time."""
    try:
        import gmsh
    except (ImportError, OSError) as e:
        # OSError: gmsh is installed but a system library (libGLU) is missing
        raise ImportError(f"{_RAISE_NO_GMSH} ({type(e).__name__}: {e})") from e

    gmsh.initialize()
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def _apply_mesh_size_and_generate(gmsh_module: Any, resolution: float) -> None:
    """Shared tail of every domain generator once its geometry is built:
    set the target element size and mesh it."""
    gmsh_module.option.setNumber("Mesh.MeshSizeMax", resolution)
    # gmsh's mesher is free to use any size up to MeshSizeMax; without a
    # floor it can locally shrink elements much further (e.g. near a sharp
    # corner) than --resolution suggests. Setting MeshSizeMin to half of
    # --resolution caps how far below the requested size it may go, so
    # --resolution behaves as an actual target, not just an upper bound.
    gmsh_module.option.setNumber("Mesh.MeshSizeMin", resolution * 0.5)
    gmsh_module.model.mesh.generate(2)


def _extract_mesh(
    gmsh_module: Any, order: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull node coordinates, triangle connectivity, and a boundary mask
    out of gmsh's current model state, after `gmsh.model.mesh.generate(2)`
    (and `setOrder(2)`, for `order=2`).

    `order=1` extracts gmsh element type 2 (3-node T3, `tri` shape
    `(T, 3)`); `order=2` extracts type 9 (6-node T6, `tri` shape
    `(T, 6)`), calling `setOrder(2)` first.
    """
    gmsh = gmsh_module
    if order == 2:
        gmsh.model.mesh.setOrder(2)
    elif order != 1:
        raise ValueError(f"order must be 1 (T3) or 2 (T6), got {order!r}.")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # gmsh node tags are 1-indexed and not necessarily contiguous/sorted.
    # Building xy below by enumerate(node_tags) doesn't actually need this
    # map (i == tag_to_row[tag] trivially there, same enumeration order on
    # both sides) -- it's the triangle connectivity remap just below, whose
    # tags come from a separate gmsh call (getElements, not getNodes) in
    # its own order, that genuinely requires an explicit tag -> row lookup
    # rather than assuming the two orders coincide.
    tag_to_row = {tag: i for i, tag in enumerate(node_tags)}
    xy = np.zeros((len(node_tags), 2))
    for i, tag in enumerate(node_tags):
        xy[tag_to_row[tag]] = node_coords[3 * i : 3 * i + 2]

    elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    gmsh_type = 2 if order == 1 else 9  # 2 = 3-node triangle, 9 = 6-node triangle
    n_nodes_per_elem = 3 if order == 1 else 6
    tri_idx = list(elem_types).index(gmsh_type)
    tri_nodes = np.array(elem_node_tags[tri_idx], dtype=np.int64).reshape(
        -1, n_nodes_per_elem
    )
    # Here is where tag_to_row actually earns its keep: tri_nodes' tags are
    # not guaranteed to appear in the same order as node_tags above.
    tri = np.array([[tag_to_row[t] for t in row] for row in tri_nodes], dtype=np.int32)

    boundary_tags: set[int] = set()
    for dim_tag in gmsh.model.getBoundary(
        gmsh.model.getEntities(dim=2), oriented=False
    ):
        b_node_tags, _, _ = gmsh.model.mesh.getNodes(
            dim_tag[0], dim_tag[1], includeBoundary=True
        )
        boundary_tags.update(int(t) for t in b_node_tags)
    boundary_mask = np.zeros(len(node_tags), dtype=bool)
    for tag in boundary_tags:
        if tag in tag_to_row:
            boundary_mask[tag_to_row[tag]] = True

    return xy, tri, boundary_mask


def _mesh_polygon(
    gmsh_module: Any, name: str, points: list[tuple[float, float]], resolution: float
) -> None:
    gmsh = gmsh_module
    gmsh.model.add(name)
    point_tags = [gmsh.model.occ.addPoint(x, y, 0, resolution) for x, y in points]
    line_tags = [
        gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
        for i in range(len(point_tags))
    ]
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    _apply_mesh_size_and_generate(gmsh, resolution)


def generate_square(
    resolution: float = 0.05, order: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit square `[0, 1]^2` -- same domain convention used by e.g.
    `run_poisson_transolver_pino_variable_coeff_kx.py`."""
    with _gmsh_session() as gmsh:
        _mesh_polygon(
            gmsh, "square", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], resolution
        )
        return _extract_mesh(gmsh, order=order)


def generate_circular(
    resolution: float = 0.05,
    radius: float = 1.0,
    cx: float = 0.0,
    cy: float = 0.0,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disk of the given radius/center -- matches the `RADIUS = 1.0`,
    origin-centered disk used by e.g.
    `run_poisson_transolver_pino_variable_coeff_kx_circular.py`."""
    with _gmsh_session() as gmsh:
        gmsh.model.add("circular")
        gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)
        gmsh.model.occ.synchronize()
        _apply_mesh_size_and_generate(gmsh, resolution)
        return _extract_mesh(gmsh, order=order)


def generate_lshape(
    resolution: float = 0.05, order: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """L-shape: `[0,2]x[0,2]` minus the `[1,2]x[1,2]` notch -- same
    domain convention as every L-shape PINO script in this repo (e.g.
    `run_poisson_transolver_pino_variable_coeff_kx_lshape.py`'s own
    `_EDGES`/`_BOUNDS`, and the magnetostatic cylinder benchmark's own
    L-shape variant). Deliberately `[0,2]`-scaled, not `[0,1]`: those
    scripts' shared boundary-vanishing ansatz is `sin(pi*x)*sin(pi*y)`,
    which is zero not only at `x,y = 0` but also at `x,y = 1` and
    `2` -- so at this domain's scale, the same simple product ansatz
    vanishes on the notch's reentrant edges (`x=1`, `y=1`) too, with no
    extra term needed. At `[0,1]` scale it would not."""
    with _gmsh_session() as gmsh:
        pts = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ]
        _mesh_polygon(gmsh, "lshape", pts, resolution)
        return _extract_mesh(gmsh, order=order)


_DOMAIN_GENERATORS: dict[
    str, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
] = {
    "square": generate_square,
    "circular": generate_circular,
    "lshape": generate_lshape,
}


def generate_domain_mesh(
    domain: str, resolution: float = 0.05, order: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to one of the named domain generators above.

    Parameters
    ----------
    domain : {"square", "circular", "lshape"}
    resolution : target element size (gmsh's own units, same scale as the
        domain's own coordinates)
    order : 1 for T3 (linear, 3-node), 2 for T6 (quadratic, 6-node) --
        see the module docstring's "Element order" section.

    Returns
    -------
    xy : np.ndarray, shape (N, 2)
    tri : np.ndarray, shape (T, 3) for order=1 or (T, 6) for order=2,
        int32, 0-indexed
    boundary_mask : np.ndarray, shape (N,), bool
    """
    if domain not in _DOMAIN_GENERATORS:
        raise ValueError(
            f"Unknown domain {domain!r} -- supported: {sorted(_DOMAIN_GENERATORS)}."
        )
    return _DOMAIN_GENERATORS[domain](resolution, order=order)
