# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Format-agnostic loading/saving of T3/T6 triangle meshes for the Galerkin
weak-form losses (`scirex.losses.galerkin`,
`scirex.experimental.losses.galerkin_t6`).

Two formats are supported, chosen by dispatching on the file's own
extension so callers never need to say which one they have:

* `.npz` -- this repo's own convention (see
  `scirex.experimental.geometry.gmsh_mesh`'s `save_mesh`): three plain arrays,
  `xy` (N,2), `tri` (T,3) for T3 or (T,6) for T6, `boundary_mask` (N,)
  bool. No mesh library needed to read it -- just `numpy` -- which is what
  makes the generate-locally-with-gmsh, train-on-a-server-without-gmsh
  workflow work: only the machine that *creates* the mesh needs
  `gmsh`/`meshio` installed; the machine that *trains* on it only ever
  calls `np.load`.
* `.msh` -- gmsh's own native mesh format, read via `meshio` (a much
  lighter dependency than `gmsh` itself -- no mesh-generation engine, just
  a file-format reader). Since a raw `.msh` file carries no ready-made
  boundary-node flag equivalent to this repo's own `boundary_mask`, one is
  derived geometrically (`boundary_mask_from_triangles`), which also
  marks a T6 mesh's mid-edge nodes as boundary when their own edge is.

Adding a third format (e.g. `.vtk`/`.xdmf`) later is a small addition,
not a redesign: both are already read by the same `meshio.read` call
`.msh` uses here, so it only needs a new branch in
`load_mesh`'s extension dispatch, reusing the same triangle-cell
extraction and boundary-detection logic.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

SUPPORTED_MESH_EXTENSIONS = (".npz", ".msh")


def boundary_mask_from_triangles(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    """A corner node is on the mesh boundary iff it is an endpoint of an
    edge that belongs to exactly one triangle (an interior edge is always
    shared by exactly two); a T6 mid-edge node inherits its own edge's
    boundary status. Purely topological -- correct for any T3/T6
    triangulation, convex or not, and needs no gmsh-side physical-group
    tagging.

    `tri` may be `(T, 3)` (T3, corners only) or `(T, 6)` (T6, gmsh
    node order: 3 corners then mid(1,2), mid(2,3), mid(3,1))."""
    has_mid_edge_nodes = tri.shape[-1] == 6
    local_edges = (
        ((0, 1, 3), (1, 2, 4), (2, 0, 5))
        if has_mid_edge_nodes
        else ((0, 1, None), (1, 2, None), (2, 0, None))
    )

    edge_count: dict[tuple[int, int], int] = {}
    edge_mid_node: dict[tuple[int, int], int] = {}
    for row in tri:
        for local_i, local_j, local_mid in local_edges:
            a, b = int(row[local_i]), int(row[local_j])
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            if local_mid is not None:
                edge_mid_node[key] = int(row[local_mid])

    boundary_mask = np.zeros(n_nodes, dtype=bool)
    for key, count in edge_count.items():
        if count == 1:
            i, j = key
            boundary_mask[i] = True
            boundary_mask[j] = True
            if key in edge_mid_node:
                boundary_mask[edge_mid_node[key]] = True
    return boundary_mask


def save_mesh(
    path: str | Path, xy: np.ndarray, tri: np.ndarray, boundary_mask: np.ndarray
) -> Path:
    """Write this repo's own `.npz` mesh convention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xy=np.asarray(xy, dtype=np.float64),
        tri=np.asarray(tri, dtype=np.int32),
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
    )
    return path


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    for key in ("xy", "tri", "boundary_mask"):
        if key not in data:
            raise ValueError(
                f"{path} is missing required array {key!r} -- expected this "
                "repo's own mesh .npz convention (xy, tri, boundary_mask), "
                "see src/scirex/experimental/geometry/gmsh_mesh.py's save_mesh."
            )
    return data["xy"], data["tri"], data["boundary_mask"]


def _load_msh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import meshio
    except ImportError as e:
        raise ImportError(
            "Reading a .msh mesh requires the 'meshio' package (a pure "
            "file-format reader, not the gmsh mesh-generation engine -- "
            "safe to install on a training server). Install it with "
            "`pip install meshio`, or pre-convert the mesh to this repo's "
            "own .npz convention (scirex.experimental.geometry.mesh_io.save_mesh) on "
            "a machine that already has meshio/gmsh."
        ) from e

    mesh = meshio.read(path)
    if "triangle6" in mesh.cells_dict:
        tri = np.asarray(mesh.cells_dict["triangle6"], dtype=np.int32)
    elif "triangle" in mesh.cells_dict:
        tri = np.asarray(mesh.cells_dict["triangle"], dtype=np.int32)
    else:
        raise ValueError(
            f"{path} has no 3-node or 6-node triangle cells -- only T3/T6 "
            "triangle meshes are supported by the Galerkin losses in this "
            "repo so far."
        )
    xy = np.asarray(mesh.points[:, :2], dtype=np.float64)
    boundary_mask = boundary_mask_from_triangles(tri, xy.shape[0])
    return xy, tri, boundary_mask


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a T3 mesh, dispatching on `path`'s extension.

    Returns
    -------
    xy : np.ndarray, shape (N, 2)
    tri : np.ndarray, shape (T, 3), int, 0-indexed
    boundary_mask : np.ndarray, shape (N,), bool
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".msh":
        return _load_msh(path)
    raise ValueError(
        f"Unsupported mesh file extension {suffix!r} for {path} -- "
        f"supported: {SUPPORTED_MESH_EXTENSIONS}."
    )


def _generate_domain_mesh_fn() -> (
    Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
):
    """Returns `gmsh_mesh.generate_domain_mesh`, via a normal relative
    import where that works (this module loaded as part of the
    `scirex` package) or by loading `gmsh_mesh.py` as a standalone
    file otherwise -- this module is itself sometimes loaded that same
    standalone way (see `scripts/mesh/generate_gmsh_mesh.py`'s own
    docstring, which loads *this* file by path to dodge
    `src/scirex/__init__.py`'s eager JAX/Flax import), and a relative
    import raises `ImportError` in that context since there is no
    enclosing package."""
    try:
        from .gmsh_mesh import generate_domain_mesh

        return generate_domain_mesh
    except ImportError as import_error:
        import importlib.util

        gmsh_mesh_path = Path(__file__).with_name("gmsh_mesh.py")
        spec = importlib.util.spec_from_file_location("gmsh_mesh", gmsh_mesh_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"couldn't find src/scirex/experimental/geometry/gmsh_mesh.py "
                f"(looked at {gmsh_mesh_path})"
            ) from import_error
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.generate_domain_mesh


def resolve_mesh(
    mesh_path: str | Path | None,
    *,
    domain: str | None = None,
    resolution: float = 0.05,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The "did the user give us a mesh" entry point every Galerkin
    benchmark script should call: load `mesh_path` if given, otherwise
    fall back to generating one on the spot via `gmsh` (only possible on
    a machine that has it installed -- see
    `scirex.experimental.geometry.gmsh_mesh`).

    Parameters
    ----------
    mesh_path : path or None
        A pre-generated mesh file (`.npz` or `.msh`). This is the path
        expected to be used on a training server: generate the mesh on a
        machine with `gmsh` (`scripts/mesh/generate_gmsh_mesh.py`), sync
        the resulting `.npz` over, and pass its path here.
    domain, resolution, order : used only when `mesh_path` is None, to
        generate a mesh in-process via `scirex.experimental.geometry.gmsh_mesh`.
        `order=1` for T3 (linear), `order=2` for T6 (quadratic) -- see
        that module's "Element order" docstring section.
    """
    if mesh_path is not None:
        return load_mesh(mesh_path)
    if domain is None:
        raise ValueError(
            "resolve_mesh needs either mesh_path (a pre-generated .npz/.msh "
            "file) or domain (to generate one via gmsh) -- got neither."
        )
    generate_domain_mesh = _generate_domain_mesh_fn()
    return generate_domain_mesh(domain, resolution, order=order)
