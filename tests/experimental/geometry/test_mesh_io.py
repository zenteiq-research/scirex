# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from scirex.experimental.geometry.mesh_io import (
    boundary_mask_from_triangles,
    load_mesh,
    resolve_mesh,
    save_mesh,
)


def _two_triangle_square() -> tuple[np.ndarray, np.ndarray]:
    """Unit square cut into two triangles along one diagonal -- every node
    is on the boundary, and the two triangles share exactly the diagonal
    edge (the one interior edge)."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    return xy, tri


def test_boundary_mask_all_nodes_on_hull_of_two_triangle_square() -> None:
    xy, tri = _two_triangle_square()
    mask = boundary_mask_from_triangles(tri, xy.shape[0])
    assert mask.dtype == bool
    assert mask.all()


def test_boundary_mask_flags_only_hull_on_a_finer_grid() -> None:
    """A 3x3 regular grid triangulated into a 2x2 array of quads (each cut
    into two triangles): the single center node is interior, the 8 hull
    nodes are boundary."""
    xs, ys = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=-1)  # row-major, 9 nodes

    def idx(i: int, j: int) -> int:
        return i * 3 + j

    tri = []
    for i in range(2):
        for j in range(2):
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            tri.append([a, b, c])
            tri.append([a, c, d])
    tri = np.array(tri)

    mask = boundary_mask_from_triangles(tri, xy.shape[0])
    center = idx(1, 1)
    assert not mask[center]
    assert mask.sum() == 8


def test_npz_round_trip(tmp_path: Path) -> None:
    xy, tri = _two_triangle_square()
    boundary_mask = boundary_mask_from_triangles(tri, xy.shape[0])
    path = save_mesh(tmp_path / "mesh.npz", xy, tri, boundary_mask)

    xy2, tri2, mask2 = load_mesh(path)
    np.testing.assert_array_equal(xy2, xy)
    np.testing.assert_array_equal(tri2, tri)
    np.testing.assert_array_equal(mask2, boundary_mask)


def test_load_mesh_rejects_unknown_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "mesh.vtk"
    bogus.write_text("not a real mesh")
    with pytest.raises(ValueError, match="Unsupported mesh file extension"):
        load_mesh(bogus)


def test_npz_missing_array_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, xy=np.zeros((3, 2)))  # missing tri, boundary_mask
    with pytest.raises(ValueError, match="missing required array"):
        load_mesh(path)


def test_resolve_mesh_uses_given_path(tmp_path: Path) -> None:
    xy, tri = _two_triangle_square()
    boundary_mask = boundary_mask_from_triangles(tri, xy.shape[0])
    path = save_mesh(tmp_path / "mesh.npz", xy, tri, boundary_mask)

    xy2, tri2, mask2 = resolve_mesh(path)
    np.testing.assert_array_equal(xy2, xy)
    np.testing.assert_array_equal(tri2, tri)
    np.testing.assert_array_equal(mask2, boundary_mask)


def test_resolve_mesh_raises_without_path_or_domain() -> None:
    with pytest.raises(ValueError, match="needs either mesh_path"):
        resolve_mesh(None)


def test_resolve_mesh_forwards_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_mesh's on-the-fly path must pass `order` through to
    generate_domain_mesh -- it silently defaulted to order=1 (T3) even for
    a caller requesting order=2 (T6) until this was fixed. Patches
    generate_domain_mesh rather than requiring gmsh to be installed."""
    calls = []

    def fake_generate_domain_mesh(
        domain: str, resolution: float, order: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append((domain, resolution, order))
        n_local = 6 if order == 2 else 3
        return (
            np.zeros((1, 2)),
            np.zeros((1, n_local), dtype=np.int32),
            np.array([True]),
        )

    import scirex.experimental.geometry.gmsh_mesh as gmsh_mesh_module

    monkeypatch.setattr(
        gmsh_mesh_module, "generate_domain_mesh", fake_generate_domain_mesh
    )

    _, tri, _ = resolve_mesh(None, domain="square", resolution=0.1, order=2)
    assert calls == [("square", 0.1, 2)]
    assert tri.shape[-1] == 6


def _two_triangle_square_t6() -> tuple[np.ndarray, np.ndarray]:
    """The same two-triangle unit square as `_two_triangle_square`,
    elevated to T6 by hand: mid-edge nodes 4 (mid of 0-1), 5 (mid of 1-2),
    6 (mid of the shared diagonal 2-0/0-2), 7 (mid of 2-3), 8 (mid of
    3-0)."""
    xy3, _tri3 = _two_triangle_square()
    mid = {
        (0, 1): 4,
        (1, 2): 5,
        (2, 0): 6,
        (0, 2): 6,
        (2, 3): 7,
        (3, 0): 8,
    }
    xy6 = np.concatenate(
        [
            xy3,
            [
                (xy3[0] + xy3[1]) / 2,  # 4
                (xy3[1] + xy3[2]) / 2,  # 5
                (xy3[2] + xy3[0]) / 2,  # 6, the shared diagonal's midpoint
                (xy3[2] + xy3[3]) / 2,  # 7
                (xy3[3] + xy3[0]) / 2,  # 8
            ],
        ]
    )
    tri6 = np.array(
        [
            [0, 1, 2, mid[(0, 1)], mid[(1, 2)], mid[(2, 0)]],
            [0, 2, 3, mid[(0, 2)], mid[(2, 3)], mid[(3, 0)]],
        ]
    )
    return xy6, tri6


def test_boundary_mask_t6_marks_mid_edge_nodes_by_their_own_edge() -> None:
    """A T6 mid-edge node on a hull edge (shared by exactly one triangle)
    must be flagged boundary; the mid-edge node on the interior diagonal
    (shared by both triangles) must not be, even though both its own
    endpoints are boundary corners -- boundary status of a mid-edge node
    comes from its edge, not its endpoints."""
    xy6, tri6 = _two_triangle_square_t6()
    mask = boundary_mask_from_triangles(tri6, xy6.shape[0])

    assert mask[:4].all()  # every corner is on the hull
    assert mask[4] and mask[5] and mask[7] and mask[8]  # hull-edge midpoints
    assert not mask[6]  # the shared diagonal's midpoint is interior


def test_resolve_mesh_works_when_mesh_io_is_loaded_standalone() -> None:
    """resolve_mesh's fallback to gmsh_mesh.py used a bare relative import
    (`from .gmsh_mesh import ...`), which raises ImportError when mesh_io.py
    is loaded via importlib.util.spec_from_file_location rather than as
    part of the scirex package -- exactly how scripts/mesh/
    generate_gmsh_mesh.py loads this same file (to avoid src/scirex/
    __init__.py's eager JAX/Flax import). Uses an invalid domain name so
    this exercises the whole import-fallback path and reaches
    generate_domain_mesh's own dispatch, without requiring gmsh itself to
    be installed (loading gmsh_mesh.py doesn't need gmsh; only actually
    generating a mesh does)."""
    repo_root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "mesh_io_standalone",
        repo_root / "src" / "scirex" / "experimental" / "geometry" / "mesh_io.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="Unknown domain"):
        module.resolve_mesh(None, domain="hexagon")
