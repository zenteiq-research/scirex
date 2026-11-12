# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for MeshDataModule (JAX).

neuraloperator has no dedicated MeshDataModule test, so coverage follows the
Phase 1 layer-test structure (CPU/GPU backend parametrization for the numeric
helpers). The full ``MeshDataModule.__init__`` requires open3d plus an
on-disk mesh dataset (.ply / .npy), so it is reproduced as a data-gated,
skipped test. The pure, side-effect-free helpers and the lazy data loader are
exercised directly by bypassing ``__init__`` with ``__new__``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.data.datasets.mesh_datamodule import MeshDataModule, _LazyLoader


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


@pytest.fixture
def module() -> MeshDataModule:
    """A MeshDataModule instance with __init__ bypassed (helpers only)."""
    return MeshDataModule.__new__(MeshDataModule)


# ---------------------------------------------------------------------------
# range_normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_range_normalize_maps_endpoints(module, backend) -> None:
    """[min_b, max_b] maps linearly onto [new_min, new_max]."""
    with jax.default_device(jax.devices(backend)[0]):
        data = jnp.array([0.0, 5.0, 10.0])
        out = module.range_normalize(data, 0.0, 10.0, 0.0, 1.0)
        assert jnp.allclose(out, jnp.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_range_normalize_custom_target_range(module, backend) -> None:
    """A non-[0,1] target range is honoured at both ends."""
    with jax.default_device(jax.devices(backend)[0]):
        data = jnp.array([2.0, 6.0])
        out = module.range_normalize(data, 2.0, 6.0, 1e-6, 1.0)
        assert jnp.allclose(out[0], 1e-6, atol=1e-9)
        assert jnp.allclose(out[1], 1.0)


# ---------------------------------------------------------------------------
# compute_triangle_centroids
# ---------------------------------------------------------------------------


def test_compute_triangle_centroids(module) -> None:
    """Centroid is the vertex mean; area matches the analytic triangle area."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2]])
    centroids, areas = module.compute_triangle_centroids(vertices, triangles)
    assert np.allclose(centroids[0], [1 / 3, 1 / 3, 0.0])
    assert np.allclose(areas, [0.5])  # right triangle, legs of length 1


def test_compute_triangle_centroids_multiple(module) -> None:
    """Centroids/areas are computed independently per triangle."""
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    centroids, areas = module.compute_triangle_centroids(vertices, triangles)
    assert centroids.shape == (2, 3)
    assert np.allclose(areas, [2.0, 2.0])  # two triangles, area 2 each


# ---------------------------------------------------------------------------
# get_bounds_from_data / get_global_bounding_box
# ---------------------------------------------------------------------------


def test_get_bounds_from_data(module) -> None:
    """Bounds are the global min/max of a key across all samples."""
    data = [
        {"d": np.array([1.0, 4.0])},
        {"d": np.array([-2.0, 9.0])},
        {"d": np.array([0.0, 3.0])},
    ]
    lo, hi = module.get_bounds_from_data(data, "d")
    assert lo == -2.0
    assert hi == 9.0


def test_get_global_bounding_box(module) -> None:
    """Global bbox is the per-axis min/max across all mesh bounds."""

    class _FakeMesh:
        def __init__(self, mn, mx):
            self._mn = np.array(mn, dtype=float)
            self._mx = np.array(mx, dtype=float)

        def get_min_bound(self):
            return self._mn

        def get_max_bound(self):
            return self._mx

    meshes = [_FakeMesh([0, 0, 0], [1, 2, 3]), _FakeMesh([-1, 0, 1], [0, 5, 2])]
    min_b, max_b = module.get_global_bounding_box(meshes)
    assert np.allclose(min_b, [-1, 0, 0])
    assert np.allclose(max_b, [1, 5, 3])


# ---------------------------------------------------------------------------
# _LazyLoader / _create_loader
# ---------------------------------------------------------------------------


def _dict_data(n):
    return [{"x": np.array([i, i + 1]), "y": np.array([i])} for i in range(n)]


def test_lazy_loader_len_and_stacking() -> None:
    """_LazyLoader yields stacked batches following its index groups."""
    data = _dict_data(4)
    index_groups = [[0, 1], [2, 3]]
    loader = _LazyLoader(data, index_groups)
    assert len(loader) == 2
    batches = list(loader)
    assert batches[0]["x"].shape == (2, 2)
    assert np.allclose(batches[0]["x"], np.array([[0, 1], [1, 2]]))
    # exposes a minimal .dataset with __len__ for the trainer
    assert len(loader.dataset) == 2


def test_lazy_loader_dict_valued_key_single() -> None:
    """A dict-valued key in a size-1 batch is passed through unstacked."""
    data = [{"neighbors": {"idx": np.arange(3)}, "x": np.ones(2)}]
    loader = _LazyLoader(data, [[0]])
    batch = next(iter(loader))
    assert isinstance(batch["neighbors"], dict)
    assert np.allclose(batch["neighbors"]["idx"], np.arange(3))


@pytest.mark.parametrize("n,batch_size,expected", [(5, 2, 3), (4, 2, 2), (3, 5, 1)])
def test_create_loader_batch_counts(module, n, batch_size, expected) -> None:
    """_create_loader splits the data into ceil(n / batch_size) batches."""
    loader = module._create_loader(_dict_data(n), batch_size=batch_size, shuffle=False)
    assert isinstance(loader, _LazyLoader)
    assert len(loader) == expected
    assert sum(b["x"].shape[0] for b in loader) == n


def test_create_loader_shuffle_preserves_samples(module) -> None:
    """Shuffling keeps the same multiset of samples (only order changes)."""
    loader = module._create_loader(_dict_data(6), batch_size=2, shuffle=True)
    seen = np.concatenate([b["x"][:, 0] for b in loader])
    assert np.allclose(np.sort(seen), np.arange(6))


# ---------------------------------------------------------------------------
# Full init (open3d + on-disk mesh dataset gated)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full init with a synthetic open3d mesh dataset (when open3d is available)
# ---------------------------------------------------------------------------

o3d = pytest.importorskip("open3d")


def _build_mesh_dataset(
    root, item_dir_name="processed-car-pressure-data", inds=("m0", "m1")
):
    """Write a minimal mesh dataset: train/test.txt + tri_mesh.ply + press.npy."""
    ds_root = root / item_dir_name
    data_dir = ds_root / "data"
    for ind in inds:
        d = data_dir / ind
        d.mkdir(parents=True)
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(d / "tri_mesh.ply"), mesh)
        press = np.random.default_rng(0).random((200, 120), dtype=np.float32)
        np.save(str(d / "press.npy"), press)
    (ds_root / "train.txt").write_text(inds[0])
    (ds_root / "test.txt").write_text(inds[1])
    return ds_root


@pytest.fixture
def mesh_root(tmp_path) -> "object":
    _build_mesh_dataset(tmp_path)
    return tmp_path


def test_mesh_datamodule_init_full(mesh_root) -> None:
    """Full __init__ loads meshes, computes signed distances, and normalizes attrs."""
    mdm = MeshDataModule(
        root_dir=mesh_root,
        item_dir_name="processed-car-pressure-data",
        n_train=1,
        n_test=1,
        query_res=[4, 4, 4],
        attributes=["press"],
    )
    assert len(mdm.train_data) == 1
    assert len(mdm.test_data) == 1
    assert "press" in mdm.normalizers
    # train_loader / test_loader build lazy loaders over the datasets
    assert len(mdm.train_loader(batch_size=1)) == 1
    assert len(mdm.test_loader(batch_size=1)) == 1


def test_mesh_datamodule_init_no_attributes(mesh_root) -> None:
    """With attributes=None, no normalizers are built."""
    mdm = MeshDataModule(
        root_dir=str(mesh_root),  # also exercises the str->Path branch
        item_dir_name=["processed-car-pressure-data"],  # list item_dir_name branch
        n_train=1,
        n_test=1,
        query_res=[4, 4, 4],
        attributes=None,
    )
    assert mdm.normalizers is None


def test_mesh_datamodule_truncates_indices(tmp_path) -> None:
    """n_train/n_test smaller than the index files truncate the index lists."""
    _build_mesh_dataset(tmp_path, inds=("m0", "m1"))
    # train.txt/test.txt each hold one index; ask for the same and add extra indices
    ds_root = tmp_path / "processed-car-pressure-data"
    (ds_root / "train.txt").write_text("m0,m1")  # 2 train indices
    (ds_root / "test.txt").write_text("m0,m1")  # 2 test indices
    mdm = MeshDataModule(
        root_dir=tmp_path,
        item_dir_name="processed-car-pressure-data",
        n_train=1,  # < 2 -> truncation branch
        n_test=1,
        query_res=[4, 4, 4],
        attributes=None,
    )
    assert len(mdm.train_data) == 1


def test_mesh_datamodule_empty_attributes_sets_normalizers_none(tmp_path) -> None:
    """attributes=[] takes the branch where no normalizer keys are produced."""
    _build_mesh_dataset(tmp_path)
    mdm = MeshDataModule(
        root_dir=tmp_path,
        item_dir_name="processed-car-pressure-data",
        n_train=1,
        n_test=1,
        query_res=[4, 4, 4],
        # non-None but empty -> normalizer_keys empty -> normalizers=None
        attributes=[],
    )
    assert mdm.normalizers is None


def test_mesh_datamodule_rejects_non_list_query_res(tmp_path) -> None:
    """A non-list/tuple query_res raises TypeError after the meshes load."""
    _build_mesh_dataset(tmp_path)
    with pytest.raises(TypeError):
        MeshDataModule(
            root_dir=tmp_path,
            item_dir_name="processed-car-pressure-data",
            n_train=1,
            n_test=1,
            query_res=8,  # int, not list/tuple
            attributes=None,
        )


def test_compute_distances_unsigned(module) -> None:
    """compute_distances(signed_distance=False) takes the unsigned distance branch."""
    box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    query = np.zeros((2, 2, 2, 3), dtype=np.float32)
    dist, closest = module.compute_distances(box, query, signed_distance=False)
    assert np.asarray(dist).shape == (2, 2, 2)
    assert np.asarray(closest).shape[-1] == 3


def test_are_watertight_method(module) -> None:
    """are_watertight returns True for a closed mesh and False for an open one."""
    closed = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    assert module.are_watertight([closed]) is True

    open_mesh = o3d.geometry.TriangleMesh()
    open_mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    )
    open_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    assert module.are_watertight([open_mesh]) is False
