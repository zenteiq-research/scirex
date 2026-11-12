# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""JAX/Flax version of Mesh DataModule for handling mesh-based datasets"""

import logging
import random
from collections.abc import Iterator
from pathlib import Path
from timeit import default_timer
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scirex.data.transforms.normalizers import UnitGaussianNormalizer

from ..containers.dict_dataset import DictDataset

try:
    import open3d as o3d

    o3d_warn = False
except ModuleNotFoundError:
    o3d_warn = True

logger = logging.getLogger(__name__)


class _LazyLoader:
    """Lazy data loader that builds each batch on demand.

    Batches are constructed only during iteration, so no GPU memory is
    consumed at construction time.  `batch_size=1` is assumed for dict-
    valued keys such as `neighbors_in` / `neighbors_out`.
    """

    def __init__(
        self,
        data: DictDataset | list[dict],
        index_groups: list[list[int]],
        shuffle: bool = False,
    ) -> None:
        self._data = data
        self._index_groups = index_groups
        self._shuffle = shuffle
        # Expose a minimal `.dataset` object so the trainer can call len()
        _n = len(index_groups)
        self.dataset = type("Dataset", (), {"__len__": lambda _: _n})()

    def __len__(self) -> int:
        return len(self._index_groups)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        index_groups = self._index_groups
        if self._shuffle:
            flat = [idx for group in index_groups for idx in group]
            random.shuffle(flat)
            sizes = [len(group) for group in index_groups]
            index_groups = []
            pos = 0
            for size in sizes:
                index_groups.append(flat[pos : pos + size])
                pos += size

        for batch_indices in index_groups:
            batch: dict[str, Any] = {}
            for key in self._data[0].keys():
                values = [self._data[idx][key] for idx in batch_indices]
                if isinstance(values[0], dict):
                    batch[key] = values[0] if len(values) == 1 else values
                else:
                    batch[key] = np.stack(values)
            yield batch


class MeshDataModule:
    """JAX/Flax version of MeshDataModule for irregular coordinate meshes.

    Provides a general dataset for irregular coordinate meshes for use in
    GNO-based architectures with JAX/Flax.

    Parameters
    ----------
    root_dir : Union[str, Path]
        str or Path to root directory of CFD dataset
    item_dir_name : Union[str, Path]
        directory in which individual item subdirs are stored
    n_train : int, optional
        hard limit on number of training examples
    n_test : int, optional
        hard limit on number of test examples
    query_res : List[int], optional
        resolution of latent query points along each dimension
    attributes : List[str], optional
        list of string keys for attributes in the dataset to return
    """

    def __init__(
        self,
        root_dir: str | Path,
        item_dir_name: str | Path,
        n_train: int | None = None,
        n_test: int | None = None,
        query_res: list[int] | None = None,
        attributes: list[str] | None = None,
    ) -> None:
        """MeshDataModule initialization."""
        if o3d_warn:
            raise ModuleNotFoundError(
                "MeshDataModule requires the open3d package "
                '(pip install "scirex[open3d]").'
            )

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        root_dir = root_dir.expanduser().resolve()
        assert root_dir.exists(), "Path does not exist"
        assert root_dir.is_dir(), "Path is not a directory"

        if isinstance(item_dir_name, list | tuple):
            item_dir_name = item_dir_name[0]

        dataset_root = (root_dir / item_dir_name).expanduser().resolve()
        assert dataset_root.exists(), "Dataset folder does not exist"

        # Read train and test indices
        with open(dataset_root / "train.txt") as file:
            train_ind = file.readline().split(",")
        with open(dataset_root / "test.txt") as file:
            test_ind = file.readline().split(",")

        if n_train is not None and n_train < len(train_ind):
            train_ind = train_ind[0:n_train]
        if n_test is not None and n_test < len(test_ind):
            test_ind = test_ind[0:n_test]

        train_ind = train_ind[0:n_train]
        test_ind = test_ind[0:n_test]
        n_train = len(train_ind)
        n_test = len(test_ind)
        logger.debug(f"[INIT] n_train={n_train}, n_test={n_test}")

        mesh_ind = train_ind + test_ind
        mesh_ind = [x.rstrip() for x in mesh_ind]

        data_dir = dataset_root / "data"
        mesh_path_check = data_dir / train_ind[0] / "tri_mesh.ply"
        if not mesh_path_check.exists():
            data_dir = data_dir / "data"

        # Load meshes
        meshes = []

        for ind in mesh_ind:
            mesh_path = str(data_dir / (ind + "/tri_mesh.ply"))
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            meshes.append(mesh)

        # One bounding box over every mesh read (the n_train + n_test cars);
        # all vertices, centroids and the query grid are scaled with it below,
        # not with a per-car box.
        min_b, max_b = self.get_global_bounding_box(meshes)

        are_watertight = True

        # Create query points
        if isinstance(query_res, list | tuple):
            tx = np.linspace(min_b[0], max_b[0], query_res[0])
            ty = np.linspace(min_b[1], max_b[1], query_res[1])
            tz = np.linspace(min_b[2], max_b[2], query_res[2])

            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        else:
            raise TypeError(f"query_res must be list/tuple, got {type(query_res)}")

        # Process data from meshes
        data: list[dict[str, Any]] = []
        deleted_meshes = []
        self.time_to_distance = 0.0

        for i, mesh in enumerate(meshes):
            item_dict: dict[str, Any] = {}

            mesh = mesh.compute_triangle_normals()
            mesh = mesh.compute_vertex_normals()

            item_dict["vertices"] = np.asarray(mesh.vertices)
            item_dict["vertex_normals"] = np.asarray(mesh.vertex_normals)
            item_dict["triangle_normals"] = np.asarray(mesh.triangle_normals)

            centroids, area = self.compute_triangle_centroids(
                item_dict["vertices"], np.asarray(mesh.triangles)
            )
            item_dict["centroids"] = centroids
            item_dict["triangle_areas"] = area

            # Scale vertices and centroids into [0, 1] with the dataset-wide
            # box (min_b, max_b), not this car's own extent.
            item_dict["vertices"] = self.range_normalize(
                item_dict["vertices"], min_b, max_b, 0, 1
            )
            item_dict["centroids"] = self.range_normalize(
                item_dict["centroids"], min_b, max_b, 0, 1
            )

            # Compute distances
            if query_points is not None:
                tt = default_timer()
                try:
                    distance, closest = self.compute_distances(
                        mesh, query_points, are_watertight
                    )
                    elapsed = default_timer() - tt
                except Exception:
                    deleted_meshes.append(mesh_ind[i])
                    continue

                self.time_to_distance += elapsed
                item_dict["distance"] = np.expand_dims(distance, -1)
                item_dict["closest_points"] = closest
                item_dict["closest_points"] = self.range_normalize(
                    closest, min_b, max_b, 0, 1
                )

            data.append(item_dict)

        self.time_to_distance /= len(meshes) if len(meshes) > 0 else 1

        del meshes
        # Normalize data
        n_train -= len(deleted_meshes)

        # SDF and triangle-area bounds: training cars only (data[0:n_train]).
        min_dist, max_dist = self.get_bounds_from_data(data[0:n_train], "distance")

        min_area, max_area = self.get_bounds_from_data(
            data[0:n_train], "triangle_areas"
        )

        for data_dict in data:
            data_dict["distance"] = self.range_normalize(
                data_dict["distance"], min_dist, max_dist, 1e-6, 1
            )

            data_dict["normalized_triangle_areas"] = self.range_normalize(
                data_dict["triangle_areas"], min_area, max_area, 1e-6, 1
            )

        # Convert numpy arrays to jax arrays
        for data_dict in data:
            for key, value in data_dict.items():
                data_dict[key] = jnp.array(value, dtype=jnp.float32)

        # Load non-mesh data
        self.normalizers: dict[str, UnitGaussianNormalizer] | None = None
        if attributes is not None:
            for j, ind in enumerate(mesh_ind):
                if ind in deleted_meshes:
                    continue
                for attr in attributes:
                    path = str(data_dir / (ind + "/" + attr + ".npy"))
                    attr_data = np.load(path)

                    data[j][attr] = jnp.array(attr_data, dtype=jnp.float32)

            # Compute the Gaussian normalizers (mean/std). NOTE: `data` holds
            # every loaded car, train AND test (n_train + n_test), so these
            # statistics include the test cars, as in neuraloperator's
            # reference loader. They are fitted on the raw attribute, which
            # for Car-CFD still has all 3,682 pressure entries (the slice to
            # 3,586 happens afterwards, in CarCFDDatasetjax).
            normalizer_keys = []
            for attr in attributes:
                if isinstance(data[0][attr], jax.Array) or isinstance(
                    data[0][attr], jnp.ndarray
                ):
                    normalizer_keys.append(attr)

            # returns keyed dict of UnitGaussianNormalizer instances
            self.normalizers = UnitGaussianNormalizer.from_dataset(
                data, axis=[1], keys=normalizer_keys
            )

            # Encode all data
            for attr in normalizer_keys:
                for j in range(len(data)):
                    data_elem = data[j][attr]
                    if data_elem.shape[0] != 1:
                        data_elem = jnp.expand_dims(data_elem, axis=0)
                    data[j][attr] = self.normalizers[attr].transform(data_elem)

            if not bool(self.normalizers):
                self.normalizers = None
        else:
            self.normalizers = None

        # Set-up constant dict
        query_points_norm = self.range_normalize(query_points, min_b, max_b, 0, 1)
        query_points_jnp = jnp.array(query_points_norm, dtype=jnp.float32)

        constant = {"query_points": query_points_jnp}

        # Datasets
        self.train_data = DictDataset(data[0:n_train], constant)
        self.test_data = DictDataset(data[n_train:], constant)

    def get_global_bounding_box(self, meshes: list) -> tuple[np.ndarray, np.ndarray]:
        min_b = np.zeros((3, len(meshes)))
        max_b = np.zeros((3, len(meshes)))
        for j, mesh in enumerate(meshes):
            try:
                min_b[:, j] = mesh.get_min_bound()
                max_b[:, j] = mesh.get_max_bound()
            except IndexError:
                pass

        min_b = min_b.min(axis=1)
        max_b = max_b.max(axis=1)

        return min_b, max_b

    def are_watertight(self, meshes: list) -> bool:
        for mesh in meshes:
            if not mesh.is_watertight():
                return False
        return True

    def compute_triangle_centroids(
        self, vertices: np.ndarray, triangles: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        centroids = (A + B + C) / 3
        areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2

        return centroids, areas

    def compute_distances(
        self, mesh: Any, query_points: np.ndarray, signed_distance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        if signed_distance:
            dist = scene.compute_signed_distance(query_points).numpy()
        else:
            dist = scene.compute_distance(query_points).numpy()

        closest = scene.compute_closest_points(query_points)["points"].numpy()

        return dist, closest

    def range_normalize(
        self,
        data: jax.Array | np.ndarray,
        min_b: jax.Array | np.ndarray | float,
        max_b: jax.Array | np.ndarray | float,
        new_min: float,
        new_max: float,
    ) -> jax.Array | np.ndarray:
        # Normalize from [min_b, max_b] → [new_min, new_max]
        data = (data - min_b) / (max_b - min_b)
        data = (new_max - new_min) * data + new_min
        return data

    def get_bounds_from_data(self, data: list[dict], key: str) -> tuple[Any, Any]:
        global_min = data[0][key].min()
        global_max = data[0][key].max()

        for j in range(1, len(data)):
            current_min = data[j][key].min()
            current_max = data[j][key].max()

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

        return global_min, global_max

    # ------------------------------------------------------------------
    # Lazy loader helpers
    # ------------------------------------------------------------------
    def train_loader(self, batch_size: int = 1, shuffle: bool = True) -> _LazyLoader:
        """Return training dataloader."""
        return self._create_loader(self.train_data, batch_size, shuffle)

    def test_loader(self, batch_size: int = 1, shuffle: bool = False) -> _LazyLoader:
        """Return test dataloader."""
        return self._create_loader(self.test_data, batch_size, shuffle)

    def _create_loader(
        self,
        data: DictDataset | list[dict],
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> _LazyLoader:
        """Return a lazy data loader that builds each batch on demand.

        Batches are constructed only when iterated, avoiding the GPU OOM
        that would result from pre-building all batches as JAX arrays at once.
        """
        n_samples = len(data)
        indices = list(range(n_samples))

        # Store only index groups; actual tensors (and, if requested, a fresh
        # shuffle) are built in __iter__ so each epoch gets a new order.
        index_groups = [
            indices[i : i + batch_size] for i in range(0, n_samples, batch_size)
        ]

        return _LazyLoader(data, index_groups, shuffle=shuffle)
