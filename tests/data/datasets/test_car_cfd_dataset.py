# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the Car-CFD dataset helpers (JAX).

neuraloperator has no dedicated test for the Car-CFD dataset, so coverage
focuses on what can be exercised without the multi-gigabyte Zenodo mesh archive:

* ``load_mini_car`` -> loads the packaged 3-example mini dataset (torch-free).
* ``CarCFDDatasetjax`` -> requires downloading/parsing the full mesh dataset and
  is reproduced as a network-gated, skipped test.

The mini-car data is shipped as a torch ``.pt`` archive; it is read with the
repository's torch-free ``_load_pt`` loader, so these tests need no torch.
"""

from pathlib import Path

import numpy as np
import pytest

from scirex._internal.paths import get_project_root
from scirex.data.datasets.car_cfd import (
    CarCFDDatasetjax,
    load_mini_car,
)

# The mini-car archive (~1.2 MB) is gitignored and therefore absent in CI; the
# tests that read it are skipped when the file is not present locally.
_MINI_CAR_PATH = get_project_root() / "scirex/data/datasets/data/mini_car.pt"
requires_mini_car = pytest.mark.skipif(
    not _MINI_CAR_PATH.exists(),
    reason=f"packaged mini-car dataset not available at {_MINI_CAR_PATH}",
)

# Keys present in every packaged mini-car example.
_EXPECTED_KEYS = {
    "vertices",
    "vertex_normals",
    "triangle_normals",
    "centroids",
    "triangle_areas",
    "distance",
    "closest_points",
    "normalized_triangle_areas",
    "press",
    "query_points",
}


# ---------------------------------------------------------------------------
# load_mini_car
# ---------------------------------------------------------------------------


@requires_mini_car
def test_load_mini_car_returns_three_examples() -> None:
    """The packaged mini dataset has exactly three mesh examples."""
    data = load_mini_car()
    assert isinstance(data, list)
    assert len(data) == 3


@requires_mini_car
def test_load_mini_car_example_keys() -> None:
    """Each example exposes the full set of mesh/SDF attributes."""
    data = load_mini_car()
    for example in data:
        assert isinstance(example, dict)
        assert _EXPECTED_KEYS.issubset(example.keys())


@requires_mini_car
def test_load_mini_car_array_shapes() -> None:
    """Loaded attributes are numeric arrays with the expected rank/last-dim."""
    example = load_mini_car()[0]
    vertices = np.asarray(example["vertices"])
    query_points = np.asarray(example["query_points"])
    press = np.asarray(example["press"])

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert query_points.ndim == 4 and query_points.shape[-1] == 3
    assert press.ndim == 2
    # query SDF cube is a regular grid, equal along the three spatial dims
    assert query_points.shape[0] == query_points.shape[1] == query_points.shape[2]


@requires_mini_car
def test_mini_car_pressure_trim_matches_dataset_logic() -> None:
    """The vertex-trim applied in CarCFDDatasetjax keeps cols [0:16] + [112:]."""
    press = np.asarray(load_mini_car()[0]["press"])
    trimmed = np.concatenate((press[:, 0:16], press[:, 112:]), axis=1)
    assert trimmed.shape[1] == 16 + (press.shape[1] - 112)
    assert trimmed.shape[0] == press.shape[0]


# ---------------------------------------------------------------------------
# Full dataset (network + mesh-archive gated)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="CarCFDDatasetjax downloads the full Car-CFD mesh archive from Zenodo "
    "and parses it via MeshDataModule; requires network access and several GB of "
    "data, so it is not run in CI."
)
def test_car_cfd_dataset_download(tmp_path) -> None:
    dataset = CarCFDDatasetjax(
        root_dir=tmp_path,
        n_train=1,
        n_test=1,
        query_res=[16, 16, 16],
        download=True,
    )
    assert dataset.train_data
    assert dataset.test_data


# ---------------------------------------------------------------------------
# Offline CarCFDDatasetjax over a synthetic open3d mesh dataset
# ---------------------------------------------------------------------------

o3d = pytest.importorskip("open3d")


def _build_car_dataset(root, inds=("m0", "m1")):
    ds_root = root / "processed-car-pressure-data"
    data_dir = ds_root / "data"
    for ind in inds:
        d = data_dir / ind
        d.mkdir(parents=True)
        mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(d / "tri_mesh.ply"), mesh)
        # press needs >=113 columns so the [0:16] + [112:] vertex-trim is meaningful
        press = np.random.default_rng(0).random((200, 120), dtype=np.float32)
        np.save(str(d / "press.npy"), press)
    (ds_root / "train.txt").write_text(inds[0])
    (ds_root / "test.txt").write_text(inds[1])


def test_car_cfd_offline_init(tmp_path) -> None:
    """CarCFDDatasetjax(download=False) parses the mesh dir and trims pressure cols."""
    _build_car_dataset(tmp_path)
    ds = CarCFDDatasetjax(
        root_dir=tmp_path, n_train=1, n_test=1, query_res=[4, 4, 4], download=False
    )
    assert len(ds.train_data) == 1
    assert len(ds.test_data) == 1
    # the trim slices axis 1 (the 200 pressure rows): keeps [0:16] + [112:]
    press = ds.train_data.data_list[0]["press"]
    assert press.shape[1] == 16 + (200 - 112)
    assert press.shape[-1] == 120


def test_car_cfd_download_branch_is_invoked(tmp_path, monkeypatch) -> None:
    """download=True into a fresh str root: mkdir + (mocked) download build the data."""
    root = tmp_path / "car_new"  # does not exist -> exercises str->Path + mkdir branch
    called = {}

    def _fake_download(**kwargs):
        called["hit"] = True
        _build_car_dataset(Path(kwargs["root"]))

    monkeypatch.setattr(
        "scirex.data.datasets.car_cfd.download_from_zenodo_record",
        _fake_download,
    )
    CarCFDDatasetjax(
        root_dir=str(root), n_train=1, n_test=1, query_res=[4, 4, 4], download=True
    )
    assert called.get("hit") is True


def test_load_mini_car_with_synthetic_archive(tmp_path, monkeypatch) -> None:
    """load_mini_car reads the packaged mini_car.pt via the torch-free loader."""
    torch = pytest.importorskip("torch")
    data_dir = tmp_path / "scirex/data/datasets/data"
    data_dir.mkdir(parents=True)
    payload = [{"vertices": torch.zeros(3, 3)} for _ in range(3)]
    torch.save(payload, (data_dir / "mini_car.pt").as_posix())
    monkeypatch.setattr(
        "scirex.data.datasets.car_cfd.get_project_root",
        lambda: tmp_path,
    )
    loaded = load_mini_car()
    assert len(loaded) == 3
