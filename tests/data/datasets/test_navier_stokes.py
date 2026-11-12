# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for NavierStokesDataset and load_navier_stokes_pt (JAX).

Mirrors ``neuralop/data/datasets/tests/test_ptdatasets.py::test_NSDownload``.
The reference test downloads the NS forcing archive from Zenodo and asserts that
``train_db`` / ``test_dbs`` / ``data_processor`` are populated. Network access is
unavailable in CI, so:

* the Zenodo download is reproduced as a network-gated, skipped test, and
* the dataset contract is verified offline with synthetic ``nsforcing_*.pt``
  fixtures (resolution 128, the smallest available archive resolution).

Note ``load_navier_stokes_pt`` always constructs the dataset with
``download=True``; because ``download_from_zenodo_record`` short-circuits on an
empty file list, it runs fully offline once the fixtures already exist.
"""

from pathlib import Path

import pytest

from scirex.data.datasets import navier_stokes
from scirex.data.datasets.darcy import DataLoader
from scirex.data.datasets.navier_stokes import (
    NavierStokesDataset,
    load_navier_stokes_pt,
)

RES = 128


@pytest.fixture
def ns_dir(tmp_path) -> Path:
    """Write synthetic nsforcing_{train,test}_128.pt fixtures (requires torch)."""
    torch = pytest.importorskip("torch")
    for split in ("train", "test"):
        t = torch.arange(4 * RES * RES, dtype=torch.float32).reshape(4, RES, RES)
        torch.save(
            {"x": t, "y": t * 2.0},
            (tmp_path / f"nsforcing_{split}_{RES}.pt").as_posix(),
        )
    return tmp_path


def test_ns_offline_contract(ns_dir) -> None:
    """NavierStokesDataset(download=False) exposes the required attributes."""
    from scirex.data.transforms.data_processors import DefaultDataProcessor

    dataset = NavierStokesDataset(
        root_dir=ns_dir,
        n_train=3,
        n_tests=[2],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=RES,
        test_resolutions=[RES],
        download=False,
    )
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor is not None
    assert isinstance(dataset.data_processor, DefaultDataProcessor)
    assert len(dataset.train_db) == 3
    assert len(dataset.test_dbs[RES]) == 2
    assert dataset.train_db[0]["x"].shape == (1, RES, RES)


def test_ns_rejects_unavailable_resolution(ns_dir) -> None:
    """Resolutions outside [128, 1024] raise AssertionError."""
    with pytest.raises(AssertionError):
        NavierStokesDataset(
            root_dir=ns_dir,
            n_train=1,
            n_tests=[1],
            batch_size=1,
            test_batch_sizes=[1],
            train_resolution=64,  # not in [128, 1024]
            test_resolutions=[64],
            download=False,
        )


def test_load_navier_stokes_pt_offline(ns_dir) -> None:
    """load_navier_stokes_pt returns loaders offline (empty download list)."""
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        n_train=3,
        n_tests=[2],
        batch_size=1,
        test_batch_sizes=[1],
        data_root=ns_dir,
        train_resolution=RES,
        test_resolutions=[RES],
    )
    assert isinstance(train_loader, DataLoader)
    assert set(test_loaders.keys()) == {RES}
    assert data_processor is not None
    assert len(train_loader) == 3  # batch_size=1 over 3 train samples


def test_ns_download_branch_writes_and_loads(tmp_path, monkeypatch) -> None:
    """download=True triggers the (mocked) Zenodo fetch into a fresh root dir."""
    torch = pytest.importorskip("torch")
    root = tmp_path / "ns_new"  # does not exist -> exercises str + mkdir branch

    def _fake_download(record_id, root, files_to_download) -> None:
        for split in ("train", "test"):
            t = torch.arange(4 * RES * RES, dtype=torch.float32).reshape(4, RES, RES)
            torch.save(
                {"x": t, "y": t * 2.0},
                (Path(root) / f"nsforcing_{split}_{RES}.pt").as_posix(),
            )

    monkeypatch.setattr(navier_stokes, "download_from_zenodo_record", _fake_download)

    dataset = NavierStokesDataset(
        root_dir=str(root),
        n_train=3,
        n_tests=[2],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=RES,
        test_resolutions=[RES],
        download=True,
    )
    assert dataset.train_db
    assert dataset.test_dbs[RES]


@pytest.mark.skip(
    reason="Downloads the Navier-Stokes archive from Zenodo; requires network "
    "access and is not run in CI. Offline coverage is provided by "
    "test_ns_offline_contract."
)
def test_ns_download(tmp_path) -> None:
    dataset = NavierStokesDataset(
        root_dir=tmp_path,
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=128,
        test_resolutions=[128],
        download=True,
    )
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
