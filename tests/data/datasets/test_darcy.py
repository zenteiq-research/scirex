# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for DarcyDataset, the torch-free DataLoader, and load_darcy_pt (JAX).

Mirrors ``neuralop/data/datasets/tests/test_ptdatasets.py::test_DarcyDatasetDownload``.
The reference test downloads the Darcy archive from Zenodo and then asserts that
``train_db`` / ``test_dbs`` / ``data_processor`` are populated. Network downloads
are not available in CI, so:

* the actual Zenodo download is reproduced as a network-gated, skipped test, and
* the same dataset contract is verified against ``DarcyDataset(download=False)``
  using small synthetic ``darcy_*.pt`` fixtures written with ``torch.save``.

The torch-free ``DataLoader`` (a drop-in for ``torch.utils.data.DataLoader``)
needs no torch and is tested directly.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest

from scirex.data.containers.tensor_dataset import TensorDataset
from scirex.data.datasets import darcy
from scirex.data.datasets.darcy import (
    DarcyDataset,
    DataLoader,
    load_darcy_flow_small,
    load_darcy_pt,
)

# ---------------------------------------------------------------------------
# DataLoader (torch-free batching loader)
# ---------------------------------------------------------------------------


def _tensor_dataset(n, dim=2):
    x = jnp.arange(n * dim, dtype=jnp.float32).reshape(n, dim)
    return TensorDataset(x, x * 10.0)


@pytest.mark.parametrize("n,batch_size,expected", [(6, 2, 3), (7, 3, 3), (5, 5, 1)])
def test_dataloader_len(n, batch_size, expected) -> None:
    """__len__ is ceil(n / batch_size)."""
    dl = DataLoader(_tensor_dataset(n), batch_size=batch_size)
    assert len(dl) == expected


def test_dataloader_batches_cover_all_samples() -> None:
    """Iterating yields every sample exactly once, last batch is the remainder."""
    dl = DataLoader(_tensor_dataset(7), batch_size=3)
    batches = list(dl)
    assert len(batches) == 3
    assert batches[0]["x"].shape == (3, 2)
    assert batches[-1]["x"].shape == (1, 2)  # 7 = 3 + 3 + 1
    total = sum(b["x"].shape[0] for b in batches)
    assert total == 7


def test_dataloader_shuffle_preserves_set_of_samples() -> None:
    """Shuffling changes order but keeps the same multiset of first-column values."""
    dl = DataLoader(_tensor_dataset(6), batch_size=2, shuffle=True)
    seen = jnp.concatenate([b["x"][:, 0] for b in dl])
    assert jnp.allclose(jnp.sort(seen), jnp.arange(6) * 2.0)


# ---------------------------------------------------------------------------
# DarcyDataset (offline, via synthetic fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture
def darcy_dir(tmp_path) -> Path:
    """Write synthetic darcy_{train,test}_16.pt fixtures (requires torch)."""
    torch = pytest.importorskip("torch")
    for split in ("train", "test"):
        t = torch.arange(5 * 16 * 16, dtype=torch.float32).reshape(5, 16, 16)
        torch.save(
            {"x": t, "y": t * 2.0}, (tmp_path / f"darcy_{split}_16.pt").as_posix()
        )
    return tmp_path


def _darcy(root, **kwargs):
    return DarcyDataset(
        root_dir=root,
        n_train=4,
        n_tests=[3],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=16,
        test_resolutions=[16],
        download=False,
        **kwargs,
    )


def test_darcy_offline_contract(darcy_dir) -> None:
    """DarcyDataset(download=False) exposes the required attributes (ref contract)."""
    from scirex.data.transforms.data_processors import DefaultDataProcessor

    dataset = _darcy(darcy_dir)
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor is not None
    assert isinstance(dataset.data_processor, DefaultDataProcessor)
    assert len(dataset.train_db) == 4
    assert len(dataset.test_dbs[16]) == 3
    assert dataset.train_db[0]["x"].shape == (1, 16, 16)


def test_darcy_rejects_unavailable_resolution(darcy_dir) -> None:
    """Resolutions outside the Zenodo archive set raise AssertionError."""
    with pytest.raises(AssertionError):
        DarcyDataset(
            root_dir=darcy_dir,
            n_train=1,
            n_tests=[1],
            batch_size=1,
            test_batch_sizes=[1],
            train_resolution=17,  # not in [16, 32, 64, 128, 421]
            test_resolutions=[17],
            download=False,
        )


def test_load_darcy_pt_returns_loaders(darcy_dir) -> None:
    """load_darcy_pt returns (train_loader, test_loaders dict, data_processor)."""
    train_loader, test_loaders, data_processor = load_darcy_pt(
        n_train=4,
        n_tests=[3],
        batch_size=2,
        test_batch_sizes=[1],
        data_root=darcy_dir,
        train_resolution=16,
        test_resolutions=[16],
    )
    assert isinstance(train_loader, DataLoader)
    assert set(test_loaders.keys()) == {16}
    assert data_processor is not None
    # batch_size=2 over 4 train samples -> 2 batches
    assert len(train_loader) == 2


# ---------------------------------------------------------------------------
# Zenodo download (network-gated, mirrors test_DarcyDatasetDownload)
# ---------------------------------------------------------------------------


def test_darcy_download_branch_writes_and_loads(tmp_path, monkeypatch) -> None:
    """download=True triggers the (mocked) Zenodo fetch into a fresh root dir."""
    torch = pytest.importorskip("torch")
    root = tmp_path / "darcy_new"  # does not exist -> exercises str + mkdir branch

    def _fake_download(record_id, root, files_to_download) -> None:
        # the offline mock "downloads" by writing the expected .pt fixtures
        for split in ("train", "test"):
            t = torch.arange(5 * 16 * 16, dtype=torch.float32).reshape(5, 16, 16)
            torch.save(
                {"x": t, "y": t * 2.0}, (Path(root) / f"darcy_{split}_16.pt").as_posix()
            )

    monkeypatch.setattr(darcy, "download_from_zenodo_record", _fake_download)

    dataset = DarcyDataset(
        root_dir=str(root),
        n_train=4,
        n_tests=[3],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=16,
        test_resolutions=[16],
        download=True,
    )
    assert dataset.train_db
    assert dataset.test_dbs[16]


def test_load_darcy_flow_small_returns_loaders(darcy_dir, monkeypatch) -> None:
    """load_darcy_flow_small builds the dataset and legacy dataloaders (offline)."""
    monkeypatch.setattr(darcy, "download_from_zenodo_record", lambda **kwargs: None)
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=4,
        n_tests=[3],
        batch_size=2,
        test_batch_sizes=[1],
        data_root=darcy_dir,
        test_resolutions=[16],
    )
    assert isinstance(train_loader, DataLoader)
    assert set(test_loaders.keys()) == {16}
    assert data_processor is not None


@pytest.mark.skip(
    reason="Downloads the Darcy archive from Zenodo; requires network access and "
    "is not run in CI. Offline coverage is provided by test_darcy_offline_contract."
)
def test_darcy_dataset_download(tmp_path) -> None:
    resolution = 16
    dataset = DarcyDataset(
        root_dir=tmp_path,
        n_train=5,
        n_tests=[5],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=resolution,
        test_resolutions=[resolution, resolution * 2],
        download=True,
    )
    assert dataset.train_db
    assert dataset.test_dbs
    assert dataset.data_processor
