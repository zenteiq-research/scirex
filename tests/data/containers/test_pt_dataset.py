# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for PTDataset and the torch-free ``.pt`` loader (JAX).

Mirrors ``neuralop/data/datasets/tests/test_ptdatasets.py``. The reference test
exercises the download-backed Darcy/Burgers/NS datasets and then asserts the
required attributes ``train_db`` / ``test_dbs`` / ``data_processor`` exist.
scirex's ``PTDataset`` is the torch-free base class (no network download), so we
generate small synthetic ``.pt`` fixtures locally with ``torch.save`` and assert
the same dataset contract plus the channel/subsampling/encoder behaviour.

The whole module is skipped when PyTorch is unavailable, since the ``.pt``
fixtures (and the loader's torch-zip format) require it to be written.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import io  # noqa: E402

from scirex.data.containers.pt_dataset import (  # noqa: E402
    PTDataset,
    _load_pt,
    _PtUnpickler,
    _rebuild_tensor,
)
from scirex.data.containers.tensor_dataset import TensorDataset  # noqa: E402

# ---------------------------------------------------------------------------
# torch-free loader internals (unit-level, no .pt file needed)
# ---------------------------------------------------------------------------


def test_rebuild_tensor_without_stride() -> None:
    """The no-stride path reshapes a flat storage slice to the requested shape."""
    storage = np.arange(6, dtype=np.float32)
    out = _rebuild_tensor(storage, 0, (2, 3))  # no stride args -> size/reshape branch
    assert out.shape == (2, 3)
    assert np.allclose(out, storage.reshape(2, 3))


def test_pt_unpickler_typed_storage_pid() -> None:
    """A torch>=2 typed-storage persistent id (len>=6) resolves its explicit dtype."""
    raw = {"3": np.array([1.0], dtype=np.float32).tobytes()}
    up = _PtUnpickler(io.BytesIO(b""), raw)
    arr = up.persistent_load(("storage", None, "3", "cpu", 1, "torch.float32"))
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0])


def test_pt_unpickler_find_class_special_cases() -> None:
    """find_class resolves storage classes, the rebuild fn, and torch.Size."""
    up = _PtUnpickler(io.BytesIO(b""), {})
    storage_cls = up.find_class("torch", "FloatStorage")
    assert storage_cls.__name__ == "FloatStorage"
    assert up.find_class("torch._utils", "_rebuild_tensor_v2") is _rebuild_tensor
    assert up.find_class("torch", "Size") is tuple


DATASET_NAME = "synth"
N_TOTAL = 6


def _write_pt(path: Path, n: int, res: int) -> None:
    """Write a synthetic Darcy-like {'x','y'} sample of shape (n, res, res)."""
    x = torch.arange(n * res * res, dtype=torch.float32).reshape(n, res, res)
    y = x * 2.0 + 1.0
    torch.save({"x": x, "y": y}, path.as_posix())


@pytest.fixture
def dataset_dir(tmp_path) -> Path:
    """Create train/test .pt files for resolutions 8 and 16."""
    for res in (8, 16):
        _write_pt(tmp_path / f"{DATASET_NAME}_train_{res}.pt", N_TOTAL, res)
        _write_pt(tmp_path / f"{DATASET_NAME}_test_{res}.pt", N_TOTAL, res)
    return tmp_path


def _make_dataset(root, n_train=4, n_tests=(3,), test_resolutions=(8,), **kwargs):
    return PTDataset(
        root_dir=root,
        dataset_name=DATASET_NAME,
        n_train=n_train,
        n_tests=list(n_tests),
        batch_size=1,
        test_batch_sizes=[1] * len(n_tests),
        train_resolution=8,
        test_resolutions=list(test_resolutions),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# torch-free loader
# ---------------------------------------------------------------------------


def test_load_pt_round_trip(dataset_dir) -> None:
    """_load_pt recovers tensors saved by torch.save as numpy arrays."""
    loaded = _load_pt((dataset_dir / f"{DATASET_NAME}_train_8.pt").as_posix())
    assert set(loaded.keys()) == {"x", "y"}
    expected_x = np.arange(N_TOTAL * 8 * 8, dtype=np.float32).reshape(N_TOTAL, 8, 8)
    assert np.allclose(loaded["x"], expected_x)
    assert np.allclose(loaded["y"], expected_x * 2.0 + 1.0)


# ---------------------------------------------------------------------------
# PTDataset contract (mirrors the reference dataset assertions)
# ---------------------------------------------------------------------------


def test_ptdataset_exposes_required_attrs(dataset_dir) -> None:
    """train_db, test_dbs and data_processor are populated (reference contract)."""
    from scirex.data.transforms.data_processors import DefaultDataProcessor

    ds = _make_dataset(dataset_dir)
    assert ds.train_db
    assert ds.test_dbs
    assert ds.data_processor is not None
    assert isinstance(ds.train_db, TensorDataset)
    assert isinstance(ds.data_processor, DefaultDataProcessor)


def test_ptdataset_counts_and_channel_dim(dataset_dir) -> None:
    """n_train/n_test slice sizes and the inserted channel dim are correct."""
    ds = _make_dataset(dataset_dir, n_train=4, n_tests=(3,), test_resolutions=(8,))
    assert len(ds.train_db) == 4
    assert len(ds.test_dbs[8]) == 3
    item = ds.train_db[0]
    # channels_squeezed=True (default) inserts a channel at dim 1
    assert item["x"].shape == (1, 8, 8)
    assert item["y"].shape == (1, 8, 8)
    assert item["x"].dtype == jnp.float32


def test_ptdataset_multiple_test_resolutions(dataset_dir) -> None:
    """test_dbs is keyed by each requested test resolution."""
    ds = _make_dataset(dataset_dir, n_train=4, n_tests=(3, 2), test_resolutions=(8, 16))
    assert set(ds.test_dbs.keys()) == {8, 16}
    assert ds.test_dbs[16][0]["x"].shape == (1, 16, 16)


@pytest.mark.parametrize("encode_input,encode_output", [(False, False), (True, True)])
def test_ptdataset_encoders(dataset_dir, encode_input, encode_output) -> None:
    """Encoder flags control whether the data_processor receives fitted normalizers."""
    ds = _make_dataset(
        dataset_dir, encode_input=encode_input, encode_output=encode_output
    )
    assert (ds.data_processor.in_normalizer is not None) == encode_input
    assert (ds.data_processor.out_normalizer is not None) == encode_output


def test_ptdataset_pixel_wise_encoding_and_str_root(dataset_dir) -> None:
    """pixel-wise encoding reduces only the batch dim; a str root_dir is accepted."""
    ds = PTDataset(
        root_dir=dataset_dir.as_posix(),  # str root_dir -> Path conversion branch
        dataset_name=DATASET_NAME,
        n_train=4,
        n_tests=[3],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=8,
        test_resolutions=[8],
        encode_input=True,
        encode_output=True,
        encoding="pixel-wise",
    )
    # pixel-wise reduces over axis [0] only, so the encoder mean keeps the spatial dims
    assert ds.data_processor.in_normalizer is not None
    assert ds.data_processor.out_normalizer is not None
    assert ds.data_processor.in_normalizer.mean.shape[0] == 1


def test_ptdataset_subsampling(dataset_dir) -> None:
    """Subsampling decimates the spatial axis (matches upstream ndim-2 indexing)."""
    ds = _make_dataset(dataset_dir, input_subsampling_rate=2, output_subsampling_rate=2)
    # Saved data is (N, H, W); upstream computes data_dims = ndim - 2 = 1, so a
    # single subsampling slice is applied -> one spatial axis is halved.
    assert ds.train_db[0]["x"].shape == (1, 4, 8)


# ---------------------------------------------------------------------------
# _rebuild_tensor rejects views that leave the storage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, shape, stride",
    [
        (0, (4, 4), (4, 1)),  # needs 16 elements, storage has 8
        (6, (4,), (1,)),  # offset pushes past the end
        (-1, (2,), (1,)),  # negative offset
        (0, (-2, 2), (2, 1)),  # negative extent
        (0, (2, 2), (-2, 1)),  # negative stride
        (0, (2, 2), (1,)),  # stride rank differs from shape
    ],
)
def test_rebuild_tensor_rejects_out_of_range_views(offset, shape, stride) -> None:
    from scirex.data.containers.pt_dataset import _rebuild_tensor

    storage = np.arange(8, dtype=np.float32)
    with pytest.raises(ValueError):
        _rebuild_tensor(storage, offset, shape, stride)


def test_rebuild_tensor_rejects_short_contiguous_storage() -> None:
    from scirex.data.containers.pt_dataset import _rebuild_tensor

    with pytest.raises(ValueError):
        _rebuild_tensor(np.arange(3, dtype=np.float32), 0, (2, 2))


def test_rebuild_tensor_valid_strided_view_still_works() -> None:
    from scirex.data.containers.pt_dataset import _rebuild_tensor

    storage = np.arange(8, dtype=np.float32)
    out = _rebuild_tensor(storage, 1, (2, 3), (3, 1))
    np.testing.assert_array_equal(out, storage[1:7].reshape(2, 3))
