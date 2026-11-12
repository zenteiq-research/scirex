# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Base PTDataset class and a torch-free reader for PyTorch `.pt` data archives."""

import io
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import jax.numpy as jnp
import numpy as np

from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer
from .tensor_dataset import TensorDataset

logger = logging.getLogger(__name__)

# ── Torch-free .pt file loader ────────────────────────────────────────────────
# PyTorch saves .pt files as zip archives. Storages are referenced via
# persistent_load (not find_class), so we override that to deserialize
# directly as numpy arrays - no torch dependency needed.

_STORAGE_DTYPES: dict[str, type[np.generic]] = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "BFloat16Storage": np.float32,  # upcast: numpy has no bfloat16
}

# torch >= 2.0 stores explicit dtype strings in the persistent id
_TORCH_DTYPE_MAP: dict[str, type[np.generic]] = {
    "torch.float32": np.float32,
    "torch.float": np.float32,
    "torch.float64": np.float64,
    "torch.double": np.float64,
    "torch.float16": np.float16,
    "torch.half": np.float16,
    "torch.int64": np.int64,
    "torch.long": np.int64,
    "torch.int32": np.int32,
    "torch.int": np.int32,
    "torch.int16": np.int16,
    "torch.short": np.int16,
    "torch.uint8": np.uint8,
    "torch.bool": np.bool_,
    "torch.bfloat16": np.float32,
}

# Globals a tensor archive may reference besides those handled in find_class.
_ALLOWED_GLOBALS = {("collections", "OrderedDict")}


def _rebuild_tensor(
    storage: np.ndarray, offset: int, shape: Any, *args: Any
) -> np.ndarray:
    # args: stride, requires_grad, backward_hooks, optional metadata
    shape = tuple(int(s) for s in shape)
    offset = int(offset)
    stride = tuple(int(s) for s in args[0]) if args and args[0] else None
    if offset < 0 or any(n < 0 for n in shape):
        raise ValueError(f"Invalid tensor in .pt file: offset={offset}, shape={shape}")
    if stride is not None and shape:
        if len(stride) != len(shape) or any(st < 0 for st in stride):
            raise ValueError(
                f"Invalid tensor in .pt file: shape={shape}, strides={stride}"
            )
        if all(n > 0 for n in shape):
            last = offset + sum((n - 1) * st for n, st in zip(shape, stride))
            if last >= storage.size:
                raise ValueError(
                    "Tensor in .pt file reads outside its storage: "
                    f"needs element {last}, storage has {storage.size}"
                )
        byte_strides = tuple(s * storage.dtype.itemsize for s in stride)
        arr = np.lib.stride_tricks.as_strided(
            storage[offset:], shape=shape, strides=byte_strides
        )
        return arr.copy()
    size = int(np.prod(shape)) if shape else 1
    window = storage[offset : offset + size]
    if window.size != size:
        raise ValueError(
            "Tensor in .pt file reads outside its storage: "
            f"needs {size} elements from {offset}, storage has {storage.size}"
        )
    return window.reshape(shape).copy()


class _PtUnpickler(pickle.Unpickler):
    def __init__(self, file: BinaryIO, raw_storages: dict[str, bytes]) -> None:
        super().__init__(file)
        self._raw = raw_storages

    def persistent_load(self, pid: Any) -> np.ndarray:
        # Classic format: ("storage", StorageCls, key, location, numel)
        # New format (torch>=2): ("storage", TypedStorage, key, location, numel, dtype)
        storage_cls, key = pid[1], str(pid[2])
        if len(pid) >= 6:
            dtype = _TORCH_DTYPE_MAP.get(str(pid[5]), np.float32)
        else:
            dtype = _STORAGE_DTYPES.get(
                getattr(storage_cls, "__name__", ""), np.float32
            )
        if key not in self._raw:
            raise ValueError(f"Storage {key!r} is missing from the .pt archive")
        return np.frombuffer(self._raw[key], dtype=dtype).copy()

    def find_class(self, module: str, name: str) -> Any:
        if name in _STORAGE_DTYPES and module in ("torch", "torch.storage", "torch._C"):

            class _S:
                pass

            _S.__name__ = name
            return _S
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name == "Size":
            return tuple
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked global {module}.{name} in .pt file")


def _load_pt(path: str) -> dict:
    """Load a PyTorch .pt zip archive and return tensors as numpy arrays."""
    with zipfile.ZipFile(path) as zf:
        prefix = zf.namelist()[0].split("/")[0]
        raw: dict[str, bytes] = {}
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[1] == "data":
                with zf.open(name) as f:
                    raw[parts[2]] = f.read()
        with zf.open(f"{prefix}/data.pkl") as f:
            return _PtUnpickler(io.BytesIO(f.read()), raw).load()


# ─────────────────────────────────────────────────────────────────────────────


class PTDataset:
    """PTDataset is a base Dataset class for our library.
        PTDatasets contain input-output pairs a(x), u(x) and may also
        contain additional information, e.g. function parameters,
        input geometry or output query points.

        datasets may implement a download flag at init, which provides
        access to a number of premade datasets for sample problems provided
        in our Zenodo archive.

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    dataset_name : str
        prefix of pt data files to store/access
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_sizes : List[int]
        batch size of test sets
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int]
        resolution of data for testing sets
    encode_input : bool, optional
        whether to normalize inputs in provided DataProcessor,
        by default False
    encode_output : bool, optional
        whether to normalize outputs in provided DataProcessor,
        by default True
    encoding : str, optional
        parameter for input/output normalization. Whether
        to normalize by channel ("channel-wise") or
        by pixel ("pixel-wise"), default "channel-wise"
    input_subsampling_rate : int or List[int], optional
        rate at which to subsample each input dimension, by default None
    output_subsampling_rate : int or List[int], optional
        rate at which to subsample each output dimension, by default None
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    channels_squeezed : bool, optional
        If the channels dim is 1, whether that is explicitly kept in the saved tensor.
        If not, we need to unsqueeze it to explicitly have a channel dim.
        Only applies when there is only one data channel, as in our example problems
        Defaults to True

    All datasets are required to expose the following attributes after init:

    train_db: torch.utils.data.Dataset of training examples
    test_db: torch.utils.data.Dataset of test examples
    data_processor: scirex.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Path | str,
        dataset_name: str,
        n_train: int,
        n_tests: list[int],
        batch_size: int,
        test_batch_sizes: list[int],
        train_resolution: int,
        test_resolutions: list[int],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        input_subsampling_rate: int | list[int] | None = None,
        output_subsampling_rate: int | list[int] | None = None,
        channel_dim: int = 1,
        channels_squeezed: bool = True,
    ) -> None:
        """Initialize the PTDataset.

        See class docstring for detailed parameter descriptions.
        """
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        # Load train data

        data = _load_pt(
            Path(root_dir)
            .joinpath(f"{dataset_name}_train_{train_resolution}.pt")
            .as_posix()
        )

        x_train = jnp.array(data["x"], dtype=jnp.float32)
        if channels_squeezed:
            x_train = jnp.expand_dims(x_train, axis=channel_dim)

        # optionally subsample along data indices
        ## Input subsampling
        input_data_dims = data["x"].ndim - 2  # batch and channels
        # convert None and 0 to 1
        if not input_subsampling_rate:
            input_subsampling_rate = 1
        if not isinstance(input_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            input_subsampling_rate = [input_subsampling_rate] * input_data_dims
        # make sure there is one subsampling rate per data dim
        assert len(input_subsampling_rate) == input_data_dims, (
            "Error: length mismatch between input_subsampling_rate and "
            "dimensions of data. input_subsampling_rate must be one int "
            "shared across all dims, or an iterable of length "
            f"{input_data_dims}, got {input_subsampling_rate}"
        )
        # Construct full indices along which to grab X
        train_input_indices_list = [slice(0, n_train, None)] + [
            slice(None, None, rate) for rate in input_subsampling_rate
        ]
        train_input_indices_list.insert(channel_dim, slice(None))
        train_input_indices = tuple(train_input_indices_list)
        x_train = x_train[train_input_indices]

        y_train = jnp.array(data["y"])
        if channels_squeezed:
            y_train = jnp.expand_dims(y_train, axis=channel_dim)

        ## Output subsampling
        output_data_dims = data["y"].ndim - 2
        # convert None and 0 to 1
        if not output_subsampling_rate:
            output_subsampling_rate = 1
        if not isinstance(output_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            output_subsampling_rate = [output_subsampling_rate] * output_data_dims
        # make sure there is one subsampling rate per data dim
        assert len(output_subsampling_rate) == output_data_dims, (
            "Error: length mismatch between output_subsampling_rate and "
            "dimensions of data. input_subsampling_rate must be one int "
            "shared across all dims, or an iterable of length "
            f"{output_data_dims}, got {output_subsampling_rate}"
        )

        # Construct full indices along which to grab Y
        train_output_indices_list = [slice(0, n_train, None)] + [
            slice(None, None, rate) for rate in output_subsampling_rate
        ]
        train_output_indices_list.insert(channel_dim, slice(None))
        train_output_indices = tuple(train_output_indices_list)
        y_train = y_train[train_output_indices]

        del data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer.create(axis=tuple(reduce_dims))
            input_encoder = input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer.create(axis=tuple(reduce_dims))
            output_encoder = output_encoder.fit(y_train)
        else:
            output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        # create DataProcessor
        self._data_processor = DefaultDataProcessor(
            in_normalizer=input_encoder, out_normalizer=output_encoder
        )

        # load test data
        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            logger.info(f"Loading test db for resolution {res} with {n_test} samples ")
            data = _load_pt(
                Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt").as_posix()
            )

            x_test = jnp.array(data["x"], dtype=jnp.float32)
            if channels_squeezed:
                x_test = jnp.expand_dims(x_test, axis=channel_dim)
            # optionally subsample along data indices
            test_input_indices_list = [slice(0, n_test, None)] + [
                slice(None, None, rate) for rate in input_subsampling_rate
            ]
            test_input_indices_list.insert(channel_dim, slice(None))
            test_input_indices = tuple(test_input_indices_list)
            x_test = x_test[test_input_indices]

            y_test = jnp.array(data["y"])
            if channels_squeezed:
                y_test = jnp.expand_dims(y_test, axis=channel_dim)
            test_output_indices_list = [slice(0, n_test, None)] + [
                slice(None, None, rate) for rate in output_subsampling_rate
            ]
            test_output_indices_list.insert(channel_dim, slice(None))
            test_output_indices = tuple(test_output_indices_list)
            y_test = y_test[test_output_indices]

            del data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self) -> DefaultDataProcessor:
        return self._data_processor

    @property
    def train_db(self) -> TensorDataset:
        return self._train_db

    @property
    def test_dbs(self) -> dict[int, TensorDataset]:
        return self._test_dbs
