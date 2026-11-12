# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""JAX/Flax loaders for the 2D Darcy flow dataset."""

import logging
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scirex._internal.paths import get_project_root
from scirex._internal.web_utils import download_from_zenodo_record

from ..containers.pt_dataset import PTDataset
from ..containers.tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor

logger = logging.Logger(name=__name__)
logger.setLevel(logging.root.level)


class DataLoader:
    """Simple batching data loader replacing torch.utils.data.DataLoader."""

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[dict[str, Any]]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.default_rng().shuffle(indices)
        for start in range(0, n, self.batch_size):
            yield self.dataset[indices[start : start + self.batch_size]]

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)


class DarcyDataset(PTDataset):
    """
    DarcyDataset stores data generated according to Darcy's Law.
    Input is a coefficient function and outputs describe flow.

    Data source: https://zenodo.org/records/12784353

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
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
    test_resolutions : sequence of int, optional
        resolution of data for testing sets, by default [16,32]
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
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    subsampling_rate : int or List[int], optional
        rate at which to subsample each input dimension, by default None
    download : bool, optional
        whether to download data if not present, by default True

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Path | str,
        n_train: int,
        n_tests: list[int],
        batch_size: int,
        test_batch_sizes: list[int],
        train_resolution: int,
        test_resolutions: Sequence[int] = (16, 32),
        encode_input: bool = False,
        encode_output: bool = True,
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        subsampling_rate: int | list[int] | None = None,
        download: bool = True,
    ) -> None:
        """Initialize the DarcyDataset.

        See class docstring for detailed parameter descriptions.
        """
        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        # Zenodo record ID for Darcy-Flow dataset
        zenodo_record_id = "12784353"

        # List of resolutions needed for dataset object
        resolutions = set([*test_resolutions, train_resolution])

        # We store data at these resolutions on the Zenodo archive
        available_resolutions = [16, 32, 64, 128, 421]
        for res in resolutions:
            assert (
                res in available_resolutions
            ), f"Error: resolution {res} not available"

        # download darcy data from zenodo archive if passed
        if download:
            files_to_download = []
            already_downloaded_files = [x.name for x in root_dir.iterdir()]
            for res in resolutions:
                if (
                    f"darcy_train_{res}.pt" not in already_downloaded_files
                    or f"darcy_test_{res}.pt" not in already_downloaded_files
                ):
                    files_to_download.append(f"darcy_{res}.tgz")
            download_from_zenodo_record(
                record_id=zenodo_record_id,
                root=root_dir,
                files_to_download=files_to_download,
            )

        # once downloaded/if files already exist, init PTDataset
        super().__init__(
            root_dir=root_dir,
            dataset_name="darcy",
            n_train=n_train,
            n_tests=n_tests,
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes,
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
            encode_input=encode_input,
            encode_output=encode_output,
            encoding=encoding,
            channel_dim=channel_dim,
            input_subsampling_rate=subsampling_rate,
            output_subsampling_rate=subsampling_rate,
        )


# legacy Small Darcy Flow example
example_data_root = get_project_root() / "neuralop/data/datasets/data"


def load_darcy_flow_small(
    n_train: int,
    n_tests: list[int],
    batch_size: int,
    test_batch_sizes: list[int],
    data_root: Path | str = example_data_root,
    test_resolutions: Sequence[int] = (16, 32),
    encode_input: bool = False,
    encode_output: bool = True,
    encoding: str = "channel-wise",
    channel_dim: int = 1,
) -> tuple[DataLoader, dict[int, DataLoader], DefaultDataProcessor]:
    """Load the small 2D Darcy flow dataset (16x16 training resolution).

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_tests : list of int
        Number of test samples for each test resolution.
    batch_size : int
        Batch size of the training loader.
    test_batch_sizes : list of int
        Batch size of the test loader for each test resolution.
    data_root : Path or str, optional
        Directory holding (or receiving) the data files; missing files are
        downloaded from Zenodo.
    test_resolutions : sequence of int, optional
        Resolutions of the test sets, by default `[16, 32]`.
    encode_input : bool, optional
        Whether the returned data processor normalizes inputs, by default False.
    encode_output : bool, optional
        Whether the returned data processor normalizes outputs, by default True.
    encoding : str, optional
        `"channel-wise"` or `"pixel-wise"` normalization, by default channel-wise.
    channel_dim : int, optional
        Axis that holds data channels, by default 1.

    Returns
    -------
    train_loader : DataLoader
        Batches of the training set.
    test_loaders : dict of int to DataLoader
        Test loaders keyed by test resolution.
    data_processor : DefaultDataProcessor
        Processor holding the fitted normalizers.
    """
    dataset = DarcyDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=16,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
        download=True,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders: dict[int, DataLoader] = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor


# legacy pt Darcy Flow loader
def load_darcy_pt(
    n_train: int,
    n_tests: list[int],
    batch_size: int,
    test_batch_sizes: list[int],
    data_root: Path | str = "./neuralop/data/datasets/data",
    train_resolution: int = 16,
    test_resolutions: Sequence[int] = (16, 32),
    encode_input: bool = False,
    encode_output: bool = True,
    encoding: str = "channel-wise",
    channel_dim: int = 1,
    num_workers: int = 1,
) -> tuple[DataLoader, dict[int, DataLoader], DefaultDataProcessor]:
    """Load the 2D Darcy flow dataset from local `.pt` files (no download).

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_tests : list of int
        Number of test samples for each test resolution.
    batch_size : int
        Batch size of the training loader.
    test_batch_sizes : list of int
        Batch size of the test loader for each test resolution.
    data_root : Path or str, optional
        Directory that already contains the `darcy_train_<res>.pt` and
        `darcy_test_<res>.pt` files.
    train_resolution : int, optional
        Resolution of the training set, by default 16.
    test_resolutions : sequence of int, optional
        Resolutions of the test sets, by default `[16, 32]`.
    encode_input : bool, optional
        Whether the returned data processor normalizes inputs, by default False.
    encode_output : bool, optional
        Whether the returned data processor normalizes outputs, by default True.
    encoding : str, optional
        `"channel-wise"` or `"pixel-wise"` normalization, by default channel-wise.
    channel_dim : int, optional
        Axis that holds data channels, by default 1.
    num_workers : int, optional
        Accepted for API compatibility; the loader batches in-process. By default 1.

    Returns
    -------
    train_loader : DataLoader
        Batches of the training set.
    test_loaders : dict of int to DataLoader
        Test loaders keyed by test resolution.
    data_processor : DefaultDataProcessor
        Processor holding the fitted normalizers.
    """
    dataset = DarcyDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
        download=False,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders: dict[int, DataLoader] = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor
