# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Data processors that wrap a model to apply (inverse) normalization and patching
during training.
"""

import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class DataProcessor(nn.Module, metaclass=ABCMeta):
    """Base class of the processors that prepare data for a model and decode its output.

    A processor implements `preprocess` (applied to a batch before the forward pass),
    `postprocess` (applied to the output before the loss) and `wrap`, which joins a model
    with both steps into one forward pass. Subclass it to add your own pre- and
    post-processing; `DefaultDataProcessor` is the ready-made implementation with
    optional input and output normalization.
    """

    def setup(self) -> None:
        """
        DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with JAX convention (JAX handles device placement automatically)
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - __call__(self, x, training=True):
            forward pass providing that a model has been wrapped
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "DataProcessor":
        pass

    @abstractmethod
    def preprocess(self, x: dict) -> dict:
        pass

    # Every concrete processor (and every trainer.py call site) invokes
    # postprocess(out, data_dict, training=...), not the single-arg
    # postprocess(x) this signature previously declared -- widened to match
    # the real contract.
    @abstractmethod
    def postprocess(
        self, out: jnp.ndarray, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        pass

    # default wrap method
    def wrap(self, model: nn.Module) -> "DataProcessor":
        self.model = model
        return self

    @abstractmethod
    def __call__(self, x: dict, training: bool = True) -> tuple[jnp.ndarray, dict]:
        pass


class DefaultDataProcessor:
    """DefaultDataProcessor is a simple processor
    to pre/post process data before training/inferencing a model.

    Plain Python class (not a Flax nn.Module) so that training mode can be
    tracked as ordinary mutable state via train() / eval() calls.

    Parameters
    ----------
    in_normalizer : Transform, optional, default is None
        normalizer (e.g. UnitGaussianNormalizer) for the input samples
    out_normalizer : Transform, optional, default is None
        normalizer (e.g. UnitGaussianNormalizer) for the target and predicted samples
    """

    def __init__(
        self, in_normalizer: Any | None = None, out_normalizer: Any | None = None
    ) -> None:
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.training = True
        self.device = "cpu"
        self.model: nn.Module | None = None

    def train(self) -> None:
        """Switch to training mode (targets are normalized in preprocessing)."""
        self.training = True

    def eval(self) -> None:
        """Switch to evaluation mode (outputs are denormalized in postprocessing)."""
        self.training = False

    def to(self, device: str) -> "DefaultDataProcessor":
        """Record the target device name and return `self`."""
        self.device = device
        return self

    def wrap(self, model: nn.Module) -> "DefaultDataProcessor":
        """Attach a model so the processor can run a full forward pass via __call__."""
        self.model = model
        return self

    def preprocess(self, data_dict: dict, batched: bool = True) -> dict:
        """Pre-process a batch: convert to JAX arrays and apply input normalization.

        During training also normalizes the target y so the loss is computed in
        normalized space.

        Parameters
        ----------
        data_dict : dict
            Must contain keys 'x' and 'y'.
        batched : bool, optional
            Unused; kept for API compatibility, by default True

        Returns
        -------
        dict
            Preprocessed data_dict with JAX arrays.
        """
        x = jnp.asarray(data_dict["x"])
        y = jnp.asarray(data_dict["y"])

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict

    def postprocess(
        self, output: jnp.ndarray, data_dict: dict, training: bool | None = None
    ) -> tuple[jnp.ndarray, dict]:
        """Post-process model output: denormalize during evaluation.

        Parameters
        ----------
        output : jnp.ndarray
        data_dict : dict
        training : bool or None, optional
            If None (default), uses self.training set by train()/eval() calls.

        Returns
        -------
        output, data_dict
        """
        use_training = training if training is not None else self.training
        if self.out_normalizer is not None and not use_training:
            output = self.out_normalizer.inverse_transform(output)
        return output, data_dict

    def __call__(
        self, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        """Run preprocessing, the wrapped model and postprocessing on a batch.

        Parameters
        ----------
        data_dict : dict
            Batch with keys `'x'` and `'y'`.
        training : bool, optional
            Sets the processor's training mode for this call, by default True.

        Returns
        -------
        output, data_dict
            Model output (denormalized unless `training`) and the preprocessed batch.
        """
        self.training = training
        data_dict = self.preprocess(data_dict)
        assert self.model is not None, (
            "Error: DefaultDataProcessor.__call__ requires a model attached via"
            " wrap(model) before it can run a forward pass"
        )
        output = self.model(data_dict["x"])
        output, data_dict = self.postprocess(output, data_dict, training=training)
        return output, data_dict


class IncrementalDataProcessor(nn.Module):
    """Data processor that raises the input resolution during training (curriculum learning).

    Training starts on subsampled inputs and moves to finer resolutions every `epoch_gap`
    epochs, which makes training on large high-resolution data cheaper.
    """

    def __init__(
        self,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
        device: str = "cpu",
        subsampling_rates: Sequence[int] = (2, 1),
        dataset_resolution: int = 16,
        dataset_indices: Sequence[int] = (2, 3),
        epoch_gap: int = 10,
        verbose: bool = False,
    ) -> None:
        """An incremental processor to pre/post process data before
        training/inferencing a model. In particular this processor first
        regularizes the input resolution based on the sub_list and
        dataset_indices in the spatial domain based on a fixed number of
        epochs. We incrementally increase the resolution like done in
        curriculum learning to train the model. This is useful for training
        models on large datasets with high resolution data.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        device : str, optional, default is 'cpu'
            device 'cuda' or 'cpu' where computations are performed
        subsampling_rates : sequence of int, optional, default is (2, 1)
            list of subsampling rates to use
        dataset_resolution : int, optional, default is 16
            resolution of the input data
        dataset_indices : sequence of int, optional, default is (2, 3)
            list of indices of the dataset to slice to regularize the input
            resolution - Spatial Dimensions
        epoch_gap : int, optional, default is 10
            number of epochs to wait before increasing the resolution
        verbose : bool, optional, default is False
            if True, print the current resolution
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = device
        self.sub_list = subsampling_rates
        self.dataset_resolution = dataset_resolution
        self.dataset_indices = dataset_indices
        self.epoch_gap = epoch_gap
        self.verbose = verbose
        self.epoch = 0
        self.model: nn.Module | None = None

        self.current_index = 0
        self.current_logged_epoch = 0
        self.current_sub = self.index_to_sub_from_table(self.current_index)
        self.current_res = int(self.dataset_resolution / self.current_sub)

        logger.info(f"Original Incre Res: change index to {self.current_index}")
        logger.info(f"Original Incre Res: change sub to {self.current_sub}")
        logger.info(f"Original Incre Res: change res to {self.current_res}")

    def to(self, device: str) -> "IncrementalDataProcessor":
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def wrap(self, model: nn.Module) -> "IncrementalDataProcessor":
        """Attach a model so the processor can run a full forward pass via __call__."""
        self.model = model
        return self

    def epoch_wise_res_increase(self, epoch: int) -> None:
        # Update the current_sub and current_res values based on the epoch
        if (
            epoch % self.epoch_gap == 0
            and epoch != 0
            and (self.current_logged_epoch != epoch)
        ):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = int(self.dataset_resolution / self.current_sub)
            self.current_logged_epoch = epoch

            if self.verbose:
                logger.info(f"Incre Res Update: change index to {self.current_index}")
                logger.info(f"Incre Res Update: change sub to {self.current_sub}")
                logger.info(f"Incre Res Update: change res to {self.current_res}")

    def index_to_sub_from_table(self, index: int) -> int:
        # Get the sub value from the sub_list based on the index
        if index >= len(self.sub_list):
            return self.sub_list[-1]
        else:
            return self.sub_list[index]

    def regularize_input_res(
        self, x: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Regularize the input data based on the current_sub and dataset_name
        for idx in self.dataset_indices:
            indexes = jnp.arange(0, x.shape[idx], self.current_sub)
            x = jnp.take(x, indexes, axis=idx)
            y = jnp.take(y, indexes, axis=idx)
        return x, y

    def step(
        self,
        loss: float | None = None,
        epoch: int | None = None,
        x: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        if x is not None and y is not None:
            assert (
                epoch is not None
            ), "Error: step() requires an epoch whenever x and y are provided"
            self.epoch_wise_res_increase(epoch)
            return self.regularize_input_res(x, y)
        return None

    def preprocess(self, data_dict: dict, batched: bool = True) -> dict:
        x = jnp.asarray(data_dict["x"])
        y = jnp.asarray(data_dict["y"])

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        if self.training:
            stepped = self.step(epoch=self.epoch, x=x, y=y)
            assert stepped is not None
            x, y = stepped

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(
        self, output: jnp.ndarray, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        y = data_dict["y"]
        if self.out_normalizer and not training:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["y"] = y
        return output, data_dict

    def __call__(
        self, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        self.training = training
        data_dict = self.preprocess(data_dict)
        assert self.model is not None, (
            "Error: IncrementalDataProcessor.__call__ requires a model attached"
            " via wrap(model) before it can run a forward pass"
        )
        output = self.model(data_dict["x"])
        output, data_dict = self.postprocess(output, data_dict, training=training)
        return output, data_dict
