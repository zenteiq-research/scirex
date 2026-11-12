# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit-Gaussian normalizers for single- and dictionary-valued data."""

from collections.abc import Iterable, Sequence

import jax.numpy as jnp
from flax import struct

from ...utils import count_tensor_params


@struct.dataclass
class Normalizer:
    """Normalization by given statistics.

    `transform` returns `(data - mean) / (std + eps)` and `inverse_transform` undoes it.
    Use `UnitGaussianNormalizer` to compute the statistics from data.

    Parameters
    ----------
    mean : jnp.ndarray
        Mean to subtract.
    std : jnp.ndarray
        Standard deviation to divide by.
    eps : float, optional
        Added to `std` to avoid division by zero. Default: 1e-6.
    """

    mean: jnp.ndarray
    std: jnp.ndarray
    eps: float = 1e-6

    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: jnp.ndarray) -> jnp.ndarray:
        return (data * (self.std + self.eps)) + self.mean


@struct.dataclass
class UnitGaussianNormalizer:
    """
    Normalizes data to zero mean and unit std.
    Implemented as a Flax PyTreeNode for functional purity and JAX compatibility.

    Every method that updates statistics returns a new instance; the original is
    never modified. Create one with `create`, fit it with `fit` or
    `partial_fit`, then use `transform` and `inverse_transform`.
    """

    # Dynamic variables (JAX arrays) that update during fitting
    mean: jnp.ndarray | None = None
    std: jnp.ndarray | None = None
    squared_mean: jnp.ndarray | None = None
    n_elements: int | jnp.ndarray = 0
    mask: jnp.ndarray | None = None

    # Static parameters (Excluded from JAX's state tree)
    eps: float = struct.field(pytree_node=False, default=1e-7)
    axis: Sequence[int] | None = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        eps: float = 1e-7,
        axis: int | Sequence[int] | None = None,
        mask: jnp.ndarray | None = None,
    ) -> "UnitGaussianNormalizer":
        """Initialize the normalizer."""
        if isinstance(axis, int):
            axis = (axis,)
        elif axis is not None:
            axis = tuple(axis)
        return cls(eps=eps, axis=axis, mask=mask)

    def fit(self, data_batch: jnp.ndarray) -> "UnitGaussianNormalizer":
        """Returns a NEW instance of the normalizer with updated stats."""
        return self.update_mean_std(data_batch)

    def partial_fit(
        self, data_batch: jnp.ndarray, batch_size: int = 1
    ) -> "UnitGaussianNormalizer":
        """Incrementally fits the normalizer and returns the updated instance."""
        if 0 in data_batch.shape:
            return self

        count = 0
        n_samples = len(data_batch)
        state = self  # We must carry the new state through the loop

        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            if state.n_elements > 0:
                state = state.incremental_update_mean_std(samples)
            else:
                state = state.update_mean_std(samples)
            count += batch_size

        return state

    def update_mean_std(self, data_batch: jnp.ndarray) -> "UnitGaussianNormalizer":
        """Compute the statistics of `data_batch` from scratch.

        Parameters
        ----------
        data_batch : jnp.ndarray
            Data to take the mean and standard deviation of, reduced over `axis`.

        Returns
        -------
        UnitGaussianNormalizer
            A new normalizer holding the mean, std, squared mean and element
            count of `data_batch`. Previously accumulated statistics are discarded.
        """
        n_elements: int | jnp.ndarray
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.axis)
            mean = jnp.mean(data_batch, axis=self.axis, keepdims=True)
            squared_mean = jnp.mean(data_batch**2, axis=self.axis, keepdims=True)
            std = jnp.std(data_batch, axis=self.axis, keepdims=True)
        else:
            assert self.axis is not None, (
                "Error: axis must be provided (via create()) whenever mask is set,"
                " since masked reduction needs to know which axes to reduce over"
            )
            batch_size = data_batch.shape[0]
            axis_no_batch = tuple(i - 1 for i in self.axis if i > 0)

            # Pure equivalent of: data_batch[:, self.mask == 1] = 0
            masked_data = jnp.where(self.mask == 1, 0.0, data_batch)

            n_elements = jnp.count_nonzero(self.mask, axis=axis_no_batch) * batch_size

            mean = jnp.sum(masked_data, axis=axis_no_batch, keepdims=True) / n_elements
            squared_mean = (
                jnp.sum(masked_data**2, axis=axis_no_batch, keepdims=True) / n_elements
            )
            std = jnp.std(data_batch, axis=self.axis, keepdims=True)

        # Instead of `self.mean = mean`, we return an updated copy (Functional Purity)
        # flax.struct.dataclass injects `replace()` dynamically; mypy can't see it.
        return self.replace(  # type: ignore[attr-defined]
            mean=mean, std=std, squared_mean=squared_mean, n_elements=n_elements
        )

    def incremental_update_mean_std(
        self, data_batch: jnp.ndarray
    ) -> "UnitGaussianNormalizer":
        """Merge the statistics of `data_batch` into the existing ones.

        Requires statistics from an earlier `update_mean_std` call (as in
        `partial_fit`).

        Parameters
        ----------
        data_batch : jnp.ndarray
            New samples to fold into the running mean and standard deviation.

        Returns
        -------
        UnitGaussianNormalizer
            A new normalizer with the combined statistics and element count.
        """
        assert self.mean is not None and self.squared_mean is not None, (
            "Error: incremental_update_mean_std requires stats from a prior"
            " update_mean_std call (reached via partial_fit once n_elements > 0)"
        )
        n_elements: int | jnp.ndarray
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.axis)
            axis = self.axis
            masked_data = data_batch
        else:
            assert self.axis is not None, (
                "Error: axis must be provided (via create()) whenever mask is set,"
                " since masked reduction needs to know which axes to reduce over"
            )
            axis = tuple(i - 1 for i in self.axis if i > 0)
            n_elements = jnp.count_nonzero(self.mask, axis=axis) * data_batch.shape[0]
            masked_data = jnp.where(self.mask == 1, 0.0, data_batch)

        total_elements = self.n_elements + n_elements

        mean = (1.0 / total_elements) * (
            self.n_elements * self.mean + jnp.sum(masked_data, axis=axis, keepdims=True)
        )
        squared_mean = (1.0 / total_elements) * (
            self.n_elements * self.squared_mean
            + jnp.sum(masked_data**2, axis=axis, keepdims=True)
        )

        # jnp.maximum avoids negative sqrt input from float precision limits
        variance = jnp.maximum(0.0, squared_mean - mean**2)
        std = jnp.sqrt(variance) * (
            total_elements / jnp.maximum(1.0, total_elements - 1)
        )

        return self.replace(  # type: ignore[attr-defined]
            mean=mean, squared_mean=squared_mean, std=std, n_elements=total_elements
        )

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalize `x` to `(x - mean) / (std + eps)`."""
        assert self.mean is not None and self.std is not None, (
            "Error: normalizer must be fit (via fit()/partial_fit()) or"
            " constructed with explicit mean/std before calling transform()"
        )
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Undo `transform`: return `x * (std + eps) + mean`."""
        assert self.mean is not None and self.std is not None, (
            "Error: normalizer must be fit (via fit()/partial_fit()) or"
            " constructed with explicit mean/std before calling inverse_transform()"
        )
        return x * (self.std + self.eps) + self.mean

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Alias for `transform`."""
        return self.transform(x)

    @classmethod
    def from_dataset(
        cls,
        dataset: Iterable[dict],
        axis: int | Sequence[int] | None = None,
        keys: Iterable[str] | None = None,
        mask: jnp.ndarray | None = None,
    ) -> dict[str, "UnitGaussianNormalizer"]:
        """Fit one normalizer per dictionary key by streaming over `dataset`.

        Parameters
        ----------
        dataset : iterable of dict
            Samples, each a dict of arrays (a sample is fit as a batch of one).
        axis : int or sequence of int, optional
            Axes to reduce over, see `create`.
        keys : iterable of str, optional
            Entries to normalize. Defaults to all keys of the first sample.
        mask : jnp.ndarray, optional
            Mask passed to every normalizer, see `create`.

        Returns
        -------
        dict of str to UnitGaussianNormalizer
            One fitted normalizer per key.
        """
        instances: dict[str, UnitGaussianNormalizer] = {}
        resolved_keys: list[str] | None = list(keys) if keys is not None else None
        for i, data_dict in enumerate(dataset):
            if i == 0:
                if resolved_keys is None:
                    resolved_keys = list(data_dict.keys())
                instances = {
                    key: cls.create(axis=axis, mask=mask) for key in resolved_keys
                }
            assert resolved_keys is not None

            for key, sample in data_dict.items():
                if key in resolved_keys:
                    # sample.unsqueeze(0) -> jnp.expand_dims
                    sample_batched = jnp.expand_dims(jnp.asarray(sample), axis=0)
                    # Must re-assign to capture the updated state
                    instances[key] = instances[key].partial_fit(sample_batched)
        return instances


@struct.dataclass
class DictUnitGaussianNormalizer:
    """
    Composes multiple UnitGaussianNormalizers to normalize different
    fields of a tensor to Gaussian distributions with mean 0 and unit variance.
    """

    # JAX natively treats dictionaries of PyTrees as PyTrees themselves!
    normalizer_dict: dict[str, "UnitGaussianNormalizer"]

    # Python slice objects cannot be JAX arrays, so they MUST be marked as static fields
    input_mappings: dict[str, slice] = struct.field(pytree_node=False)
    return_mappings: dict[str, slice] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        normalizer_dict: dict[str, "UnitGaussianNormalizer"],
        input_mappings: dict[str, slice],
        return_mappings: dict[str, slice],
    ) -> "DictUnitGaussianNormalizer":
        assert set(normalizer_dict.keys()) == set(
            input_mappings.keys()
        ), "Error: normalizers and model input fields must be keyed identically"
        assert set(normalizer_dict.keys()) == set(
            return_mappings.keys()
        ), "Error: normalizers and model output fields must be keyed identically"

        return cls(
            normalizer_dict=normalizer_dict,
            input_mappings=input_mappings,
            return_mappings=return_mappings,
        )

    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies normalizations to specific slices of the tensor."""
        result = x
        for key, norm in self.normalizer_dict.items():
            in_slice = self.input_mappings[key]
            out_slice = self.return_mappings[key]

            # Extract the slice, transform it, and place it in the new tensor
            normalized_field = norm.transform(x[in_slice])
            result = result.at[out_slice].set(normalized_field)

        return result

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reverses the normalization on specific slices of the tensor."""
        result = x
        for key, norm in self.normalizer_dict.items():
            in_slice = self.input_mappings[key]
            out_slice = self.return_mappings[key]

            # Assuming return_mappings are the locations in the output tensor
            denormalized_field = norm.inverse_transform(x[out_slice])
            result = result.at[in_slice].set(denormalized_field)

        return result

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)

    @classmethod
    def from_dataset(
        cls,
        dataset: Iterable[dict],
        input_mappings: dict[str, slice],
        return_mappings: dict[str, slice],
        axis: int | Sequence[int] | None = None,
        keys: Iterable[str] | None = None,
        mask: jnp.ndarray | None = None,
    ) -> "DictUnitGaussianNormalizer":
        """
        Initializes and functionally fits the composite normalizer
        using data from a dataset.
        """
        # Single pass over dataset: a `next(iter(dataset))` peek to determine
        # keys, followed by a separate `for data_dict in dataset` loop, would
        # silently drop the first batch from fitting whenever dataset is a
        # single-pass iterator (e.g. a generator). Determine keys from the
        # first item of the same loop that fits, matching the sibling
        # UnitGaussianNormalizer.from_dataset above.
        base_instances: dict[str, UnitGaussianNormalizer] = {}
        resolved_keys: list[str] | None = list(keys) if keys is not None else None
        for i, data_dict in enumerate(dataset):
            if i == 0:
                if resolved_keys is None:
                    resolved_keys = [k for k in data_dict.keys() if k in input_mappings]
                base_instances = {
                    key: UnitGaussianNormalizer.create(axis=axis, mask=mask)
                    for key in resolved_keys
                }
            assert resolved_keys is not None

            for key in resolved_keys:
                if key in data_dict:
                    sample = jnp.asarray(data_dict[key])
                    sample_batched = jnp.expand_dims(sample, axis=0)

                    # Functional update: Must re-assign the returned updated object
                    base_instances[key] = base_instances[key].partial_fit(
                        sample_batched
                    )

        return cls.create(
            normalizer_dict=base_instances,
            input_mappings=input_mappings,
            return_mappings=return_mappings,
        )
