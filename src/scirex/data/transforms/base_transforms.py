# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Base Transform classes, including composite and dictionary-valued transforms."""

from abc import abstractmethod

import jax.numpy as jnp
from flax import linen as nn


class Transform(nn.Module):
    """
    Applies transforms or inverse transforms to
    model inputs or outputs, respectively
    """

    @abstractmethod
    def transform(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def to(self, device: str) -> "Transform":
        # In JAX, device placement is handled via jax.device_put
        # Subclasses can override if needed
        return self

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)


class CompositeTransform(Transform):
    """Composite transform composes a list of
    Transforms into one Transform object.

    Transformations are not assumed to be commutative

    Parameters
    ----------
    transforms : List[Transform]
        list of transforms to be applied to data
        in order
    """

    transforms: list

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)

    def transform(self, data_dict: jnp.ndarray) -> jnp.ndarray:
        for tform in self.transforms:
            data_dict = tform.transform(data_dict)
        return data_dict

    def inverse_transform(self, data_dict: jnp.ndarray) -> jnp.ndarray:
        for tform in self.transforms[::-1]:
            data_dict = tform.inverse_transform(data_dict)
        return data_dict

    def to(self, device: str) -> "CompositeTransform":
        # JAX device placement: apply jax.device_put to each transform's arrays
        # Flax modules are stateless; parameters are moved via jax.device_put
        # on param dicts
        return self


class DictTransform(Transform):
    """When a model has multiple input and output fields,
        apply a different transform to each field,
        tries to apply the inverse_transform to each output

    Parameters
    ----------
    transforms : dict
        dictionary of output encoders (keyed by field name)
    input_mappings : dict[tuple(slice)]
        indices of an output array x to use for
        each field, such that x[mappings[field]]
        returns the correct slice of x.
    return_mappings : dict[tuple(slice)], optional
        same as above. if only certain indices
        of encoder output are important, this indexes those.
    """

    transforms: dict
    input_mappings: dict
    return_mappings: dict | None = None

    def setup(self) -> None:
        assert self.transforms.keys() == self.input_mappings.keys(), (
            f"Error: expected keys in transforms and input_mappings to match,"
            f" received {list(self.transforms.keys())=}\n"
            f"{list(self.input_mappings.keys())=}"
        )
        if self.return_mappings is not None:
            assert self.transforms.keys() == self.return_mappings.keys(), (
                f"Error: expected keys in transforms and return_mappings to match,"
                f" received {list(self.transforms.keys())=}\n"
                f"{list(self.return_mappings.keys())=}"
            )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)

    def transform(self, tensor_dict: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        tensor_dict : jnp.ndarray dict
            model output, indexed according to self.input_mappings
        """
        out = jnp.zeros_like(tensor_dict)

        for field, indices in self.input_mappings.items():
            encoded = self.transforms[field].transform(tensor_dict[indices])
            if self.return_mappings is not None:
                encoded = encoded[self.return_mappings[field]]
            out = out.at[indices].set(encoded)

        return out

    def inverse_transform(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : jnp.ndarray
            model output, indexed according to self.input_mappings
        """
        out = jnp.zeros_like(x)
        for field, indices in self.input_mappings.items():
            decoded = self.transforms[field].inverse_transform(x[indices])
            if self.return_mappings is not None:
                decoded = decoded[self.return_mappings[field]]
            out = out.at[indices].set(decoded)

        return out

    def to(self, device: str) -> "DictTransform":
        # JAX device placement is done via jax.device_put on parameter pytrees,
        # not on the module itself. This is a no-op placeholder.
        return self
