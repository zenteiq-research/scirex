# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Simple in-memory tensor datasets backed by JAX arrays."""

from collections.abc import Callable, Sequence

import jax
import numpy as np


class TensorDataset:
    """In-memory dataset pairing an input array `x` with a target array `y`.

    Indexing returns a dictionary `{"x": ..., "y": ...}` so samples can be
    passed to a data processor and the trainer directly.

    Parameters
    ----------
    x : jax.Array
        Input samples; the first axis indexes samples.
    y : jax.Array
        Target samples; must have the same size along the first axis as `x`.
    transform_x : callable, optional
        Applied to each selected `x` when it is accessed.
    transform_y : callable, optional
        Applied to each selected `y` when it is accessed.
    """

    def __init__(
        self,
        x: jax.Array,
        y: jax.Array,
        transform_x: Callable[[jax.Array], jax.Array] | None = None,
        transform_y: Callable[[jax.Array], jax.Array] | None = None,
    ) -> None:
        assert x.shape[0] == y.shape[0], "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index: int | np.ndarray) -> dict[str, jax.Array]:
        """Return sample(s) at `index` as `{"x": ..., "y": ...}`, transformed."""
        x = self.x[index]
        y = self.y[index]

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return {"x": x, "y": y}

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.x.shape[0]


class GeneralTensorDataset:
    """Dataset of several tensors with a common first (sample) axis.

    Indexing returns the sample of every tensor, each passed through its own
    optional transform. With one tensor, a single array is returned instead of a list.

    Parameters
    ----------
    sets : Sequence[jax.Array]
        Tensors to index together. All must have the same number of samples.
    transforms : Sequence[Callable | None]
        One transform per tensor, applied to the indexed sample, or None for no transform.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> ds = GeneralTensorDataset([jnp.zeros((10, 3)), jnp.ones((10, 1))], [None, None])
    >>> len(ds), len(ds[0])
    (10, 2)
    """

    def __init__(
        self,
        sets: Sequence[jax.Array],
        transforms: Sequence[Callable[[jax.Array], jax.Array] | None],
    ) -> None:
        assert len(sets) == len(
            transforms
        ), "Size mismatch between number of tensors and transforms"
        self.n = len(sets)
        if self.n > 1:
            for j in range(1, self.n):
                assert (
                    sets[j].shape[0] == sets[0].shape[0]
                ), "Size mismatch between tensors"

        self.sets = sets
        self.transforms = transforms

    def __getitem__(self, index: int | np.ndarray) -> list[jax.Array] | jax.Array:
        if self.n > 1:
            items: list[jax.Array] = []
            for j in range(self.n):
                item = self.sets[j][index]
                transform = self.transforms[j]
                if transform is not None:
                    item = transform(item)
                items.append(item)
            return items

        single = self.sets[0][index]
        single_transform = self.transforms[0]
        if single_transform is not None:
            single = single_transform(single)
        return single

    def __len__(self) -> int:
        return self.sets[0].shape[0]
