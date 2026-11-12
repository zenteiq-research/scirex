# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Parameter-counting utilities for JAX/Flax models and tensors.
"""

from collections.abc import Sequence
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["count_model_params", "count_tensor_params"]


def count_model_params(params: Any) -> int:
    """Returns the total number of parameters of a JAX/Flax model.

    Notes
    -----
    One complex number is counted as two parameters (real and imaginary parts).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> count_model_params({"w": jnp.ones((3, 4)), "b": jnp.ones(4)})
    16
    """
    leaves = jax.tree_util.tree_leaves(params)
    return sum(p.size * 2 if np.iscomplexobj(p) else p.size for p in leaves)


def count_tensor_params(tensor: jnp.ndarray, dims: Sequence[int] | None = None) -> int:
    """Returns the number of parameters (elements) in a single tensor,
    optionally, along certain dimensions only

    Parameters
    ----------
    tensor : jax.Array
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the
        number of parameters (elements)

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if np.iscomplexobj(tensor):
        return 2 * n_params
    return n_params
