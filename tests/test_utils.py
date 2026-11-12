# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the general utilities (JAX).

Mirrors ``neuralop/tests/test_utils.py`` for the helpers that are ported:

* ``test_count_model_params`` -> the JAX ``count_model_params`` counts every leaf
  of a parameter pytree (complex leaves counted twice, for real + imaginary);
* ``test_count_tensor_params`` -> reproduced as-is (real / complex, dim subsets).

The remaining helpers that used to live in ``scirex.utils`` are tested under
``tests/_internal`` and ``tests/experimental/test_analysis.py``.
"""

from typing import Any

import jax.numpy as jnp

from scirex.utils import count_model_params, count_tensor_params

# ---------------------------------------------------------------------------
# count_model_params
# ---------------------------------------------------------------------------


def _nested_params(n_submodels: int, dtype: jnp.dtype = jnp.float32) -> dict:
    """A params pytree with (n_submodels + 1) leaves each of shape (2, 3, 4)."""
    # recursively nested, so the value type can't be a single flat Array
    params: dict[str, Any] = {"param": jnp.zeros((2, 3, 4), dtype=dtype)}
    node = params
    for i in range(n_submodels):
        node["model"] = {"param": jnp.zeros((2, 3, 4), dtype=dtype)}
        node = node["model"]
    return params


def test_count_model_params() -> None:
    """Every leaf is counted; complex leaves count double (real + imaginary)."""
    n_submodels = 2
    params = _nested_params(n_submodels)
    assert count_model_params(params) == (n_submodels + 1) * 2 * 3 * 4

    params_cx = _nested_params(n_submodels, dtype=jnp.complex64)
    assert count_model_params(params_cx) == 2 * (n_submodels + 1) * 2 * 3 * 4


# ---------------------------------------------------------------------------
# count_tensor_params
# ---------------------------------------------------------------------------


def test_count_tensor_params_real() -> None:
    """All elements (or only those along selected dims) of a real tensor."""
    x = jnp.zeros((2, 3, 4, 5, 6), dtype=jnp.float32)
    assert count_tensor_params(x) == 2 * 3 * 4 * 5 * 6
    assert count_tensor_params(x, dims=[1, 3]) == 3 * 5


def test_count_tensor_params_complex() -> None:
    """A complex tensor counts each element twice (real + imaginary)."""
    x = jnp.zeros((2, 3, 4, 5, 6), dtype=jnp.complex64)
    assert count_tensor_params(x) == 2 * 3 * 4 * 5 * 6 * 2
    assert count_tensor_params(x, dims=[1, 3]) == 3 * 5 * 2
