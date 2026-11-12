# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Mappings that copy, reduce, scatter and gather tensors across the model-parallel
region.
"""

import jax
import jax.numpy as jnp

from .comm import get_model_parallel_group

# helper functions
from .helpers import _gather, _reduce, _split


# model parallel
class _CopyToModelParallelRegion:
    """Pass the input to the model parallel region."""

    @staticmethod
    def apply(input_: jnp.ndarray) -> jnp.ndarray:
        # JAX custom_vjp: forward = identity, backward = all-reduce gradient
        @jax.custom_vjp
        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return x

        def fn_fwd(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            return x, None

        def fn_bwd(_: None, g: jnp.ndarray) -> tuple[jnp.ndarray]:
            return (_reduce(g, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _ReduceFromModelParallelRegion:
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def apply(input_: jnp.ndarray) -> jnp.ndarray:
        # JAX custom_vjp: forward = all-reduce, backward = identity
        @jax.custom_vjp
        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return _reduce(x, group=get_model_parallel_group())

        def fn_fwd(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            return fn(x), None

        def fn_bwd(_: None, g: jnp.ndarray) -> tuple[jnp.ndarray]:
            return (g,)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _ScatterToModelParallelRegion:
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def apply(input_: jnp.ndarray, dim_: int) -> jnp.ndarray:
        # dim_ is closed over so fn takes only the differentiable array input
        @jax.custom_vjp
        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return _split(x, dim_, group=get_model_parallel_group())

        def fn_fwd(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            return fn(x), None

        def fn_bwd(_: None, g: jnp.ndarray) -> tuple[jnp.ndarray]:
            return (_gather(g, dim_, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


class _GatherFromModelParallelRegion:
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def apply(input_: jnp.ndarray, dim_: int) -> jnp.ndarray:
        # dim_ is closed over so fn takes only the differentiable array input
        @jax.custom_vjp
        def fn(x: jnp.ndarray) -> jnp.ndarray:
            return _gather(x, dim_, group=get_model_parallel_group())

        def fn_fwd(x: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            return fn(x), None

        def fn_bwd(_: None, g: jnp.ndarray) -> tuple[jnp.ndarray]:
            return (_split(g, dim_, group=get_model_parallel_group()),)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn(input_)


# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_model_parallel_region(input_: jnp.ndarray) -> jnp.ndarray:
    """Pass the input into the model-parallel region (identity forward, all-reduce in the backward pass)."""
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: jnp.ndarray) -> jnp.ndarray:
    """Sum the input over the model-parallel group (all-reduce forward, identity in the backward pass)."""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Split the input along `dim` and keep the chunk of this rank (gather in the backward pass)."""
    return _ScatterToModelParallelRegion.apply(input_, dim)


def gather_from_model_parallel_region(input_: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Gather the chunks of all ranks along `dim` (split in the backward pass)."""
    return _GatherFromModelParallelRegion.apply(input_, dim)
