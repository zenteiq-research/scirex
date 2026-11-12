# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Low-level tensor helpers for model-parallel splitting, gathering and reduction."""

import jax
import jax.numpy as jnp


def get_memory_format(tensor: jnp.ndarray) -> None:
    # JAX arrays have no memory_format attribute; always row-major (C-contiguous)
    """Placeholder kept for API parity with the PyTorch implementation; it does nothing in JAX."""
    return None


def pad_helper(
    tensor: jnp.ndarray, dim: int, new_size: int, mode: str = "zero"
) -> jnp.ndarray:
    """Pad `tensor` along `dim` up to `new_size`, with zeros or by the given `mode`."""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    # Convert PyTorch F.pad flat reversed list to jnp.pad (before, after) pairs
    # in forward order
    n_pairs = len(output_shape) // 2
    pairs = [(output_shape[2 * i], output_shape[2 * i + 1]) for i in range(n_pairs)]
    jnp_pad_widths = [(0, 0)] * (ndim - n_pairs) + list(reversed(pairs))
    tensor_pad = jnp.pad(tensor, jnp_pad_widths, mode="constant")

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad = tensor_pad.at[tuple(lhs_slice)].set(
            jnp.flip(jnp.conj(tensor_pad[tuple(rhs_slice)]), axis=dim)
        )

    return tensor_pad


def truncate_helper(tensor: jnp.ndarray, dim: int, new_size: int) -> jnp.ndarray:
    """Cut `tensor` along `dim` down to `new_size`."""
    get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[tuple(output_slice)]

    return tensor_trunc


def split_tensor_along_dim(
    tensor: jnp.ndarray, dim: int, num_chunks: int
) -> list[jnp.ndarray]:
    """Split `tensor` into `num_chunks` chunks along `dim` and return them as a list."""
    assert (
        dim < tensor.ndim
    ), f"Error, tensor dimension is {tensor.ndim} which cannot be split along {dim}"
    assert tensor.shape[dim] % num_chunks == 0, (
        f"Error, cannot split dim {dim} evenly. Dim size is "
        f"{tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    )
    tensor_list = jnp.split(tensor, num_chunks, axis=dim)

    return tensor_list


# distributed primitives
def _reduce(
    input_: jnp.ndarray, use_fp32: bool = True, group: list[int] | None = None
) -> jnp.ndarray:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    comm_size = len(group) if group is not None else jax.process_count()
    if comm_size == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.astype(jnp.float32)
        inputf_ = jax.lax.psum(inputf_, axis_name="batch")
        input_ = inputf_.astype(dtype)
    else:
        input_ = jax.lax.psum(input_, axis_name="batch")

    return input_


def _split(
    input_: jnp.ndarray, dim_: int, group: list[int] | None = None
) -> jnp.ndarray:
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = len(group) if group is not None else jax.process_count()
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: jnp.split does not create contiguous tensors by default.
    rank = jax.process_index()
    output = input_list[rank]

    return output


def _gather(
    input_: jnp.ndarray, dim_: int, group: list[int] | None = None
) -> jnp.ndarray:
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    get_memory_format(input_)

    comm_size = len(group) if group is not None else jax.process_count()
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert (
        dim_ < input_.ndim
    ), f"Error, cannot gather along {dim_} for tensor with {input_.ndim} dimensions."

    # Size and dimension.
    comm_rank = jax.process_index()

    tensor_list = [jnp.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    gathered = jax.lax.all_gather(input_, axis_name="batch")

    # Note: jnp.concatenate already creates a contiguous tensor.
    output = jnp.concatenate([gathered[i] for i in range(comm_size)], axis=dim_)

    return output
