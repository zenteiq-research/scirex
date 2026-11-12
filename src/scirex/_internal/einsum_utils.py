# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Einsum utilities for half-precision complex tensors
(JAX analogues of torch `view_as_real` / `view_as_complex`).
"""

import jax.numpy as jnp
import opt_einsum
import tensorly as tl
from tensorly.plugins import use_opt_einsum

from .tensorly_backend import use_jax_backend

use_jax_backend()
use_opt_einsum("optimal")


def _view_as_real(a: jnp.ndarray) -> jnp.ndarray:
    """JAX equivalent of torch.view_as_real: complex (...) -> real (..., 2)"""
    return jnp.stack([a.real, a.imag], axis=-1)


def _view_as_complex(a: jnp.ndarray) -> jnp.ndarray:
    """JAX equivalent of torch.view_as_complex: real (..., 2) -> complex (...)"""
    return a[..., 0] + 1j * a[..., 1]


def einsum_complexhalf_two_input(
    eq: str, a: jnp.ndarray, b: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute (two-input) einsum for complexhalf tensors.
    Because jnp.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in jnp.einsum
    """
    assert len(eq.split(",")) == 2, "Equation must have two inputs."

    # cast both tensors to "view as real" form, and half precision
    a = _view_as_real(a)
    b = _view_as_real(b)
    a = a.astype(jnp.float16)
    b = b.astype(jnp.float16)

    # create a new einsum equation that takes into account "view as real" form
    input_output = eq.split("->")
    new_output = "xy" + input_output[1]
    input_terms = input_output[0].split(",")
    new_inputs = [input_terms[0] + "x", input_terms[1] + "y"]
    new_eqn = new_inputs[0] + "," + new_inputs[1] + "->" + new_output

    # convert back to complex form
    tmp = tl.einsum(new_eqn, a, b)
    res = jnp.stack(
        [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], axis=-1
    )
    return _view_as_complex(res)


def einsum_complexhalf(eq: str, *args: jnp.ndarray) -> jnp.ndarray:
    """
    Compute einsum for complexhalf tensors.
    Because jnp.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in jnp.einsum
    """
    if len(args) == 2:
        # if there are two inputs, it is faster to call this method
        return einsum_complexhalf_two_input(eq, *args)

    # find the optimal path
    _, path_info = opt_einsum.contract_path(eq, *args)
    partial_eqns = [
        contraction_info[2] for contraction_info in path_info.contraction_list
    ]

    # create a dict of the input tensors by their label in the einsum equation
    input_labels = eq.split("->")[0].split(",")
    output_label = eq.split("->")[1]
    tensors = dict(zip(input_labels, args))

    # convert all tensors to half precision and "view as real" form
    for key, tensor in tensors.items():
        tensor = _view_as_real(tensor)
        tensor = tensor.astype(jnp.float16)
        tensors[key] = tensor

    for partial_eq in partial_eqns:
        # get the input tensors to partial_eq
        in_labels, out_label = partial_eq.split("->")
        in_labels = in_labels.split(",")
        in_tensors = [tensors[label] for label in in_labels]

        # create new einsum equation that takes into account "view as real" form
        input_output = partial_eq.split("->")
        new_output = "xy" + input_output[1]
        input_terms = input_output[0].split(",")
        new_inputs = [input_terms[0] + "x", input_terms[1] + "y"]
        new_eqn = new_inputs[0] + "," + new_inputs[1] + "->" + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = tl.einsum(new_eqn, *in_tensors)
        result = jnp.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], axis=-1
        )
        tensors[out_label] = result

    return _view_as_complex(tensors[output_label])
