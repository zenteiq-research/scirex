# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Functionality for handling complex-valued spatial data
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def cgelu(x: jnp.ndarray) -> jnp.ndarray:
    """Complex GELU activation function
    Follows the formulation of CReLU from [1].
    Applies GELU to real and imaginary parts of the input
    separately, then combine as complex number


    Parameters
    ----------
    x : jnp.ndarray (dtype=complex)
        pre-activation inputs

    References
    ----------
    [1] :

    Trabelsi, C., et al. (2018). "Deep Complex Networks".
        ICLR 2018, https://openreview.net/pdf?id=H1T2hmZAb.
    """
    return jax.lax.complex(
        nn.gelu(x.real).astype(jnp.float32), nn.gelu(x.imag).astype(jnp.float32)
    )


def cselu(x: jnp.ndarray) -> jnp.ndarray:
    """Complex-valued SELU activation
    Apply SELU to real and imag part of the input separately, then combine as
    complex number
    Args:
        x: complex array
    """
    return jax.lax.complex(
        nn.selu(x.real).astype(jnp.float32), nn.selu(x.imag).astype(jnp.float32)
    )


def ctanh(x: jnp.ndarray) -> jnp.ndarray:
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as
    complex number
    Args:
        x: complex array
    """
    return jax.lax.complex(
        jnp.tanh(x.real).astype(jnp.float32), jnp.tanh(x.imag).astype(jnp.float32)
    )


def apply_complex(
    real_func: Callable[[jnp.ndarray], jnp.ndarray],
    imag_func: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    dtype: jnp.dtype = jnp.complex64,
) -> jnp.ndarray:
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    re = (real_func(x.real) - imag_func(x.imag)).astype(jnp.float32)
    im = (real_func(x.imag) + imag_func(x.real)).astype(jnp.float32)
    return jax.lax.complex(re, im).astype(dtype)


class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    module: nn.Module

    def setup(self) -> None:
        self.fr = self.module.clone()
        self.fi = self.module.clone()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return apply_complex(self.fr, self.fi, x)
