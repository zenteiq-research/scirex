# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared test helpers for the JAX layers test suite."""

from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp


def available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


def init_apply(
    module: nn.Module, x: jnp.ndarray, key: jax.Array | None = None
) -> jnp.ndarray:
    """Initialize a single-input Flax module and apply it to ``x``."""
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = module.init(key, x)
    return cast(jnp.ndarray, module.apply(variables, x))
