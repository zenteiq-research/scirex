# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Select tensorly's JAX backend without changing JAX's float64 setting.

``tensorly.set_backend("jax")`` calls ``jax.config.update("jax_enable_x64", True)``
every time it runs. JAX's configuration is process-wide, so that silently made every
later computation 64-bit: Transolver's ShapeNetCar training regressed about 48 times
(22 s to 1070 s per epoch on a GPU with weak FP64 throughput), and the test tolerances,
which depend on the default dtype, tightened to float64 values for float32 results.
Every module that needs the backend calls :func:`use_jax_backend`, so the flag is the
same after the call as before it.
"""

import jax
import tensorly as tl


def use_jax_backend() -> None:
    """Make ``jax`` tensorly's backend and put ``jax_enable_x64`` back as it was."""
    x64_before = bool(jax.config.read("jax_enable_x64"))
    tl.set_backend("jax")
    jax.config.update("jax_enable_x64", x64_before)
