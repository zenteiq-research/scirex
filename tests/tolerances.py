# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Numerical tolerances that depend on the JAX backend and the float precision.

A test states the tolerance it needs on the CPU in float32, and this module returns what to
use on the device the test is running on::

    from tests import tolerances

    assert jnp.allclose(got, want, atol=tolerances.atol(1e-5))
    np.testing.assert_allclose(got, want, **tolerances.both(rtol=1e-5, atol=1e-6))

**float32.** The value is multiplied by ``FLOAT32_SCALE[backend]``. Matrix products and
reductions round differently on a GPU (TF32, a different summation order) and on a TPU
(default matmuls use reduced precision), so the same code gives larger differences there.

**float64.** The value is replaced by ``FLOAT64_RTOL`` / ``FLOAT64_ATOL`` when they are
smaller. Float64 is on only inside a ``needs_x64`` test (``jax_enable_x64``), where float32
tolerances would hide a real error. The scale does not apply: the three backends agree to
rounding at this precision.

The precision is the default float dtype of JAX at the time of the call, or ``dtype`` when
it is given.

.. warning::
   The GPU and TPU factors are **provisional**: they are the starting guesses, not
   measurements. CI has no GPU or TPU. Measure them (run ``pytest tests -m "not gpu"`` on the
   device and lower or raise the factor until it is as tight as it can be and still passes)
   before relying on them.

``RELATIVE_TOLERANCE``, ``ABSOLUTE_TOLERANCE`` and ``NUMPY_TOLERANCE`` in the environment
still override the ``rel_tol``, ``abs_tol`` and ``np_tol`` fixtures of ``conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: float32 multiplier over the CPU value, by ``jax.default_backend()``. Provisional (see above).
FLOAT32_SCALE: dict[str, float] = {"cpu": 1.0, "gpu": 10.0, "tpu": 1000.0}
#: Largest tolerances used for float64 arrays.
FLOAT64_RTOL = 1e-10
FLOAT64_ATOL = 1e-12


def backend() -> str:
    """``cpu``, ``gpu`` or ``tpu``; any other name (for example ``rocm``) counts as ``gpu``."""
    name = jax.default_backend()
    return name if name in FLOAT32_SCALE else "gpu"


def _is_float64(dtype) -> bool:
    # With x64 off, jnp.zeros(dtype=float64) silently gives float32, so use numpy for a
    # stated dtype and JAX only for the default.
    dtype = np.dtype(dtype) if dtype is not None else jnp.zeros(()).dtype
    return dtype in (np.dtype("float64"), np.dtype("complex128"))


def _scaled(base: float, cap: float, dtype) -> float:
    if _is_float64(dtype):
        return min(base, cap)
    return base * FLOAT32_SCALE[backend()]


def rtol(base: float = 1e-5, dtype=None) -> float:
    """Relative tolerance for a test that needs ``base`` on the CPU in float32."""
    return _scaled(base, FLOAT64_RTOL, dtype)


def atol(base: float = 1e-5, dtype=None) -> float:
    """Absolute tolerance for a test that needs ``base`` on the CPU in float32."""
    return _scaled(base, FLOAT64_ATOL, dtype)


def both(rtol: float = 1e-5, atol: float = 1e-6, dtype=None) -> dict[str, float]:
    """``{"rtol": ..., "atol": ...}`` for ``allclose`` and ``assert_allclose``."""
    return {
        "rtol": _scaled(rtol, FLOAT64_RTOL, dtype),
        "atol": _scaled(atol, FLOAT64_ATOL, dtype),
    }
