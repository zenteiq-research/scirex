# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Importing the spectral layers must not turn on JAX's float64 mode.

tensorly's ``set_backend("jax")`` enables ``jax_enable_x64`` for the whole process. If it
stays on, every later test sees float64 as the default dtype, and the tolerances in
``tests/tolerances.py`` tighten to float64 values for float32 results.
"""

import subprocess
import sys
import textwrap

import pytest

MODULES = [
    "scirex._internal.einsum_utils",
    "scirex.layers.blocks.spectral_convolution",
    "scirex.models.fno",
    "scirex.models.wno",
    "scirex.models.gino",
]


@pytest.mark.parametrize("module", MODULES)
def test_importing_a_tensorly_module_leaves_x64_as_it_was(module: str) -> None:
    code = textwrap.dedent(
        f"""
        import jax
        before = jax.config.jax_enable_x64
        import {module}
        assert jax.config.jax_enable_x64 == before, "{module} changed jax_enable_x64"
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stderr[-600:]
