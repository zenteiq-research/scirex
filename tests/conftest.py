# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

import os

# JAX defaults for the test run; they must be set before jax is imported.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# A headless backend: the default one starts Tk on Windows and can fail with
# "Can't find a usable init.tcl" when other tests run alongside.
os.environ.setdefault("MPLBACKEND", "Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest

from tests import tolerances


@pytest.fixture(scope="session", autouse=True)
def xla_warmup() -> None:
    dummy = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)
    jax.jit(lambda x: x + 1)(dummy).block_until_ready()


def _tolerance(env: str, scaled: float) -> float:
    """The environment variable if set, else the per-backend value (see tests/tolerances.py)."""
    return float(os.environ[env]) if env in os.environ else scaled


@pytest.fixture
def rel_tol() -> float:
    return _tolerance("RELATIVE_TOLERANCE", tolerances.rtol(1e-5))


@pytest.fixture
def abs_tol() -> float:
    return _tolerance("ABSOLUTE_TOLERANCE", tolerances.atol(1e-7))


@pytest.fixture
def np_tol() -> float:
    return _tolerance("NUMPY_TOLERANCE", tolerances.atol(1e-6))


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(autouse=True)
def _enable_x64_when_marked(request: pytest.FixtureRequest):
    """Run tests marked ``needs_x64`` with float64 switched on, then restore it.

    These tests used to pass only when an earlier test had enabled float64, so
    they failed on their own or in a different order.
    """
    if request.node.get_closest_marker("needs_x64") is None:
        yield
        return
    from jax.experimental import enable_x64

    with enable_x64():
        yield


def pytest_configure(config: pytest.Config) -> None:
    """Register any global pytest options or markers for operator tests."""
    config.addinivalue_line(
        "markers",
        "unit: fast tests with JIT disabled, no GPU required",
    )
    config.addinivalue_line(
        "markers",
        "integration: JIT-compiled tests using XLA cache",
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires physical NVIDIA GPU",
    )
    config.addinivalue_line(
        "markers",
        "needs_x64: runs with jax float64 enabled for the duration of the test",
    )
