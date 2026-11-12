# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the MPU tensor helpers (JAX).

neuraloperator's ``mpu`` package has no test suite, so coverage follows the
Phase 1 ``layers/tests`` structure (CPU/GPU backend parametrization for the
numeric helpers). The pure helpers (``pad_helper``, ``truncate_helper``,
``split_tensor_along_dim``) are tested directly; the distributed primitives
(``_reduce`` / ``_split`` / ``_gather``) are exercised on their single-process
fast path, where they short-circuit to the identity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.distributed.helpers import (
    _gather,
    _reduce,
    _split,
    get_memory_format,
    pad_helper,
    split_tensor_along_dim,
    truncate_helper,
)


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# get_memory_format
# ---------------------------------------------------------------------------


def test_get_memory_format_is_none() -> None:
    """JAX arrays expose no memory format, so the helper always returns None."""
    assert get_memory_format(jnp.zeros((2, 2))) is None


# ---------------------------------------------------------------------------
# pad_helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_pad_helper_zero_mode(backend) -> None:
    """Zero padding extends the chosen dim and fills the new entries with zeros."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(24.0).reshape(2, 3, 4)
        padded = pad_helper(x, dim=2, new_size=6, mode="zero")
        assert padded.shape == (2, 3, 6)
        # original content preserved, the two new columns are zero
        assert np.allclose(padded[..., :4], x)
        assert np.allclose(padded[..., 4:], 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pad_helper_conj_mode_is_hermitian(backend) -> None:
    """conj mode fills the padded region with the flipped complex conjugate."""
    with jax.default_device(jax.devices(backend)[0]):
        x = (jnp.arange(6.0) + 1j * jnp.arange(6.0)).reshape(1, 6).astype(jnp.complex64)
        padded = pad_helper(x, dim=1, new_size=10, mode="conj")
        assert padded.shape == (1, 10)
        # the original entries are unchanged
        assert np.allclose(padded[:, :6], x)
        # the appended region is the conjugate of a flipped slice (Hermitian)
        assert np.allclose(padded[:, 6:], jnp.conj(jnp.flip(x[:, 1:5], axis=1)))


# ---------------------------------------------------------------------------
# truncate_helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_truncate_helper(backend) -> None:
    """Truncation keeps the first new_size entries along the chosen dim."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(24.0).reshape(2, 3, 4)
        truncated = truncate_helper(x, dim=2, new_size=2)
        assert truncated.shape == (2, 3, 2)
        assert np.allclose(truncated, x[..., :2])


# ---------------------------------------------------------------------------
# split_tensor_along_dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_split_tensor_along_dim(backend) -> None:
    """Splitting yields num_chunks equal pieces that concatenate back to x."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(24.0).reshape(2, 6, 2)
        chunks = split_tensor_along_dim(x, dim=1, num_chunks=3)
        assert len(chunks) == 3
        assert all(c.shape == (2, 2, 2) for c in chunks)
        assert np.allclose(jnp.concatenate(chunks, axis=1), x)


def test_split_tensor_uneven_raises() -> None:
    """Splitting a dim that is not divisible by num_chunks asserts."""
    x = jnp.zeros((2, 5, 2))
    with pytest.raises(AssertionError):
        split_tensor_along_dim(x, dim=1, num_chunks=2)


def test_split_tensor_bad_dim_raises() -> None:
    """Splitting along an out-of-range dim asserts."""
    x = jnp.zeros((2, 4))
    with pytest.raises(AssertionError):
        split_tensor_along_dim(x, dim=5, num_chunks=2)


# ---------------------------------------------------------------------------
# distributed primitives - single-process identity fast path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_distributed_primitives_single_process_identity(backend) -> None:
    """With one process, _reduce / _split / _gather return the input unchanged."""
    with jax.default_device(jax.devices(backend)[0]):
        x = jnp.arange(12.0).reshape(3, 4)
        assert np.allclose(_reduce(x), x)
        assert np.allclose(_split(x, 0), x)
        assert np.allclose(_gather(x, 0), x)


# ---------------------------------------------------------------------------
# distributed primitives - multi-rank collective paths via a named vmap axis
# ---------------------------------------------------------------------------
#
# The multi-rank bodies of _reduce / _gather call jax.lax collectives
# (psum / all_gather) over the axis "batch". A real multi-process
# launch is unavailable in CI, but mapping over a named "batch" axis with
# jax.vmap provides that collective axis on a single device, and passing an
# explicit 2-rank group forces comm_size > 1 so the collective path is taken.


@pytest.mark.parametrize("use_fp32", [True, False])
def test_reduce_collective_path_sums_across_named_axis(use_fp32) -> None:
    """_reduce's psum path all-reduces across the mapped axis (group size 2)."""
    xb = jnp.arange(2 * 3.0).reshape(2, 3)
    out = jax.vmap(
        lambda x: _reduce(x, use_fp32=use_fp32, group=[0, 1]), axis_name="batch"
    )(xb)
    # every row holds the column-wise sum over the two batch rows
    expected = jnp.sum(xb, axis=0)
    assert np.allclose(out[0], expected)
    assert np.allclose(out[1], expected)


def test_gather_collective_path_concatenates_across_named_axis() -> None:
    """_gather's all_gather path concatenates the per-rank tensors (group size 2)."""
    xb = jnp.arange(2 * 3.0).reshape(2, 3)
    out = jax.vmap(lambda x: _gather(x, 0, group=[0, 1]), axis_name="batch")(xb)
    # each row gathers both rows -> width doubles from 3 to 6
    assert out.shape == (2, 6)


def test_split_with_explicit_group_keeps_local_rank_chunk() -> None:
    """Passing an explicit multi-rank group selects this process's local chunk.

    ``comm_size`` is derived from ``len(group)`` rather than ``process_count``,
    so the split path is exercised single-process: with a 2-rank group the
    tensor is halved along ``dim_`` and rank 0 keeps the first half.
    """
    x = jnp.arange(12.0).reshape(2, 6)
    out = _split(x, dim_=1, group=[0, 1])
    assert out.shape == (2, 3)
    assert np.allclose(out, x[:, :3])
