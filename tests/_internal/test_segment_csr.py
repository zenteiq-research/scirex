# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Tests for segment_csr (JAX CSR segment reduction).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex._internal.segment_csr import segment_csr

BACKENDS = ["cpu"] + (
    ["gpu"] if any(d.platform == "gpu" for d in jax.devices()) else []
)


# ---------------------------------------------------------------------------
# 1:1 mirror of test_native_segcsr_shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_segcsr_shapes(batch_size: int, backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        rng = np.random.default_rng(0)
        n_pts = 25
        n_channels = 5
        max_nbrhd_size = 7

        # randomly generate neighbourhood sizes → cumulative splits (like indptr)
        nbrhd_sizes = [0]
        while sum(nbrhd_sizes) < n_pts:
            pick = int(rng.integers(0, max_nbrhd_size + 1))
            pick = min(pick, n_pts - sum(nbrhd_sizes))
            nbrhd_sizes.append(pick)
            max_nbrhd_size = min(max_nbrhd_size, n_pts - sum(nbrhd_sizes))
        indptr = np.cumsum(nbrhd_sizes)  # length = n_out + 1
        n_out = len(indptr) - 1

        counts = jnp.array(np.diff(indptr), dtype=jnp.int32)
        # build segment_ids from counts
        seg_ids = jnp.repeat(jnp.arange(n_out, dtype=jnp.int32), counts)

        if batch_size > 1:
            src = jnp.array(
                rng.standard_normal((batch_size, n_pts, n_channels)), dtype=jnp.float32
            )
        else:
            src = jnp.array(rng.standard_normal((n_pts, n_channels)), dtype=jnp.float32)

        out = segment_csr(src, seg_ids, counts, reduction="sum")

        if batch_size == 1:
            assert out.shape == (n_out, n_channels)
        else:
            assert out.shape == (batch_size, n_out, n_channels)


# ---------------------------------------------------------------------------
# 1:1 mirror of test_native_segcsr_reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_segcsr_reductions(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        src = jnp.ones([10, 3])
        counts = jnp.array([3, 5, 2], dtype=jnp.int32)
        seg_ids = jnp.repeat(jnp.arange(3, dtype=jnp.int32), counts)

        out_sum = segment_csr(src, seg_ids, counts, reduction="sum")
        assert out_sum.shape == (3, 3)
        expected_sum = jnp.array([[3, 3, 3], [5, 5, 5], [2, 2, 2]], dtype=jnp.float32)
        assert jnp.allclose(out_sum, expected_sum)

        out_mean = segment_csr(src, seg_ids, counts, reduction="mean")
        assert out_mean.shape == (3, 3)
        assert jnp.allclose(out_mean, jnp.ones([3, 3]))


# ---------------------------------------------------------------------------
# Extra coverage: invalid reduction raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_reduction_raises(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        src = jnp.ones([6, 2])
        counts = jnp.array([3, 3], dtype=jnp.int32)
        seg_ids = jnp.repeat(jnp.arange(2, dtype=jnp.int32), counts)
        with pytest.raises(ValueError, match="reduce"):
            # Deliberately invalid at the type level too, to exercise the
            # runtime ValueError guard for a bad reduction string.
            segment_csr(src, seg_ids, counts, reduction="max")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Extra coverage: batched sum and mean
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_sum(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        B, n_edges, C = 3, 10, 4
        counts = jnp.array([3, 5, 2], dtype=jnp.int32)
        seg_ids = jnp.repeat(jnp.arange(3, dtype=jnp.int32), counts)
        src = jnp.ones([B, n_edges, C])
        out = segment_csr(src, seg_ids, counts, reduction="sum")
        assert out.shape == (B, 3, C)
        expected = jnp.array([[[3] * C, [5] * C, [2] * C]] * B, dtype=jnp.float32)
        assert jnp.allclose(out, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_mean(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        B, C = 2, 3
        counts = jnp.array([4, 6], dtype=jnp.int32)
        n_edges = int(counts.sum())
        seg_ids = jnp.repeat(jnp.arange(2, dtype=jnp.int32), counts)
        src = jnp.ones([B, n_edges, C])
        out = segment_csr(src, seg_ids, counts, reduction="mean")
        assert out.shape == (B, 2, C)
        assert jnp.allclose(out, jnp.ones([B, 2, C]))


# ---------------------------------------------------------------------------
# Extra coverage: zero-count segment gets 0 (not NaN)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_count_segment(backend: str) -> None:
    with jax.default_device(jax.devices(backend)[0]):
        counts = jnp.array([3, 0, 2], dtype=jnp.int32)
        seg_ids = jnp.array([0, 0, 0, 2, 2], dtype=jnp.int32)
        src = jnp.ones([5, 2])
        out = segment_csr(src, seg_ids, counts, reduction="mean")
        assert out.shape == (3, 2)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.allclose(out[1], jnp.zeros(2))  # empty segment → 0
