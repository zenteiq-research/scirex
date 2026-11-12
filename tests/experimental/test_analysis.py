# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit tests for the experimental spectrum and low-rank analysis helpers."""

import jax
import jax.numpy as jnp
import numpy as np

from scirex.experimental.analysis import (
    compute_explained_variance,
    compute_rank,
    compute_stable_rank,
    spectrum_2d,
)


def test_compute_rank_full_rank_matrix() -> None:
    """compute_rank returns the matrix rank (full for an identity matrix)."""
    eye = jnp.eye(4)
    assert int(compute_rank(eye)) == 4


def test_compute_stable_rank_bounded_by_rank() -> None:
    """Stable rank is positive and no greater than the true rank."""
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 5))
    stable = float(compute_stable_rank(x))
    assert 0.0 < stable <= int(np.linalg.matrix_rank(np.array(x)))


# ---------------------------------------------------------------------------
# compute_rank / compute_stable_rank / compute_explained_variance
# (source lines 217-218, 223-226, 231-232)
# ---------------------------------------------------------------------------


def test_compute_rank_identity() -> None:
    """compute_rank of an identity matrix equals its size."""
    with jax.default_device(jax.devices("cpu")[0]):
        eye = jnp.eye(4)
        assert int(compute_rank(eye)) == 4


def test_compute_stable_rank_positive() -> None:
    """compute_stable_rank is positive for a full-rank matrix."""
    with jax.default_device(jax.devices("cpu")[0]):
        x = jax.random.normal(jax.random.PRNGKey(42), (6, 6))
        assert float(compute_stable_rank(x)) > 0.0


def test_compute_explained_variance_all_modes() -> None:
    """compute_explained_variance with all modes selected is 1.0."""
    with jax.default_device(jax.devices("cpu")[0]):
        s = jnp.array([3.0, 2.0, 1.0, 0.5], dtype=jnp.float32)
        # keeping all modes - explained variance should be exactly 1
        ev = float(compute_explained_variance(len(s), s))
        assert abs(ev - 1.0) < 1e-5


def test_compute_explained_variance_zero_modes() -> None:
    """compute_explained_variance with 0 modes is less than 1.0."""
    with jax.default_device(jax.devices("cpu")[0]):
        s = jnp.array([3.0, 2.0, 1.0, 0.5], dtype=jnp.float32)
        ev = float(compute_explained_variance(0, s))
        assert ev < 1.0


# ---------------------------------------------------------------------------
# spectrum_2d (source lines 119-157)
# ---------------------------------------------------------------------------


def test_spectrum_2d_normalize_true() -> None:
    """spectrum_2d with normalize=True uses ortho-normalised FFT."""
    with jax.default_device(jax.devices("cpu")[0]):
        # signal has shape (T, n*n) where T=signal.shape[0]; reshape to (T, n, n)
        n, T = 8, 3
        signal = jax.random.normal(jax.random.PRNGKey(0), (T, n * n))
        spec = spectrum_2d(signal, n, normalize=True)
        assert spec.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(spec)))


def test_spectrum_2d_normalize_false() -> None:
    """spectrum_2d with normalize=False uses backward-normalised rfft2."""
    with jax.default_device(jax.devices("cpu")[0]):
        n, T = 8, 3
        signal = jax.random.normal(jax.random.PRNGKey(1), (T, n * n))
        spec = spectrum_2d(signal, n, normalize=False)
        assert spec.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(spec)))


def test_spectrum_2d_multiple_time_steps() -> None:
    """spectrum_2d averages over T time-steps to give a 1-D spectrum."""
    with jax.default_device(jax.devices("cpu")[0]):
        n, T = 8, 4
        signal = jax.random.normal(jax.random.PRNGKey(2), (T, n * n))
        spec = spectrum_2d(signal, n, normalize=True)
        assert spec.shape == (n,)
