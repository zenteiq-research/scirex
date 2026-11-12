# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Spectrum and low-rank analysis helpers. Experimental: semantics may change."""

import jax.numpy as jnp


def spectrum_2d(
    signal: jnp.ndarray, n_observations: int, normalize: bool = True
) -> jnp.ndarray:
    """This function computes the spectrum of a 2D signal using the
    Fast Fourier Transform (FFT).

    Parameters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT.
        If True, normalizes the outputs by `1/n_observations`
        (actually `1/sqrt(n_observations * n_observations)`).
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular
        coefficient is the coefficient's location along the diagonal, indexed
        from the top-left corner of the 2D FFT output.
    """
    T = signal.shape[0]
    signal = signal.reshape(T, n_observations, n_observations)

    if normalize:
        signal = jnp.fft.fft2(signal, norm="ortho")
    else:
        signal = jnp.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2D wavenumbers following fft convention
    k_max = n_observations // 2
    wavenumbers = jnp.concatenate(
        (
            jnp.arange(start=0, stop=k_max, step=1),
            jnp.arange(start=-k_max, stop=0, step=1),
        )
    )
    wavenumbers = jnp.tile(wavenumbers, (n_observations, 1))
    k_x = wavenumbers.T
    k_y = wavenumbers

    # Sum wavenumbers
    sum_k = jnp.sqrt(k_x**2 + k_y**2)

    # Remove symmetric components from wavenumbers
    index = -1.0 * jnp.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index = index.at[0:k_max1, 0:k_max1].set(sum_k[0:k_max1, 0:k_max1])

    spectrum = jnp.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = jnp.where(index == j)
        spectrum = spectrum.at[:, j - 1].set(
            (jnp.abs(signal[:, ind[0], ind[1]]) ** 2).sum(axis=1)
        )

    spectrum = spectrum.mean(axis=0)
    return spectrum


def compute_rank(tensor: jnp.ndarray) -> jnp.ndarray:
    # Compute the matrix rank of a tensor
    """Return the matrix rank of `tensor`.

    Parameters
    ----------
    tensor : jnp.ndarray
        A matrix.

    Returns
    -------
    jnp.ndarray
        The number of singular values above the numerical tolerance.
    """
    rank = jnp.linalg.matrix_rank(tensor)
    return rank


def compute_stable_rank(tensor: jnp.ndarray) -> jnp.ndarray:
    # Compute the stable rank of a tensor
    """Return the stable rank of `tensor`, the squared Frobenius norm over the squared spectral norm.

    Parameters
    ----------
    tensor : jnp.ndarray
        A matrix.

    Returns
    -------
    jnp.ndarray
        A scalar between 1 and the matrix rank.
    """
    fro_norm = jnp.linalg.norm(tensor, ord="fro") ** 2
    l2_norm = jnp.linalg.norm(tensor, ord=2) ** 2
    rank = fro_norm / l2_norm
    return rank


def compute_explained_variance(frequency_max: int, s: jnp.ndarray) -> jnp.ndarray:
    # Compute the explained variance based on frequency_max and singular values (s)
    """Return the variance explained by the first `frequency_max` singular values.

    Parameters
    ----------
    frequency_max : int
        Number of leading singular values kept.
    s : jnp.ndarray
        Singular values.

    Returns
    -------
    jnp.ndarray
        A scalar, 1 minus the variance of the discarded part over the variance of `s`.
    """
    s_current = s.at[frequency_max:].set(0)
    return 1 - jnp.var(s - s_current) / jnp.var(s)
