# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The original 2D Fourier spectral-convolution design from Li et al.
("Fourier Neural Operator for Parametric Partial Differential
Equations", 2020/2021): a real FFT, and two independently-learned
complex weight tensors for the low-frequency and wrapped high-frequency
row bands of the spectrum, rather than deriving one band from the other
by symmetry.

This is a smaller, purpose-built alternative to
`SpectralConv`, which
is a more general, feature-rich implementation (Tucker factorization,
separable channels, complex-valued data, resolution rescaling). Kept as
its own module rather than folded into that one, since none of that
generality applies here -- this is a narrow, exact port of one specific
published design, retained standalone so its behavior stays pinned to
that design rather than drifting as the general layer evolves.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class ClassicSpectralConv2d(nn.Module):
    """2D spectral convolution via a real FFT, with two independently
    learned complex weight tensors (for the low-frequency and wrapped
    high-frequency row bands of the spectrum).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_modes1 : int
        Number of retained Fourier modes along the row (height) axis,
        for *each* of the low- and high-frequency bands (so the total
        row-mode budget is `2 * n_modes1`).
    n_modes2 : int
        Number of retained Fourier modes along the column (width) axis.
        A real FFT already halves this axis by construction; only the
        low-frequency half is kept, with no separate high-frequency band
        needed there.
    init_scale : float or None, optional
        Scale of the uniform weight initialization. Defaults to
        `1 / (in_channels * out_channels)`, matching the original
        published design.
    split_complex_params : bool, optional
        Whether to store each complex weight as a pair of REAL parameters
        (`<name>_re` and `<name>_im`) instead of one complex one, by
        default False.

        The two are mathematically identical; what differs is the
        optimizer state built from them. Optax's moment accumulators are
        shaped like the parameters, so complex weights give complex
        accumulators, and Adam's second moment is a real quantity that
        then has to be cast back -- which JAX currently warns about
        ("Casting from complex to real dtypes will soon raise a
        ValueError") and will eventually reject. Set this when the layer
        is trained with a moment-based optimizer; leave it off to keep
        the original design's parameter layout.
    """

    in_channels: int
    out_channels: int
    n_modes1: int
    n_modes2: int
    init_scale: float | None = None
    split_complex_params: bool = False

    def setup(self) -> None:
        scale = (
            self.init_scale
            if self.init_scale is not None
            else 1.0 / (self.in_channels * self.out_channels)
        )
        shape = (self.in_channels, self.out_channels, self.n_modes1, self.n_modes2)

        def _cx_init(rng: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
            rng_r, rng_i = jax.random.split(rng)
            return (
                scale * jax.random.uniform(rng_r, shape, dtype=jnp.float32)
                + 1j * scale * jax.random.uniform(rng_i, shape, dtype=jnp.float32)
            ).astype(jnp.complex64)

        if self.split_complex_params:
            # Same U(0, scale) marginals as _cx_init, one real parameter each.
            # float32 explicitly: under jax_enable_x64 the default would be
            # float64, and the pair would then recombine into complex128.
            real_init = nn.initializers.uniform(scale, dtype=jnp.float32)
            self.weight_parts: list[tuple[jax.Array, jax.Array]] = [
                (
                    self.param(f"{name}_re", real_init, shape),
                    self.param(f"{name}_im", real_init, shape),
                )
                for name in ("weights1", "weights2")
            ]
        else:
            self.weights1: jax.Array = self.param("weights1", _cx_init, shape)
            self.weights2: jax.Array = self.param("weights2", _cx_init, shape)

    def _weights(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """The low- and high-band complex weight tensors, however stored."""
        if self.split_complex_params:
            # jax.lax.complex, not `re + 1j * im`: under jax_enable_x64 the
            # Python complex literal would promote these float32 parts to
            # complex128 and the scatter below would then downcast them.
            return tuple(jax.lax.complex(re, im) for re, im in self.weight_parts)  # type: ignore[return-value]
        return self.weights1, self.weights2

    @staticmethod
    def _compl_mul2d(input_ft: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("bixy,ioxy->boxy", input_ft, weights)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """`x`: `(batch, in_channels, H, W)` real-valued ->
        `(batch, out_channels, H, W)`.
        """
        batch_size, _, h, w = x.shape
        # Fail fast on a mode budget the grid cannot carry: the two row bands
        # are sliced as [:n_modes1] and [-n_modes1:], so they overlap (and the
        # second write silently clobbers the first) once 2 * n_modes1 > h.
        if self.n_modes1 > h // 2 or self.n_modes2 > w // 2 + 1:
            raise ValueError(
                f"n_modes ({self.n_modes1}, {self.n_modes2}) exceed what a "
                f"{h}x{w} grid supports (at most {h // 2} row modes per band "
                f"and {w // 2 + 1} column modes)"
            )
        # Explicit cast: under jax_enable_x64 (flipped on globally as an
        # import-time side effect by tensorly, see spectral_convolution.py's
        # own note on this), rfft2 promotes to complex128 even for float32
        # input, which otherwise silently upcasts every downstream op here to
        # match this layer's explicitly complex64 weights -- a scatter dtype
        # mismatch JAX currently warns about and will make a hard error in a
        # future release.
        x_ft = jnp.fft.rfft2(x.astype(jnp.float32)).astype(jnp.complex64)

        weights1, weights2 = self._weights()
        out_ft = jnp.zeros(
            (batch_size, self.out_channels, h, w // 2 + 1), dtype=jnp.complex64
        )
        out_ft = out_ft.at[:, :, : self.n_modes1, : self.n_modes2].set(
            self._compl_mul2d(x_ft[:, :, : self.n_modes1, : self.n_modes2], weights1)
        )
        out_ft = out_ft.at[:, :, -self.n_modes1 :, : self.n_modes2].set(
            self._compl_mul2d(x_ft[:, :, -self.n_modes1 :, : self.n_modes2], weights2)
        )
        return jnp.fft.irfft2(out_ft, s=(h, w)).astype(jnp.float32)
