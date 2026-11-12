# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Spectral convolution: the frequency-domain layer at the heart of the Fourier Neural
Operator.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorly as tl
from tensorly.plugins import use_opt_einsum

from ..._internal.scaling import Number, validate_scaling_factor
from ..._internal.tensorly_backend import use_jax_backend
from ..common.resample import resample
from .base_spectral_conv import BaseSpectralConv

# tensorly's set_backend("jax") turns jax_enable_x64 on for the whole process;
# use_jax_backend() puts it back (see scirex/_internal/tensorly_backend.py).
use_jax_backend()
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class JAXTuckerTensor:
    """Lightweight Tucker tensor for JAX - replaces tltorch.TuckerTensor.

    Stores a core tensor and a list of factor matrices as plain JAX arrays.
    The interface mirrors what `_contract_tucker` expects (.core, .factors).
    """

    def __init__(self, core: jnp.ndarray, factors: Sequence[jnp.ndarray]) -> None:
        self.core = core
        self.factors = list(factors)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the full (reconstructed) tensor, derived from factor rows."""
        return tuple(f.shape[0] for f in self.factors)

    def __getitem__(self, slices: int | tuple | slice) -> "JAXTuckerTensor":
        """Slice the Tucker tensor by slicing each factor matrix's row dimension."""
        if not isinstance(slices, tuple):
            slices = (slices,)
        new_factors = []
        for i, factor in enumerate(self.factors):
            s = slices[i] if i < len(slices) else slice(None)
            new_factors.append(factor[s, :])  # slice rows; keep rank dim intact
        return JAXTuckerTensor(self.core, new_factors)


def _contract_dense(
    x: jnp.ndarray, weight: Any, separable: bool = False
) -> jnp.ndarray:
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0], *list(weight_syms)]
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not isinstance(weight, jnp.ndarray):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


def _contract_dense_separable(
    x: jnp.ndarray, weight: Any, separable: bool
) -> jnp.ndarray:
    if not isinstance(weight, jnp.ndarray):
        weight = weight.to_tensor()
    return x * weight


def _contract_tucker(
    x: jnp.ndarray, tucker_weight: "JAXTuckerTensor", separable: bool = False
) -> jnp.ndarray:
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]
    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def get_contract_fun(
    weight: Any, implementation: str = "reconstructed", separable: bool = False
) -> Callable[..., jnp.ndarray]:
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input
        (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors.
        if False,

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    # JAX-native Tucker is always contracted factorized (no reconstruction)
    if isinstance(weight, JAXTuckerTensor):
        return _contract_tucker

    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if isinstance(weight, jnp.ndarray):
            return _contract_dense
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f"Got implementation={implementation}, expected "
            '"reconstructed" or "factorized"'
        )


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_modes : int or int tuple
    complex_data : bool, optional
    max_n_modes : int tuple or None, optional
    bias : bool, optional
    separable : bool, optional
    resolution_scaling_factor : float, list of float, or None, optional
    fno_block_precision : str, optional
    rank : float, optional
    factorization : str or None, optional
    implementation : {'factorized', 'reconstructed'}, optional
    enforce_hermitian_symmetry : bool, optional
    fixed_rank_modes : bool, optional
    decomposition_kwargs : dict or None, optional
    init_std : float or 'auto', optional
    fft_norm : str, optional
    split_complex_params : bool, optional
        Store the dense spectral weight as two real parameters, `weight_re`
        and `weight_im`, instead of one complex `weight`, by default False.
        The forward pass is unchanged; only the parameter layout differs.

        Set this when training with a moment-based optimiser. optax shapes its
        moment estimates like the parameters, so a complex parameter produces a
        complex Adam state, and `optax.tree.cast_like` then raises JAX's
        "Casting from complex to real dtypes will soon raise a ValueError".

        Leaving it at False keeps the original parameter names, so existing
        checkpoints load unchanged. Supported for the dense factorization only.
    """

    in_channels: int
    out_channels: int
    n_modes: int | tuple
    complex_data: bool = False
    max_n_modes: int | tuple | None = None
    bias: bool = True
    separable: bool = False
    resolution_scaling_factor: Number | list[Number] | None = None
    fno_block_precision: str = "full"
    rank: float = 1.0
    factorization: str | None = None
    implementation: str = "reconstructed"
    enforce_hermitian_symmetry: bool = True
    fixed_rank_modes: bool | list[int] = False
    decomposition_kwargs: dict | None = None
    init_std: str | float = "auto"
    fft_norm: str = "forward"
    split_complex_params: bool = False

    def setup(self) -> None:
        """Normalise the mode and scaling settings and create the spectral weights."""
        n_modes: list[int]
        if isinstance(self.n_modes, int):
            n_modes = [self.n_modes]
        else:
            n_modes = list(self.n_modes)
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes
        self.order = len(self._n_modes)

        max_n_modes: list[int] | tuple
        if self.max_n_modes is None:
            max_n_modes = self._n_modes
        elif isinstance(self.max_n_modes, int):
            max_n_modes = [self.max_n_modes]
        else:
            max_n_modes = self.max_n_modes
        self._max_n_modes = list(max_n_modes)

        self._resolution_scaling_factor = validate_scaling_factor(
            self.resolution_scaling_factor, self.order
        )

        init_std: float
        if self.init_std == "auto":
            init_std = (2 / (self.in_channels + self.out_channels)) ** 0.5
        elif isinstance(self.init_std, str):
            raise ValueError(
                f"Got init_std={self.init_std!r}, expected a float or the string "
                "'auto'"
            )
        else:
            init_std = self.init_std
        self._init_std = init_std

        fixed_rank_modes: list[int] | None
        if isinstance(self.fixed_rank_modes, bool):
            fixed_rank_modes = [0] if self.fixed_rank_modes else None
        else:
            fixed_rank_modes = self.fixed_rank_modes
        self._fixed_rank_modes = fixed_rank_modes

        # Use the config factorization (tucker or dense). CP/TT are PyTorch
        # tltorch options with no JAX port; silently falling back to dense
        # used to hide that a requested factorization was never applied.
        factorization = (
            self.factorization if self.factorization is not None else "Dense"
        )
        if factorization.lower() not in ("dense", "tucker"):
            raise NotImplementedError(
                f"JAX port only supports 'dense' or 'tucker' factorization, "
                f"got '{factorization}'. CP and TT factorizations are not "
                "implemented."
            )

        if self.separable:
            if self.in_channels != self.out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={self.in_channels} and "
                    f"out_channels={self.out_channels}",
                )
            weight_shape = (self.in_channels, *self._max_n_modes)
        else:
            weight_shape = (self.in_channels, self.out_channels, *self._max_n_modes)

        # Create spectral weight tensor as native Flax parameters.
        # Dense: one complex array of shape weight_shape.
        # Tucker: core tensor + per-dim factor matrices - 70x smaller than dense.
        def _cx_init(rng: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
            std = self._init_std
            rng_r, rng_i = jax.random.split(rng)
            return (
                std * jax.random.normal(rng_r, shape, dtype=jnp.float32)
                + 1j * std * jax.random.normal(rng_i, shape, dtype=jnp.float32)
            ).astype(jnp.complex64)

        if factorization.lower() == "dense":
            if self.split_complex_params:
                # optax shapes its moment estimates like the parameters, so a
                # complex parameter gives Adam a complex state and
                # optax.tree.cast_like then hits JAX's "casting from complex
                # to real dtypes will soon raise a ValueError". Storing the
                # two components as separate real parameters keeps the
                # optimiser state real. jax.lax.complex, rather than
                # `re + 1j * im`, because tensorly enables jax_enable_x64 at
                # import and the literal would promote to complex128.
                def _re_init(rng: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
                    return self._init_std * jax.random.normal(
                        rng, shape, dtype=jnp.float32
                    )

                weight_re = self.param("weight_re", _re_init, weight_shape)
                weight_im = self.param("weight_im", _re_init, weight_shape)
                self.weight: jax.Array | JAXTuckerTensor = jax.lax.complex(
                    weight_re, weight_im
                )
            else:
                self.weight = self.param("weight", _cx_init, weight_shape)

        elif factorization.lower() == "tucker":
            if self.split_complex_params:
                raise ValueError(
                    "split_complex_params is implemented for the dense "
                    "factorization only, but factorization='tucker' was "
                    "requested; pass factorization=None to use it"
                )
            # Use tensorly's validate_tucker_rank for parity with PyTorch's
            # tltorch.TuckerTensor.new(...), which calls the same function.
            # For a float rank in (0, 1], it is treated as a compression ratio
            # (target Tucker params = rank * full tensor params), NOT a
            # per-dim multiplier. A simple `round(rank * d)` would under-shoot
            # by a large factor and produce a different shape than PyTorch.

            from tensorly.tucker_tensor import validate_tucker_rank

            tucker_ranks = tuple(
                int(r)
                for r in validate_tucker_rank(
                    weight_shape,
                    rank=self.rank,
                    fixed_modes=self._fixed_rank_modes,
                )
            )

            # Core tensor
            self._w_core = self.param("w_core", _cx_init, tucker_ranks)

            # Factor matrices - one per dimension of weight_shape
            for i, (d, r) in enumerate(zip(weight_shape, tucker_ranks)):
                setattr(self, f"_w_U{i}", self.param(f"w_U{i}", _cx_init, (d, r)))

            n_dims = len(weight_shape)
            self.weight = JAXTuckerTensor(
                self._w_core,
                [getattr(self, f"_w_U{i}") for i in range(n_dims)],
            )
        # No other branch: factorization is validated to be "dense" or
        # "tucker" above.

        self._contract = get_contract_fun(
            self.weight, implementation=self.implementation, separable=self.separable
        )

        if self.bias:
            # Flax parameter: shape (out_channels, 1, 1, ...) with `order` trailing 1s
            # [PARITY-FIX] Match PyTorch SpectralConv bias: init_std * randn(...).
            # JAX originally used zeros; PyTorch uses random normal with same std.
            # To restore the original zero init, use zeros scaled by _init_std.
            self._bias: jax.Array | None = cast(
                "jax.Array",
                self.param(
                    "bias",
                    lambda rng, shape: self._init_std
                    * jax.random.normal(rng, shape, dtype=jnp.float32),
                    tuple([self.out_channels]) + (1,) * self.order,
                ),
            )
            # [/PARITY-FIX]
        else:
            self._bias = None

    def transform(
        self, x: jnp.ndarray, output_shape: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        """Resample `x` to `output_shape`, or by the resolution scaling factor."""
        in_shape = list(x.shape[2:])

        out_shape: Sequence[int]
        if self._resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                round(s * r)
                for (s, r) in zip(in_shape, self._resolution_scaling_factor)
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == list(out_shape):
            return x
        else:
            return resample(
                x, 1.0, list(range(2, x.ndim)), output_shape=tuple(out_shape)
            )

    @nn.nowrap
    def _get_n_modes(self) -> list[int]:
        return self._n_modes

    def __call__(
        self, x: jnp.ndarray, output_shape: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : jnp.ndarray
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = (
                fft_size[-1] // 2 + 1
            )  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        # TPUs have no float64/complex128 FFT. tensorly flips jax_enable_x64 on
        # at import and numpy data arrives as float64, so pin the FFT input to
        # 32-bit regardless of the global flag.
        if self.fno_block_precision == "half":
            x = x.astype(jnp.float16)
        elif self.complex_data:
            x = x.astype(jnp.complex64)
        else:
            x = x.astype(jnp.float32)

        if self.complex_data:
            x = jnp.fft.fftn(x, norm=self.fft_norm, axes=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = jnp.fft.rfftn(x, norm=self.fft_norm, axes=fft_dims)
            dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = jnp.fft.fftshift(x, axes=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            x = x.astype(jnp.complex64)  # JAX has no chalf; use complex64

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = jnp.complex64  # JAX has no complex32/chalf
        else:
            out_dtype = jnp.complex64

        out_fft = jnp.zeros([batchsize, self.out_channels, *fft_size], dtype=out_dtype)

        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(
                fft_size, self._n_modes, self._max_n_modes
            )
        ]

        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels

        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        weight = self.weight[tuple(slices_w)]
        # Factorized weights (e.g. Tucker) are tensorly objects with no dtype;
        # their parameters are created 32-bit already.
        w_dtype = getattr(weight, "dtype", None)
        if w_dtype is not None and isinstance(weight, jnp.ndarray):
            if jnp.issubdtype(w_dtype, jnp.complexfloating):
                weight = weight.astype(jnp.complex64)
            elif w_dtype == jnp.float64:
                weight = weight.astype(jnp.float32)

        if self.separable:
            weight_start_idx = 1
        else:
            weight_start_idx = 2

        slices_x = [slice(None), slice(None)]  # batch_size, channels

        for all_modes, kept_modes in zip(
            fft_size, list(weight.shape[weight_start_idx:])
        ):
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x_t = tuple(slices_x)
        out_fft = out_fft.at[slices_x_t].set(
            self._contract(x[slices_x_t], weight, separable=self.separable).astype(
                out_fft.dtype
            )
        )

        mode_sizes_out: Sequence[int] = mode_sizes
        if self._resolution_scaling_factor is not None and output_shape is None:
            mode_sizes_out = tuple(
                round(s * r)
                for (s, r) in zip(mode_sizes, self._resolution_scaling_factor)
            )

        if output_shape is not None:
            mode_sizes_out = output_shape

        if self.order > 1:
            out_fft = jnp.fft.ifftshift(out_fft, axes=fft_dims[:-1])

        # Inverse FFT
        if self.complex_data:
            x = jnp.fft.ifftn(
                out_fft, s=mode_sizes_out, axes=fft_dims, norm=self.fft_norm
            )
        else:
            if self.enforce_hermitian_symmetry:
                out_fft = jnp.fft.ifftn(
                    out_fft,
                    s=mode_sizes_out[:-1],
                    axes=fft_dims[:-1],
                    norm=self.fft_norm,
                )

                # Enforce Hermitian symmetry conditions for irfft
                # 0th frequency must be real
                out_fft = out_fft.at[..., 0].set(
                    (out_fft[..., 0].real + 0j).astype(out_fft.dtype)
                )

                # Nyquist frequency must be real if the spatial size is even
                if mode_sizes_out[-1] % 2 == 0:
                    out_fft = out_fft.at[..., -1].set(
                        (out_fft[..., -1].real + 0j).astype(out_fft.dtype)
                    )

                x = jnp.fft.irfft(
                    out_fft, n=mode_sizes_out[-1], axis=fft_dims[-1], norm=self.fft_norm
                )
            else:
                x = jnp.fft.irfftn(
                    out_fft, s=mode_sizes_out, axes=fft_dims, norm=self.fft_norm
                )

        # Keep the result 32-bit (see the note before the forward FFT).
        x = x.astype(jnp.complex64 if self.complex_data else jnp.float32)

        if self._bias is not None:
            x = x + self._bias.astype(x.dtype)

        return x
