# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""FNOBlocks: a stack of Fourier layers forming the core of the FNO."""

from collections.abc import Callable, Sequence
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..._internal.scaling import validate_scaling_factor
from ..common.channel_mlp import ChannelMLP
from ..common.complex import ComplexValued, cgelu, ctanh
from ..common.normalization import AdaIN, BatchNorm, InstanceNorm
from ..common.skip_connections import skip_connection
from .spectral_convolution import SpectralConv


class FNOBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers.

    The Fourier layers are first described in [1], and the exact implementation details
    of the Fourier layer architecture are discussed in [2].

    Parameters
    ----------
    in_channels : int
        Number of input channels to Fourier layers
    out_channels : int
        Number of output channels after Fourier layers
    n_modes : int or List[int]
        Number of modes to keep along each dimension in frequency space.
        Can either be specified as an int (for all dimensions) or an iterable
        with one number per dimension
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        Factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        Number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int or List[int], optional
        Maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        Floating point precision to use for computations. Options: "full", "half",
        "mixed", by default "full"
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        Dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : callable, optional
        Nonlinear activation function to use between layers, by default nn.gelu
    stabilizer : Literal["tanh"], optional
        Stabilizing module to use between certain layers. Options: "tanh", None,
        by default None
    norm : Literal["ada_in", "group_norm", "instance_norm", "batch_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm",
        "batch_norm", None, by default None
    ada_in_features : int, optional
        Number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        Whether to call forward pass with pre-activation, by default False
        If True, call nonlinear activation and norm before Fourier convolution
        If False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        Module to use for FNO skip connections. Options: "linear", "soft-gating",
        "identity", None, by default "linear"
        If None, no skip connection is added. See layers.skip_connections for more
        details
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Options: "linear",
        "soft-gating", "identity", None, by default "soft-gating"
        If None, no skip connection is added. See layers.skip_connections for more
        details

    Other Parameters
    ----------------
    complex_data : bool, optional
        Whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        Separable parameter for SpectralConv, by default False
    factorization : str, optional
        Factorization parameter for SpectralConv. Options: None or "dense",
        and "tucker"; "cp" and "tt" are not supported, by default None
    rank : float, optional
        Rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        Module to use for convolutions in FNO block, by default SpectralConv
    fixed_rank_modes : bool, optional
        Fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        Implementation parameter for SpectralConv. Options: "factorized",
        "reconstructed", by default "factorized"
    decomposition_kwargs : dict, optional
        Kwargs for tensor decomposition in SpectralConv, by default dict()
    split_complex_params : bool, optional
        Store each spectral weight as a real/imaginary pair of real parameters
        rather than one complex parameter, by default False. Needed when
        training with a moment-based optimiser; see
        `SpectralConv`. Only
        used when `conv_module` is `SpectralConv` or a subclass.
    enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data. Only used when `conv_module` is `SpectralConv`
        or a subclass; ignored otherwise. When True, explicitly enforces that the 0th
        frequency and Nyquist frequency are real-valued before calling irfft. When
        False,
        relies on cuFFT's irfftn to handle symmetry automatically, which may fail on
        certain GPUs or input sizes, causing line artifacts. By default True.

    References
    ----------
    [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    in_channels: int
    out_channels: int
    n_modes: Any  # int or List[int]
    resolution_scaling_factor: Any = None
    n_layers: int = 1
    max_n_modes: Any = None
    fno_block_precision: str = "full"
    use_channel_mlp: bool = True
    channel_mlp_dropout: float = 0
    channel_mlp_expansion: float = 0.5
    non_linearity: Callable = nn.gelu
    stabilizer: Any = None
    norm: Any = None
    ada_in_features: Any = None
    preactivation: bool = False
    fno_skip: Any = "linear"
    channel_mlp_skip: Any = "soft-gating"
    complex_data: bool = False
    separable: bool = False
    factorization: Any = None
    rank: float = 1.0
    conv_module: Any = SpectralConv
    fixed_rank_modes: bool = False
    implementation: str = "factorized"
    decomposition_kwargs: Any = None
    enforce_hermitian_symmetry: bool = True
    split_complex_params: bool = False

    def setup(self) -> None:
        """Construct the per-layer spectral convolutions, skip connections,
        channel MLPs, and normalization layers. Flax lifecycle hook.
        """
        n_modes = self.n_modes
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        decomposition_kwargs = self.decomposition_kwargs or {}

        resolution_scaling_factor = validate_scaling_factor(
            self.resolution_scaling_factor, self.n_dim, self.n_layers
        )

        # apply real nonlin if data is real, otherwise cgelu
        if self.complex_data:
            self._non_linearity = cgelu
        else:
            self._non_linearity = self.non_linearity

        # One conv per layer. Only resolution_scaling_factor varies by layer index
        convs = [
            self.conv_module(
                self.in_channels,
                self.out_channels,
                n_modes,
                # Per-layer scaling for super-resolution, or None if disabled
                resolution_scaling_factor=(
                    resolution_scaling_factor[i]
                    if self.resolution_scaling_factor is not None
                    else None
                ),
                max_n_modes=self.max_n_modes,
                rank=self.rank,
                fixed_rank_modes=self.fixed_rank_modes,
                implementation=self.implementation,
                separable=self.separable,
                factorization=self.factorization,
                fno_block_precision=self.fno_block_precision,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=self.complex_data,
                # Only SpectralConv (and subclasses) accept these two
                **(
                    {
                        "enforce_hermitian_symmetry": self.enforce_hermitian_symmetry,
                        "split_complex_params": self.split_complex_params,
                    }
                    if issubclass(self.conv_module, SpectralConv)
                    else {}
                ),
            )
            for i in range(self.n_layers)
        ]
        self.convs = convs

        if self.fno_skip is not None:
            fno_skips: list[nn.Module] = [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=self.fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(self.n_layers)
            ]
            if self.complex_data:
                fno_skips = [ComplexValued(module=x) for x in fno_skips]
            self.fno_skips: list[nn.Module] | None = fno_skips
        else:
            self.fno_skips = None

        if self.use_channel_mlp:
            channel_mlp: list[nn.Module] = [
                ChannelMLP(
                    in_channels=self.out_channels,
                    hidden_channels=round(
                        self.out_channels * self.channel_mlp_expansion
                    ),
                    dropout=self.channel_mlp_dropout,
                    n_dim=self.n_dim,
                )
                for _ in range(self.n_layers)
            ]
            if self.complex_data:
                channel_mlp = [ComplexValued(module=x) for x in channel_mlp]
            self.channel_mlp = channel_mlp

            if self.channel_mlp_skip is not None:
                channel_mlp_skips: list[nn.Module] = [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=self.channel_mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(self.n_layers)
                ]
                if self.complex_data:
                    channel_mlp_skips = [
                        ComplexValued(module=x) for x in channel_mlp_skips
                    ]
                self.channel_mlp_skips: list[nn.Module] | None = channel_mlp_skips
            else:
                self.channel_mlp_skips = None

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        norm: list[nn.Module] | None
        if self.norm is None:
            norm = None
        elif self.norm == "instance_norm":
            norm = [InstanceNorm() for _ in range(self.n_layers * self.n_norms)]
        elif self.norm == "group_norm":
            norm = [
                nn.GroupNorm(num_groups=1) for _ in range(self.n_layers * self.n_norms)
            ]
        elif self.norm == "batch_norm":
            norm = [
                BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                for _ in range(self.n_layers * self.n_norms)
            ]
        elif self.norm == "ada_in":
            norm = [
                AdaIN(self.ada_in_features, self.out_channels)
                for _ in range(self.n_layers * self.n_norms)
            ]
        else:
            raise ValueError(
                f"Got norm={self.norm} but expected None or one of "
                "[instance_norm, group_norm, batch_norm, ada_in]"
            )

        # Wrap in ComplexValued while the list is still a local variable, then
        # bind to self once. Binding the raw norms to self._norm first and then
        # reassigning the wrapped list would reuse the auto-generated submodule
        # names (_norm_0, ...) and raise NameInUseError (mirrors the
        # channel_mlp construction above).
        if self.complex_data and norm is not None:
            norm = [ComplexValued(module=x) for x in norm]
        self._norm = norm

    def __call__(
        self,
        x: jnp.ndarray,
        index: int = 0,
        output_shape: tuple[int, ...] | None = None,
        ada_in_embeddings: Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Apply Fourier layer `index` to `x`.

        Dispatches to the pre- or post-activation forward pass depending on
        `self.preactivation`.

        Parameters
        ----------
        x : jnp.ndarray
            input activation of shape (batch, channels, d1, ..., dN)
        index : int, optional
            index of the Fourier layer to apply, by default 0
        output_shape : tuple of int, optional
            explicit output spatial shape for this block, by default None
        ada_in_embeddings : list of jnp.ndarray, optional
            adaptive instance-norm embeddings, used only when `norm == "ada_in"`,
            by default None

        Returns
        -------
        jnp.ndarray
            output activation of the Fourier layer
        """
        if self.preactivation:
            return self.forward_with_preactivation(
                x, index, output_shape, ada_in_embeddings
            )
        else:
            return self.forward_with_postactivation(
                x, index, output_shape, ada_in_embeddings
            )

    def _apply_norm(
        self,
        x: jnp.ndarray,
        norm_idx: int,
        ada_in_embeddings: Sequence[jnp.ndarray] | None,
        layer_index: int,
    ) -> jnp.ndarray:
        """Helper to apply norm at norm_idx, handling ada_in embedding lookup."""
        if self._norm is None:
            return x
        norm_layer = self._norm[norm_idx]
        if self.norm == "ada_in" and ada_in_embeddings is not None:
            # pick the embedding: either one shared or one per norm call
            if len(ada_in_embeddings) == 1:
                emb = ada_in_embeddings[0]
            else:
                emb = ada_in_embeddings[norm_idx]
            return norm_layer(x, emb)
        return norm_layer(x)

    def forward_with_postactivation(
        self,
        x: jnp.ndarray,
        index: int = 0,
        output_shape: tuple[int, ...] | None = None,
        ada_in_embeddings: Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Fourier-layer forward pass applying the nonlinearity after the
        convolution and channel MLP (standard FNO ordering).

        Parameters
        ----------
        x : jnp.ndarray
            input activation of shape (batch, channels, d1, ..., dN)
        index : int, optional
            index of the Fourier layer to apply, by default 0
        output_shape : tuple of int, optional
            explicit output spatial shape for this block, by default None
        ada_in_embeddings : list of jnp.ndarray, optional
            adaptive instance-norm embeddings, by default None

        Returns
        -------
        jnp.ndarray
            output activation of the Fourier layer
        """
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(
                x_skip_fno, output_shape=output_shape
            )

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(
                x_skip_channel_mlp, output_shape=output_shape
            )

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = jnp.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        jax.block_until_ready(x_fno)

        x_fno = self._apply_norm(x_fno, self.n_norms * index, ada_in_embeddings, index)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self._non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        x = self._apply_norm(x, self.n_norms * index + 1, ada_in_embeddings, index)

        if index < (self.n_layers - 1):
            x = self._non_linearity(x)

        return x

    def forward_with_preactivation(
        self,
        x: jnp.ndarray,
        index: int = 0,
        output_shape: tuple[int, ...] | None = None,
        ada_in_embeddings: Sequence[jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        """Fourier-layer forward pass applying the nonlinearity (and norm) before
        the convolution (resnet-style preactivation ordering).

        Parameters
        ----------
        x : jnp.ndarray
            input activation of shape (batch, channels, d1, ..., dN)
        index : int, optional
            index of the Fourier layer to apply, by default 0
        output_shape : tuple of int, optional
            explicit output spatial shape for this block, by default None
        ada_in_embeddings : list of jnp.ndarray, optional
            adaptive instance-norm embeddings, by default None

        Returns
        -------
        jnp.ndarray
            output activation of the Fourier layer
        """
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self._non_linearity(x)

        x = self._apply_norm(x, self.n_norms * index, ada_in_embeddings, index)

        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(
                x_skip_fno, output_shape=output_shape
            )

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(
                x_skip_channel_mlp, output_shape=output_shape
            )

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = jnp.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        jax.block_until_ready(x_fno)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self._non_linearity(x)

        x = self._apply_norm(x, self.n_norms * index + 1, ada_in_embeddings, index)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        return x

    # Deliberately overrides the `n_modes: Any` dataclass field above with a
    # property, so that setting .n_modes after construction propagates to the
    # already-built self.convs (see setter below). Standard Flax idiom for
    # mutable-after-init parameters; mypy can't express field-then-property.
    @property  # type: ignore[no-redef]
    def n_modes(self) -> Any:
        """Tuple[int, ...]: Number of Fourier modes kept along each dimension."""
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: Any) -> None:
        """Set the active number of Fourier modes, propagating the value to each
        per-layer spectral convolution.
        """
        # convs is only available after setup() has been called
        try:
            convs = object.__getattribute__(self, "convs")
            for i in range(self.n_layers):
                convs[i].n_modes = n_modes
        except AttributeError:
            pass  # during construction, convs not yet initialized
        try:
            object.__setattr__(self, "_n_modes", n_modes)
        except (AttributeError, TypeError):
            pass

    def get_block(self, indices: int | slice) -> "SubModule":
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(main_module=self, indices=indices)

    def __getitem__(self, indices: int | slice) -> "SubModule":
        """Return the sub-block at `indices` via `get_block`."""
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that parameters are not duplicated:
    if the same parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    main_module: nn.Module
    indices: int | slice

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.main_module(x, self.indices)
