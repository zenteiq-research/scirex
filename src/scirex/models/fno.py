# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Fourier Neural Operator (FNO)."""

import warnings
from collections.abc import Callable
from functools import partialmethod
from typing import Any

import flax.linen as nn

from ..layers.blocks.fno_block import FNOBlocks
from ..layers.blocks.spectral_convolution import SpectralConv
from ..layers.common.channel_mlp import ChannelMLP
from ..layers.common.complex import ComplexValued
from ..layers.common.embeddings import GridEmbedding2D, GridEmbeddingND
from ..layers.common.padding import DomainPadding
from .base import BaseModel

warnings.filterwarnings("once", category=UserWarning)

Number = float | int


class FNO(BaseModel, name="FNO"):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions,
    as described in [1].

    The key component of an FNO is its SpectralConv layer (see
    `scirex.layers.SpectralConv`), which is similar to a standard CNN
    conv layer but operates in the frequency domain.

    For the theory, see the Fourier Neural Operator concept page.

    Parameters
    ----------
    n_modes : Tuple[int, ...]
        Number of modes to keep in Fourier Layer, along each dimension.
        The dimensionality of the FNO is inferred from len(n_modes).
        n_modes must be larger enough but smaller than max_resolution//2
        (Nyquist frequency)
    in_channels : int
        Number of channels in input function. Determined by the problem.
    out_channels : int
        Number of channels in output function. Determined by the problem.
    hidden_channels : int
        Width of the FNO (i.e. number of channels).
        This significantly affects the number of parameters of the FNO.
        Good starting point can be 64, and then increased if more expressivity
        is needed.
        Update lifting_channel_ratio and projection_channel_ratio accordingly
        since they are proportional to hidden_channels.
    n_layers : int, optional
        Number of Fourier Layers. Default: 4

    Other Parameters
    ----------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels.
        The number of lifting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels.
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO.
        Options:

        - "grid": Appends a grid positional embedding with default settings
          to the last channels of raw input.
          Assumes the inputs are discretized over a grid with entry
          [0,0,...] at the origin and side lengths of 1.
        - GridEmbeddingND: Uses this module directly (see
          `scirex.layers.GridEmbeddingND` for details).
        - GridEmbedding2D: Uses this module directly for 2D cases.
        - None: Does nothing.
        Default: "grid"
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: nn.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm",
        "instance_norm", None. Default: None
    complex_data : bool, optional
        Whether the data is complex-valued. If True, initializes
        complex-valued modules. Default: False
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block. Default: True
    channel_mlp_dropout : float, optional
        Dropout parameter for ChannelMLP in FNO Block. Default: 0
    channel_mlp_expansion : float, optional
        Expansion parameter for ChannelMLP in FNO Block. Default: 0.5
    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in channel-mixing mlp. Options:
        "linear", "identity", "soft-gating", None.
        Default: "soft-gating"
    fno_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in FNO layers. Options: "linear",
        "identity", "soft-gating", None.
        Default: "linear"
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Layer-wise factor by which to scale the domain resolution of function.
        Options:

        - None: No scaling
        - Single number n: Scales resolution by n at each layer
        - List of numbers [n_0, n_1,...]: Scales layer i's resolution by n_i
        Default: None
    domain_padding : Union[Number, List[Number]], optional
        Percentage of padding to use.
        Options:

        - None: No padding
        - Single number: Percentage of padding to use along all dimensions
        - List of numbers [p1, p2, ..., pN]: Percentage of padding along each dimension
        Default: None
    fno_block_precision : str, optional
        Precision mode in which to perform spectral convolution.
        Options: "full", "half", "mixed". Default: "full". Default: "full"
    stabilizer : str, optional
        Whether to use a stabilizer in FNO block. Options: "tanh", None. Default: None.
        stabilizer greatly improves performance in the case
        `fno_block_precision='mixed'`.
    max_n_modes : Tuple[int, ...], optional
        Maximum number of modes to use in Fourier domain during training.
        None means that all the n_modes are used.
        Tuple of integers: Incrementally increase the number of modes during training.
        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use.
        Options: None or "Dense" (the dense weights) and "Tucker". CP and TT
        are not supported and raise ``NotImplementedError``. Default: None
    rank : float, optional
        Tensor rank to use in factorization. Default: 1.0
        Set to float <1.0 when factorization is not None; rank 0.1 gives
        roughly 10% of the parameters of a dense FNO.
    fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    implementation : str, optional
        Implementation method for factorized tensors.
        Options: "factorized", "reconstructed". Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for the tensor decomposition. Default: {}
    separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    preactivation : bool, optional
        Whether to compute FNO forward pass with resnet-style preactivation.
        Default: False
    conv_module : nn.Module, optional
        Module to use for FNOBlock's convolutions. Default: SpectralConv
    enforce_hermitian_symmetry : bool, optional
        Whether to enforce Hermitian symmetry conditions when performing inverse FFT
        for real-valued data. Only used when `conv_module` is `SpectralConv`
        or a subclass; ignored otherwise. When True, explicitly enforces that the 0th
        frequency and Nyquist frequency are real-valued before calling
        irfft. When False, relies on cuFFT's irfftn to handle symmetry
        automatically, which may fail on
        certain GPUs or input sizes, causing line artifacts. By default True.

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from scirex.models import FNO
    >>> model = FNO(n_modes=(8, 8), in_channels=1, out_channels=1, hidden_channels=16, n_layers=2)
    >>> x = jnp.ones((2, 1, 32, 32))  # (batch, channels, height, width)
    >>> params = model.init(jax.random.PRNGKey(0), x)
    >>> model.apply(params, x).shape
    (2, 1, 32, 32)

    References
    ----------
    [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    # Flax dataclass fields (all FNO constructor parameters).
    # in_channels/out_channels/hidden_channels must precede n_modes because the
    # @property/@n_modes.setter below causes Python's dataclass to treat n_modes
    # as having a "default" (the property descriptor), so all truly required
    # no-default fields must be declared first.
    in_channels: int
    out_channels: int
    hidden_channels: int
    n_modes: Any  # Tuple[int, ...]
    n_layers: int = 4
    lifting_channel_ratio: Number = 2
    projection_channel_ratio: Number = 2
    positional_embedding: Any = "grid"  # Union[str, nn.Module, None]
    non_linearity: Callable = nn.gelu
    norm: Any = None
    complex_data: bool = False
    use_channel_mlp: bool = True
    channel_mlp_dropout: float = 0
    channel_mlp_expansion: float = 0.5
    channel_mlp_skip: Any = "soft-gating"
    fno_skip: Any = "linear"
    resolution_scaling_factor: Any = None
    domain_padding: Any = None  # Union[Number, List[Number], None]
    fno_block_precision: str = "full"
    stabilizer: Any = None
    max_n_modes: Any = None
    factorization: Any = None
    rank: float = 1.0
    fixed_rank_modes: bool = False
    implementation: str = "factorized"
    decomposition_kwargs: Any = None
    separable: bool = False
    preactivation: bool = False
    conv_module: Any = SpectralConv
    enforce_hermitian_symmetry: bool = True

    def setup(self) -> None:
        """Build the FNO submodules (optional positional embedding, lifting block,
        stack of FNO blocks, and projection block) from the configured fields.

        Flax lifecycle hook, invoked once when the module is initialized.
        """
        decomposition_kwargs = self.decomposition_kwargs or {}

        n_dim = len(self.n_modes)
        object.__setattr__(self, "n_dim", n_dim)
        object.__setattr__(self, "_n_modes", self.n_modes)

        # init lifting and projection channels using ratios w.r.t hidden channels
        lifting_channels = int(self.lifting_channel_ratio * self.hidden_channels)
        object.__setattr__(self, "lifting_channels", lifting_channels)

        projection_channels = int(self.projection_channel_ratio * self.hidden_channels)
        object.__setattr__(self, "projection_channels", projection_channels)

        ## Positional embedding
        # Flax class fields are frozen; submodule stored under
        # _positional_embedding_module to avoid overwriting the
        # 'positional_embedding' class field.
        self._positional_embedding_module: GridEmbeddingND | GridEmbedding2D | None
        _pe = self.positional_embedding
        if _pe == "grid":
            spatial_grid_boundaries = tuple((0.0, 1.0) for _ in range(n_dim))
            self._positional_embedding_module = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(_pe, GridEmbedding2D):
            if n_dim == 2:
                self._positional_embedding_module = _pe
            else:
                raise ValueError(
                    f"Error: expected {n_dim}-d positional embeddings, got {_pe}"
                )
        elif isinstance(_pe, GridEmbeddingND):
            self._positional_embedding_module = _pe
        elif _pe is None:
            self._positional_embedding_module = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {_pe},\
                              expected one of 'grid', GridEmbeddingND"
            )

        ## Domain padding
        # Submodule stored under _domain_padding_module to avoid
        # overwriting class field.
        self._domain_padding_module: DomainPadding | None
        _dp = self.domain_padding
        if _dp is not None and (
            (isinstance(_dp, list) and sum(_dp) > 0)
            or (isinstance(_dp, float | int) and _dp > 0)
        ):
            self._domain_padding_module = DomainPadding(
                domain_padding=_dp,
                resolution_scaling_factor=self.resolution_scaling_factor,
            )
        else:
            self._domain_padding_module = None

        ## Resolution scaling factor
        _rsf = self.resolution_scaling_factor
        if _rsf is not None:
            if isinstance(_rsf, float | int):
                _rsf = [_rsf] * self.n_layers

        ## FNO blocks
        self.fno_blocks = FNOBlocks(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=_rsf,
            use_channel_mlp=self.use_channel_mlp,
            channel_mlp_dropout=self.channel_mlp_dropout,
            channel_mlp_expansion=self.channel_mlp_expansion,
            non_linearity=self.non_linearity,
            stabilizer=self.stabilizer,
            norm=self.norm,
            preactivation=self.preactivation,
            fno_skip=self.fno_skip,
            channel_mlp_skip=self.channel_mlp_skip,
            complex_data=self.complex_data,
            max_n_modes=self.max_n_modes,
            fno_block_precision=self.fno_block_precision,
            rank=self.rank,
            fixed_rank_modes=self.fixed_rank_modes,
            implementation=self.implementation,
            separable=self.separable,
            factorization=self.factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=self.conv_module,
            n_layers=self.n_layers,
            enforce_hermitian_symmetry=self.enforce_hermitian_symmetry,
        )

        ## Lifting layer
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if _pe is not None:
            lifting_in_channels += n_dim
        _lifting = ChannelMLP(
            in_channels=lifting_in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=n_dim,
            non_linearity=self.non_linearity,
        )
        self.lifting: ChannelMLP | ComplexValued
        if self.complex_data:
            self.lifting = ComplexValued(module=_lifting)
        else:
            self.lifting = _lifting

        ## Projection layer
        _projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=n_dim,
            non_linearity=self.non_linearity,
        )
        self.projection: ChannelMLP | ComplexValued
        if self.complex_data:
            self.projection = ComplexValued(module=_projection)
        else:
            self.projection = _projection

    def __call__(self, x: Any, output_shape: Any = None, **kwargs: Any) -> Any:
        """FNO's forward pass

        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate
           function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence
           (SpectralConvolution + skip connections, nonlinearity)

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor

        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.

            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        if kwargs:
            warnings.warn(
                "FNO.__call__() received unexpected keyword arguments: "
                f"{list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self._positional_embedding_module is not None:
            x = self._positional_embedding_module(x)

        x = self.lifting(x)

        if self._domain_padding_module is not None:
            x = self._domain_padding_module.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self._domain_padding_module is not None:
            x = self._domain_padding_module.unpad(x)

        x = self.projection(x)

        return x

    # Deliberate Flax idiom: this property overrides the dataclass field of
    # the same name above so it can be mutated post-init; mypy's no-redef
    # check doesn't model that pattern.
    @property  # type: ignore[no-redef]
    def n_modes(self) -> Any:
        """Tuple[int, ...]: Number of Fourier modes kept along each dimension."""
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes: Any) -> None:
        """Set the active number of Fourier modes, propagating the value to the
        underlying FNO blocks.
        """
        # fno_blocks is only available after setup() has been called
        try:
            fno_blocks = object.__getattribute__(self, "fno_blocks")
            fno_blocks.n_modes = n_modes
        except AttributeError:
            pass  # during construction, fno_blocks not yet initialized
        try:
            object.__setattr__(self, "_n_modes", n_modes)
        except (AttributeError, TypeError):
            pass


def partialclass(new_name: str, cls: type, *args: Any, **kwargs: Any) -> type:
    """Create a new class with different default values

    A typical use is a preset such as `partialclass("SmallFNO", FNO, n_layers=2)`.

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    # cls is a generic factory arg (any BaseModel subclass); mypy can't
    # verify __init__ compatibility across arbitrary subclasses here.
    __init__ = partialmethod(cls.__init__, *args, **kwargs)  # type: ignore[misc]
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "__call__": cls.__call__,
        },
    )
