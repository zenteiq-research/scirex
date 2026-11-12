# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Geometry-Informed Neural Operator (GINO)."""

import warnings
from collections.abc import Callable
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from ..layers.blocks.fno_block import FNOBlocks
from ..layers.blocks.gno_block import GNOBlock
from ..layers.blocks.gno_weighting_functions import dispatch_weighting_fn
from ..layers.blocks.spectral_convolution import SpectralConv
from ..layers.common.channel_mlp import ChannelMLP
from ..layers.common.embeddings import SinusoidalEmbedding
from .base import BaseModel


class GINO(BaseModel, name="gino"):
    """
    GINO: Geometry-informed Neural Operator - learns a mapping between
    functions presented over arbitrary coordinate meshes. The model carries
    global integration through spectral convolution layers in an intermediate
    latent space, as described in [1]. Optionally enables a weighted output
    GNO.

    Parameters
    ----------
    in_channels : int
        Feature dimension of input points. Determined by the problem.
    out_channels : int
        Feature dimension of output points. Determined by the problem.
    fno_n_modes : tuple, optional
        Number of modes along each dimension to use in FNO. Default: (16, 16, 16)
    fno_hidden_channels : int, optional
        Hidden channels for use in FNO. Default: 64
    fno_n_layers : int, optional
        Number of layers in FNO. Default: 4
    latent_feature_channels : int, optional
        Number of channels in optional latent feature map. Default: None
    projection_channel_ratio : int, optional
        Ratio of pointwise projection channels to fno_hidden_channels. Default: 4
    out_query_feature_channels : int, optional
        Number of per-point features given at the output queries and
        concatenated to the output-GNO result before the projection, so the
        pointwise decoder sees each query's own features (e.g. its distance to
        a wall). Default: None (off)
    gno_coord_dim : int, optional
        Geometric dimension of input/output queries. Default: 3
    in_gno_radius : float, optional
        Radius in input space for GNO neighbor search. Default: 0.033
    out_gno_radius : float, optional
        Radius in output space for GNO neighbor search. Default: 0.033
    fno_split_complex_params : bool, optional
        Store the latent FNO's spectral weights as real/imaginary pairs rather
        than complex arrays. Default: False. Set this to train with Adam or
        another moment-based optimiser; see
        `SpectralConv`.

    References
    ----------
    [1] : Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S.,
        Nabian, M., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A. (2023)
        Geometry-Informed Neural Operator for Large-Scale 3D PDEs. NeurIPS 2023.
    """

    in_channels: int
    out_channels: int
    latent_feature_channels: int | None = None
    # When False, no input GNO is built or run: the latent-grid features
    # (the SDF) are the entire FNO input and the GNO is used solely for
    # output. This is the GINO paper's decoder-only variant; the default
    # True is the encoder-decoder topology the reference neuralop GINO
    # also uses unconditionally.
    use_in_gno: bool = True
    projection_channel_ratio: int = 4
    out_query_feature_channels: int | None = None
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    in_gno_radius: float = 0.033
    out_gno_radius: float = 0.033
    gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_pos_embed_type: str = "nerf"
    fno_domain_padding: float = 0.125
    gno_weighting_function: Any = None
    gno_weight_function_scale: float = 1
    in_gno_pos_embed_type: str | None = "nerf"
    out_gno_pos_embed_type: str | None = "nerf"
    fno_in_channels: int = 3
    fno_n_modes: Any = (16, 16, 16)
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 2
    fno_n_layers: int = 4
    gno_embed_channels: int = 32
    gno_embed_max_positions: int = 10000
    in_gno_channel_mlp_hidden_layers: Any = (80, 80, 80)
    out_gno_channel_mlp_hidden_layers: Any = (512, 256)
    gno_channel_mlp_non_linearity: Callable = nn.gelu
    gno_use_open3d: bool = True
    gno_use_torch_scatter: bool = False
    out_gno_tanh: Any = None
    fno_resolution_scaling_factor: Any = None
    fno_block_precision: str = "full"
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_dropout: float = 0
    fno_channel_mlp_expansion: float = 0.5
    fno_non_linearity: Callable = nn.gelu
    fno_stabilizer: Any = None
    fno_norm: Any = None
    fno_ada_in_features: int | None = 4
    fno_ada_in_dim: int = 1
    fno_preactivation: bool = False
    fno_skip: str = "linear"
    fno_channel_mlp_skip: str = "soft-gating"
    fno_separable: bool = False
    fno_factorization: Any = None
    fno_rank: float = 1.0
    fno_fixed_rank_modes: bool = False
    fno_implementation: str = "factorized"
    fno_decomposition_kwargs: Any = None
    fno_conv_module: Any = SpectralConv
    fno_enforce_hermitian_symmetry: bool = True
    fno_split_complex_params: bool = False

    def setup(self) -> None:
        """Build the GINO submodules: input GNO, lifting block, latent-space FNO
        blocks, output GNO, and projection block. Flax lifecycle hook, run once
        when the module is initialized.
        """
        fno_hidden_channels = self.fno_hidden_channels
        fno_n_modes = self.fno_n_modes
        decomposition_kwargs = self.fno_decomposition_kwargs or {}

        lifting_channels = self.fno_lifting_channel_ratio * fno_hidden_channels

        # If the input GNO performs a nonlinear kernel, the GNO's output
        # features must be the same dimension as its input.
        if self.in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = self.fno_in_channels

        # The actual input channels to the FNO are computed here.
        # Decoder-only contributes no input-GNO channels at all.
        fno_in_channels = in_gno_out_channels if self.use_in_gno else 0
        if self.latent_feature_channels is not None:
            fno_in_channels += self.latent_feature_channels
        self._fno_in_channels = fno_in_channels

        gno_use_open3d = self.gno_use_open3d
        if self.gno_coord_dim != 3 and gno_use_open3d:
            warnings.warn(
                f"GNO expects {self.gno_coord_dim}-d data but Open3d expects 3D data",
                UserWarning,
                stacklevel=2,
            )
            gno_use_open3d = False

        in_coord_dim = len(fno_n_modes)
        if in_coord_dim != self.gno_coord_dim:
            warnings.warn(
                f"FNO expects {in_coord_dim}-d data while input GNO "
                f"expects {self.gno_coord_dim}-d data",
                UserWarning,
                stacklevel=2,
            )
        self._in_coord_dim = in_coord_dim
        self._in_coord_dim_forward_order = list(range(in_coord_dim))
        # spatial indices start at 2 (skip batch and channel dims)
        self._in_coord_dim_reverse_order = [
            j + 2 for j in self._in_coord_dim_forward_order
        ]

        self.adain_pos_embed: SinusoidalEmbedding | None
        if self.fno_norm == "ada_in":
            if (
                self.fno_ada_in_features is not None
                and self.out_gno_pos_embed_type is not None
            ):
                self.adain_pos_embed = SinusoidalEmbedding(
                    in_channels=self.fno_ada_in_dim,
                    num_frequencies=self.fno_ada_in_features,
                    max_positions=10000,
                    embedding_type=self.out_gno_pos_embed_type,
                )
                ada_in_dim = self.adain_pos_embed.out_channels
            else:
                ada_in_dim = self.fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            ada_in_dim = None
        self._ada_in_dim = ada_in_dim

        ### input GNO (skipped entirely in the decoder-only variant)
        self.gno_in = (
            None
            if not self.use_in_gno
            else GNOBlock(
                in_channels=self.in_channels,
                out_channels=in_gno_out_channels,
                coord_dim=self.gno_coord_dim,
                pos_embedding_type=self.in_gno_pos_embed_type,
                pos_embedding_channels=self.gno_embed_channels,
                pos_embedding_max_positions=self.gno_embed_max_positions,
                radius=self.in_gno_radius,
                reduction="mean",
                weighting_fn=None,
                channel_mlp_layers=list(self.in_gno_channel_mlp_hidden_layers),
                channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
                transform_type=self.in_gno_transform_type,
                use_torch_scatter_reduce=self.gno_use_torch_scatter,
                use_open3d_neighbor_search=gno_use_open3d,
            )
        )

        ### Lifting layer before FNOBlocks
        self.lifting = ChannelMLP(
            in_channels=fno_in_channels,
            hidden_channels=lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=2,
        )

        ### FNOBlocks in latent space
        self.fno_blocks = FNOBlocks(
            n_modes=fno_n_modes,
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            n_layers=self.fno_n_layers,
            resolution_scaling_factor=self.fno_resolution_scaling_factor,
            fno_block_precision=self.fno_block_precision,
            use_channel_mlp=self.fno_use_channel_mlp,
            channel_mlp_expansion=self.fno_channel_mlp_expansion,
            channel_mlp_dropout=self.fno_channel_mlp_dropout,
            non_linearity=self.fno_non_linearity,
            stabilizer=self.fno_stabilizer,
            norm=self.fno_norm,
            ada_in_features=ada_in_dim,
            preactivation=self.fno_preactivation,
            fno_skip=self.fno_skip,
            channel_mlp_skip=self.fno_channel_mlp_skip,
            separable=self.fno_separable,
            factorization=self.fno_factorization,
            rank=self.fno_rank,
            fixed_rank_modes=self.fno_fixed_rank_modes,
            implementation=self.fno_implementation,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=self.fno_conv_module,
            enforce_hermitian_symmetry=self.fno_enforce_hermitian_symmetry,
            split_complex_params=self.fno_split_complex_params,
        )

        ### output GNO
        if self.gno_weighting_function is not None:
            weight_fn = dispatch_weighting_fn(
                self.gno_weighting_function,
                sq_radius=self.out_gno_radius**2,
                scale=self.gno_weight_function_scale,
            )
        else:
            weight_fn = None

        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction="sum",
            weighting_fn=weight_fn,
            pos_embedding_type=self.out_gno_pos_embed_type,
            pos_embedding_channels=self.gno_embed_channels,
            pos_embedding_max_positions=self.gno_embed_max_positions,
            channel_mlp_layers=list(self.out_gno_channel_mlp_hidden_layers),
            channel_mlp_non_linearity=self.gno_channel_mlp_non_linearity,
            transform_type=self.out_gno_transform_type,
            use_torch_scatter_reduce=self.gno_use_torch_scatter,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        projection_channels = self.projection_channel_ratio * fno_hidden_channels
        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels + (self.out_query_feature_channels or 0),
            out_channels=self.out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=self.fno_non_linearity,
        )

    def latent_embedding(self, in_p: Any, ada_in: Any = None) -> Any:
        """Lift the input onto the latent grid and apply the FNO blocks.

        Parameters
        ----------
        in_p : jnp.ndarray
            latent-grid input of shape (batch, n_1, ..., n_k, in_channels + k)
        ada_in : jnp.ndarray, optional
            adaptive instance-normalization scalar of shape (fno_ada_in_dim,),
            used only when `fno_norm == "ada_in"`, by default None

        Returns
        -------
        jnp.ndarray
            latent embedding of shape (batch, fno_hidden_channels, n_1, ..., n_k)
        """
        # in_p has shape (batch, n_1, ..., n_k, in_channels + k);
        # ada_in has shape (fno_ada_in_dim,)

        # permute (b, n_1, ..., n_k, c) -> (b, c, n_1, ..., n_k)
        perm = (0, len(in_p.shape) - 1, *list(range(1, len(in_p.shape) - 1)))
        in_p = jnp.transpose(in_p, perm)

        # Compute Ada IN embedding
        ada_in_embeddings = None
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = jnp.squeeze(ada_in, axis=0)
            if self.adain_pos_embed is not None:
                ada_in_embed = jnp.squeeze(
                    self.adain_pos_embed(jnp.expand_dims(ada_in, axis=0)), axis=0
                )
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                ada_in_embeddings = [ada_in_embed]

        # Apply FNO blocks
        in_p = self.lifting(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx, ada_in_embeddings=ada_in_embeddings)

        return in_p

    def encode(
        self,
        input_geom: Any,
        latent_queries: Any,
        x: Any = None,
        latent_features: Any = None,
        ada_in: Any = None,
        neighbors_in: dict | None = None,
    ) -> tuple[Any, Any]:
        """Input GNO --> FNOBlocks: input mesh -> latent-grid embedding.

        Split out of `__call__` (which now just calls `encode` then
        `decode`) so a caller that needs to differentiate the decode step
        alone w.r.t. `output_queries` -- e.g. an AD-based physics-residual
        loss -- can call `encode` once and reuse its result across many
        `decode` calls, instead of recomputing the (comparatively
        expensive) encoder + latent FNO backbone for every derivative
        evaluation.

        Parameters
        ----------
        input_geom : jnp.ndarray
            input domain coordinate mesh, shape (1, n_in, gno_coord_dim)
        latent_queries : jnp.ndarray
            latent geometry on which to compute FNO latent embeddings,
            shape (1, n_gridpts_1, ..., n_gridpts_n, gno_coord_dim)
        x : jnp.ndarray, optional
            input function defined on the input domain, shape (batch, n_in, in_channels)
        latent_features : jnp.ndarray, optional
            optional feature map to concatenate onto latent embedding
        ada_in : jnp.ndarray, optional
            adaptive scalar instance parameter
        neighbors_in : dict, optional
            precomputed neighbor search result for the input GNO

        Returns
        -------
        latent_embed : jnp.ndarray
            shape (batch, n_1*...*n_k, fno_hidden_channels)
        latent_queries : jnp.ndarray
            `latent_queries` with its leading singleton axis squeezed --
            `decode` needs this same squeezed array (for its own
            `y=...reshape(...)` argument) and must not re-squeeze it.
        """
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        if latent_features is not None:
            assert (
                self.latent_feature_channels is not None
            ), "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels
            assert latent_features.ndim == self.gno_coord_dim + 2, (
                f"Expected latent_features ndim={self.gno_coord_dim + 2}, "
                f"got shape {latent_features.shape}"
            )
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = jnp.tile(
                        latent_features,
                        (batch_size, *[1] * (latent_features.ndim - 1)),
                    )

        input_geom = jnp.squeeze(input_geom, axis=0)
        latent_queries = jnp.squeeze(latent_queries, axis=0)

        grid_shape = latent_queries.shape[:-1]  # disregard positional encoding dim

        if self.use_in_gno:
            # setup() builds gno_in exactly when use_in_gno is set.
            assert self.gno_in is not None
            # Pass through input GNOBlock
            in_p = self.gno_in(
                y=input_geom,
                x=latent_queries.reshape((-1, latent_queries.shape[-1])),
                f_y=x,
                neighbors=neighbors_in,
            )

            # shape (batch_size, grid1, ...gridn, -1)
            in_p = in_p.reshape((batch_size, *grid_shape, -1))

            if latent_features is not None:
                in_p = jnp.concatenate((in_p, latent_features), axis=-1)
        else:
            # Decoder-only: the latent-grid features are the whole FNO input.
            assert latent_features is not None, (
                "use_in_gno=False requires latent_features: with no input GNO "
                "they are the only thing feeding the FNO."
            )
            in_p = latent_features

        # apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, ada_in=ada_in)

        # Integrate latent space to output queries
        # latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute (b, c, n...) -> (b, n..., c) then flatten spatial dims
        perm = (0, *self._in_coord_dim_reverse_order, 1)
        latent_embed = jnp.transpose(latent_embed, perm).reshape(
            batch_size, -1, self.fno_hidden_channels
        )

        if self.out_gno_tanh in ["latent_embed", "both"]:
            latent_embed = jnp.tanh(latent_embed)

        return latent_embed, latent_queries

    def decode(
        self,
        latent_embed: Any,
        latent_queries: Any,
        output_queries: Any,
        neighbors_out: dict | None = None,
        output_query_features: Any = None,
    ) -> Any:
        """Output GNO + projection: latent-grid embedding -> output queries.

        Parameters
        ----------
        latent_embed : jnp.ndarray
            first element of `encode(...)`'s return value.
        latent_queries : jnp.ndarray
            second element of `encode(...)`'s return value -- already
            squeezed; do not pass the original `(1, ...)` array here.
        output_queries : jnp.ndarray | dict[jnp.ndarray]
            points at which to query the final GNO layer to get output,
            shape (1, n_out, gno_coord_dim) per tensor.
        neighbors_out : dict, optional
            precomputed neighbor search result for the output GNO
        output_query_features : jnp.ndarray, optional
            per-query features of shape (1, n_out, out_query_feature_channels),
            concatenated to the output-GNO result before the projection;
            required exactly when `out_query_feature_channels` is set.
            Not supported with dict `output_queries`.

        Returns
        -------
        out : jnp.ndarray | dict[jnp.ndarray]
            Function over the output query coordinates
        """
        assert (output_query_features is None) == (
            self.out_query_feature_channels is None
        ), "pass output_query_features exactly when out_query_feature_channels is set"
        out: Any
        if isinstance(output_queries, dict):
            assert output_query_features is None, "not supported for dict queries"
            out = {}
            for key, out_p in output_queries.items():
                out_p = jnp.squeeze(out_p, axis=0)

                sub_output = self.gno_out(
                    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                    x=out_p,
                    f_y=latent_embed,
                    neighbors=neighbors_out,
                )
                sub_output = jnp.transpose(sub_output, (0, 2, 1))

                # Project pointwise to out channels
                sub_output = jnp.transpose(self.projection(sub_output), (0, 2, 1))

                out[key] = sub_output
        else:
            output_queries = jnp.squeeze(output_queries, axis=0)

            out = self.gno_out(
                y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                x=output_queries,
                f_y=latent_embed,
                neighbors=neighbors_out,
            )
            out = jnp.transpose(out, (0, 2, 1))
            if output_query_features is not None:
                out = jnp.concatenate(
                    [out, jnp.transpose(output_query_features, (0, 2, 1))], axis=1
                )

            # Project pointwise to out channels
            out = jnp.transpose(self.projection(out), (0, 2, 1))

        return out

    def __call__(
        self,
        input_geom: Any,
        latent_queries: Any,
        output_queries: Any,
        x: Any = None,
        latent_features: Any = None,
        ada_in: Any = None,
        neighbors_in: dict | None = None,
        neighbors_out: dict | None = None,
        output_query_features: Any = None,
        **kwargs: Any,
    ) -> Any:
        """The GINO's forward call:
        Input GNO --> FNOBlocks --> output GNO + projection to output queries.

        Thin wrapper around `encode` then `decode` (see those for the
        full parameter docs) -- kept as a single entry point for the normal
        forward-pass use case; behaviorally identical to the previous
        monolithic implementation.

        Returns
        -------
        out : jnp.ndarray | dict[jnp.ndarray]
            Function over the output query coordinates
        """
        if kwargs:
            warnings.warn(
                f"GINO.__call__() received unexpected keyword arguments: "
                f"{list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        latent_embed, latent_queries = self.encode(
            input_geom=input_geom,
            latent_queries=latent_queries,
            x=x,
            latent_features=latent_features,
            ada_in=ada_in,
            neighbors_in=neighbors_in,
        )
        return self.decode(
            latent_embed=latent_embed,
            latent_queries=latent_queries,
            output_queries=output_queries,
            neighbors_out=neighbors_out,
            output_query_features=output_query_features,
        )

    def precompute_neighbors(
        self,
        dataset: Any,
        output_n_points: int | None = None,
        log_dir: Any = None,
        split: str = "train",
    ) -> None:
        """Precompute and cache neighbor search results for all samples.

        Parameters
        ----------
        dataset : DictDataset
            The train or test dataset whose data_list will be updated in-place.
        output_n_points : int, optional
            Number of output query points. If None, all vertices are used.
        log_dir : str or Path, optional
            Directory to write per-sample neighbor count log files.
        split : str, optional
            Label for the dataset split. Default: "train".
        """
        from pathlib import Path

        latent_queries = dataset.constant["query_points"]  # shape [d1, d2, d3, 3]
        if hasattr(latent_queries, "numpy"):
            latent_queries = jnp.array(latent_queries.detach().cpu().numpy())
        else:
            latent_queries = jnp.array(latent_queries)
        latent_queries_flat = latent_queries.reshape(
            -1, latent_queries.shape[-1]
        )  # [m, 3]

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        log_file = None
        if log_dir is not None:
            log_path = log_dir / f"{split}.txt"
            log_file = open(log_path, "w")
            log_file.write(f"Neighbor search summary - {split}\n")
            log_file.write("=" * 60 + "\n\n")

        # Directly call the standalone neighbor search function
        # we cannot instantiate nn.Module subclasses (like NeighborSearch) outside
        # of Flax's init/apply lifecycle, and Flax wraps every method on this class.
        from ..layers.blocks.neighbor_search import kdtree_neighbor_search as _ns_fn

        return_norm = self.gno_weighting_function is not None

        def _to_jnp(t: Any) -> Any:
            if hasattr(t, "numpy"):  # torch.Tensor
                return jnp.array(t.detach().cpu().numpy())
            return jnp.array(t)

        try:
            for i, sample in enumerate(dataset.data_list):
                vertices = _to_jnp(sample["vertices"])  # [n, 3]

                if self.use_in_gno:
                    sample["neighbors_in"] = _ns_fn(
                        data=np.asarray(vertices),
                        queries=np.asarray(latent_queries_flat),
                        radius=self.in_gno_radius,
                        return_norm=return_norm,
                    )

                out_p = (
                    vertices if output_n_points is None else vertices[:output_n_points]
                )
                sample["neighbors_out"] = _ns_fn(
                    data=np.asarray(latent_queries_flat),
                    queries=np.asarray(out_p),
                    radius=self.out_gno_radius,
                    return_norm=return_norm,
                )

                if log_file is not None:
                    if self.use_in_gno:
                        splits_in = sample["neighbors_in"]["neighbors_row_splits"]
                        counts_in = (splits_in[1:] - splits_in[:-1]).tolist()
                    else:
                        counts_in = None

                    splits_out = sample["neighbors_out"]["neighbors_row_splits"]
                    counts_out = (splits_out[1:] - splits_out[:-1]).tolist()

                    log_file.write(f"Sample {i+1:03d}\n")
                    if counts_in is None:
                        log_file.write("  gno_in  - skipped (decoder-only)\n")
                    else:
                        log_file.write(
                            f"  gno_in  - total: {sum(counts_in)}  "
                            f"min: {min(counts_in)}  max: {max(counts_in)}  "
                            f"mean: {sum(counts_in)/len(counts_in):.2f}\n"
                        )
                    log_file.write(
                        f"  gno_out - total: {sum(counts_out)}  "
                        f"min: {min(counts_out)}  max: {max(counts_out)}  "
                        f"mean: {sum(counts_out)/len(counts_out):.2f}\n\n"
                    )
        finally:
            if log_file is not None:
                log_file.close()
