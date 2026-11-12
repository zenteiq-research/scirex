# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from Transolver (https://github.com/thuml/Transolver),
# Copyright (c) 2024 THUML @ Tsinghua University, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Transolver: a Transformer solver with physics attention for PDEs."""

import warnings
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from scirex.layers.blocks.transolver_block import TransolverBlock
from scirex.layers.common.embeddings import timestep_embedding
from scirex.layers.common.mlp import MLP
from scirex.models.base import BaseModel


class Transolver(BaseModel, name="transolver"):
    """Transolver: a Transformer solver built on physics attention.

    Maps point-wise inputs on an irregular mesh or a structured 2D/3D grid to
    point-wise outputs. Every block attends over a small number of learned
    "slices" of the points (physics attention) instead of over all pairs of
    points, so the cost grows linearly with the number of points.

    Input widths are inferred from the arrays passed to `__call__`; `space_dim`
    and `fun_dim` are accepted for parity with the reference implementation but
    are not used by the model.

    Parameters
    ----------
    n_layers : int
        Number of Transolver blocks.
    n_hidden : int
        Hidden width of the embedding and of every block.
    n_head : int
        Number of attention heads; must divide `n_hidden`.
    slice_num : int
        Number of slices (physical states) in the physics attention.
    mlp_ratio : int
        Expansion factor of the hidden layer of each block's MLP.
    out_dim : int
        Number of output channels, produced by the last block.
    dropout_rate : float
        Dropout probability. It only acts when `__call__` receives
        `deterministic=False` and a `"dropout"` RNG; otherwise `__call__`
        warns that it has no effect.
    act : Callable
        Activation function.
    geometry : str
        `"irregular"` (point clouds), `"structured_2d"` (needs `H` and `W`)
        or `"structured_3d"` (needs `H`, `W` and `D`).
    H, W, D : int, optional
        Grid resolution for the structured geometries.
    kernel : int
        Kernel size of the convolutions inside the physics attention on
        structured grids.
    conv_init : str
        Initialisation of the structured-2D slice convolutions, `"flax"` or
        `"torch"` (PyTorch's default).
    time_input : bool
        Whether `__call__` takes a time `T` that is embedded and added to the
        hidden state.
    unified_pos : bool
        Replace the input coordinates by their distance to a `ref` x `ref`
        (x `ref` in 3D) reference grid.
    ref : int
        Number of reference points per axis when `unified_pos` is set.
    concat_pos : bool
        With `unified_pos`, concatenate the distances to the input instead of
        replacing it.
    grid_range : tuple, optional
        `(low, high)` per spatial dimension of the reference grid; defaults to
        `(0, 1)` in every dimension.
    layer_norm_epsilon : float
        Epsilon of every LayerNorm (Flax default 1e-6; PyTorch uses 1e-5).

    Examples
    --------
    >>> import jax, jax.numpy as jnp
    >>> from scirex.models import Transolver
    >>> model = Transolver(space_dim=2, n_layers=2, n_hidden=16, n_head=2, slice_num=8)
    >>> x = jnp.ones((2, 100, 2))  # (batch, points, coordinates)
    >>> f = jnp.ones((2, 100, 1))  # (batch, points, input features)
    >>> params = model.init(jax.random.PRNGKey(0), x, f)
    >>> model.apply(params, x, f).shape
    (2, 100, 1)

    Notes
    -----
    Based on Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General
    Geometries" (ICML 2024), https://github.com/thuml/Transolver.
    """

    space_dim: int = 1
    n_layers: int = 5
    n_hidden: int = 256
    # NOTE: wired through to every TransolverBlock/MLP/attention Dropout
    # layer, but currently a no-op end-to-end: no train_transolver_*_jax.py
    # script calls model.apply(..., deterministic=False, rngs={"dropout": ...}),
    # so deterministic defaults to True everywhere and Dropout never fires.
    # __call__ warns if dropout_rate > 0 and deterministic is left True.
    dropout_rate: float = 0.0
    n_head: int = 8
    time_input: bool = False
    act: Callable = nn.gelu
    mlp_ratio: int = 1
    fun_dim: int = 1
    out_dim: int = 1
    slice_num: int = 32
    ref: int = 8
    unified_pos: bool = False
    geometry: str = "irregular"  # "irregular" | "structured_2d" | "structured_3d"
    # False: unified_pos replaces x. True: concatenates onto x (Airfoil/Car).
    concat_pos: bool = False
    # tuple of (low, high) pairs, one per spatial dim. Defaults to (0., 1.) per dim.
    grid_range: tuple | None = None
    H: int | None = None
    W: int | None = None
    D: int | None = None
    kernel: int = 3
    # Epsilon of every block's LayerNorm. The default is Flax's (1e-6);
    # torch.nn.LayerNorm uses 1e-5.
    layer_norm_epsilon: float = 1e-6
    # Init of the structured-2D slice convolutions: "flax" (default) or
    # "torch" (torch.nn.Conv2d's default), see StructuredMesh2D.
    conv_init: str = "flax"

    def setup(self) -> None:
        """Validate the geometry; create the embedding MLP, blocks and placeholder."""
        if self.geometry not in ("irregular", "structured_2d", "structured_3d"):
            raise ValueError(f"Unknown geometry: {self.geometry}")
        if self.geometry == "structured_2d" and (self.H is None or self.W is None):
            raise ValueError("geometry='structured_2d' requires H and W to be set.")
        if self.geometry == "structured_3d" and (
            self.H is None or self.W is None or self.D is None
        ):
            raise ValueError("geometry='structured_3d' requires H, W, and D to be set.")

        self.preprocess = MLP(
            features=[self.n_hidden * 2, self.n_hidden],
            activation=self.act,
            dropout_rate=0.0,
            activation_last=False,
            residual=False,
        )
        if self.time_input:
            self.time_fc = MLP(
                features=[self.n_hidden, self.n_hidden],
                activation=nn.silu,
                activation_last=False,
            )
        self.blocks = [
            TransolverBlock(
                num_heads=self.n_head,
                hidden_dim=self.n_hidden,
                dropout_rate=self.dropout_rate,
                geometry=self.geometry,
                act=self.act,
                mlp_ratio=self.mlp_ratio,
                out_dim=self.out_dim,
                slice_num=self.slice_num,
                H=self.H,
                W=self.W,
                D=self.D,
                kernel=self.kernel,
                layer_norm_epsilon=self.layer_norm_epsilon,
                conv_init=self.conv_init,
                last_layer=(i == self.n_layers - 1),
            )
            for i in range(self.n_layers)
        ]
        self.placeholder = self.param(
            "placeholder",
            lambda key, shape: (1.0 / self.n_hidden)
            * jax.random.uniform(key, shape, dtype=jnp.float32),
            (self.n_hidden,),
        )

    def _grid_range_2d(self) -> tuple:
        return (
            self.grid_range if self.grid_range is not None else ((0.0, 1.0), (0.0, 1.0))
        )

    def _grid_range_3d(self) -> tuple:
        return (
            self.grid_range
            if self.grid_range is not None
            else ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        )

    def _get_grid_irregular(self, pos: jnp.ndarray) -> jnp.ndarray:
        # pos: (B, N, ndim) real point coordinates -- read fresh every forward
        # call, since irregular meshes can have different point coordinates
        # per sample. Distinct from `x` (the function/feature values) when
        # concat_pos=True (Airfoil/Car pass a separate coordinate tensor here).
        # ndim is read from pos itself (2D for most PDE-Solving datasets, 3D
        # for e.g. car-mesh coordinates) rather than hardcoded, so this
        # supports any irregular-mesh coordinate dimensionality.
        ndim = pos.shape[-1]
        ref_bounds = (
            self.grid_range if self.grid_range is not None else ((0.0, 1.0),) * ndim
        )
        ref_axes = [
            jnp.linspace(lo, hi, self.ref, dtype=jnp.float32) for lo, hi in ref_bounds
        ]
        ref_mesh = jnp.meshgrid(*ref_axes, indexing="ij")
        grid_ref = jnp.stack(ref_mesh, axis=-1).reshape(self.ref**ndim, ndim)
        diff = pos[:, :, None, :] - grid_ref[None, None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (B, N, ref**ndim)

    def _get_grid_structured(self, batch_size: int) -> jnp.ndarray:
        # Independent of the real input -- positions are implied by the fixed
        # H x W [x D] resolution, not read from data.
        if self.geometry == "structured_2d":
            assert (
                self.H is not None and self.W is not None
            ), "geometry='structured_2d' requires H and W to be set."
            (lox, hix), (loy, hiy) = self._grid_range_2d()
            gx, gy = jnp.meshgrid(
                jnp.linspace(lox, hix, self.H, dtype=jnp.float32),
                jnp.linspace(loy, hiy, self.W, dtype=jnp.float32),
                indexing="ij",
            )
            grid = jnp.stack([gx, gy], axis=-1)  # (H, W, 2)
            n_points, ndim = self.H * self.W, 2
            ref_bounds = self._grid_range_2d()
        else:  # structured_3d
            assert (
                self.H is not None and self.W is not None and self.D is not None
            ), "geometry='structured_3d' requires H, W, and D to be set."
            (lox, hix), (loy, hiy), (loz, hiz) = self._grid_range_3d()
            gx, gy, gz = jnp.meshgrid(
                jnp.linspace(lox, hix, self.H, dtype=jnp.float32),
                jnp.linspace(loy, hiy, self.W, dtype=jnp.float32),
                jnp.linspace(loz, hiz, self.D, dtype=jnp.float32),
                indexing="ij",
            )
            grid = jnp.stack([gx, gy, gz], axis=-1)  # (H, W, D, 3)
            n_points, ndim = self.H * self.W * self.D, 3
            ref_bounds = self._grid_range_3d()

        ref_axes = [
            jnp.linspace(lo, hi, self.ref, dtype=jnp.float32) for lo, hi in ref_bounds
        ]
        ref_mesh = jnp.meshgrid(*ref_axes, indexing="ij")
        grid_ref = jnp.stack(ref_mesh, axis=-1).reshape(self.ref**ndim, ndim)

        diff = grid.reshape(n_points, 1, ndim) - grid_ref[None, :, :]
        pos = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # (n_points, ref**ndim)
        return jnp.broadcast_to(pos, (batch_size, *pos.shape))

    def __call__(
        self,
        x: jnp.ndarray,
        fx: jnp.ndarray | None = None,
        pos: jnp.ndarray | None = None,
        T: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply the model.

        Parameters
        ----------
        x : jnp.ndarray
            Point-wise inputs of shape `(batch, n_points, channels)`; the
            coordinates when `fx` is given.
        fx : jnp.ndarray, optional
            Point-wise function values of shape `(batch, n_points, channels)`,
            concatenated to `x`.
        pos : jnp.ndarray, optional
            Point coordinates `(batch, n_points, ndim)` used for `unified_pos`
            on irregular meshes; defaults to `x`.
        T : jnp.ndarray, optional
            Time of shape `(batch,)`, used when `time_input` is set.
        deterministic : bool
            Disable dropout when `True`.

        Returns
        -------
        jnp.ndarray
            Outputs of shape `(batch, n_points, out_dim)`.
        """
        if self.dropout_rate > 0 and deterministic:
            warnings.warn(
                f"dropout_rate={self.dropout_rate} has no effect: "
                "deterministic=True disables dropout. Call with "
                "deterministic=False and an RNG under "
                'rngs={"dropout": ...} to use it.',
                stacklevel=2,
            )

        if self.unified_pos:
            if self.geometry == "irregular":
                grid_pos = pos if pos is not None else x
                new_pos = self._get_grid_irregular(grid_pos)
                x = (
                    jnp.concatenate([x, new_pos], axis=-1)
                    if self.concat_pos
                    else new_pos
                )
            else:
                new_pos = self._get_grid_structured(x.shape[0])
                x = (
                    jnp.concatenate([x, new_pos], axis=-1)
                    if self.concat_pos
                    else new_pos
                )

        if fx is not None:
            fx = jnp.concatenate([x, fx], axis=-1)
            fx = self.preprocess(fx)
            if self.geometry == "irregular":
                fx = fx + self.placeholder[None, None, :]
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            time_emb = self.time_fc(timestep_embedding(T, self.n_hidden))
            fx = fx + time_emb[:, None, :]

        for block in self.blocks:
            fx = block(fx, deterministic=deterministic)

        return fx
