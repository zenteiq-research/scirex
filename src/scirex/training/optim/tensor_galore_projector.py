# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Tensor-GaLore low-rank gradient projection for memory-efficient training."""

from typing import Any

import jax
import jax.numpy as jnp
from tensorly import tenalg
from tensorly.decomposition import tucker


class TensorGaLoreProjector:
    """TensorGaLoreProjector implements low-rank gradient projection [1]
    for higher-order tensors as described in [2]. The original tensor is
    projected into a low-rank subspace using low-rank mode-wise factors
    obtained by Tucker decomposition. The parameters are optimized in this
    space to save memory and are then projected back into the full-rank
    space for use in a model.

    Parameters
    ----------
        rank : float, int or int tuple
            Goal rank of the transformed gradient tensor.
            Either a float corresponding to a percentage of params
            to preserve, or an list of int ranks corresponding to
            each mode of the tensor. If a single int is given, it is used
            for all modes.
        update_proj_gap : int, optional
            How often (number of iterations) to update the projection
            tensor used to project gradients to low rank, by default 200
        scale : float, optional
            An additional LR-like scalar factor by which the learned
            low-rank gradients are multiplied after projection back to full rank
            before being added to the original weights, by default 1.0
        warm_restart : bool, optional
            If True, uses the last computed projection tensor as the starting point
            for High-Order Iteration (HOI) computation of the new projection tensor,
            by default False
        activation_checkpoint : bool, optional
            Whether to use activation checkpointing for Tucker decomposition
            for maximum memory savings, by default False

    References
    ----------
    .. _[1] : Zhao, J, Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., Tian Y.
        (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
        Projection. ICML 2024, https://arxiv.org/abs/2403.03507.

    .. _[2] : George, R., Pitt, D., Zhao, J., Kossaifi, J., Luo, C., Tian, Y.,
        Anandkumar, A (2024). Tensor-GaLore: Memory-Efficient Training via
        Gradient Tensor Decomposition. arXiv preprint,
        https://arxiv.org/pdf/2501.02379.
    """

    def __init__(
        self,
        rank: float | int | tuple[int, ...],
        update_proj_gap: int = 200,
        scale: float = 1.0,
        tucker_n_iter_max: int = 10,
        warm_restart: bool = False,
        activation_checkpoint: bool = False,
    ) -> None:
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.warm_restart = warm_restart
        self.tucker_n_iter_max = tucker_n_iter_max
        self.activation_checkpoint = activation_checkpoint

        self.proj_tensor: list[jnp.ndarray] | None = None

    def state_dict(self) -> dict[str, Any]:
        """Returns projector config and the current projection tensor.

        Used to checkpoint a projector embedded in optimizer state without
        pickling the live object (VA finding F-03).
        """
        return {
            "rank": self.rank,
            "update_proj_gap": self.update_proj_gap,
            "scale": self.scale,
            "warm_restart": self.warm_restart,
            "tucker_n_iter_max": self.tucker_n_iter_max,
            "activation_checkpoint": self.activation_checkpoint,
            "proj_tensor": self.proj_tensor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores config and the projection tensor from `state_dict`."""
        self.rank = state_dict["rank"]
        self.update_proj_gap = state_dict["update_proj_gap"]
        self.scale = state_dict["scale"]
        self.warm_restart = state_dict["warm_restart"]
        self.tucker_n_iter_max = state_dict["tucker_n_iter_max"]
        self.activation_checkpoint = state_dict["activation_checkpoint"]
        self.proj_tensor = state_dict.get("proj_tensor")

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "TensorGaLoreProjector":
        """Reconstructs a projector from a dict produced by `state_dict()`.

        Unlike `load_state_dict`, this builds a new instance -- used when
        restoring optimizer state that has no live projector to load into.
        """
        obj = cls(
            rank=state_dict["rank"],
            update_proj_gap=state_dict["update_proj_gap"],
            scale=state_dict["scale"],
            tucker_n_iter_max=state_dict["tucker_n_iter_max"],
            warm_restart=state_dict["warm_restart"],
            activation_checkpoint=state_dict["activation_checkpoint"],
        )
        obj.proj_tensor = state_dict.get("proj_tensor")
        return obj

    def project(self, full_rank_grad: jnp.ndarray, iter: int) -> jnp.ndarray:
        if self.proj_tensor is None and iter % self.update_proj_gap == 0:
            self.proj_tensor = self.get_projection_tensor(full_rank_grad, self.rank)
        # iter=0 is always divisible by update_proj_gap, so proj_tensor is
        # guaranteed to be set by the first call to project().
        assert self.proj_tensor is not None
        # In JAX, arrays are on the correct device automatically
        # No need for .to(device) conversion unlike PyTorch
        transformed_low_rank = self.transform(self.proj_tensor, full_rank_grad)
        return transformed_low_rank

    def project_back(self, low_rank_grad: jnp.ndarray) -> jnp.ndarray:
        # project() must run before project_back() in the same step, so
        # proj_tensor is already initialized here.
        assert self.proj_tensor is not None
        full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)
        return full_rank_grad * self.scale

    # Tucker decomp: higher-order SVD
    def get_projection_tensor(
        self, weights: jnp.ndarray, rank: float | int | tuple[int, ...]
    ) -> list[jnp.ndarray]:
        # Convert to complex if needed
        if jnp.iscomplexobj(weights):
            matrix = weights.astype(jnp.complex64)
        else:
            matrix = weights

        # if warm_restart is True, initialize with
        # existing projection tensor if it exists
        init: str | list[jnp.ndarray]
        if self.warm_restart and self.proj_tensor is not None:
            init = self.proj_tensor
        else:
            init = "svd"  # default setting

        if self.activation_checkpoint:
            # JAX checkpoint for memory-efficient gradient computation
            def _tucker_decomp() -> tuple[jnp.ndarray, list[jnp.ndarray]]:
                return tucker(matrix, rank=rank, init=init)

            core, factors = jax.checkpoint(_tucker_decomp)()
        else:
            core, factors = tucker(matrix, rank=rank, init=init)
        del core
        return factors

    def transform(self, factors: list[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        # unpack/drop core
        if self.activation_checkpoint:
            # JAX checkpoint for memory-efficient computation
            def _transform() -> jnp.ndarray:
                return tenalg.multi_mode_dot(x, factors, transpose=True)

            return jax.checkpoint(_transform)()
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(
        self, factors: list[jnp.ndarray], x: jnp.ndarray
    ) -> jnp.ndarray:
        if self.activation_checkpoint:
            # JAX checkpoint for memory-efficient computation
            def _inverse_transform() -> jnp.ndarray:
                return tenalg.multi_mode_dot(x, factors)

            return jax.checkpoint(_inverse_transform)()
        return tenalg.multi_mode_dot(x, factors)
