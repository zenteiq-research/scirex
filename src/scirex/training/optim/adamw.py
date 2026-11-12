# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""AdamW optimizer with correct complex-parameter handling, built on optax."""

import math
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .tensor_galore_projector import TensorGaLoreProjector

# ── Complex-aware optax AdamW ─────────────────────────────────────────────────
# PyTorch uses grad * conj(grad) = |g|² (real) for the second moment of complex
# parameters. Standard optax.adam uses jnp.square(g) = g*g (complex), which
# breaks the adaptive scaling for SpectralConv weights. This implementation
# matches PyTorch by keeping the second moment real for complex parameters.


class _AdamCxState(NamedTuple):
    # `count` shadows tuple.count's inherited signature; matches optax's own
    # ScaleByAdamState convention (same field name, same trade-off).
    count: jnp.ndarray  # type: ignore[assignment]
    mu: optax.Updates
    nu: optax.Updates  # always real-dtype


def scale_by_adam_cx(
    b1: float = 0.9, b2: float = 0.999, eps: float = 1e-6
) -> optax.GradientTransformation:
    """Adam scale transform that uses |g|² for complex params (matches PyTorch).

    JAX jax.grad returns conj(PyTorch_grad) for complex parameters (Wirtinger
    convention difference). We conjugate complex gradients here so that the
    first moment accumulates in the same direction as PyTorch's optimizer,
    and weight decay shrinks (rather than grows) the imaginary component.
    |g|² is unchanged by conjugation so nu is not affected.
    """

    def init_fn(params: optax.Params) -> _AdamCxState:
        mu = jax.tree_util.tree_map(jnp.zeros_like, params)
        nu = jax.tree_util.tree_map(
            lambda p: jnp.zeros(
                p.shape, dtype=jnp.float32 if jnp.iscomplexobj(p) else p.dtype
            ),
            params,
        )
        return _AdamCxState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(
        updates: optax.Updates,
        state: _AdamCxState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, _AdamCxState]:
        count = state.count + jnp.ones([], jnp.int32)
        # Conjugate complex gradients to match PyTorch's Wirtinger convention.
        # jax.grad returns ∂L/∂w for complex w; PyTorch stores ∂L/∂w̄ = conj(∂L/∂w).
        # Conjugating here makes mu accumulate in the correct descent direction.
        updates = jax.tree_util.tree_map(
            lambda g: jnp.conj(g) if jnp.iscomplexobj(g) else g, updates
        )
        mu = jax.tree_util.tree_map(
            lambda m, g: b1 * m + (1.0 - b1) * g, state.mu, updates
        )
        nu = jax.tree_util.tree_map(
            lambda v, g: (
                b2 * v + (1.0 - b2) * jnp.abs(g) ** 2
                if jnp.iscomplexobj(g)
                else b2 * v + (1.0 - b2) * g**2
            ),
            state.nu,
            updates,
        )
        bc1 = 1.0 - b1**count
        bc2 = 1.0 - b2**count
        # PyTorch-style bias correction: step = sqrt(bc2)/bc1, denom uses raw
        # (un-corrected) second moment.  This differs from the standard optax
        # approach (which bias-corrects both m and v independently) only when
        # |g| << eps - a regime where the raw second moment is tiny and eps
        # dominates the denominator.  Using the PyTorch formula keeps the
        # effective step in that regime proportional to sqrt(bc2)/bc1,
        # matching PyTorch.
        step = jnp.sqrt(bc2) / bc1
        new_updates = jax.tree_util.tree_map(
            lambda m, v: step * m / (jnp.sqrt(v) + eps), mu, nu
        )
        return new_updates, _AdamCxState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def adamw_cx(
    learning_rate: float,
    weight_decay: float = 0.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
) -> optax.GradientTransformation:
    """AdamW that handles complex-valued parameters, as an Optax transformation.

    Compatible with `optax.inject_hyperparams`. For complex parameters the
    second moment uses `|g|**2` (real), as PyTorch does; see
    `scale_by_adam_cx`. Weight decay is decoupled from the Adam update.

    Parameters
    ----------
    learning_rate : float
        Step size.
    weight_decay : float, optional
        Decoupled weight-decay coefficient, by default 0.0.
    b1 : float, optional
        Exponential decay rate of the first moment, by default 0.9.
    b2 : float, optional
        Exponential decay rate of the second moment, by default 0.999.
    eps : float, optional
        Term added to the denominator for numerical stability, by default 1e-6.

    Returns
    -------
    optax.GradientTransformation
        The chained Adam scaling, weight decay and learning-rate scaling.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import optax
    >>> params = {"w": jnp.ones(3, dtype=jnp.complex64)}
    >>> opt = adamw_cx(learning_rate=1e-3)
    >>> state = opt.init(params)
    >>> grads = {"w": jnp.ones(3, dtype=jnp.complex64)}
    >>> updates, state = opt.update(grads, state, params)
    >>> params = optax.apply_updates(params, updates)
    >>> params["w"].dtype
    dtype('complex64')
    """
    return optax.chain(
        scale_by_adam_cx(b1=b1, b2=b2, eps=eps),
        optax.add_decayed_weights(weight_decay),
        optax.scale(-learning_rate),
    )


class AdamW:
    """
    Implements AdamW (Adam with weight decay fix [1]), and offers
    optional Tensor-GaLore projection (see [2] and [3]) for memory-efficient training.

    Note: This JAX implementation maintains optimizer state but requires
    external parameter update logic (typically handled by training loop).

    Parameters
    ----------
    params : Iterable, optional
        Iterable of parameters to optimize or dictionaries defining parameter groups.
    lr : float, optional, defaults to 0.001
        The learning rate to use.
    betas : Tuple[float,float], optional, defaults to (0.9, 0.999)
        Adam's betas parameters (b1, b2).
    eps : float, optional, defaults to 1e-06
        Adam's epsilon for numerical stability.
    weight_decay : float, optional, defaults to 0.0
        Decoupled weight decay to apply.
    correct_bias : bool, optional, defaults to True
        Whether or not to correct bias in Adam.
    galore_params : Iterable, optional
        Iterable of parameters to optimize in low-rank subspace
    galore_rank : float, int or int tuple
        Rank of low-rank subspace in which to optimize galore_params.
        Either a float corresponding to a percentage of params
        to preserve, or an list of int ranks corresponding to
        each mode of the tensor. If a single int is given, it is used
        for all modes (see `src/scirex/training/optim/tensor_galore_projector.py`)
    galore_update_proj_gap : int, defaults to 50
        Number of optimizer steps before projection tensors are recomputed.
    galore_scale : float, defaults to 1.0
        Additional lr-like scalar by which galore parameters are multiplied
        before update
    activation_checkpoint : bool, default False
        Whether to use activation checkpointing during projection
    warm_restart : bool, default True
        Whether to use warm restart for GaLore projection

    References
    ----------
    .. _[1] : Loschchilov, I. and Hutter, F. (2019). Decoupled Decay Regularization.
         ICLR 2019, https://arxiv.org/pdf/1711.05101.

    .. _[2] : Zhao, J, Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., Tian Y.
        (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
        Projection. ICML 2024, https://arxiv.org/abs/2403.03507.

    .. _[3] : George, R., Pitt, D., Zhao, J., Kossaifi, J., Luo, C., Tian, Y.,
        Anandkumar, A (2024). Tensor-GaLore: Memory-Efficient Training via
        Gradient Tensor Decomposition. arXiv preprint,
        https://arxiv.org/pdf/2501.02379.
    """

    def __init__(
        self,
        params: Iterable | None = None,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        galore_params: Iterable | None = None,
        galore_rank: float | int | tuple[int, ...] = 1.0,
        galore_update_proj_gap: int = 50,
        galore_scale: float = 1.0,
        activation_checkpoint: bool = False,
        warm_restart: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        self.param_groups: list[dict[str, Any]] = []

        # Add default param group
        self.param_groups.append(
            {
                "params": params if params is not None else [],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "correct_bias": correct_bias,
                "galore": False,
            }
        )

        # Keep optional GaLore parameters separate for projection
        if galore_params is not None:
            self.add_param_group(
                {
                    "params": galore_params,
                    "rank": galore_rank,
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "correct_bias": correct_bias,
                    "galore": True,
                }
            )

        self.galore_rank = galore_rank
        self.activation_checkpoint = activation_checkpoint
        self.warm_restart = warm_restart
        self.galore_update_proj_gap = galore_update_proj_gap
        self.galore_scale = galore_scale

        # Optimizer state
        self.state: dict[tuple[int, int], dict[str, Any]] = {}
        self.learning_rate = lr

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add a param group to the optimizer."""
        self.param_groups.append(param_group)

    def step(
        self, closure: Callable | None = None
    ) -> tuple[Any, dict[tuple[int, int], Any]]:
        """
        Performs a single optimization step.

        Arguments:
            closure (Callable, optional): A closure that reevaluates the
                model and returns the loss.

        Returns
        -------
        loss : float or None
            Loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        updates: dict[tuple[int, int], Any] = {}

        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                if not hasattr(p, "grad") or p.grad is None:
                    continue

                grad = p.grad

                # Use (group_idx, param_idx) as unique key for parameter state
                param_key = (group_idx, param_idx)

                if param_key not in self.state:
                    self.state[param_key] = {}

                state = self.state[param_key]

                if "step" not in state:
                    state["step"] = 0

                # GaLore Projection
                if group.get("galore", False):
                    if "projector" not in state:
                        state["projector"] = TensorGaLoreProjector(
                            rank=self.galore_rank,
                            update_proj_gap=self.galore_update_proj_gap,
                            scale=self.galore_scale,
                            activation_checkpoint=self.activation_checkpoint,
                            warm_restart=self.warm_restart,
                        )

                    # track tensor shape for projection back
                    proj_input = grad
                    grad = state["projector"].project(proj_input, state["step"])

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = jnp.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = jnp.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
                if jnp.iscomplexobj(grad):
                    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * jnp.real(
                        grad * jnp.conj(grad)
                    )
                else:
                    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * (grad * grad)

                denom = jnp.sqrt(exp_avg_sq) + group["eps"]

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # compute norm gradient
                norm_grad = exp_avg / denom

                # GaLore Projection Back
                if group.get("galore", False):
                    norm_grad = state["projector"].project_back(norm_grad)

                # Compute parameter update
                param_update = -step_size * norm_grad

                # Add weight decay (decoupled weight decay, not L2 regularization)
                # This is the correct way to use weight decay with Adam
                if group["weight_decay"] > 0.0:
                    param_update = (
                        param_update - group["lr"] * group["weight_decay"] * p
                    )

                # Store update for this parameter
                updates[param_key] = param_update

                # Update state
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

        return loss, updates

    def state_dict(self) -> dict[str, Any]:
        """Returns optimizer state as a serialization-safe dictionary.

        `self.state` is keyed by `tuple[int, int]` and may embed a live
        `TensorGaLoreProjector`; `self.param_groups` embeds the live
        parameter arrays. Neither is safe to pass to a non-pickle
        serializer as-is (VA finding F-03), so this encodes state keys as
        `[group, index]` pairs, converts any embedded projector to its
        own state dict, and drops `params` from the param groups --
        matching PyTorch's own `optimizer.state_dict()` convention:
        parameters belong to the model checkpoint, not the optimizer's.
        `load_state_dict` restores onto an already-constructed optimizer
        that already has them.
        """
        encoded_state = []
        for key, per_param_state in self.state.items():
            entry = dict(per_param_state)
            projector = entry.get("projector")
            if projector is not None and hasattr(projector, "state_dict"):
                entry["projector"] = projector.state_dict()
            encoded_state.append([list(key), entry])

        param_groups = [
            {k: v for k, v in group.items() if k != "params"}
            for group in self.param_groups
        ]

        return {
            "state": encoded_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores optimizer state from a dict produced by `state_dict()`.

        Must be called on an optimizer already constructed with the same
        params: `param_groups[i]["params"]` is kept from the live object,
        not the checkpoint (`state_dict()` never saves it).
        """
        self.state = {}
        for key, entry in state_dict.get("state", []):
            entry = dict(entry)
            projector = entry.get("projector")
            if isinstance(projector, dict):
                entry["projector"] = TensorGaLoreProjector.from_state_dict(projector)
            self.state[tuple(key)] = entry

        for group, saved_group in zip(
            self.param_groups, state_dict.get("param_groups", []), strict=False
        ):
            group.update({k: v for k, v in saved_group.items() if k != "params"})
