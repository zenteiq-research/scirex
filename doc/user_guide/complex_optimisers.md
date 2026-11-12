<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Complex-Parameter Optimisers

Some ScireX models have **complex-valued weights**. A standard Optax optimiser
applied to them with JAX gradients moves those weights in the wrong direction,
without any error or warning. This page explains why, and how to train such
models correctly with [`scirex.training.adamw_cx`](../api/training.md#scirex.training.adamw_cx).

## When this matters

| Model | Complex weights | Optimiser |
|---|---|---|
| FNO | Yes, in every spectral convolution | [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) |
| FNOClassic2d | Yes, unless `split_complex_params=True` | [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) |
| GINO | Yes, in its FNO block, unless `fno_split_complex_params=True` | [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) |
| GIWO (experimental) | No; its WNO latent operator is real | either |
| WNO | No | either |
| Transolver | No | either |

For a model with only real weights, [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) and `optax.adamw` give the same
updates, apart from a bias-correction detail that matters only when the
gradients are smaller than `eps`. [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) is therefore a safe default for
every model.

To check your own model, look for complex leaves in its parameters:

```python
import jax
import jax.numpy as jnp

from scirex.models import FNO

model = FNO(in_channels=1, out_channels=1, hidden_channels=16, n_modes=(8, 8))
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 1, 32, 32)))
print(any(jnp.iscomplexobj(p) for p in jax.tree_util.tree_leaves(params)))  # True
```

## The JAX convention

For a real loss `L` and a complex weight `w = x + iy`, the direction of
steepest ascent is `∂L/∂x + i ∂L/∂y`. PyTorch stores this as the gradient,
and Optax's transforms assume they are given it.

`jax.grad` returns its **complex conjugate**, `∂L/∂x - i ∂L/∂y`:

```python
jax.grad(lambda w: jnp.abs(w) ** 2)(jnp.complex64(1 + 1j))
# Array(2.-2.j, dtype=complex64); the steepest ascent of x² + y² at (1, 1) is 2+2j
```

Both conventions are correct; they differ in what they call the gradient. The
problem appears when a JAX gradient is passed to an optimiser written for the
other convention. The real part of the update then descends, but the imaginary
part **ascends**. On the loss `|w|²` from `w = 1 + 1j`, with learning rate 0.1:

| Optimiser | Loss after 1 step | after 10 steps | after 50 steps | `w` after 50 steps |
|---|---|---|---|---|
| `optax.adamw` | 2.01 | 3.47 | 42.9 | `-0.01 + 6.55j` |
| [`scirex.training.adamw_cx`](../api/training.md#scirex.training.adamw_cx) | 1.73 | 0.21 | 0.0073 | `0.06 + 0.06j` |

The second moment is not the problem: Optax 0.2.8 and later already use
`|g|²` for complex gradients. Only the direction is.

In a network the error is less visible: the real parts still learn, so the
loss falls, only more slowly and to a worse value. On the Pipe benchmark, a
line-for-line Flax port of Geo-FNO's FNO reached 21.77% test error after 25
epochs with plain Optax, and 2.41% with the gradients conjugated first
(PyTorch: 2.14%); see the
[Pipe FNO page](../benchmarks/pipe/results/comparative.md#optimiser-for-complex-weights).

## `adamw_cx`

[`scirex.training.adamw_cx`](../api/training.md#scirex.training.adamw_cx) is AdamW for complex weights, as an Optax
gradient transformation. It

1. conjugates complex gradients, so the update follows PyTorch's convention;
2. keeps the second moment real (`|g|²`, stored as float32);
3. applies decoupled weight decay, as AdamW does;
4. uses PyTorch's bias correction.

Real weights are updated exactly as by AdamW. Use it wherever you would use
`optax.adamw`:

```python
import optax

from scirex.training import adamw_cx

schedule = optax.cosine_decay_schedule(1e-3, decay_steps=10_000)
tx = optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
    learning_rate=schedule, weight_decay=1e-4, eps=1e-8
)
tx = optax.chain(optax.clip_by_global_norm(1.0), tx)  # optional clipping
opt_state = tx.init(params)
```

Three details:

- **`hyperparam_dtype=jnp.float32`.** `optax.inject_hyperparams` stores the
  hyperparameters in the dtype of the parameters. With complex parameters the
  learning rate would become `complex64`; pass `hyperparam_dtype` to keep it
  real.
- **`eps`.** The default is `1e-6`. The benchmarks use `1e-8`, the PyTorch
  default. With small gradients, as with the Ritz energy, `eps` sets the size
  of the step (see [Models and Losses](models_and_losses.md#the-ritz-energy)).
- **Schedules.** [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) takes a number as `learning_rate`, not a
  schedule; [`adamw_cx(schedule)`](../api/training.md#scirex.training.adamw_cx) raises a `TypeError` on the first update.
  Wrap it in `optax.inject_hyperparams`, as above, or build the chain from
  [`scale_by_adam_cx`](../api/training.md#scirex.training.optim.scale_by_adam_cx), as the Pipe benchmark does. Either way the weight decay
  is multiplied by the current learning rate, as in `torch.optim.AdamW`:

```python
from scirex.training.optim import scale_by_adam_cx

tx = optax.chain(
    scale_by_adam_cx(eps=1e-8),
    optax.add_decayed_weights(1e-5),
    optax.scale_by_learning_rate(schedule),
)
```

## `make_optimizer`

ScireX 1.0 has no library function that builds the optimiser of a run. Each
benchmark script has its own `make_optimizer`, which turns the run's recipe
into an Optax transformation in one place. The pattern, simplified from
[`scripts/benchmarks/darcy/train.py`](../../scripts/benchmarks/darcy/train.py) (which also offers a step-decay
schedule):

```python
OPTIMIZERS = {"adamw": optax.adamw, "adamw_cx": adamw_cx}


def make_optimizer(opt: dict, steps_per_epoch: int) -> optax.GradientTransformation:
    """The run's optimiser: a learning-rate schedule, optionally behind
    global-norm gradient clipping."""
    schedule = optax.cosine_onecycle_schedule(
        transition_steps=steps_per_epoch * opt["epochs"], peak_value=opt["lr"]
    )
    tx = optax.inject_hyperparams(OPTIMIZERS[opt["optimizer"]])(
        learning_rate=schedule, weight_decay=opt["weight_decay"], eps=1e-8
    )
    if opt["grad_clip"] is not None:
        tx = optax.chain(optax.clip_by_global_norm(opt["grad_clip"]), tx)
    return tx
```

The Darcy recipe names `adamw` for its real-weight models (WNO, Transolver)
and [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) for FNO and GINO. The Pipe benchmark accepts only [`adamw_cx`](../api/training.md#scirex.training.adamw_cx)
and raises for any other name. When you write your own, keep the optimiser
name in the config, so the choice is saved with the checkpoint.

## The guard and `allow_raw_complex`

ScireX 1.0 has **no guard**: nothing stops a plain Optax optimiser from being
used on complex weights, and there is no `allow_raw_complex` switch. Until
there is one, add the check to your own `make_optimizer`:

```python
def check_optimizer(name: str, params, allow_raw_complex: bool = False) -> None:
    """Refuse a non-complex-aware optimiser for a model with complex weights."""
    has_complex = any(jnp.iscomplexobj(p) for p in jax.tree_util.tree_leaves(params))
    if has_complex and name != "adamw_cx" and not allow_raw_complex:
        raise ValueError(
            f"optimizer {name!r} on complex weights moves their imaginary parts "
            "the wrong way; use 'adamw_cx', or pass allow_raw_complex=True"
        )
```

Pass `allow_raw_complex=True` only when you need the raw behaviour on purpose,
for example to reproduce a run trained before the fix.

## Alternative: store complex weights as real pairs

[`FNOClassic2d(split_complex_params=True)`](../api/models.md#scirex.models.FNOClassic2d) and
[`GINO(fno_split_complex_params=True)`](../api/models.md#scirex.models.GINO) store every complex weight as two real
arrays, its real and imaginary parts. All parameters are then real, so any
Optax optimiser updates them correctly. In the Pipe check above, the Flax port
with its complex weights stored this way reached 2.29% with plain Optax,
against 2.41% with conjugated gradients. [`FNO`](../api/models.md#scirex.models.FNO) has no such option; use
[`adamw_cx`](../api/training.md#scirex.training.adamw_cx) with it.

## Checklist

- Does the model have complex weights? Check with `jnp.iscomplexobj` on the
  parameter leaves.
- If yes, use [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) (or split the weights into real pairs where the model
  allows it).
- With `optax.inject_hyperparams`, pass `hyperparam_dtype=jnp.float32`.
- Set `eps` deliberately: `1e-8` matches PyTorch and the benchmarks.
- Save the optimiser's name and settings with the checkpoint.

## See also

- [Workflow](workflow.md#4-optimiser): the optimiser in a complete run.
- [API reference: `scirex.training`](../api/training.md).
- [Pipe FNO page](../benchmarks/pipe/results/comparative.md#optimiser-for-complex-weights)
  and [Navier-Stokes comparison](../benchmarks/navier_stokes/results/comparative.md#optimiser-for-complex-weights):
  how the issue was found and its effect on the benchmarks.
