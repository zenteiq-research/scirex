<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# JAX Patterns

A short reference for the JAX/Flax idioms used throughout ScireX. These
are the patterns that most often differ from PyTorch.

## Stateless modules and explicit parameters

Flax modules hold **no parameters of their own**. Parameters live in a separate
pytree produced by `init` and threaded explicitly through `apply`:

```python
params = model.init(key, x)       # build parameters
out = model.apply(params, x)      # pure function: (params, x) -> out
```

This functional separation is what makes models `jit`/`grad`/`vmap` composable.

## Immutability and functional updates

JAX arrays are immutable. Replace in-place mutation with the `.at[]` API:

```python
# PyTorch:  x[mask] = 0
x = x.at[mask].set(0)
```

## `jit`, `vmap`, `grad`

- `jax.jit` compiles a function with XLA; the first call traces and compiles, and
  subsequent calls with the same input shapes/dtypes reuse the compiled artifact.
- `jax.vmap` vectorizes over a batch axis without a Python loop.
- `jax.value_and_grad` returns both the loss and its gradient in one pass – see the
  training step in the Darcy benchmark's
  [`train.py`](../../scripts/benchmarks/darcy/train.py).

## Optimisers with optax

Optimisation state is explicit, as with model parameters:

```python
import optax

# `optax.adamw` is right for real weights. For complex weights (FNO, GINO) use
# `scirex.training.adamw_cx`, which takes the same arguments.
tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
opt_state = tx.init(params)

grads = jax.grad(loss_fn)(params)
updates, opt_state = tx.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

## Randomness with explicit keys

JAX has no global RNG. Thread a `PRNGKey` and `split` it whenever you need fresh
randomness:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
```

## Data layout

Flax `Conv` expects channels-last `(B, H, W, C)`, whereas PyTorch uses
channels-first `(B, C, H, W)`. Keep the convention explicit and transpose only at
module boundaries.

## Common pitfalls

- Avoid `jax.block_until_ready()` outside of profiling – it forces host–device
  syncs and hides real performance.
- Pass `dtype=jnp.float32` explicitly to initializers to avoid silent promotion.
- Recompilation on every call usually means input shapes/dtypes are changing:
  pad or bucket inputs to keep shapes static.
