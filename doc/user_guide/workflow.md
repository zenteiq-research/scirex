<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Workflow

Every ScireX run goes through the same seven steps:

```text
data  ->  model  ->  loss  ->  optimiser  ->  training loop  ->  checkpoint  ->  inference
```

This page explains each step with one small example: the 1D Poisson problem
`-u''(x) = f(x)` on `[0, 1]` with `u(0) = u(1) = 0`, solved by a 1D FNO. Its
exact solution is known, the data are generated in a second, and every block
below runs on a CPU. The code blocks run in order as one script. The full
version, with plots and a command line, is
[`scripts/tutorials/poisson1d_fno.py`](../../scripts/tutorials/poisson1d_fno.py),
explained in the [worked example](../tutorials/NewExample/Poisson1D_FNO.md).

!!! note "ScireX 1.0 has no packaged training entry point"
    The library provides the models, losses, optimiser, data loaders and
    checkpoint format. The training loop is your own short JAX loop, as below,
    or one of the benchmark scripts (see [Training Scripts](training.md)).

## 1. Data

A model takes arrays, so the first step is to produce input and target arrays
with the layout the model expects. FNO is **channels-first**:
`(batch, channels, *spatial)`. Here every sample is one source `f` and its
solution `u`, each with one channel and 128 grid points.

```python
import numpy as np


def poisson1d(n_samples, n_points=128, n_terms=16, seed=0):
    """Sources f and exact solutions u of -u'' = f, u(0) = u(1) = 0."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_points)
    k = np.arange(1, n_terms + 1)
    c = rng.standard_normal((n_samples, n_terms)) / k
    basis = np.sin(np.pi * k[:, None] * x[None, :])  # (n_terms, n_points)
    f = c @ basis
    u = (c / (np.pi * k) ** 2) @ basis
    # channels-first: (batch, channels, n_points)
    return f[:, None].astype(np.float32), u[:, None].astype(np.float32)


f_train, u_train = poisson1d(200, seed=0)
f_test, u_test = poisson1d(50, seed=1)
```

For the released datasets, [`scirex.data`](../api/data.md) has loaders ([`load_darcy_flow_small`](../api/data.md#scirex.data.load_darcy_flow_small),
[`load_darcy_pt`](../api/data.md#scirex.data.load_darcy_pt), [`load_navier_stokes_pt`](../api/data.md#scirex.data.load_navier_stokes_pt), [`load_cavity_kepsilon`](../api/data.md#scirex.data.load_cavity_kepsilon),
[`load_airfrans`](../api/data.md#scirex.data.load_airfrans), [`load_shapenet_car`](../api/data.md#scirex.data.load_shapenet_car)) and normalisers ([`UnitGaussianNormalizer`](../api/data.md#scirex.data.UnitGaussianNormalizer),
[`DefaultDataProcessor`](../api/data.md#scirex.data.DefaultDataProcessor)). Each benchmark's `_data.py` shows how its data are
read, split and normalised; the [Datasets](../dataset_guide/darcy_2d.md) pages
describe the files.

Rules that hold for every problem:

- **Split before you normalise.** Compute normalisation statistics on the
  training set only, and apply the same statistics to the test set.
- **Check the layout.** FNO and FNOClassic2d are channels-first; WNO is
  channels-last, `(batch, *spatial, channels)`; Transolver takes point lists,
  `(batch, n_points, channels)`; GINO takes coordinates and query points. See
  [Models and Losses](models_and_losses.md).
- **Keep float32.** JAX computes in float32 by default and ScireX does not
  switch float64 on. The examples still call
  `jax.config.update("jax_enable_x64", False)` after the imports, so that code
  copied into a program that enabled float64 earlier behaves the same.

## 2. Model

Models are [Flax](https://flax.readthedocs.io/) modules. A module holds only
the architecture; its weights (`params`) are a separate pytree, created by
`init` and passed to every `apply`.

```python
import jax
import jax.numpy as jnp

from scirex.models import FNO

jax.config.update("jax_enable_x64", False)

model = FNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_modes=(16,),  # one entry per dimension: a 1D FNO
    n_layers=4,
)
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 1, 128)))
print(model.apply(params, jnp.asarray(f_test[:4])).shape)  # (4, 1, 128)
```

`init` needs one sample input to trace the shapes; a zero array of the right
shape is enough. Building a model from a config object instead of keyword
arguments is described in [Config](config.md).

## 3. Loss

A loss is a function of the parameters and one batch, returning a scalar. The
standard data loss for operator learning is the **relative L2 error**,
`||pred - u|| / ||u||` per sample, which does not depend on the scale of `u`.

```python
from scirex.losses import LpLoss

l2 = LpLoss(d=1, p=2, reduction="mean")  # relative L2, averaged over the batch


def loss_fn(params, f, u):
    return l2(model.apply(params, f), u)
```

`d` is the number of spatial dimensions; the loss treats the last `d` axes as
space. The default reduction is `"sum"` over the batch, so pass
`reduction="mean"` if you want the loss independent of the batch size.
Physics losses that need no target `u` (the Ritz energy, the equation
residual) are described in [Models and Losses](models_and_losses.md).

## 4. Optimiser

Optimisers are [Optax](https://optax.readthedocs.io/) gradient
transformations. Use [`scirex.training.adamw_cx`](../api/training.md#scirex.training.adamw_cx) for any model with complex
weights (the spectral layers of FNO, FNOClassic2d and GINO). `jax.grad`
returns the complex conjugate of the gradient Optax expects, so plain
`optax.adamw` moves the imaginary part of every complex weight the wrong way;
[`adamw_cx`](../api/training.md#scirex.training.adamw_cx) conjugates the gradients first. For real weights it is the same as
AdamW. Its default `eps` is `1e-6`. See
[Complex-Parameter Optimisers](complex_optimisers.md) for the details.

```python
import optax

from scirex.training import adamw_cx

epochs, batch_size = 10, 50
steps_per_epoch = len(f_train) // batch_size
schedule = optax.cosine_decay_schedule(3e-3, decay_steps=epochs * steps_per_epoch)
tx = optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
    learning_rate=schedule, weight_decay=1e-4
)
opt_state = tx.init(params)
```

`inject_hyperparams` makes the learning rate readable from `opt_state` (useful
for logging). With complex parameters, pass `hyperparam_dtype=jnp.float32`;
without it the injected learning rate takes the parameters' complex dtype.
Gradient clipping is added with
`optax.chain(optax.clip_by_global_norm(1.0), tx)`.

## 5. Training loop

One optimiser step is: compute the loss and its gradient, turn the gradient
into an update, apply the update. Wrap the step in `jax.jit` so it is compiled
once; the first call is slow because it compiles.

```python
@jax.jit
def train_step(params, opt_state, f, u):
    loss, grads = jax.value_and_grad(loss_fn)(params, f, u)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


key = jax.random.PRNGKey(1)
for epoch in range(epochs):
    key, sub = jax.random.split(key)
    perm = np.asarray(jax.random.permutation(sub, len(f_train)))
    for b in range(steps_per_epoch):
        idx = perm[b * batch_size : (b + 1) * batch_size]
        params, opt_state, loss = train_step(
            params, opt_state, f_train[idx], u_train[idx]
        )
    print(f"epoch {epoch}: loss {float(loss):.4f}")
```

These 10 epochs on 200 samples only show the loop working; the loss is still
falling. The full example trains 100 epochs on 1000 samples.

The benchmark scripts use the same loop, with evaluation every few epochs and
a history saved next to the checkpoint. The experimental
[`scirex.experimental.training.Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) automates the loop for a model wrapper;
only the Car-CFD benchmark uses it (see [Training Scripts](training.md)).

## 6. Checkpoint

Every ScireX model can save its weights together with the constructor
arguments it was built with:

```python
import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp()) / "poisson1d_fno"
model.save_checkpoint(out, "fno", params)
print(sorted(p.name for p in out.iterdir()))
# ['fno_metadata.msgpack', 'fno_params.bytes']
```

`<name>_params.bytes` holds the weights and `<name>_metadata.msgpack` the
constructor arguments. Metadata is stored as msgpack, not pickle, so loading a
checkpoint does not run code from it. Save the rest of the run next to it
yourself, for example the config, the normalisation statistics and the seed as
JSON; you need them to rebuild the same test data.

The benchmark scripts write one directory per run with `params.msgpack`,
`history.json` and `config.json` (`config.json` last, so a directory without it
is an unfinished run); see each benchmark's `_checkpoint.py`.

## 7. Inference

Loading needs a *template*: a params pytree of the right structure, from a
fresh `init`. `from_checkpoint` then rebuilds the model from the saved
constructor arguments and fills the template with the saved weights.

```python
template = FNO(
    in_channels=1, out_channels=1, hidden_channels=32, n_modes=(16,), n_layers=4
).init(jax.random.PRNGKey(0), jnp.zeros((1, 1, 128)))
model2, params2 = FNO.from_checkpoint(out, "fno", template)

predict = jax.jit(model2.apply)
pred = predict(params2, jnp.asarray(f_test))
err = LpLoss(d=1, p=2, reduction="mean")(pred, jnp.asarray(u_test))
print(f"test relative L2: {float(err):.2%}")
```

`from_checkpoint` returns the weights as JAX arrays, ready for `apply`.
Compile `predict` once and reuse it: calling `jax.jit(model.apply)` inside a loop compiles it again on
every call.

If the inputs were normalised in step 1, normalise the test inputs with the
training statistics and de-normalise the predictions before computing the
error, so the error is measured in the units of the data.

## Where to go next

- [Config](config.md): keep every setting of a run in one checked object.
- [Models and Losses](models_and_losses.md): which model and which loss to
  choose.
- [Training Scripts](training.md): the benchmark scripts that follow this
  workflow for the released datasets.
- [Add a New Example](../tutorials/NewExample/Overview.md): turn your own
  problem into an example with tests.
