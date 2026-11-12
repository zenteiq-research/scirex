<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Required Tests

Every new model, and every new example, comes with these four tests. They run
on a CPU in a few seconds each, so they run in CI on every pull request. The
worked example has all four in [`tests/docs/test_tutorial_poisson1d_fno.py`](../../../tests/docs/test_tutorial_poisson1d_fno.py);
the code below is taken from there.

| Test | What it catches |
|---|---|
| 1. Output shape | a wrong layout (not the one the model documents), a lost or extra dimension, a wrong dtype |
| 2. Finite gradients | `NaN` or `inf` from a division, a `sqrt` or a log at zero; parameters that get no gradient at all |
| 3. Tiny overfit | a model or loss that cannot learn: a detached branch, a wrong sign, a broken optimiser |
| 4. Unknown config key refused | a misspelt key that is silently ignored, so a run trains with the default instead of the intended value |

## Where they go

| What is new | File |
|---|---|
| A model in `src/scirex/experimental/<model>.py` | `tests/experimental/test_<model>.py` |
| A model in `src/scirex/models/<model>.py` | `tests/models/test_<model>.py` |
| A tutorial script | `tests/docs/test_tutorial_<problem>_<model>.py` |
| A benchmark | `scripts/tests/test_<problem>_benchmark.py` |

`tests/` and [`scripts/tests/`](../../../scripts/tests) run in separate pytest processes (see
`pytest.ini`), because many scripts switch off float64 when they are
imported.

## 1. Output shape

Build the model at a tiny size, apply it to a batch of more than one sample,
and check the shape and the dtype:

```python
def test_output_shape() -> None:
    tutorial = load_script()
    config = tutorial.make_config(**TINY)
    model, params, _ = tutorial.build_model(config, seed=0)
    f = jnp.ones((3, 1, config["n_points"]), jnp.float32)
    out = model.apply(params, f)
    assert out.shape == (3, 1, config["n_points"])
    assert out.dtype == jnp.float32
```

For a model, test every dimension it supports (1D, 2D, 3D) and more than one
output channel; [`tests/models/test_wno.py`](../../../tests/models/test_wno.py) is a good template.

## 2. Finite gradients

Take the gradient of the real loss on real-looking data and check every leaf:

```python
def test_gradients_are_finite() -> None:
    tutorial = load_script()
    config = tutorial.make_config(n_train=4, n_test=2, **TINY)
    data = tutorial.load_data(config, seed=0)
    model, params, _ = tutorial.build_model(config, seed=0)
    loss_fn = tutorial.make_loss(model)
    grads = jax.grad(loss_fn)(params, data["f_train"], data["u_train"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(float(jnp.abs(g).max()) > 0 for g in leaves), "all gradients are zero"
```

`jnp.isfinite` works for complex leaves too (FNO's spectral weights).

## 3. Tiny overfit

Train on a handful of samples, with the test set equal to the training set,
and require the loss to fall far below where it started. Write down the
values you measured, so the thresholds have a reason:

```python
def test_tiny_overfit() -> None:
    """Eight samples, one batch: the training loss must fall well below its start."""
    tutorial = load_script()
    config = tutorial.make_config(
        n_train=8, n_test=8, batch_size=8, epochs=300, learning_rate=1e-2, **TINY
    )
    data = tutorial.load_data(config, seed=0)
    data["f_test"], data["u_test"] = data["f_train"], data["u_train"]
    model, params, key = tutorial.build_model(config, seed=0)
    _, history = tutorial.train(model, params, key, data, config)
    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    # Measured: 5.43 -> 0.035 (CPU, seed 0).
    assert last < 0.1, f"loss {first:.3f} -> {last:.3f}: the model cannot fit 8 samples"
    assert last < first / 20
```

Fix the seed. Keep the model tiny so the test takes about a second; if it
needs many steps, check that `jax.jit` is not being recreated inside the loop.

## 4. An unknown config key is refused

Check both places a key can enter: the problem config and the model
constructor.

```python
def test_unknown_config_key_is_rejected() -> None:
    tutorial = load_script()
    with pytest.raises(ValueError, match=r"unknown config key\(s\) \['epoch'\]"):
        tutorial.make_config(epoch=3)
    with pytest.raises(ValueError, match="unknown config key"):
        tutorial.main(["--smoke", "--set", "epoch=3"])
    # The model itself refuses a constructor argument it does not declare.
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_mode'"):
        tutorial.FNO(in_channels=1, out_channels=1, hidden_channels=8, n_mode=(8,))
```

For a registered model, also go through the factory, since that is the path a
config takes. zencfg keeps the unknown key, and [`get_model_jax`](../../api/models.md#scirex.models.get_model_jax) refuses it with
a `ValueError` that names the key:

```python
from zencfg.bunch import Bunch

from scirex.models import get_model_jax


def test_get_model_jax_refuses_an_unknown_key() -> None:
    model = dict(model_arch="pointmlp", data_channels=1, out_channels=1)
    config = Bunch(dict(model=dict(model, hidden_chanels=8)))  # typo
    with pytest.raises(ValueError, match="hidden_chanels"):
        get_model_jax(config)
```

`Bunch` is the type `ConfigBase.to_dict()` returns: a dict whose keys are
also attributes, which is what [`get_model_jax`](../../api/models.md#scirex.models.get_model_jax) expects.

## Beyond the four

The four tests are the minimum. The worked example also checks:

- the data against the exact solution (`-u'' = f` by finite differences);
- that training and test samples differ;
- the metric against its definition;
- a checkpoint round trip: the loaded model gives the same output;
- the whole `--smoke` run, then `--infer` on its checkpoint, within the
  two-minute CI budget.

A benchmark adds a test that the tutorial's sizes and recipe match `RUNS`,
and a train, checkpoint, infer round trip on synthetic data.

Run them with

```bash
pytest tests/docs/test_tutorial_poisson1d_fno.py      # the worked example
pytest tests/experimental/test_<model>.py             # your model
pytest scripts/tests/test_<problem>_benchmark.py      # your benchmark
```
