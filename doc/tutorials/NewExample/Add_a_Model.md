<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Add a Model

Skip this page if your example uses a model ScireX already has ([`FNO`](../../api/models.md#scirex.models.FNO),
[`FNOClassic2d`](../../api/models.md#scirex.models.FNOClassic2d), [`WNO`](../../api/models.md#scirex.models.WNO), [`GINO`](../../api/models.md#scirex.models.GINO), [`Transolver`](../../api/models.md#scirex.models.Transolver), or the experimental [`GIWO`](../../api/experimental.md#scirex.experimental.giwo.GIWO)).
[`scirex.models.available_models()`](../../api/models.md#scirex.models.available_models) lists them.

The running example on this page is `PointMLP`, a small pointwise network. Its
code below was run against this repository: it registers, builds through the
factory and refuses unknown keys as shown.

## 1. File location

A new model starts in the **experimental** tier:

```text
src/scirex/experimental/<model>.py        e.g. src/scirex/experimental/point_mlp.py
tests/experimental/test_<model>.py        its tests
```

The [API policy](../../dev_guide/api_policy.md#adding-a-public-name) says why:
a name in [`scirex.models`](../../api/models.md) is covered by the 1.x stability promise, a name in
[`scirex.experimental`](../../api/experimental.md) is not. [`GIWO`](../../api/experimental.md#scirex.experimental.giwo.GIWO) ([`src/scirex/experimental/giwo.py`](../../../src/scirex/experimental/giwo.py)) is
the model to copy. A model moves to [`src/scirex/models/`](../../../src/scirex/models) only in a release
that promotes it (step 5 below).

Reusable layers go in [`src/scirex/layers/`](../../../src/scirex/layers), not in the model file.

## 2. Class interface

A model is a Flax `linen` module that subclasses [`scirex.models.BaseModel`](../../api/models.md#scirex.models.BaseModel):

```python
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""PointMLP: a pointwise multilayer perceptron, the smallest operator baseline."""

import flax.linen as nn
import jax.numpy as jnp

from ..models.base import BaseModel


class PointMLP(BaseModel, name="PointMLP"):
    """Pointwise MLP: the same small network applied at every grid point.

    It sees one point at a time, so it cannot learn a non-local operator; it
    is a baseline that a neural operator must beat.

    Parameters
    ----------
    in_channels : int
        Channels of the input function. Determined by the problem.
    out_channels : int
        Channels of the output function. Determined by the problem.
    hidden_channels : int, optional
        Width of every hidden layer. Default: 32.
    n_layers : int, optional
        Number of hidden layers. Default: 2.

    Notes
    -----
    Input and output are channels-first, ``(batch, channels, *spatial)``,
    like ``FNO`` and ``FNOClassic2d``.
    """

    in_channels: int
    out_channels: int
    hidden_channels: int = 32
    n_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map ``(batch, in_channels, *spatial)`` to ``(batch, out_channels, *spatial)``."""
        h = jnp.moveaxis(x, 1, -1)
        for _ in range(self.n_layers):
            h = nn.gelu(nn.Dense(self.hidden_channels)(h))
        return jnp.moveaxis(nn.Dense(self.out_channels)(h), -1, 1)
```

The rules behind it:

- **Subclass [`BaseModel`](../../api/models.md#scirex.models.BaseModel) with a `name=`**. `BaseModel.__init_subclass__`
  registers the class under `name.lower()` (`"pointmlp"`), and
  `BaseModel.__new__` records the constructor arguments, so
  `save_checkpoint` and `from_checkpoint` work without extra code.
- **Hyperparameters are dataclass fields** with type annotations. They are
  the model's allowed keys: Flax's generated `__init__` refuses any other
  keyword with a `TypeError`.
- **Use `in_channels` and `out_channels`** for the input and output widths if
  the model has them. [`get_model_jax`](../../api/models.md#scirex.models.get_model_jax) fills `in_channels` from the config key
  `data_channels`. A model with another parameterisation (Transolver's
  `space_dim`, `fun_dim`, `out_dim`) is fine: the factory then drops
  `data_channels` and `out_channels`.
- **A documented layout.** This guide uses channels-first, `(batch, channels, *spatial)`, like
  [`FNO`](../../api/models.md#scirex.models.FNO); [`WNO`](../../api/models.md#scirex.models.WNO) is channels-last, [`Transolver`](../../api/models.md#scirex.models.Transolver) takes point lists and [`GINO`](../../api/models.md#scirex.models.GINO) takes coordinates and
  query points. Pick the one that fits the model and state it in the docstring.
- **Choose a name nobody uses.** The registry is keyed by the lower-case
  name, and a second class with the same name silently replaces the first.
  Check [`available_models()`](../../api/models.md#scirex.models.available_models) first.
- Follow the [JAX/Flax rules](../../dev_guide/contributing.md#porting-rules-jaxflax):
  `setup()` or `@nn.compact`, no in-place mutation, explicit `dtype=jnp.float32`
  on dtype-sensitive initialisers, no `block_until_ready` in model code.
- **Bump `_version`** (a class attribute, `"0.1.0"` by default) whenever a
  change makes old checkpoints load into a different network; loading then
  warns about the mismatch.

## 3. Registration with allowed config keys

Registration has three parts.

**a. The model registry.** Subclassing [`BaseModel`](../../api/models.md#scirex.models.BaseModel) registers the class, but
only once its module has been imported. So that [`get_model_jax`](../../api/models.md#scirex.models.get_model_jax) can build the
model without the caller importing it, add it to `_BUILTIN_ARCHS` in
[`src/scirex/models/base.py`](../../../src/scirex/models/base.py):

```python
_BUILTIN_ARCHS: dict[str, str] = {
    "fno": "scirex.models.fno",
    ...
    "giwo": "scirex.experimental.giwo",
    "pointmlp": "scirex.experimental.point_mlp",
}
```

and add it to `_BUILTIN` and `_CONFIG_MODELS` in [`tests/models/test_init.py`](../../../tests/models/test_init.py),
which build every built-in architecture in a fresh interpreter.

**b. The config class.** The allowed config keys of a model are the fields of
its config class, a subclass of [`scirex.config.ModelConfig`](../../api/config.md#scirex.config.ModelConfig) in
[`src/scirex/config/models.py`](../../../src/scirex/config/models.py). `model_arch` is the registry name; every other
field is a constructor argument of the model, except `data_channels`, which
the factory renames to `in_channels`:

```python
class PointMLPConfig(ModelConfig):
    """Hyperparameters of :class:`scirex.experimental.point_mlp.PointMLP`."""

    model_arch: str = "pointmlp"
    data_channels: int = 1
    out_channels: int = 1
    hidden_channels: int = 32
    n_layers: int = 2
```

Presets for a problem subclass it and change defaults only, like
[`FNOSmall2d`](../../api/config.md#scirex.config.FNOSmall2d) or [`TransolverPipe`](../../api/config.md#scirex.config.TransolverPipe). Export the config class from
[`src/scirex/config/__init__.py`](../../../src/scirex/config/__init__.py); that module is imported eagerly so that
presets register with zencfg and can be selected by name.

**c. Building it.** Any experiment config with a `model` field then builds
the model through the factory:

```python
from zencfg import ConfigBase

from scirex.config import ModelConfig
from scirex.models import get_model_jax


class Experiment(ConfigBase):
    model: ModelConfig = PointMLPConfig()


model = get_model_jax(Experiment().to_dict())  # a PointMLP with in_channels=1
```

**Where unknown keys are caught.** zencfg does *not* refuse an unknown key:
`PointMLPConfig(hidden_chanels=8)` (note the typo) is accepted and the key is
kept. The typo is caught one step later, when [`get_model_jax`](../../api/models.md#scirex.models.get_model_jax) checks the keys of
`config.model` against the constructor:

```text
ValueError: config.model has key(s) ['hidden_chanels'] not accepted by 'pointmlp' (accepts [...]).
```

So every config key must be a constructor argument, and the
[required test](Required_Tests.md#4-an-unknown-config-key-is-refused) checks
that an unknown key fails. A problem config that is not a model config (data
split, training recipe) must check its keys itself, the way
[`make_config`](Poisson1D_FNO.md#1-the-config) does in the worked example.

## 4. Docstring

The docstring is numpy-style, and it is what the API reference shows:

- a one-line summary, then what the model does and when to use it;
- `Parameters`: every constructor argument, with its type, its meaning and its
  default; put rarely changed ones under `Other Parameters`, as [`FNO`](../../api/models.md#scirex.models.FNO) does;
- `Notes`: the input and output layout and anything a user must know (for
  example that the dimension is set by `len(n_modes)`);
- `References`: the paper the model comes from, if any;
- a docstring on `__call__` saying the input and output shapes.

A public name without a docstring fails [`tests/test_public_api.py`](../../../tests/test_public_api.py); Ruff's
`D` rules check the format in public modules.

## 5. Making it stable (later)

Promotion to [`scirex.models`](../../api/models.md) is a separate, deliberate change. Follow
[Adding a public name](../../dev_guide/api_policy.md#adding-a-public-name):
move the file to [`src/scirex/models/`](../../../src/scirex/models), add the name to `__all__`,
`_LAZY_EXPORTS` and the `TYPE_CHECKING` block of
[`src/scirex/models/__init__.py`](../../../src/scirex/models/__init__.py), add `::: scirex.models.<Name>` to
`doc/api/models.md`, regenerate the snapshot with
`python scripts/dev/update_api_snapshot.py`, update the expected list in
[`tests/models/test_init.py`](../../../tests/models/test_init.py), and add the name under *Added* in
`CHANGELOG.md`.

## Next

[Required tests](Required_Tests.md) lists the four tests every model needs.
