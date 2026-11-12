<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Config

A config holds every setting of a run in one object: the model's size, the
training recipe, the data split. Keeping them in one place, and checking them
when the run starts, is what makes a run repeatable and a typo visible.

ScireX 1.0 has two kinds of config:

| Kind | Where | Used for |
|---|---|---|
| **Config classes** | [`scirex.config`](../api/config.md) | Typed, validated model and optimiser settings, built with [zencfg](https://github.com/jeankossaifi/zencfg); [`get_model_jax`](../api/models.md#scirex.models.get_model_jax) builds a model from them |
| **Example settings** | each benchmark's `_config.py`, each tutorial's `DEFAULT_CONFIG` | The data split, the published runs and their recipes |

## Structure

### Config classes

Every class in [`scirex.config`](../api/config.md) is a `zencfg.ConfigBase`. Fields are class
attributes with a type annotation and, optionally, a default:

```text
ModelConfig                  model_arch, data_channels, out_channels
├── FNOConfig                n_modes, hidden_channels, n_layers, ...
│   └── SimpleFNOConfig
│       ├── FNOSmall2d       preset: n_modes [16, 16], hidden_channels 24
│       ├── FNOMedium2d      preset: n_modes [64, 64], hidden_channels 64
│       ├── FNOLarge2d, FNOHuge2d, FNOMedium3d
├── GINOConfig
│   └── GINOSmall3d, GINOPoisson2d, GINODarcy2d
└── TransolverConfig
    └── TransolverDarcy2dAligned, TransolverPipe, TransolverAirfRANS,
        TransolverShapeNetCar, TransolverLaplace2d, ...

OptimizationConfig           n_epochs, learning_rate, weight_decay, scheduler,
                             max_grad_norm, save_dir, ...
PatchingConfig               multigrid patching of the input domain
```

A **preset** is a subclass that only sets defaults, for example [`FNOSmall2d`](../api/config.md#scirex.config.FNOSmall2d).
Use one as it is, or override single fields when you create it:

```python
from scirex.config import FNOSmall2d, OptimizationConfig

model_cfg = FNOSmall2d(hidden_channels=32)  # the preset with one field changed
print(model_cfg.n_modes, model_cfg.hidden_channels)  # [16, 16] 32

opt_cfg = OptimizationConfig(n_epochs=500, learning_rate=1e-3)
```

Fields without a default are required; [`OptimizationConfig`](../api/config.md#scirex.config.OptimizationConfig) requires
`n_epochs`. There is no config class for WNO or FNOClassic2d; build those
models from keyword arguments.

A run combines sections into one top-level config. [`get_model_jax`](../api/models.md#scirex.models.get_model_jax) reads the
`model` section, picks the architecture named by `model_arch` and passes the
other fields to its constructor (`data_channels` becomes `in_channels`):

```python
from zencfg import ConfigBase

from scirex.config import FNOSmall2d, ModelConfig, OptimizationConfig
from scirex.models import get_model_jax


class Experiment(ConfigBase):
    model: ModelConfig = FNOSmall2d()
    opt: OptimizationConfig = OptimizationConfig(n_epochs=500)


cfg = Experiment()
model = get_model_jax(cfg.to_dict())  # an FNO with FNOSmall2d's settings
```

`cfg.to_dict()` gives a plain nested dictionary, which is what you save next to
a checkpoint (see [Workflow](workflow.md#6-checkpoint)).

### Example settings

The benchmark scripts keep their settings in plain Python, one file per
problem (`scripts/benchmarks/<problem>/_config.py`):

- `PROTOCOL`: the data split shared by every run (files, `n_train`,
  `n_test`, subsampling, evaluation interval).
- `RunConfig`: a frozen dataclass for one run: `name`, `model`, `arch` (the
  model's size), `opt` (its recipe) and `seed`.
- `RUNS`: the published runs, by name. `train.py --run fno` trains one,
  `--run all` trains every one.

The tutorials use one dictionary, `DEFAULT_CONFIG`, and a `make_config`
function that applies overrides to it (see the
[worked example](../tutorials/NewExample/Poisson1D_FNO.md)).

## Errors

A config is checked when it is built, before any data are read or any step is
trained. A wrong setting fails at once with a message that names it, instead
of failing hours into a run or, worse, being ignored.

### Wrong types and missing fields

Values are converted to the annotated type, and a value that cannot be
converted is an error:

```python
FNOSmall2d(hidden_channels="big")
# TypeError: Type Error during validation, got invalid value for field
# 'FNOSmall2d.hidden_channels'. Expected type <class 'int'>, but got str='big'

OptimizationConfig(n_epochs=10, scheduler="Cosine")
# TypeError: ... Expected type typing.Literal['StepLR', 'ReduceLROnPlateau',
# 'CosineAnnealingLR', 'OneCycleLR'], but got str='Cosine'

OptimizationConfig()
# ValueError: Missing required field 'n_epochs', of type '<class 'int'>'
```

### Unknown keys

A key the config does not declare is an error when the config is built from a
dictionary or from the command line, the two ways settings arrive from outside
the code:

```python
from zencfg import make_config_from_nested_dict

make_config_from_nested_dict(
    Experiment, {"opt": {"n_epochs": 10, "max_grad_nrom": 1.0}}, strict=True
)
# ValueError: Got key 'opt.max_grad_nrom' that is not a valid field for
# OptimizationConfig. Check for typos or remove unused config keys.
# Valid fields are: {...}
```

Without this check the typo would be dropped and the run would train without
gradient clipping. The same check applies to a misspelled section
(`{"optim": ...}`) and to command-line overrides through
`zencfg.make_config_from_cli` (`--opt.max_grad_nrom 1.0`).

A section given as a dictionary is rebuilt from its class defaults, not from
the instance in the parent config, so repeat any required field without a
class default (here `n_epochs`).

!!! warning "Keyword arguments are not checked for unknown keys"
    [`FNOSmall2d(hiden_channels=32)`](../api/config.md#scirex.config.FNOSmall2d) does **not** raise: the misspelled name is
    stored as an extra attribute and `hidden_channels` keeps its default.
    Build configs that come from users or files with
    `make_config_from_nested_dict`, which checks every key.

The example settings are checked the same way:

| Where | Unknown key | Result |
|---|---|---|
| Benchmark scripts | `train.py --run fnoo` | `error: unknown run(s) ['fnoo']; choose from ['fno', 'wno', 'gino', 'transolver']` |
| Benchmark scripts | an unknown flag, `--lr 1e-3` | `error: unrecognized arguments: --lr 1e-3` |
| Tutorials | `--set epoch=10` | `ValueError: unknown config key(s) ['epoch']; allowed keys: [...]` |

## Permissive mode

Building a config has **no permissive mode**: an unknown key is always an error, and
there is no switch that turns it into a warning or ignores it.

The one exception is building a model.
[`get_model_jax(config, permissive=True)`](../api/models.md#scirex.models.get_model_jax)
warns about and drops the keys of `config.model` that the architecture does not accept; by
default it raises a `ValueError` that names them.

`zencfg`'s `strict` argument (`make_config_from_nested_dict(..., strict=...)`,
`make_config_from_cli(..., strict=...)`) does not relax this. It only changes
how a missing required field is reported: with `strict=True` it is a
`ValueError` that names the field; with `strict=False` the field is set to
`None` and then fails type validation with a `TypeError`. Unknown keys and
wrong types are errors in both modes.

If a config legitimately needs a key that the class does not declare, add the
field to a subclass rather than relaxing the check:

```python
from scirex.config import OptimizationConfig


class MyOptimization(OptimizationConfig):
    warmup_steps: int = 0
```

## See also

- [API reference: `scirex.config`](../api/config.md): every class and preset
  with all its fields.
- [Workflow](workflow.md): where the config is used in a run.
- [Models and Losses](models_and_losses.md): what the model settings mean.
