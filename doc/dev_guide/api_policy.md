<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# API Policy

This page defines what is public API in ScireX, what we promise about it, and how to
add, change or remove a public name. It applies from release 1.0.0. Earlier 0.x
releases carry no guarantee.

## What is public

A name is public when both of these hold:

1. It is listed in the `__all__` of a documented package, and
2. it appears in the [API reference](../api/index.md).

You import a public name from the package, not from the module that defines it:

```python
from scirex.models import FNO
from scirex.losses import Q4Galerkin
from scirex.training import adamw_cx
```

The module that defines a name (`scirex.losses.galerkin_q4`, `scirex.training.optim.adamw`)
is an implementation detail. We move code between modules without notice, and a deep
import path can break in any release. Package-level imports keep working.

The stable packages are `scirex`, [`scirex.models`](../api/models.md), [`scirex.layers`](../api/layers.md), [`scirex.losses`](../api/losses.md),
[`scirex.data`](../api/data.md), [`scirex.training`](../api/training.md), [`scirex.config`](../api/config.md) and [`scirex.utils`](../api/utils.md).
The sub-packages [`scirex.layers.blocks`](../api/layers.md), [`scirex.layers.common`](../api/layers.md), [`scirex.data.containers`](../api/data.md),
[`scirex.data.transforms`](../api/data.md), [`scirex.data.datasets`](../api/data.md) and [`scirex.training.optim`](../api/training.md) export subsets
of their parent's names and are covered by the same rules.

Config key names (the attributes of the classes in [`scirex.config`](../api/config.md), and the YAML or
command-line keys that set them) and the names of the config presets are part of the
public API.

## Tiers

| Tier | Where | Promise |
|---|---|---|
| Stable | Names in `__all__` of the packages above | Semantic Versioning (next section). Removal only after a deprecation period |
| Experimental | `scirex.experimental.*` | None. A name can change its signature or behaviour, move, or disappear in any release, with no deprecation period |
| Private | `scirex._internal.*`, and any module, class, function or attribute whose name starts with an underscore | None. Do not import it. It is not documented and can change in any release |

Experimental code is for features whose interface is not settled, such as an
architecture that is not yet published or a multi-GPU path we are still changing.
A name moves from experimental to stable in a minor release, announced under *Added* in
the CHANGELOG. If you are unsure whether a new feature is ready, make it experimental.
Promoting a name is easy; withdrawing a stable one costs two minor releases.

## Versioning rules for 1.x

ScireX uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Given `MAJOR.MINOR.PATCH`:

- **PATCH** releases fix bugs. They do not change the public API.
- **MINOR** releases add features and may deprecate names. Code that works on `1.x`
  keeps working on `1.y` for `y > x`. It may emit `DeprecationWarning`s.
- **MAJOR** releases may break the API. Everything removed or changed incompatibly is listed
  in the CHANGELOG.

### What counts as a breaking change

For a stable name, each of these is breaking:

- Removing or renaming it, or removing it from the package's `__all__`.
- Making it impossible to import from its package.
- Changing a signature incompatibly: removing, renaming or reordering parameters,
  adding a required parameter, or narrowing the accepted types.
- Changing behaviour in a way that makes code written against the documented
  behaviour wrong: the shape, dtype or meaning of a return value, the default of a
  parameter, or the meaning of a config key.
- Renaming or removing a config key or a config preset.
- Renaming a model constructor argument, or changing the names or shapes of the
  parameters a released model creates (see [Checkpoints](#checkpoints)).
- Making a checkpoint written by an earlier 1.x release fail to load.

These are not breaking:

- Adding a public name, a method, or an optional parameter with a default that keeps the
  old behaviour.
- Bug fixes that make the behaviour match its documentation.
- Floating-point differences from a changed operation order or a new JAX version.
- Anything in [`scirex.experimental`](../api/experimental.md) or `scirex._internal`.
- The wording of error and warning messages. The exception types are part of the contract.

The one allowed removal in a minor release is the removal of a name that went through
the deprecation policy below.

### Checkpoints

A model checkpoint (`BaseModel.save_checkpoint`) stores three things: the Flax
parameters, the keyword arguments the model was constructed with (by argument name), and
references to any class or function passed as an argument (such as an activation), by
module path and qualified name. `BaseModel.from_checkpoint` rebuilds the model from
those arguments and references.

We promise that a checkpoint written by a 1.x release loads in every later 1.x release.

To keep this, within 1.x we do not rename a model's constructor arguments (config keys
may differ from them; [`get_model_jax`](../api/models.md#scirex.models.get_model_jax) maps one to the other), and we do not change the
names or shapes of a released model's parameters. When a module, class or function that a
checkpoint can reference moves, the old path is added to the move table in
`scirex/_internal/aliases.py` (which `_legacy_refs.py` reuses), with a test.

We do not promise:

- That a checkpoint written by a newer release loads in an older one.
- Anything about checkpoints of experimental models such as [`GIWO`](../api/experimental.md#scirex.experimental.giwo.GIWO).
- That a reference can point outside `scirex`, `jax` or `flax`. Loading refuses any
  other module on purpose, because checkpoint metadata is data, not code.

If the class version stored in a checkpoint differs from the code's, `from_checkpoint` emits a
warning and still loads.

## Deprecation policy

The policy covers every stable public name and its documented signature.
Experimental and private code is exempt.

To retire a stable name:

1. **Warn.** Keep the name working and make it emit a `DeprecationWarning` with the
   standard message produced by `scirex._internal.deprecation`:

    > `scirex.training.resolve_dtype is deprecated since ScireX 1.0.0 and will be removed in 1.2.0 or later. Use scirex._internal.wno_train_utils.resolve_dtype (it is not public API) instead.`

2. **Announce.** List the name under *Deprecated* in the CHANGELOG of the release that
   introduces the warning.
3. **Wait.** Keep the name for at least **two minor releases and at least three
   months**, whichever is longer. A name deprecated in 1.0.0 can be removed in 1.2.0 at the
   earliest.
4. **Remove.** Remove it only in the release the warning names (or later), and list it under
   *Removed* in the CHANGELOG.

`DeprecationWarning` is hidden by default outside `__main__` and test runners. To see
the warnings in your own code, run Python with `-W default::DeprecationWarning`, or run
your tests with pytest, which shows them. To turn them into errors, use `-W error::DeprecationWarning`.

Inside ScireX, a `DeprecationWarning` attributed to a `scirex` module is an error in the
test suite (`filterwarnings` in `pytest.ini`), so ScireX code cannot depend on a deprecated name.

### When the policy starts

The policy applies from 1.0.0, the first release.

## Adding a public name

1. Implement it in the module where it belongs.
2. Give it a numpy-style docstring (summary, `Parameters`, `Returns`). Ruff's `D` rules apply
   to public modules, and a test fails for an exported name without a docstring.
3. Export it from the package: add it to `__all__` (keep the list in ruff `RUF022` order). In
   a lazy package also add it to `_LAZY_EXPORTS` and to the `TYPE_CHECKING` import block.
   Tests check that `_LAZY_EXPORTS` and `__all__` agree.
4. Add a `::: scirex.<package>.<Name>` entry to `doc/api/<package>.md`, under a fitting
   heading.
5. Regenerate the API snapshot and commit it with the change:

    ```bash
    python scripts/dev/update_api_snapshot.py
    ```

    [`tests/test_public_api.py`](../../tests/test_public_api.py) compares the live `__all__` lists with [`tests/api_snapshot.json`](../../tests/api_snapshot.json),
    so an unintended addition or removal fails the test.
6. Add the name under *Added* in `CHANGELOG.md`.

A new name that is not ready for the promise goes under [`scirex.experimental`](../api/experimental.md) instead and
is not listed in `__all__`.

## Deprecating a public name

Pick the version it is deprecated in (`since`) and the earliest removal version
(`removal`, at least two minors later). Then use the helpers in `scirex._internal.deprecation`.

A function or class that stays in place:

```python
from scirex._internal.deprecation import deprecated


@deprecated(since="1.1.0", removal="1.3.0", alternative="scirex.losses.NewLoss")
class OldLoss:
    ...
```

A name that left `__all__` but must still resolve from its package (a module-level
`__getattr__`, as [`scirex.losses`](../api/losses.md) and [`scirex.training`](../api/training.md) do):

```python
import importlib

from scirex._internal.deprecation import warn_deprecated


def __getattr__(name):
    if name in _DEPRECATED_EXPORTS:
        warn_deprecated(
            f"scirex.losses.{name}",
            since="1.1.0",
            removal="1.3.0",
            alternative="scirex.losses.NewLoss",
            stacklevel=3,  # point at the line that did the import
        )
        module = importlib.import_module(_DEPRECATED_EXPORTS[name])
        return getattr(module, name)
    ...
```

A parameter or behaviour inside a function: call `warn_deprecated("scirex.x.f(old_arg)", ...)`
from the function body with the default `stacklevel=2`.

Then:

1. Remove the name from `__all__` if it is no longer exported, and update the API page and the snapshot.
2. Add a test that the warning is emitted, using `pytest.warns(DeprecationWarning)`.
3. Add the name to the table below and to *Deprecated* in the CHANGELOG.
4. Run [`tests/test_public_api.py`](../../tests/test_public_api.py). It fails if a deprecated name is missing from the table
   below or from a *Deprecated* section of the CHANGELOG, if the versions in the table differ
   from the ones in the warning, or if a new file calls the deprecation helpers without being
   listed in `DEPRECATION_SITES` in that test.

## Currently deprecated names

| Name | Deprecated in | Earliest removal | Use instead |
|---|---|---|---|
| `scirex.training.append_grid_nhwc`, `append_grid_nwc`, `create_train_state`, `make_schedule`, `per_frame_relative_l2`, `plot_loss_curves`, `relative_l2_sum`, `resolve_dtype`, `write_metrics` | 1.0.0 | 1.2.0 | These were never meant to be public. They live in `scirex._internal.wno_train_utils`, which is private: copy what you need |
| `scirex.losses.ADDiff`, [`AdvectionTerm`](../api/experimental.md#scirex.experimental.losses.AdvectionTerm), [`AdvectionTermAD`](../api/experimental.md#scirex.experimental.losses.AdvectionTermAD), [`BatchedVariableCoefficientGalerkinResidualLossT3`](../api/experimental.md#scirex.experimental.losses.BatchedVariableCoefficientGalerkinResidualLossT3), [`BatchedVariableCoefficientGalerkinResidualLossT6`](../api/experimental.md#scirex.experimental.losses.BatchedVariableCoefficientGalerkinResidualLossT6), [`DarcyHybridLossGINO`](../api/experimental.md#scirex.experimental.losses.DarcyHybridLossGINO), [`DiffusionTermAD`](../api/experimental.md#scirex.experimental.losses.DiffusionTermAD), [`DivergenceTerm`](../api/experimental.md#scirex.experimental.losses.DivergenceTerm), [`FourierDiff`](../api/experimental.md#scirex.experimental.losses.FourierDiff), [`GradientTerm`](../api/experimental.md#scirex.experimental.losses.GradientTerm), [`NavierStokesEqnLoss`](../api/experimental.md#scirex.experimental.losses.NavierStokesEqnLoss), [`PDEResidualLoss`](../api/experimental.md#scirex.experimental.losses.PDEResidualLoss), [`ReactionTerm`](../api/experimental.md#scirex.experimental.losses.ReactionTerm), [`ReactionTermAD`](../api/experimental.md#scirex.experimental.losses.ReactionTermAD), [`SourceTermAD`](../api/experimental.md#scirex.experimental.losses.SourceTermAD), [`T3Galerkin`](../api/experimental.md#scirex.experimental.losses.T3Galerkin), [`T6Galerkin`](../api/experimental.md#scirex.experimental.losses.T6Galerkin), [`TransientTerm`](../api/experimental.md#scirex.experimental.losses.TransientTerm) | 1.0.0 | 1.2.0 | These losses are experimental. Import them from [`scirex.experimental.losses`](../api/experimental.md) (no stability guarantee) |
| `scirex.training.SimpleDataLoader`, [`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer), [`load_training_state`](../api/experimental.md#scirex.experimental.training.load_training_state), [`save_training_state`](../api/experimental.md#scirex.experimental.training.save_training_state) | 1.0.0 | 1.2.0 | These are experimental. Import them from [`scirex.experimental.training`](../api/experimental.md) (no stability guarantee) |
| `scirex.data.web_utils`, `scirex.layers.common.einsum_utils`, `scirex.layers.common.segment_csr`, `scirex.training.torch_setup`, `scirex.training.trainer`, `scirex.training.training_state`, `scirex.training.wno_train_utils`, `scirex.distributed`, `scirex.geometry`, `scirex.training.patching`, `scirex.tuning`, `scirex.config.distributed`, `scirex.models.giwo` (module paths) | 1.0.0 | 1.2.0 | These modules moved. The old path still imports the same module and warns; the new paths are `scirex._internal.<name>` (private) and [`scirex.experimental.distributed`](../api/experimental.md), `.geometry`, `.patching` |

[`GIWO`](../api/experimental.md#scirex.experimental.giwo.GIWO) is not in this table. It is experimental, so it is not exported from
[`scirex.models`](../api/models.md). Import it from [`scirex.experimental.giwo`](../api/experimental.md). [`get_model_jax`](../api/models.md#scirex.models.get_model_jax) still builds it
under the name `"giwo"`.
