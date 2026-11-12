<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Module Changes - Phase 4 (training tests)

This document records every change made to **production** source files while
writing the Phase 4 training test suite (`test_trainer.py`,
`test_patching.py`, `test_adamw.py`,
`test_tensor_galore_projector.py`, `test_torch_setup.py`,
`test_training_state.py`). Each entry follows the convention:
Why / Location / Lines removed / Lines added / How to revert.

---

## 1. unfold_jax: `TypeError` from an invalid `slice_in_dim` keyword

**Why**
`test_patching.py::test_make_patches` mirrors the upstream
`neuralop/training/tests/test_patching.py::test_make_patches`. Any call that
actually creates patches (`n > 1`) routed through `make_patches -> unfold_jax`
raised:

```
TypeError: slice_in_dim() got an unexpected keyword argument 'dimension'
```

`jax.lax.slice_in_dim`'s signature is
`slice_in_dim(operand, start_index, limit_index, stride=1, axis=0)` - the axis
argument is named `axis`, not `dimension`. The window is sliced along the
dimension that was moved to the front (axis 0), so the correct call passes
`axis=0`. This bug meant `make_patches` (and therefore every patching path) had
never run successfully.

**Location**
`src/scirex/experimental/patching.py`, `unfold_jax` (the sliding-window slice,
around line 65).

**Lines removed**
```python
        window = jax.lax.slice_in_dim(
            x_moved, i * stride, i * stride + kernel_size, 1, dimension=0
        )
```

**Lines added**
```python
        window = jax.lax.slice_in_dim(
            x_moved, i * stride, i * stride + kernel_size, 1, axis=0
        )
```

**How to revert**
Change `axis=0` back to `dimension=0`. Note this reintroduces the `TypeError`
for any `make_patches` call with `n > 1`.

---

## Skipped reference coverage (no production change)

* `test_patching.py::test_full_mgp2d` - mirrors the upstream
  `test_full_mgp2d`, which drives `MultigridPatching2D.patch` / `unpatch`. That
  Flax `nn.Module` is not usable as written under Flax's frozen-dataclass model:
  - `setup()` reassigns the declared field `padding_fraction`
    (`SetAttributeInModuleSetupError`);
  - `_make_mg_patches()` reassigns the declared fields `padding_height` /
    `padding_width` during a method call;
  - the derived `n_patches` state is computed only in `setup()`, which does not
    run under the imperative `patcher.patch(x, y)` usage the reference (and
    `MGPatchingDataProcessor`) rely on.

  Making `test_full_mgp2d` pass requires a structural refactor of the class
  (e.g. `object.__setattr__` for derived/stateful attributes plus lazy /
  `__post_init__` initialization) that also affects the deferred
  `MGPatchingDataProcessor`. This was deferred by explicit decision; the pure
  building blocks it relies on (`make_patches`, `unfold_jax`) are covered
  directly. The 1D `make_patches` path is also broken (the final reshape/
  transpose hardcodes the 2D case), but the upstream reference only covers 2D,
  so it is out of scope here.

* `test_trainer.py::test_incremental` - mirrors the upstream
  `test_incremental`, which exercises `IncrementalFNOTrainer`. The JAX port ships
  only the base `Trainer`, so there is no analogue.
