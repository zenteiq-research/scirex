<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Module Changes - Phase 3 (models tests)

This document records every change made to **production** source files while
writing the Phase 3 model test suite (`test_base_model.py`,
`test_fno.py`, `test_gino.py`). Each entry follows the convention:
Why / Location / Lines removed / Lines added / How to revert.

---

## 1. FNOBlocks: `NameInUseError` for `complex_data=True` combined with a norm

**Why**
`test_fno.py::test_fno_norm` mirrors the upstream
`neuralop/models/tests/test_fno.py::test_fno_advanced_params`, which
parametrizes `norm` (`None`/`group_norm`/`instance_norm`) **together with**
`complex_data` (`True`/`False`). The `complex_data=True` + non-`None` norm cases
raised:

```
flax.errors.NameInUseError: Could not create submodule "_norm_0" in Module FNOBlocks: Name in use.
```

The norm submodules were first bound to `self._norm` (Flax auto-names them
`_norm_0`, `_norm_1`, ...), and then a second statement re-wrapped them in
`ComplexValued` and reassigned `self._norm`. The wrapper submodules requested
the same `_norm_*` names that the inner norms already held, so Flax rejected the
duplicate. The `channel_mlp` construction immediately above already does this
correctly: it builds a **local** list, wraps it in `ComplexValued` while still
local, and assigns to `self` exactly once.

**Location**
`src/scirex/layers/blocks/fno_block.py`, `FNOBlocks.setup` (norm construction block,
around lines 241-272).

**Lines removed**
```python
        if self.norm is None:
            self._norm = None
        elif self.norm == "instance_norm":
            self._norm = [InstanceNorm() for _ in range(self.n_layers * self.n_norms)]
        elif self.norm == "group_norm":
            self._norm = [
                nn.GroupNorm(num_groups=1) for _ in range(self.n_layers * self.n_norms)
            ]
        elif self.norm == "batch_norm":
            self._norm = [
                BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                for _ in range(self.n_layers * self.n_norms)
            ]
        elif self.norm == "ada_in":
            self._norm = [
                AdaIN(self.ada_in_features, self.out_channels)
                for _ in range(self.n_layers * self.n_norms)
            ]
        else:
            raise ValueError(...)

        if self.complex_data and self._norm is not None:
            self._norm = [ComplexValued(module=x) for x in self._norm]
```

**Lines added**
The same `if/elif` chain now assigns to a **local** variable `norm` instead of
`self._norm`; the `ComplexValued` wrapping is applied to that local list, and the
result is bound once via `self._norm = norm`:

```python
        if self.norm is None:
            norm = None
        elif self.norm == "instance_norm":
            norm = [InstanceNorm() for _ in range(self.n_layers * self.n_norms)]
        ...  # group_norm / batch_norm / ada_in likewise assign to `norm`
        else:
            raise ValueError(...)

        if self.complex_data and norm is not None:
            norm = [ComplexValued(module=x) for x in norm]
        self._norm = norm
```

(An explanatory comment was added before the wrapping step.)

**How to revert**
Restore the `if/elif` chain to assign directly to `self._norm` and move the
`ComplexValued` wrapping back into a trailing `self._norm = [...]` reassignment.
Note this reintroduces the `NameInUseError` for `complex_data=True` + norm.

**Behaviour preserved**
The non-complex path is unchanged (the same list is bound to `self._norm` once,
with identical auto-generated names). Verified by the Phase 1 suite
`tests/layers/blocks/test_fno_block.py` (26 passed, no regression).

---

## Skipped reference coverage (no production change)

* `test_fno.py::test_fno_group_norm_norm_groups_not_ported` - the upstream
  `test_fno_group_norm` asserts `norm_groups` propagation to `torch.nn.GroupNorm`.
  The JAX FNO exposes no `norm_groups` parameter, so only the portable part
  (group_norm forward) is covered (in `test_fno_norm`); the propagation assertion
  is a documented skip.
* `test_fno.py::test_fno_conv_bias_kernel_not_ported` - the upstream
  `test_fno_conv_bias_kernel` exercises a `conv_bias_kernel` parameter that the
  JAX FNO does not implement; documented skip.
