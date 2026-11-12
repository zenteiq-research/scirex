<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Source Module Changes Made During Test Writing - Phase 2 (data/datasets)

This file records every change made to production source modules while writing
the Phase 2 dataset tests. Each entry shows the file, why the change was needed,
the exact lines removed, and the exact lines added, so changes can be cleanly
reverted if needed.

---

## 1. `src/scirex/data/datasets/car_cfd.py` - fix `load_mini_car`

### Why
`load_mini_car` had two defects that made it always raise `FileNotFoundError`
and, even with the right path, fail to parse the file:

1. **Wrong path** - it pointed at `neuralop/data/datasets/data/mini_car.pt`, but
   after the torch→JAX port the data lives under
   `src/scirex/data/datasets/data/mini_car.pt` (the `neuralop/` tree does not
   exist in scirex).
2. **Wrong loader** - it used `pickle.load(open(...))`, but `mini_car.pt` is a
   PyTorch zip archive (upstream uses `torch.load`). `pickle.load` cannot read
   it. The rest of this package is deliberately torch-free and already provides
   a `.pt` reader (`pt_dataset_jax._load_pt`).

### Location
Function `load_mini_car`.

### Change

**Lines removed (original):**
```python
    import pickle

    return pickle.load(
        open(get_project_root() / "neuralop/data/datasets/data/mini_car.pt", "rb")
    )
```

**Lines added (replacement):**
```python
    from .pt_dataset_jax import _load_pt

    return _load_pt(
        (get_project_root() / "scirex/data/datasets/data/mini_car.pt").as_posix()
    )
```

### How to revert
Restore the original `pickle.load` body shown above.

---

## 2. `src/scirex/data/datasets/normalizers.py` - deleted (CI cleanup PR4)

### Finding (originally recorded during Phase 2 test writing)
`datasets/normalizers_jax.py` was a broken, unused duplicate of the working
JAX normalizers in `src/scirex/data/transforms/normalizers.py`:

* It failed at **import time** - line 14 imported
  `from .base_transforms_jax import DictTransform, Transform`, but
  `base_transforms_jax` lives under `data/transforms/`, not `data/datasets/`
  (`ModuleNotFoundError`).
* `UnitGaussianNormalizer` / `DictUnitGaussianNormalizer` were still unported
  PyTorch (`torch.mean`, `torch.std`, `register_buffer`, `.cuda()`,
  `.unsqueeze()`) with `torch` not imported - this is bug #1 from the
  `fix/annotation-errors` cleanup plan (14 ruff `F821` undefined-name errors).
* It was imported nowhere in `src/scirex/`.
* Upstream neuraloperator has **no** `data/datasets/normalizers` module
  normalizers live only under `data/transforms/` - so there was no counterpart
  to mirror for structural parity.

### Decision
Phase 2 review left the module untouched and skipped its mirror test,
recommending future deletion of the dead module. The `fix/annotation-errors`
CI cleanup (PR4, `src/scirex/data/datasets/`) acted on that recommendation:
`normalizers.py` and its skip-only test
`datasets/tests/test_normalizers_jax.py` were both deleted rather than
annotated, since annotating a broken, unreachable, unused duplicate would
have papered over dead code instead of removing it. Real normalizer coverage
continues to live in `transforms/tests/test_normalizers_jax.py`.

---

_No production source changes were required for the Phase 2 dataset tests._
