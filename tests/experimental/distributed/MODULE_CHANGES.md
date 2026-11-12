<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Module Changes - Phase 6 (mpu + utils tests)

This document records production source changes made while writing the Phase 6
test suite: `tests/experimental/distributed/` (`test_comm.py`, `test_helpers.py`,
`test_mappings.py`) and `tests/test_utils.py`.

## No production changes

The `mpu` communication/helper/mapping modules and the general `utils_jax`
helpers matched their expected behaviour, so no production source changes were
required.

## Coverage notes (single-process)

The `mpu` package is built for multi-process model/data parallelism, which is
unavailable in CI. The tests cover the single-process behaviour the rest of the
codebase relies on under `use_distributed=False`:

* `comm_jax` - rank/size getters return their single-process defaults
  (world/model/data size = 1, ranks = 0); the group accessors assert until
  `init()` runs; `init()` on one device is a no-op.
* `helpers_jax` - `pad_helper` / `truncate_helper` / `split_tensor_along_dim`
  are tested directly; `_reduce` / `_split` / `_gather` are exercised on their
  `process_count() == 1` identity fast path.
* `mappings_jax` - the four model-parallel region functions are identity (forward
  and backward) in single-process; `get_model_parallel_group()` is patched to
  return `None` so the primitives select that fast path without a distributed
  runtime.

## Skipped reference coverage (no production change)

* `tests/test_utils.py::test_wandb_login` - mirrors
  `neuralop/tests/test_utils.py::test_ArgparseConfig`, which requires a live
  `wandb` install and login. `wandb` is not a CI dependency, so the full login is
  skipped; `get_wandb_api_key` (the testable part) is covered.
