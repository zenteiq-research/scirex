<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Module Changes - Phase 5 (losses tests)

This document records production source changes made while writing the Phase 5
losses test suite (`test_data_losses.py`).

## No production changes

The `LpLoss` / `H1Loss` JAX port matched the upstream reference behaviour
exactly (abs/rel norms, sum/mean reduction, uniform quadrature weights), so no
production source changes were required.

## Skipped reference coverage (no production change)

The reference `neuralop/losses/tests/test_losses.py` covers classes/modules that
are not part of the scirex losses package:

* `test_data_losses.py::test_hdivloss` - mirrors `test_hdivloss`. The scirex
  losses package implements `LpLoss` and `H1Loss` only; `HdivLoss` is not ported.
* `test_data_losses.py::test_differentiation_module_not_ported` - the
  reference file also tests the `differentiation` module (`FiniteDiff`,
  `FourierDiff`, `non_uniform_fd`) via `test_nonuniform_fd`,
  `test_finite_diff_{1,2,3}d` and `test_fourier_diff_{,2d,3d}`. That module is not
  part of the scirex losses package, so those tests have no analogue here.
