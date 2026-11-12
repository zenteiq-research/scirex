# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Chip cooling benchmark: FNO, WNO, GINO and Transolver at a matched size.

Every run maps the six boundary-condition parameters of a liquid-cooled
dual-chip cold plate (two chip heat fluxes, two inlet mass flow rates, two
inlet temperatures) to the steady-state temperature field on its central
plane. The 2000 simulations are split 1600 train / 400 test, the same split
for every model, and every model has about 1.8M parameters, Transolver's
size.

FNO and WNO work on a regular 128x256 grid the fields are interpolated onto
(evaluated on the grid cells inside the domain); GINO and Transolver work on
the 11,110 mesh nodes directly.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: evaluates the saved checkpoints on the 400 test samples and
  writes ``doc/assets/benchmarks/chip_cooling_<run>/`` and
  ``doc/assets/benchmarks/chip_cooling_comparison/``.
* ``animate.py``: the best model's prediction across the test set, as a GIF.

Results, all four models side by side: ``doc/benchmarks/chip_cooling/results/comparative.md``.
"""
