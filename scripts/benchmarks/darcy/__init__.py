# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Darcy flow benchmark: piecewise-constant permeability -> pressure.

Li et al.'s released ``piececonst_r421_N1024_smooth{1,2}.mat`` data, strided
from 421x421 to 85x85; 1000 training and 200 test samples. FNO, WNO, GINO and
Transolver, each at Transolver's parameter count (~2.83M), each with its own
paper's (or reference code's) training recipe, see ``_config.py``.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run (``--animate`` also saves the frames ``animate.py``
  renders).
* ``infer.py``: evaluates the saved checkpoints on the test set and writes
  ``doc/assets/benchmarks/darcy_<model>/`` and ``darcy_comparison/``.
* ``animate.py``: renders ``animations/darcy_flow/evolution.gif``.

Results: ``doc/benchmarks/darcy/results/comparative.md``.
"""
