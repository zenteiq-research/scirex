# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Lid-driven cavity benchmark under the protocol of Costa Rocha et al. 2023.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: evaluates the saved checkpoints on the unseen Re = 10,000
  case and writes the per-model figures and ``metrics.json`` to
  ``doc/assets/benchmarks/cavity_<model>/`` and the four-model comparison
  figures to ``doc/assets/benchmarks/cavity_comparison/``.
* ``animate.py``: re-renders ``animations/cavity/evolution.gif`` from the
  frames ``train.py --animate`` caches.

Results: ``doc/benchmarks/cavity/results/comparative.md``.
"""
