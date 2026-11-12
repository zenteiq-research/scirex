# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-CFD benchmark under the GINO paper's own protocol (Li et al. 2023,
NeurIPS, Table 2, "Shape-Net Car"): GINO, Transolver and GIWO at GINO's
parameter budget.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: re-scores the saved checkpoints on the 111 held-out cars and
  writes the per-model figures and ``metrics.json`` to
  ``doc/assets/benchmarks/car_cfd_<model>/`` and the three-model comparison
  figures to ``doc/assets/benchmarks/car_cfd_comparison/``.
* ``animate.py``: renders ``animations/car_cfd/evolution.gif``, the best
  model's surface pressure across the test cars.

Distinct from ``shapenetcar/``, which runs the Transolver paper's
point-cloud protocol on the same car geometries -- the two are not
comparable.

Results, all three models side by side: ``doc/benchmarks/car_cfd/results/comparative.md``.
"""
