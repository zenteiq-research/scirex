# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Darcy flow benchmark, physics-only: the same problem as ``darcy/``, trained
without any labelled solution.

Same data, split and metric as the data-driven benchmark (Li et al.'s
``piececonst_r421_N1024_smooth{1,2}.mat``, strided 421 -> 85, 1000 training
coefficient fields, 200 test pairs, per-sample relative L2). Only the loss
changes: each network minimises the Q4 finite-element Ritz energy of
``-div(a grad u) = 1``, ``u = 0`` on the boundary, and never sees a training
solution. See ``_ritz.py`` for the model-independent pieces.

Entry points (run from the repo root):

* ``train_<model>.py`` (``transolver``, ``fno``, ``gino``, ``wno``): trains
  one run with that model's data-driven recipe and the Ritz loss, on the
  1000 released coefficient fields plus ``--extra-coefficients K`` sampled
  ones; writes its ``metrics.json`` and figures to ``--out-dir`` and its
  final weights under ``scripts/benchmarks/checkpoints/``.
* ``plot.py``: the unlabelled-scaling figures, from the committed
  ``metrics.json`` files only.
* ``animate.py``: renders ``animations/darcy_physics/evolution.gif`` from the
  best model's final weights (Transolver, 10k fields).
* ``check_gino_gradient_scale.py``, ``check_wno_gradient_scale.py``: the
  gradient scale at initialisation behind each model's loss scale.
* ``check_floor.py``: scores an exact Q4 solve against the released test
  solutions, the error a perfectly trained physics-only model would reach;
  writes ``doc/assets/benchmarks/darcy_q4_s85_spec_floor/``.
* ``check_sampler.py``: checks that the extra, unlabelled coefficient fields
  follow the law of the released ones; writes
  ``doc/assets/benchmarks/darcy_coefficient_sampler_check/``.

Results: ``doc/benchmarks/darcy_physics/results/{setup,comparative}.md``.
"""
