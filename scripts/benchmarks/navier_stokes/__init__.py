# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Navier-Stokes benchmark: neural operators compared at a matched size and recipe.

The Transolver paper's 2D Navier-Stokes task (``exp_ns.py``): vorticity on a
64x64 periodic grid at viscosity 1e-5; the model sees 10 frames and predicts
the next 10, one at a time, each prediction fed back as input. Every run is
trained with the paper's recipe and sized to Transolver's ~11.23M
parameters.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: evaluates the saved checkpoints on the 200 test samples and
  writes ``doc/assets/benchmarks/navier_stokes_<run>/`` and
  ``navier_stokes_comparison/``.
* ``animate.py``: renders ``animations/navier_stokes/evolution.gif``.

Results, all four models side by side: ``doc/benchmarks/navier_stokes/results/comparative.md``.
"""
