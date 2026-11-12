# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Pipe flow benchmark: neural operators compared at a matched size and recipe.

Every run maps the deformed pipe's ``(x, y)`` mesh coordinates to the
steady-state output field (channel 0 of ``Pipe_Q.npy``) on the 129x129 mesh,
trained with the recipe of the
Transolver Pipe benchmark (``_transolver.py``, a port of the paper's own
``exp_pipe.py``) and sized to its ~3.07M parameters.

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` (Transolver, WNO,
  FNO and GINO) once per seed and saves one checkpoint per run and seed;
  ``--run transolver_exp_pipe`` runs ``_transolver.py``, the port of the
  paper's ``exp_pipe.py``.
* ``infer.py``: evaluates every run's checkpoints on the 200 test samples and
  writes ``doc/assets/benchmarks/pipe_<run>/`` and ``pipe_comparison/``.
* ``animate.py``: renders ``animations/pipe/evolution.gif``.
* ``tune_transolver.py``: the Optuna/Ray Tune search of Transolver's
  optimisation settings.

Results, all four models side by side: ``doc/benchmarks/pipe/results/comparative.md``.
"""
