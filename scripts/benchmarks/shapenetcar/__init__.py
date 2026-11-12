# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-Design-ShapeNetCar benchmark (Wu et al. 2024, ICML, Transolver paper).

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: evaluates the saved checkpoints on the held-out fold and
  writes ``doc/assets/benchmarks/shapenetcar_<model>/``, one folder per
  model.
* ``animate.py``: re-renders ``animations/shapenetcar/evolution.gif`` from
  the frames ``train.py --animate`` caches.

Three architectures are registered: Transolver (the paper's own), GINO and
GIWO (both ours, parameter/cost-matched to Transolver -- see ``_models.py``'s
module docstring for why this benchmark's registry skips the automated
parameter-matching *search* machinery ``lid_driven_cavity``/``darcy`` use;
GINO/GIWO's sizes come from direct measurement against a fixed target
instead).

Results: ``doc/benchmarks/shapenetcar/results/comparative.md``.
"""
