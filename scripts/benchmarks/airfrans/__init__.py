# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""AirfRANS benchmark: steady RANS flow around 2-D airfoils, scored exactly as
in the Transolver paper (Wu et al., ICML 2024, Table 3).

Every run maps the official 7 node features of an AirfRANS case (position,
inflow velocity, signed distance, wall normal), plus two data-driven features
(log wall distance and log Reynolds number), to the RANS fields
``(Ux, Uy, p, nut)`` on every mesh node. Training follows the official
Transolver AirfRANS recipe (``Airfoil-Design-AirfRANS``: 720 / 80 / 200
split, 32,000 random nodes per step, volume + surface MSE, Adam with
OneCycleLR), and the test set is scored with a re-implementation of the
official ``Results_test`` (normalised volume / surface MSE, lift and drag
coefficients from integrated pressure and wall shear stress).

Entry points (run from the repo root):

* ``train.py``: trains the runs defined in ``_config.py`` and saves one
  checkpoint per run.
* ``infer.py``: evaluates the saved checkpoints on the 200 test cases and
  writes ``doc/assets/benchmarks/airfrans_<run>/``.

GINO has its own trainer, ``_gino.py`` (``train.py --run gino`` runs it with
the published settings, ``infer.py`` scores it).

Results: ``doc/benchmarks/airfrans/results/comparative.md``.
"""
