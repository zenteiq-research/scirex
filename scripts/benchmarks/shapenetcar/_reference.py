# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reported results of Wu et al. 2024 (Transolver, ICML), Table 3, for the
Car-Design-ShapeNetCar benchmark -- the targets this reproduction is measured
against.

Unlike ``lid_driven_cavity``/``darcy``'s ``_reference.py``, this dict is keyed
by *metric* rather than by *model*: this benchmark has exactly one
architecture (Transolver, the paper's own), so there is no "which models does
the paper not cover" question -- ``NOT_IN_PAPER`` is here only so
``scripts/tests/test_shapenetcar.py`` can use the same
every-model-is-covered-or-declared-absent pattern those benchmarks use.
"""

from __future__ import annotations

CITATION = (
    "Wu, H.; Luo, H.; Wang, H.; Wang, J.; Long, M. Transolver: A Fast "
    "Transformer Solver for PDEs on General Geometries. Proceedings of the "
    "41st International Conference on Machine Learning, PMLR 235, 2024, "
    "53681-53705."
)
DOI = "https://arxiv.org/abs/2402.02366"

# Table 3, Car-Design-ShapeNetCar.
TABLE3: dict[str, float] = dict(
    volume_rel_l2=0.0207,  # "Volume": velocity, relative L2 over all points
    surface_rel_l2=0.0745,  # "Surf": pressure, relative L2 over surface points
    c_d=0.0103,  # mean relative error of the drag coefficient
    rho_d=0.9935,  # Spearman rank correlation of the drag coefficient
)

#: Every model this benchmark reports Table 3 numbers for (all of them --
#: Transolver is the paper's own architecture for this benchmark).
NOT_IN_PAPER: tuple[str, ...] = ()
