# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reported results of Costa Rocha et al. 2023 (Appl. Sci. 13(5), 3165),
transcribed from the paper, as the targets this reproduction is measured
against.

All RMSE values are in m/s and pool the pressure and both velocity components,
on the held-out Re = 10,000 case (the paper's Section 2.2 explains that it
reports the pooled quantity in m/s because pressure is ~10x smaller than the
velocities). The one exception is ``ABSTRACT_VEL_ONLY``, which the abstract
states for the x and y velocity components only.
"""

from __future__ import annotations

CITATION = (
    "Costa Rocha, P.A.; Johnston, S.J.; Oliveira Santos, V.; Aliabadi, A.A.; "
    "Thé, J.V.G.; Gharabaghi, B. Deep Neural Network Modeling for CFD "
    "Simulations: Benchmarking the Fourier Neural Operator on the Lid-Driven "
    "Cavity Case. Appl. Sci. 2023, 13, 3165."
)
DOI = "https://doi.org/10.3390/app13053165"

# Table 1 -- RMSE of each optimised architecture after 50 epochs.
TABLE1_50_EPOCHS: dict[str, float] = {
    "fno": 6.83e-3,
}

# Table 2 -- validation RMSE after up to 300 epochs, per mass-conservation
# weight. Keys are the weight applied to the MC error term.
TABLE2_300_EPOCHS: dict[str, dict[float, float]] = {
    "fno": {0.0: 6.56e-3, 0.1: 6.43e-3, 1.0: 6.46e-3, 2.0: 6.49e-3, 5.0: 6.48e-3},
}

# Table 4 -- FNO predicting ONE timestep ahead, by input window size.
TABLE4_FNO_N_TO_1: dict[int, float] = {
    1: 6.07e-3,
    3: 6.43e-3,
    6: 7.15e-3,
    10: 6.81e-3,
    15: 6.70e-3,
}

# Table 5 -- FNO rolled out TEN timesteps ahead, by input window size.
TABLE5_FNO_N_TO_10: dict[int, float] = {
    1: 1.60e-2,
    3: 1.07e-2,
    6: 9.28e-3,
    10: 8.43e-3,
    15: 6.59e-3,
}

# Table 6 -- FNO reaching ten timesteps ahead from a 3-frame input, by the
# model's own output-window size (3:3 is the paper's best overall result).
TABLE6_FNO_3_TO_N: dict[int, float] = {1: 1.07e-2, 3: 8.79e-3, 10: 1.07e-2}

# Abstract / Conclusions headline figures.
ABSTRACT_VEL_ONLY = 6.0743e-3  # FNO 1:1, x and y velocity components only
BEST_OVERALL = 8.792e-3  # FNO 3:3, pooled, ten timesteps ahead

# The single-step target each of our `<model>_3to1` runs is measured against.
# Table 2 at the paper's chosen MC weight of 0.1 is the most directly
# comparable published number: same 3:1 setting, same soft constraint, same
# 300-epoch budget.
#
# A model the paper never ran is simply absent from these dicts, so callers
# look targets up with ``.get(model)`` and treat ``None`` as "no published
# number to compare against". GINO, WNO and Transolver are all such models.
# Inventing a target for one, or borrowing the FNO's, would misreport the
# paper.
SINGLE_STEP_TARGET: dict[str, float] = {
    "fno": TABLE2_300_EPOCHS["fno"][0.1],
}

#: Models this reproduction runs that Costa Rocha et al. 2023 does not report.
#: Their accuracy is compared against our own FNO and against the
#: copy-the-last-frame baseline instead.
NOT_IN_PAPER: tuple[str, ...] = ("gino", "wno", "transolver")
