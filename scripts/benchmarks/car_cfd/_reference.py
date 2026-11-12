# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reported results of Li et al. 2023 (GINO, NeurIPS), Table 2, Shape-Net
Car -- the targets this reproduction is measured against.

Train error is the normalised relative L2 the model trains on; test error is
the *de-normalised* relative L2 on the 111 held-out cars. The two are
different metrics and are never compared to each other.
"""

from __future__ import annotations

CITATION = (
    "Li, Z.; Kovachki, N. B.; Choy, C.; Li, B.; Kossaifi, J.; Otta, S. P.; "
    "Nabian, M. A.; Stadler, M.; Hundt, C.; Azizzadenesheli, K.; "
    "Anandkumar, A. Geometry-Informed Neural Operator for Large-Scale 3D "
    "PDEs. Advances in Neural Information Processing Systems 36 (NeurIPS "
    "2023)."
)
DOI = "https://arxiv.org/abs/2309.00583"

# Table 2, Shape-Net Car: (train error, test error).
TABLE2: dict[str, dict[str, float]] = {
    "gino_decoder": dict(train=0.0637, test=0.0712),
    "gino_encoder_decoder": dict(train=0.0795, test=0.0947),
    "fno_interp": dict(train=0.0965, test=0.0942),
}


def paper_row(use_in_gno: bool) -> str:
    """The Table 2 row a run's topology corresponds to."""
    return "gino_encoder_decoder" if use_in_gno else "gino_decoder"
