# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Settings of the physics-only Darcy benchmark.

:data:`PROTOCOL` is the data-driven benchmark's split (``darcy._config``),
unchanged, plus the physics-only choices that depend on no network: how each
Q4 element gets its coefficient value (``cell_mode``, see
:func:`._ritz.node_to_cell_coefficient`).

Each model's recipe lives in its ``train_<model>.py`` (that model's
data-driven recipe, with the loss replaced); :func:`run_dir` says where each
committed run is, for ``plot.py`` and the doc pages.
"""

from __future__ import annotations

from pathlib import Path

from scripts.benchmarks.darcy._config import (  # noqa: F401
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    REPO_ROOT,
)
from scripts.benchmarks.darcy._config import PROTOCOL as _DATA_PROTOCOL

PROTOCOL: dict = dict(
    _DATA_PROTOCOL,
    cell_mode="majority",  # floor 0.31% vs corner 1.07%, mean 0.61%
)

FLOOR_OUT_DIR = DEFAULT_OUT_DIR / "darcy_q4_s85_spec_floor"
SAMPLER_OUT_DIR = DEFAULT_OUT_DIR / "darcy_coefficient_sampler_check"
COMPARISON_OUT_DIR = DEFAULT_OUT_DIR / "darcy_physics_comparison"

MODELS: tuple[str, ...] = ("transolver", "fno", "gino", "wno")
#: Coefficient fields the physics loss sees: the 1000 released ones, then
#: 1000 + the first K fields of the sampled pool.
FIELDS: tuple[int, ...] = (1000, 2000, 5000, 10000, 20000)
#: The run each model's section of the Results page shows (predictions, error curve, compute cost):
#: the lowest-error point of its scaling curve.
HEADLINE_FIELDS: dict[str, int] = dict(
    transolver=10000, fno=10000, gino=20000, wno=20000
)


def run_dir(model: str, n_fields: int) -> Path:
    """Where the run of ``model`` on ``n_fields`` coefficient fields is
    committed (``train_<model>.py --out-dir``)."""
    return DEFAULT_OUT_DIR / f"darcy_physics_{model}" / f"fields_{n_fields}_seed0"
