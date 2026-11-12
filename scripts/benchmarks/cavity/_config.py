# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the lid-driven cavity benchmark, in one place.

* :data:`PROTOCOL` is the paper's training protocol (Sections 2.2 to 2.4),
  shared by every run.
* :data:`RUNS` are the named runs reported in
  ``doc/benchmarks/cavity/results/comparative.md``. ``train.py --run <name>`` trains
  one of them and ``infer.py`` evaluates whichever have a checkpoint.
* :func:`sample_search_configs` is the paper's random architecture search
  (``train.py --search <model>``). The ``best_*`` runs below are its winners,
  written out here so they can be retrained without repeating the search.

Which architectures exist, and the sizes the search draws from, live with the
models themselves in :data:`scripts.benchmarks.cavity._models.SPECS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.benchmarks.cavity._models import SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/cavity"
DEFAULT_DATA_DIR = Path(os.environ.get("CAVITY_DATA_DIR", "Dataset/Transient_k-e"))

MODELS: tuple[str, ...] = tuple(SPECS)
MODEL_LABELS: dict[str, str] = {name: spec.label for name, spec in SPECS.items()}

# Paper protocol (Appl. Sci. 13(5), 3165, Sections 2.2 to 2.4).
PROTOCOL: dict = dict(
    optimizer="adam",
    lr=1e-3,
    lr_factor=0.5,  # ReduceLROnPlateau factor
    lr_patience=10,  # epochs without validation improvement before halving
    early_stop_patience=50,
    epochs=300,
    batch_size=8,
    mc_weight=0.1,  # mass-conservation penalty weight (the paper's choice)
    val_frac=0.1,  # share of training windows held out for validation
    seed=0,
)

# The paper's search: 10 random sizes per model, up to 50 epochs each,
# ranked on validation loss.
SEARCH: dict = dict(n_configs=10, epochs=50, n_in=3, n_out=1, seed=0)


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model, its size, and its time windows."""

    name: str
    model: str
    n_in: int
    n_out: int
    arch: dict = field(default_factory=dict)
    epochs: int = PROTOCOL["epochs"]


_FNO_DEFAULT = dict(modes=16, width=32, n_layers=1)

# GINO is not in the paper, so it has no published size to copy. It is sized
# instead to spend the same parameter budget as ``fno_3to1`` (1,056,739), so
# that a difference in accuracy is a difference in architecture and not in
# capacity. Holding the paper's own spectral knobs at their maximum (3 layers,
# 16 modes) and solving for the latent width gives 34, for 1,058,861
# parameters: 0.2% above the FNO. Width is not restricted to powers of two
# here, because the paper's search space does not cover this model.
_GINO_FNO_MATCHED = dict(modes=16, width=34, n_layers=3, latent_grid=32)

# WNO is not in the paper either, and is sized the same way and for the same
# reason. Its parameter count moves in coarse steps, because every hidden
# channel added widens four subband weight tensors at once, so an exact match
# is not available: holding the layer count at the paper's maximum of 3 and
# sweeping the remaining two knobs over the search space, the closest point is
# level 5 with 21 channels, at 1,042,073 parameters, 1.4% below the FNO. The
# wavelet family is db6, this repo's own WNO reproduction's choice; see
# :class:`~scripts.benchmarks.cavity._models.WNOSpec`.
_WNO_FNO_MATCHED = dict(hidden_channels=21, n_layers=3, level=5)

# Transolver is the third model here that the paper does not run, and is sized
# by the same rule: hold the knobs the paper itself states at their maximum
# (3 layers), hold the slice count at the top of the search space, and solve
# for the width. The width has to be a multiple of ``n_head``, or the per-head
# width, an integer quotient, silently drops the remainder; the two
# neighbouring multiples of 8 give 962,964 parameters (8.9% below the FNO) and
# 1,233,076 (16.7% above), so 128 at 1,093,851 is the closest available, 3.5%
# above. The slice count is not a sizing knob at all: across the whole search
# space it moves the total by 0.2%, because a slice costs one projection
# column per head rather than a weight matrix. See
# :class:`~scripts.benchmarks.cavity._models.TransolverSpec`.
_TRANSOLVER_FNO_MATCHED = dict(n_hidden=128, n_layers=3, slice_num=64)

RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # Default sizes, 3 frames in and 1 out: the paper's headline setting.
        RunConfig("fno_3to1", "fno", 3, 1, _FNO_DEFAULT),
        RunConfig("gino_3to1", "gino", 3, 1, _GINO_FNO_MATCHED),
        RunConfig("wno_3to1", "wno", 3, 1, _WNO_FNO_MATCHED),
        RunConfig("transolver_3to1", "transolver", 3, 1, _TRANSOLVER_FNO_MATCHED),
        # Time-window study (the paper's Tables 4 to 6), FNO only.
        RunConfig("fno_1to1", "fno", 1, 1, _FNO_DEFAULT),
        RunConfig("fno_3to3", "fno", 3, 3, _FNO_DEFAULT),
        # Winners of the architecture search, trained for the full 300 epochs.
        RunConfig("best_fno_3to1", "fno", 3, 1, dict(modes=8, width=32, n_layers=1)),
        # The search picked the smallest of its ten sizes, at 52,260
        # parameters: a twentieth of ``gino_3to1``. Capacity is not what
        # limits GINO here, the trip through the latent grid is.
        RunConfig(
            "best_gino_3to1",
            "gino",
            3,
            1,
            dict(modes=4, width=16, n_layers=1, latent_grid=32),
        ),
        # The WNO search went the other way to GINO's: it picked the largest
        # of its ten sizes, 1,333,155 parameters, 28% above ``wno_3to1``.
        RunConfig(
            "best_wno_3to1",
            "wno",
            3,
            1,
            dict(hidden_channels=32, n_layers=1, level=4),
        ),
        # The Transolver search did neither of the above: it picked the same
        # size this page had already matched to the FNO, 1,091,403 parameters
        # against ``transolver_3to1``'s 1,093,851. The only difference between
        # the two runs is the slice count, 16 against 64, which is 0.2% of the
        # parameters.
        RunConfig(
            "best_transolver_3to1",
            "transolver",
            3,
            1,
            dict(n_hidden=128, n_layers=3, slice_num=16),
        ),
    )
}


def sample_search_configs(model: str, n: int, seed: int) -> list[RunConfig]:
    """``n`` distinct random sizes of ``model``, drawn without replacement
    from every size its :class:`~scripts.benchmarks.cavity._models.ArchitectureSpec`
    allows.

    Drawing from the enumerated grid, rather than rejecting repeated random
    draws, keeps this bounded: asking for more sizes than exist is an error
    here instead of a loop that never ends.
    """
    try:
        spec = SPECS[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; choose from {sorted(SPECS)}"
        ) from None

    configs = spec.enumerate_configs()
    if n > len(configs):
        raise ValueError(
            f"asked for {n} distinct {model} sizes but its search space has "
            f"only {len(configs)}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(configs), size=n, replace=False)

    runs: list[RunConfig] = []
    for index in chosen:
        arch, slug = configs[int(index)]
        runs.append(
            RunConfig(
                name=f"search_{model}_{slug}",
                model=model,
                n_in=SEARCH["n_in"],
                n_out=SEARCH["n_out"],
                arch=dict(arch),
                epochs=SEARCH["epochs"],
            )
        )
    return runs


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint.

    JSON turns tuples into lists, and Flax module attributes must be hashable,
    so list-valued architecture entries become tuples again.
    """
    arch = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw["arch"].items()
    }
    return RunConfig(
        name=raw["name"],
        model=raw["model"],
        n_in=raw["n_in"],
        n_out=raw["n_out"],
        arch=arch,
        epochs=raw["epochs"],
    )
