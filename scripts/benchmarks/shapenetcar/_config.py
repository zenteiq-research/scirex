# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the Car-Design-ShapeNetCar benchmark, in one place.

:data:`PROTOCOL` is the Transolver paper's own training protocol for this
benchmark (``Car-Design-ShapeNetCar/main.py``'s argparse defaults, and
``train.py``'s ``OneCycleLR`` construction, in the official PyTorch repo,
https://github.com/thuml/Transolver) -- confirmed against that source, not
re-derived. ``pct_start``/``div_factor`` are not set explicitly there, so
they are PyTorch ``OneCycleLR``'s own defaults, written out here rather than
left implicit.

:data:`RUNS` are the named runs. Which architectures exist lives with the
models themselves in :data:`scripts.benchmarks.shapenetcar._models.SPECS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.benchmarks.shapenetcar._models import SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/shapenetcar"

MODELS: tuple[str, ...] = tuple(SPECS)
MODEL_LABELS: dict[str, str] = {name: spec.label for name, spec in SPECS.items()}

# Car-Design-ShapeNetCar/main.py's argparse defaults + train.py's OneCycleLR
# construction (official thuml/Transolver repo).
PROTOCOL: dict = dict(
    n_epochs=200,
    learning_rate=1e-3,
    # main.py's --weight ("reg"): total = loss_velo + press_weight * loss_press
    press_weight=0.5,
    fold_id=0,  # main.py's --fold_id
    # OneCycleLR: only final_div_factor is passed explicitly in the
    # reference's train.py; pct_start/div_factor are PyTorch's own defaults.
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1000.0,
    seed=0,
)


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model and its size."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    n_epochs: int = PROTOCOL["n_epochs"]


# The paper's own model size for this benchmark (main.py's Transolver(...)
# construction): n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32.
_TRANSOLVER_PAPER_SIZE = dict(
    n_hidden=256, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32
)

# GINO is not in the Transolver paper, so it has no published size to copy
# here either. Sized instead to spend the same parameter budget as
# "transolver" (3,860,804), following the same rule
# lid_driven_cavity/_config.py's _GINO_FNO_MATCHED/_WNO_FNO_MATCHED/
# _TRANSOLVER_FNO_MATCHED use: hold the knobs that do not move the count much
# (layer count, the GNO kernel MLP width) at a reasonable value and solve for
# the one that does (here, the latent FNO's hidden width) by direct
# count_params measurement, not a closed-form formula -- GINO's parameter
# count depends on channel widths in ways that are easiest to just measure.
# latent_grid=10, modes=6, n_layers=4, gno_mlp_hidden=(64, 64) fixed;
# width=57 lands at 3,852,494 parameters, 0.22% below the Transolver target
# (width=56: 3,720,856, -3.6%; width=58: 3,986,924, +3.3% -- 57 is the
# closest of any integer width). in_gno_radius/out_gno_radius does not move
# the parameter count at all (a neighbor-search cutoff, not a model weight),
# but does set how much a sample's points cost to train on. A radius sized
# off the sparse latent grid's own spacing (e.g. 1.5x its ~1.0 grid spacing,
# matching lid_driven_cavity's GINO) finds a neighbor within a few units of
# every latent point -- around 1.57M (point, latent-point) pairs for one
# sample, since the 32,186-point car surface is dense at that scale -- which
# is the connectivity a local kernel is supposed to avoid, not spend compute
# on; it measured at roughly 11s/sample on an NVIDIA GB10 (the machine for
# all the timings in this file unless stated; not reproduced in the committed
# metrics.json), making a 200-epoch run impractical. The input GNO (1,000 latent queries searching the dense
# 32,186-point car cloud) and the output GNO (32,186 car-point queries
# searching the sparse 1,000-point latent grid) sit at very different
# densities, so one radius cannot size both -- each is set separately, to the
# smallest value whose *own* average neighbor count (over the queries that
# find any neighbor at all; an empty query means that latent cell/car point
# has no signal for this particular sample, costs nothing to compute, and is
# not a sizing concern) lands in the 5-10 range:
#   in_gno_radius=0.14:  1,005 pairs/sample, avg 7.08 over 142/1,000 active
#                        latent cells (most of the fixed grid sits outside
#                        any one car's own extent, see _gino_geometry.py)
#   out_gno_radius=0.80: 282,680 pairs/sample, avg 8.78, every one of the
#                        32,186 car points covered
_GINO_TRANSOLVER_MATCHED = dict(
    latent_grid=10,
    modes=6,
    width=57,
    n_layers=4,
    gno_mlp_hidden=(64, 64),
    in_gno_radius=0.14,
    out_gno_radius=0.8,
)

# An ad hoc comparison, not a second paper-matched run: same architecture and
# parameter count as "gino" above (radius is a neighbor-search cutoff, not a
# model weight -- see _GINO_TRANSOLVER_MATCHED's own comment), but with both
# radii raised to spend roughly 3x the per-sample edge count (and so roughly
# 3x the per-epoch wall time), to see whether a wider receptive field closes
# any of the accuracy gap against "gino" at a fixed, early epoch -- at the
# cost of leaving the "avg 5-10 neighbors" sizing the first run was built to.
# Measured on sample 0 of fold 0 (same method as _GINO_TRANSOLVER_MATCHED):
#   in_gno_radius=0.22:  4,128 edges/sample, avg 20.64 over 200/1,000 active
#                        latent cells (vs. 0.14's 1,005 edges, avg 7.08)
#   out_gno_radius=1.2:  918,955 edges/sample, avg 28.55, every one of the
#                        32,186 car points covered (vs. 0.8's 282,680 edges,
#                        avg 8.78)
# Total edges/sample: ~923,083 vs. "gino"'s ~283,685 (~3.25x). This is the
# headline run reported in doc/benchmarks/shapenetcar/results/comparative.md: trained for the
# full 200-epoch protocol (final val loss 0.3177, ~176 min on an NVIDIA
# GB10; the committed doc/assets/benchmarks/shapenetcar_gino/metrics.json
# and the results page record a later run on an NVIDIA RTX PRO 6000
# Blackwell: ~10.5 s/epoch, ~37 min), both to compare against the matched-parameter "gino" sizing above
# and because pushing the radius further (see
# _GINO_TRANSOLVER_COST_MATCHED below) turned out not to close the gap to
# Transolver any further.
_GINO_WIDE_RADIUS = dict(
    _GINO_TRANSOLVER_MATCHED, in_gno_radius=0.22, out_gno_radius=1.2
)

# A second step on the same experiment: radii raised again to spend roughly
# Transolver's own ~118.6s/epoch (train.py's measured steady-state cost for
# "transolver", _TRANSOLVER_PAPER_SIZE, on the GB10; the committed
# shapenetcar_transolver/metrics.json is ~16.8 s/epoch, ~58.6 min on an RTX
# PRO 6000 Blackwell, so the epoch times below are GB10 values), so the two architectures' epoch-20
# losses can be compared at matched *compute* cost per epoch, not just
# matched parameter count. Extrapolating a linear edges-vs-epoch_time fit
# from "gino" (283,685 edges, 17.7s) and "gino_wide_radius" (923,083 edges,
# 50.8s) -- t ~= 5.18e-5 * edges + 3.0s -- put the target at ~2.23M
# edges/sample (all epoch times in this block: GB10). Measured on sample 0
# of fold 0:
#   in_gno_radius=0.30:  11,939 edges/sample, avg 44.88 over 266/1,000 active
#                        latent cells
#   out_gno_radius=1.65: 2,255,071 edges/sample, avg 70.06, every one of the
#                        32,186 car points covered
# Total edges/sample: ~2,267,010, predicting ~120s/epoch by the fit above
# (measured: ~125.5s/epoch steady state). Run only to epoch 20 (--epochs 21)
# as a fast, matched-compute-cost comparison point, not a full 200-epoch
# run -- at epoch 20 its val loss (0.3133) was statistically the same as
# "gino_wide_radius"'s (0.3175 at epoch 200), the headline result reported
# in the docs: pushing the radius this far bought no further accuracy on
# this benchmark, so the full run used "gino_wide_radius"'s smaller, cheaper
# radii instead.
_GINO_TRANSOLVER_COST_MATCHED = dict(
    _GINO_TRANSOLVER_MATCHED, in_gno_radius=0.3, out_gno_radius=1.65
)

# GIWO ("Geo-WNO": src/scirex/experimental/giwo.py) is GINO's own GNO-in/GNO-out
# shell with its latent FNO replaced by a WNO -- see that module's docstring.
# Sized to Transolver's budget the same way GINO is, by direct count_params
# measurement, holding gno_mlp_hidden/in_gno_radius/out_gno_radius at exactly
# GINO's own values (same latent grid, same car geometry, so the same radius
# gives the same neighbor counts -- see _GINO_TRANSOLVER_MATCHED's own
# derivation, not repeated here) and n_layers=4 to match GINO's own latent
# depth, then sweeping the latent WNO's width (wno_hidden_channels) and its
# wavelet decomposition level:
#   n_layers=4, level=1, wavelet="db4": width=15 -> 3,752,638 params, -2.80%
#   below the Transolver target (width=14: 3,276,826, -15.13%; width=16:
#   4,261,236, +10.37% -- 15 is the closest of any integer width). Unlike
#   GINO's FNO (whose parameter count is set by a fixed number of Fourier
#   modes, independent of grid resolution), a WaveletConv's subband weights
#   are shaped from the *entire* coefficient array the latent grid's wavelet
#   decomposition produces, so param count grows much faster with width here
#   than it does for GINO at the same grid size -- hence the coarser spacing
#   between integer-width options (GINO's own sweep lands within 0.22%, this
#   one within 2.80%, simply because fewer integer widths sit near the
#   target).
_GIWO_TRANSOLVER_MATCHED = dict(
    latent_grid=10,
    width=15,
    n_layers=4,
    level=1,
    wavelet="db4",
    gno_mlp_hidden=(64, 64),
    in_gno_radius=0.14,
    out_gno_radius=0.8,
)

# Same question _GINO_TRANSOLVER_COST_MATCHED asks of GINO, asked of GIWO
# instead: push the radius up to spend roughly Transolver's own
# ~118.6s/epoch (GB10), and see whether a wider receptive field buys GIWO any
# accuracy at matched *compute* cost, not just matched parameter count.
# "gino_transolver_cost_matched" found no such gain for GINO at epoch 20
# (0.3133 vs. "gino_wide_radius"'s 0.3175 at epoch 200) -- this is the same
# check for GIWO, at GIWO's own per-edge cost (the two architectures spend a
# given edge count differently: GIWO's latent WNO is slightly more expensive
# per edge than GINO's latent FNO, so the two land on different radii for
# the same ~118.6s/epoch target).
#
# Measured steady-state epoch time on the GB10 (the committed
# shapenetcar_giwo/metrics.json, an RTX PRO 6000 Blackwell run of
# "giwo_transolver_cost_matched", is ~23.5 s/epoch, ~82.5 min for 200
# epochs; the figures below are not reproduced there) (same sample-0-of-fold-0
# radius/edge-count method as GINO's own derivations; edge counts are
# identical to GINO's at the same radius -- same latent grid, same car
# geometry, see _GIWO_TRANSOLVER_MATCHED's own comment):
#   "giwo" (in=0.14/out=0.8, 283,685 edges/sample):        ~20.6s/epoch
#   in=0.22/out=1.2 (923,083 edges/sample, GINO's own "wide_radius" pair):
#                                                           ~49.2s/epoch
# Linear fit t ~= 4.48e-5 * edges + 7.9s (vs. GINO's own 5.18e-5 * edges +
# 3.0s -- a shallower per-edge slope but a larger fixed overhead) puts the
# ~118.6s/epoch target at ~2.47M edges/sample. Rather than re-measure a new
# radius pair from scratch, reusing GINO's own "cost_matched" radius
# (in=0.3/out=1.65, 2,267,010 edges/sample -- 8% short of that estimate)
# measured at ~115.3s/epoch steady state, 97% of the Transolver target and
# close enough not to warrant a further radius search.
_GIWO_TRANSOLVER_COST_MATCHED = dict(
    _GIWO_TRANSOLVER_MATCHED, in_gno_radius=0.3, out_gno_radius=1.65
)

RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        RunConfig("transolver", "transolver", _TRANSOLVER_PAPER_SIZE),
        RunConfig("gino", "gino", _GINO_TRANSOLVER_MATCHED),
        RunConfig("gino_wide_radius", "gino", _GINO_WIDE_RADIUS),
        RunConfig(
            "gino_transolver_cost_matched", "gino", _GINO_TRANSOLVER_COST_MATCHED
        ),
        RunConfig("giwo", "giwo", _GIWO_TRANSOLVER_MATCHED),
        RunConfig(
            "giwo_transolver_cost_matched", "giwo", _GIWO_TRANSOLVER_COST_MATCHED
        ),
    )
}


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint.

    JSON turns tuples into lists, and Flax module attributes must be hashable,
    so list-valued architecture entries (e.g. GINO's ``gno_mlp_hidden``)
    become tuples again -- same fix as lid_driven_cavity/_config.py's own
    ``run_config_from_dict``.
    """
    arch = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw["arch"].items()
    }
    return RunConfig(
        name=raw["name"],
        model=raw["model"],
        arch=arch,
        n_epochs=raw["n_epochs"],
    )
