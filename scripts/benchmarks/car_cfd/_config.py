# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the Car-CFD benchmark, in one place.

:data:`PROTOCOL` is the GINO paper's own protocol for Shape-Net Car (Table 2,
Sec. 4.2): latent 64^3, GNO radius 0.055, Adam, 100 epochs, start LR 2.5e-4
halved at epoch 50, 500 train / 111 test cars.

:data:`RUNS` are the named runs reported in ``doc/benchmarks/car_cfd_*.md``.
Each one is a set of overrides on top of ``src/scirex/config/gino_carcfd_config.py``'s
``Default`` (and so of ``src/scirex/config/models.py``'s ``GINOSmall3d``);
:func:`build_config` applies them. ``train.py --run <name>`` trains one and
``infer.py`` evaluates whichever have a checkpoint.

Most runs are GINO (``architecture="gino"``). ``"transolver"`` trains
Transolver under the same protocol, data, loss and test metric, sized to
GINO's parameter budget (see ``_TRANSOLVER_GINO_MATCHED``); its ``model``
dict holds :class:`scirex.models.Transolver` fields instead of GINO ones.
``"giwo"`` is GINO with a WNO latent operator (``_giwo.GIWO``), also at
GINO's budget; its ``model`` dict is GINO overrides plus ``wno_*`` fields.

Why so many overrides: ``src/scirex/config/gino_carcfd_config.py`` was copied from
neuralop's own ``config/gino_carcfd_config.py``, which is a reduced-cost
*demo* config (32^3 latent, radius 0.033, constant LR 1e-3, and
``GINOSmall3d``'s tucker/rank-0.4/instance-norm FNO), not the configuration
the paper reports. ``"demo"`` below keeps that config as the starting point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/car_cfd"
DEFAULT_OUT_ROOT = REPO_ROOT / "doc/assets/benchmarks"
DEFAULT_VTK_ROOT = REPO_ROOT / "scripts/benchmarks/vtk_outputs/car_cfd"

#: The benchmark's architectures, in page order, with their display labels.
MODELS: tuple[str, ...] = ("gino", "transolver", "giwo")
MODEL_LABELS: dict[str, str] = {
    "gino": "GINO",
    "transolver": "Transolver",
    "giwo": "GIWO",
}
#: Each model's reported run: the one in the README and comparison figures.
#: "paper" is the GINO paper's own protocol; the other two keep it but for
#: an optimiser tuned on a validation split (train.py --val 50).
HEADLINE: dict[str, str] = {
    "gino": "paper",
    "transolver": "transolver_val",
    "giwo": "giwo_val",
}
#: Extra seeds of a headline run, trained with ``train.py --seed N`` into
#: ``<run>_seed<N>/``; infer.py reports their mean and spread.
SEEDS: dict[str, tuple[int, ...]] = {"transolver_val": (1, 2)}

# GINO paper, Table 2 / Sec. 4.2.
PROTOCOL: dict = dict(
    n_train=500,
    n_test=111,
    n_epochs=100,
    sdf_query_resolution=64,
    learning_rate=2.5e-4,
    lr_step=50,  # StepLR period, epochs
    lr_gamma=0.5,  # StepLR factor
    # The paper reports Adam. ``adamw_cx`` with weight_decay=0 is exactly
    # Adam; runs that carried this repo's AdamW default (1e-4) say so below.
    weight_decay=0.0,
    seed=0,
)


@dataclass(frozen=True)
class RunConfig:
    """One training run: model overrides plus the optimiser/data settings
    that differ from :data:`PROTOCOL`."""

    name: str
    #: For ``architecture="gino"``, overrides of ``GINOSmall3d``'s fields
    #: (``src/scirex/config/models.py``); for ``"transolver"``, the full Transolver size.
    model: dict = field(default_factory=dict)
    sdf_query_resolution: int = PROTOCOL["sdf_query_resolution"]
    learning_rate: float = PROTOCOL["learning_rate"]
    lr_step: int = PROTOCOL["lr_step"]
    lr_gamma: float = PROTOCOL["lr_gamma"]
    weight_decay: float = PROTOCOL["weight_decay"]
    n_epochs: int = PROTOCOL["n_epochs"]
    #: ``"gino"``, ``"transolver"`` or ``"giwo"`` (``_models.ARCHITECTURES``).
    architecture: str = "gino"
    #: ``None``: StepLR (``lr_step``/``lr_gamma``), set once per epoch. A
    #: dict: a per-step schedule peaking at ``learning_rate``, ``{"kind":
    #: "onecycle", ...}`` or ``{"kind": "warmup_cosine", ...}`` (see
    #: :func:`lr_schedule`).
    schedule: dict | None = None
    #: Cars per training step (the test pass is always one car at a time).
    batch_size: int = 1
    #: Global gradient-norm clip, or ``None`` for none.
    grad_clip: float | None = None
    #: Adam's second-moment decay.
    adam_b2: float = 0.999
    #: Adam's epsilon (``adamw_cx``'s default is 1e-6, PyTorch's 1e-8).
    adam_eps: float = 1e-6


# The paper's "GINO (decoder)" row: no input GNO, the SDF on the latent grid is
# the model's whole input. ``use_in_gno=False`` is this repo's addition -- the
# released neuralop ``GINO.encode()`` runs the input GNO unconditionally.
#
# ``gno_radius`` is a config field GINO stores but never reads: its GNO
# blocks use ``in_gno_radius``/``out_gno_radius`` (both 0.033 by default).
# Every GINO run below that sets only ``gno_radius`` -- "paper" and
# "paper_decoder_r0275" included -- therefore trained at radius 0.033, which
# their neighbour logs confirm. ``_PAPER_RADIUS`` sets the radius GINO reads.
_PAPER_DECODER = dict(use_in_gno=False, gno_radius=0.055)
_PAPER_RADIUS = dict(out_gno_radius=0.055)

# Reference ``GINO.__init__`` defaults (no factorization, rank 1.0, no norm),
# rather than the tucker/0.4/instance-norm GINOSmall3d inherited from the
# neuralop demo config. Neither the paper nor neuralop's own config sets them.
_REFERENCE_ARCH = dict(fno_factorization=None, fno_rank=1.0, fno_norm=None)

# This repo's AdamW default, which the first paper-protocol runs carried.
_ADAMW_WD = 1e-4

# Transolver is not in the GINO paper, so it has no published size for this
# benchmark. Sized instead to "paper"'s parameter budget (75,917,953), the
# rule shapenetcar/_config.py uses the other way round: keep the Transolver
# paper's own Car-Design-ShapeNetCar shape (8 layers, 8 heads, mlp_ratio 2,
# 32 slices) and solve for the width by direct count_params measurement.
# n_hidden must divide by n_head; 1136 gives 75,481,841 (-0.57%), 1144 gives
# 76,547,649 (+0.83%), so 1136 is the closest.
_TRANSOLVER_GINO_MATCHED = dict(
    n_hidden=1136, n_layers=8, n_head=8, mlp_ratio=2, slice_num=32
)

# The Transolver paper's own optimiser recipe for Car-Design-ShapeNetCar
# (shapenetcar/_config.py's PROTOCOL): Adam, peak lr 1e-3, OneCycle stepped
# every batch. Its 200-epoch budget is not copied; the run keeps this
# benchmark's 100 epochs, so it costs the same as "paper" and "transolver".
_TRANSOLVER_ONECYCLE = dict(
    kind="onecycle", pct_start=0.3, div_factor=25.0, final_div_factor=1000.0
)

# GIWO at "paper"'s parameter budget: "paper"'s exact shell (decoder-only,
# the reference architecture's output GNO and projection, and so the radius
# 0.033 "paper" actually trained at -- the finished GINO run to compare
# against) around a 4-layer db4 WNO on the 64^3 latent grid. A WaveletConv's
# weights are shaped by the coarsest subband of the whole grid, so width and
# decomposition level trade off against each other (count_params, latent
# 64^3): width 68 at level 5 (8^3 subbands) 76,161,293 (+0.32%); 49 at level
# 4 (10^3) 77,205,804 (+1.70%); 29 at level 3 (14^3) 74,197,232 (-2.27%;
# width 30 is 79,378,907, +4.56%). Level 3 is chosen: it is pywt's maximum
# useful level for a 64-point axis and db4 (dwt_max_level), its 14^3
# subbands are closest to GINO's 16^3 Fourier modes, and it is the cheapest
# -- with jaxwt's own transform, which runs each level as a dense 8^3 3-D
# convolution, on an NVIDIA GB10 a training step cost ~1.1 s/car (w29/l3), ~1.7 (w49/l4),
# ~2.4 (w68/l5), against GINO's ~0.19 (not reproduced in the committed
# metrics.json). _giwo uses _fast_dwt's separable version of the same
# transform: ~0.17 s/car at w29/l3, ~2.6 h per run on the GB10. The committed
# metrics.json (car_cfd_giwo, car_cfd_gino) and results pages are from an
# NVIDIA RTX PRO 6000 Blackwell: GIWO ~11 s/epoch, ~21 min per run; GINO
# ~17 s/epoch, ~31 min.
_GIWO_SHELL = {**_PAPER_DECODER, **_REFERENCE_ARCH}
_GIWO_GINO_MATCHED = {**_GIWO_SHELL, "fno_hidden_channels": 29, "wno_level": 3}

RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # The headline run: every setting the paper states, plus the reference
        # (non-demo) architecture and plain Adam.
        RunConfig("paper", {**_PAPER_DECODER, **_REFERENCE_ARCH}),
        # neuralop's demo config as shipped: encoder-decoder, 32^3, radius
        # 0.033, constant LR 1e-3 (its declared StepLR never reached the
        # optimiser), 301 epochs. 12.72% test, matching neuralop's own 12.80%.
        RunConfig(
            "demo",
            {},
            sdf_query_resolution=32,
            learning_rate=1e-3,
            lr_gamma=1.0,
            weight_decay=_ADAMW_WD,
            n_epochs=301,
        ),
        # The paper's protocol on the demo architecture, AdamW 1e-4: 9.08%.
        RunConfig("paper_decoder", _PAPER_DECODER, weight_decay=_ADAMW_WD),
        # As paper_decoder at an 80^3 latent grid: 8.955%, at 1.74x the cost.
        RunConfig(
            "paper_decoder_r80",
            _PAPER_DECODER,
            sdf_query_resolution=80,
            weight_decay=_ADAMW_WD,
        ),
        # Reference architecture, AdamW 1e-4: 8.340%.
        RunConfig(
            "paper_decoder_refarch",
            {**_PAPER_DECODER, **_REFERENCE_ARCH},
            weight_decay=_ADAMW_WD,
        ),
        # As paper_decoder_refarch at half the paper's radius: 8.375%.
        RunConfig(
            "paper_decoder_r0275",
            {**_PAPER_DECODER, **_REFERENCE_ARCH, "gno_radius": 0.0275},
            weight_decay=_ADAMW_WD,
        ),
        # Transolver at GINO's parameter budget, on "paper"'s protocol (Adam,
        # lr 2.5e-4 halved at epoch 50, 100 epochs). It reads the surface
        # mesh directly (vertex position + normal), not the latent SDF, so
        # the SDF grid is only built at a token 8^3.
        RunConfig(
            "transolver",
            _TRANSOLVER_GINO_MATCHED,
            sdf_query_resolution=8,
            architecture="transolver",
        ),
        # As "transolver", on Transolver's own recipe (_TRANSOLVER_ONECYCLE).
        RunConfig(
            "transolver_onecycle",
            _TRANSOLVER_GINO_MATCHED,
            sdf_query_resolution=8,
            learning_rate=1e-3,
            architecture="transolver",
            schedule=_TRANSOLVER_ONECYCLE,
        ),
        # As "transolver", with only the optimiser retuned (same parameters,
        # same 100 epochs): Adam, peak lr 7.5e-5, a 1-epoch linear warm-up
        # then cosine decay to 0, no gradient clipping. Chosen by 20-epoch
        # screens on the full split (doc/benchmarks/car_cfd/results/comparative.md): lower
        # peak lr was better down to ~5e-5-1e-4, and clipping at norm 1.0 or
        # 0.1 only slowed training. 6.80% test, below the paper's 7.12%.
        RunConfig(
            "transolver_tuned",
            _TRANSOLVER_GINO_MATCHED,
            sdf_query_resolution=8,
            learning_rate=7.5e-5,
            architecture="transolver",
            schedule=dict(kind="warmup_cosine", warmup_epochs=1, end_factor=0.0),
        ),
        # As "transolver_tuned", with the peak lr chosen without the test set:
        # 20-epoch screens trained on the first 450 training cars and scored
        # on the held-out last 50 (train.py --val 50). Validation error:
        # 1e-4 9.19%, 7.5e-5 9.25%, 5e-5 9.50%, 2.5e-5 10.04%, and 7.5e-5
        # with clip 1.0 11.76%. Trained on all 500 for seeds 0, 1, 2.
        RunConfig(
            "transolver_val",
            _TRANSOLVER_GINO_MATCHED,
            sdf_query_resolution=8,
            learning_rate=1e-4,
            architecture="transolver",
            schedule=dict(kind="warmup_cosine", warmup_epochs=1, end_factor=0.0),
        ),
        # "paper" with the paper's radius 0.055 actually applied to the output
        # GNO (see _PAPER_RADIUS); "paper" itself ran at 0.033.
        RunConfig("paper_r055", {**_PAPER_DECODER, **_REFERENCE_ARCH, **_PAPER_RADIUS}),
        # GIWO at GINO's parameter budget (_GIWO_GINO_MATCHED), on "paper"'s
        # shell and protocol.
        RunConfig("giwo", _GIWO_GINO_MATCHED, architecture="giwo"),
        # As "giwo", with only the optimiser retuned on validation (train.py
        # --val 50, 20-epoch screens; same parameters, same 100 epochs): Adam,
        # a 1-epoch linear warm-up then cosine decay to 0. Validation error by
        # peak lr: 2.5e-4 19.77%, 5e-4 18.04%, 1e-3 17.20%, 2e-3 15.90%, 4e-3
        # 14.53%, 8e-3 13.92%, 1.6e-2 diverged; "giwo"'s StepLR at 2.5e-4
        # 19.11%.
        RunConfig(
            "giwo_val",
            _GIWO_GINO_MATCHED,
            learning_rate=8e-3,
            architecture="giwo",
            schedule=dict(kind="warmup_cosine", warmup_epochs=1, end_factor=0.0),
        ),
    )
}


def build_config(run: RunConfig, n_train: int, n_test: int):
    """The zencfg config ``scirex.get_model_jax`` and the dataset take, for
    ``run``: ``config/gino_carcfd_config.Default`` with ``run``'s overrides.

    The data/optimiser sections apply to every architecture; the ``model``
    overrides are applied only for GINO and GIWO runs (GIWO's ``wno_*``
    fields are left to ``_models.build_module``).
    """
    from copy import deepcopy

    from scirex.config.gino_carcfd_config import Default
    from scirex.models import GINO

    # Default()'s sections are shared class-level instances: without the
    # copy, one run's overrides would carry into every later build_config
    # call in the same process.
    cfg = deepcopy(Default())
    cfg.data.n_train = n_train
    cfg.data.n_test = n_test
    cfg.data.sdf_query_resolution = run.sdf_query_resolution
    cfg.opt.n_epochs = run.n_epochs
    cfg.opt.learning_rate = run.learning_rate
    cfg.opt.step_size = run.lr_step
    cfg.opt.gamma = run.lr_gamma
    cfg.opt.weight_decay = run.weight_decay
    if run.architecture not in ("gino", "giwo"):
        return cfg.to_dict()
    gino_fields = GINO.__dataclass_fields__
    for key, value in run.model.items():
        if run.architecture == "giwo" and key.startswith("wno_"):
            continue
        # A GINO field GINOSmall3d does not declare (e.g. out_gno_radius) is
        # still a valid override: get_model_jax passes the config through.
        if not hasattr(cfg.model, key) and key not in gino_fields:
            raise ValueError(f"run {run.name!r}: GINOSmall3d has no field {key!r}")
        setattr(cfg.model, key, value)
    # The latent grid cannot carry more Fourier modes than it has points.
    if run.sdf_query_resolution < cfg.model.fno_n_modes[0]:
        cfg.model.fno_n_modes = [run.sdf_query_resolution] * 3
    return cfg.to_dict()


def lr_at(run: RunConfig, epoch: int) -> float:
    """StepLR: ``learning_rate * lr_gamma ** (epoch // lr_step)``."""
    return run.learning_rate * run.lr_gamma ** (epoch // run.lr_step)


def lr_schedule(run: RunConfig, steps_per_epoch: int):
    """``run``'s per-step schedule over ``n_epochs`` epochs, peaking at
    ``learning_rate``.

    ``onecycle``: PyTorch ``OneCycleLR`` (cosine annealing), with its
    ``pct_start``/``div_factor``/``final_div_factor``. ``warmup_cosine``: a
    linear warm-up from 0 over ``warmup_epochs``, then cosine decay to
    ``learning_rate * end_factor``.
    """
    import optax

    total = steps_per_epoch * run.n_epochs
    params = dict(run.schedule)
    kind = params.pop("kind")
    if kind == "onecycle":
        return optax.cosine_onecycle_schedule(
            transition_steps=total, peak_value=run.learning_rate, **params
        )
    if kind == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=run.learning_rate,
            warmup_steps=int(params["warmup_epochs"] * steps_per_epoch),
            decay_steps=total,
            end_value=run.learning_rate * params.get("end_factor", 0.0),
        )
    raise ValueError(f"run {run.name!r}: unknown schedule kind {kind!r}")


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint."""
    return RunConfig(**raw)
