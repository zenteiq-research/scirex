# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the Darcy benchmark, in one place.

* :data:`PROTOCOL` is the data split shared by every run.
* :data:`RUNS` are the published runs, each with its own architecture
  (``arch``) and training recipe (``opt``). ``train.py --run <name>`` trains
  one of them and ``infer.py`` evaluates whichever have a checkpoint.

Each recipe follows its own paper or reference code:

* FNO: Li et al. (ICLR 2021) -- Adam, lr 1e-3 halved every 100 epochs, no
  weight decay, batch 20, 500 epochs, per-pixel normalisation. The
  complex-aware ``adamw_cx`` is used for FNO's complex spectral weights.
* WNO: Tripura & Chakraborty (2022) and their reference code -- db6, level 4,
  padding 1, AdamW with weight decay 1e-6, lr halved every 50 epochs,
  batch 20, 500 epochs, per-pixel normalisation, float32 compute.
* GINO: AdamW (``adamw_cx``) with weight decay 1e-4, cosine one-cycle lr,
  gradient clipping 1.0, batch 16, 2000 epochs, per-pixel normalisation, and
  Transolver's ``rel. L2 + 0.1 x gradient`` loss. The GNO radii follow
  ``0.16 * 16 / (s - 1)`` and ``0.175 * 16 / (s - 1)``.
* Transolver: Wu et al. (ICML 2024) Table 8 and ``exp_darcy.py`` -- AdamW
  with weight decay 1e-5, cosine one-cycle lr, gradient clipping 0.1,
  batch 4, 500 epochs, ``rel. L2 + 0.1 x gradient`` loss, one global
  normalisation scalar per field, seed 2 (the best of five seeds).

FNO, WNO and GINO are resized to Transolver's parameter count; see each
model's page under ``doc/benchmarks/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/darcy"
DEFAULT_OUT_DIR = REPO_ROOT / "doc/assets/benchmarks"
DEFAULT_DATA_DIR = Path(
    os.environ.get("DARCY_DATA_DIR", "src/scirex/data/darcy_wno_paper_cache")
)
DEFAULT_ANIM_OUT = REPO_ROOT / "animations/darcy_flow/evolution.gif"

PROTOCOL: dict = dict(
    train_file="piececonst_r421_N1024_smooth1.mat",
    test_file="piececonst_r421_N1024_smooth2.mat",
    n_train=1000,  # first 1000 samples of the train file
    n_test=200,  # first 200 samples of the test file
    subsample_rate=5,  # 421 -> 85, as in Li et al.'s own code
    eval_every=10,  # epochs between test evaluations in the history
)

#: Completed epochs at which ``train.py --animate`` captures a frame: dense
#: early, where the prediction changes fastest.
ANIM_EPOCHS: tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100,
    125, 150, 175, 200, 250, 300, 350, 400, 450, 500,
)  # fmt: skip


@dataclass(frozen=True)
class RunConfig:
    """One training run: the model, its size, its recipe and its seed."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    opt: dict = field(default_factory=dict)
    seed: int = 0


RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        RunConfig(
            "fno",
            "fno",
            arch=dict(modes=20, hidden_channels=40, n_layers=4),
            opt=dict(
                optimizer="adamw_cx",
                lr=1e-3,
                weight_decay=0.0,
                schedule="step",
                decay_epochs=100,
                decay_rate=0.5,
                grad_clip=None,
                batch_size=20,
                epochs=500,
                grad_loss_weight=0.0,
            ),
        ),
        RunConfig(
            "wno",
            "wno",
            arch=dict(
                hidden_channels=28, level=4, wavelet="db6", n_layers=4, padding=1
            ),
            opt=dict(
                optimizer="adamw",
                lr=1e-3,
                weight_decay=1e-6,
                schedule="step",
                decay_epochs=50,
                decay_rate=0.5,
                grad_clip=None,
                batch_size=20,
                epochs=500,
                grad_loss_weight=0.0,
            ),
        ),
        RunConfig(
            "gino",
            "gino",
            arch=dict(hidden_channels=31, modes=16, n_layers=5),
            opt=dict(
                optimizer="adamw_cx",
                lr=1e-3,
                weight_decay=1e-4,
                schedule="onecycle",
                grad_clip=1.0,
                batch_size=16,
                epochs=2000,
                grad_loss_weight=0.1,
            ),
        ),
        RunConfig(
            "transolver",
            "transolver",
            arch=dict(
                n_hidden=128,
                n_layers=8,
                n_head=8,
                slice_num=64,
                ref=8,
                mlp_ratio=1,
                kernel=3,
            ),
            opt=dict(
                optimizer="adamw",
                lr=1e-3,
                weight_decay=1e-5,
                schedule="onecycle",
                grad_clip=0.1,
                batch_size=4,
                epochs=500,
                grad_loss_weight=0.1,
            ),
            seed=2,
        ),
    )
}

MODEL_LABELS = {"fno": "FNO", "wno": "WNO", "gino": "GINO", "transolver": "Transolver"}

#: Published reference numbers (test relative L2), where a paper reports one.
PAPER = {
    "transolver": dict(
        rel_l2=0.0057,
        reference="Wu et al., Transolver, ICML 2024, Table 2 (Darcy): 0.57%",
    ),
    "wno": dict(
        rel_l2=0.0084,
        reference="Tripura & Chakraborty, WNO, CMAME 2022, Table 3 "
        "(Darcy, rectangular): ~0.84%; not reproducible from the paper's own "
        "released code and data",
    ),
}


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint."""
    return RunConfig(
        name=raw["name"],
        model=raw["model"],
        arch=dict(raw["arch"]),
        opt=dict(raw["opt"]),
        seed=raw["seed"],
    )
