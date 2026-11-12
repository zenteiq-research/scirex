# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the chip cooling benchmark, in one place.

* :data:`PROTOCOL` is what every run shares: the data split, the seed and
  the epoch budget.
* :data:`RUNS` are the named runs reported under ``doc/benchmarks/``, each
  with its architecture (``arch``) and optimiser settings (``opt``).
  ``train.py --run <name>`` trains one of them and ``infer.py`` evaluates
  whichever have a checkpoint.

The four runs keep the recipes they were published with, which differ
between models (optimiser, learning-rate schedule, batch size); see the
training functions in ``train.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scirex.data.datasets.chipcooling import (
    DEFAULT_DATASET_DIR,
    GRID_H,
    GRID_W,
    TEST_SIZE,
    TRAIN_SIZE,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/chip_cooling"
# Each run writes to <out_dir>/chip_cooling_<run name>/.
DEFAULT_OUT_DIR = REPO_ROOT / "doc/assets/benchmarks"
# Directory holding model_setup.json and the sim_*.npz files
# (CHIPCOOLING_DATA_DIR, or <repo_root>/1_Dataset).
DEFAULT_DATA_DIR = Path(DEFAULT_DATASET_DIR)
# The 128x256 grid FNO and WNO train on, interpolated once from the mesh.
DEFAULT_CACHE_DIR = Path(
    os.environ.get("CHIPCOOLING_CACHE_DIR", "Dataset_cache/chip_cooling")
)

PROTOCOL: dict = dict(
    ntrain=TRAIN_SIZE,  # first 1600 samples
    ntest=TEST_SIZE,  # last 400 samples
    grid=(GRID_H, GRID_W),
    epochs=500,
    seed=42,
)

# Test sample shown in every model's contour.png.
CONTOUR_SAMPLE = 99


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model, its size and its optimiser."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    opt: dict = field(default_factory=dict)
    epochs: int = PROTOCOL["epochs"]


RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # 1,829,496 parameters. AdamW with correct complex-weight updates
        # (adamw_cx), learning rate on a per-epoch cosine from 1e-3 to 0.
        RunConfig(
            "fno",
            "fno",
            arch=dict(n_modes=(24, 24), hidden_channels=27),
            opt=dict(
                optimizer="adamw_cx",
                lr=1e-3,
                weight_decay=1e-4,
                schedule="per-epoch cosine to 0",
                batch_size=8,
            ),
        ),
        # 1,920,481 parameters. Adam with weight decay added to the gradients,
        # learning rate on a per-step cosine from 1e-3 to 0; bfloat16 compute.
        RunConfig(
            "wno",
            "wno",
            arch=dict(
                hidden_channels=16,
                n_layers=4,
                level=4,
                wavelet="db6",
                padding=1,
                compute_dtype="bfloat16",
                param_dtype="float32",
                projection_hidden_channels=128,
                grid_range=(1.0, 1.0),
            ),
            opt=dict(
                optimizer="adam + add_decayed_weights",
                lr=1e-3,
                weight_decay=1e-4,
                schedule="per-step cosine to 0",
                batch_size=8,
            ),
        ),
        # 1,821,028 parameters. One sample per step; AdamW with correct
        # complex-weight updates (adamw_cx), learning rate on a per-epoch
        # cosine from 1e-3 to 0, stepped after each epoch.
        RunConfig(
            "gino",
            "gino",
            arch=dict(
                latent_h=64,
                latent_w=128,
                latent_feature_channels=6,
                gno_coord_dim=2,
                gno_coord_embed_dim=16,
                in_gno_radius=0.05,
                out_gno_radius=0.05,
                in_gno_transform_type="linear",
                out_gno_transform_type="linear",
                in_gno_pos_embed_type=None,
                out_gno_pos_embed_type="transformer",
                gno_embed_channels=16,
                gno_embed_max_positions=600,
                in_gno_channel_mlp_hidden_layers=(80, 80, 80),
                out_gno_channel_mlp_hidden_layers=(512, 256),
                fno_n_modes=(20, 20),
                fno_hidden_channels=30,
                fno_lifting_channel_ratio=4,
                fno_n_layers=4,
                fno_use_channel_mlp=True,
                fno_channel_mlp_expansion=0.5,
                fno_norm="group_norm",
                fno_ada_in_features=8,
                fno_factorization=None,
                fno_rank=0.8,
                fno_domain_padding=0.0,
                projection_channel_ratio=4,
            ),
            opt=dict(
                optimizer="adamw_cx",
                lr=1e-3,
                weight_decay=1e-4,
                schedule="per-epoch cosine to 0, stepped after each epoch",
                batch_size=1,
            ),
        ),
        # 1,807,561 parameters, the size the other three are matched to. Adam
        # with weight decay added to the gradients and gradient clipping 1.0,
        # learning rate warmed up linearly over 50 epochs to 1e-4, then a
        # cosine down to 1e-6.
        RunConfig(
            "transolver",
            "transolver",
            arch=dict(
                n_layers=5,
                n_hidden=256,
                n_head=8,
                mlp_ratio=1,
                dropout_rate=0.0,
                slice_num=32,
                ref=8,
                space_dim=2,
                fun_dim=6,
                out_dim=1,
                time_input=False,
                unified_pos=False,
                geometry="irregular",
            ),
            opt=dict(
                optimizer="adam + add_decayed_weights + clip_by_global_norm",
                lr=1e-4,
                weight_decay=1e-6,
                grad_clip=1.0,
                warmup_epochs=50,
                lr_floor=1e-6,
                schedule="linear warmup, then per-step cosine",
                batch_size=4,
            ),
        ),
    )
}

MODEL_LABELS: dict[str, str] = {
    "fno": "FNO",
    "wno": "WNO",
    "gino": "GINO",
    "transolver": "Transolver",
}


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint."""

    def untuple(d: dict) -> dict:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}

    return RunConfig(
        name=raw["name"],
        model=raw["model"],
        arch=untuple(raw["arch"]),
        opt=untuple(raw["opt"]),
        epochs=raw["epochs"],
    )
