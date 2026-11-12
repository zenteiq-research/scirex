# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the AirfRANS benchmark, in one place.

* :data:`PROTOCOL` is the shared training recipe: the official Transolver
  AirfRANS recipe (``Airfoil-Design-AirfRANS``: ``main.py``, ``train.py``,
  ``params.yaml``), plus the data-driven additions listed under
  ``additions``.
* :data:`RUNS` are the named runs reported under ``doc/benchmarks/``.
  ``train.py --run <name>`` trains one of them and ``infer.py`` evaluates
  whichever have a checkpoint.
* :data:`PAPER_REFERENCE` is the Transolver row of the paper's Table 3.

Which architectures exist, and how each is built, lives with the models in
:data:`scripts.benchmarks.airfrans._models.SPECS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.benchmarks.airfrans._models import SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/airfrans"
# Each run writes to <out_dir>/airfrans_<run name>/.
DEFAULT_OUT_DIR = REPO_ROOT / "doc/assets/benchmarks"
# The unzipped AirfRANS dataset: manifest.json and one folder per case.
DEFAULT_DATA_DIR = Path(os.environ.get("AIRFRANS_DATA_DIR", "Dataset"))
# Per-case npz cache of the node features (built on first use).
DEFAULT_CACHE_DIR = Path(
    os.environ.get("AIRFRANS_CACHE_DIR", "Dataset_cache/airfrans_benchmark")
)

MODELS: tuple[str, ...] = tuple(SPECS)
MODEL_LABELS: dict[str, str] = {name: spec.label for name, spec in SPECS.items()}

PROTOCOL: dict = dict(
    # Split: manifest["full_train"] minus its last 80 cases / those 80 /
    # manifest["full_test"], exactly as the official main.py.
    n_train=720,
    n_val=80,
    n_test=200,
    subsample=32_000,  # random nodes per case per step (params.yaml)
    batch_size=1,
    optimizer="adam",  # torch.optim.Adam(lr), betas (0.9, 0.999), eps 1e-8
    lr=1e-3,  # peak of the one-cycle schedule
    # torch.optim.lr_scheduler.OneCycleLR defaults: two phases, cosine
    # annealing, beta1 cycled 0.95 -> 0.85 -> 0.95 (cycle_momentum=True),
    # total_steps = (n_train // batch_size + 1) * epochs.
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    base_momentum=0.85,
    max_momentum=0.95,
    surf_weight=1.0,  # loss = vol MSE + surf_weight * surf MSE (main.py -w)
    epochs=500,
    seed=0,
    val_every=10,  # validation loss on the 80 cases, logged only
    val_repeats=20,  # random 32k subsets per validation case (train.py)
    additions=dict(
        features="log10(|wall distance| + 1e-6) and log10(Re), Re = U_inf / 1.56e-5",
        targets="(Ux, Uy) / U_inf, p / U_inf^2, nut / U_inf, then normalised",
        ema_decay=0.9995,  # the EMA weights are the reported model
        grad_clip_global_norm=10.0,  # cuts spikes only
        matmul_precision="highest",  # true FP32, as PyTorch; XLA defaults to TF32
        epochs="500 (official: 398)",
    ),
)

# The Transolver paper's own AirfRANS results (Wu et al., ICML 2024,
# arXiv:2402.02366, Table 3, "Transolver" row).
PAPER_REFERENCE = dict(
    volume=0.0037,
    surf=0.0142,
    relerr_cl=0.1030,
    spearman_cl=0.9978,
    n_params_used=3_893_572,  # the official Transolver(...) call in main.py
    source="Wu et al., Transolver, ICML 2024, Table 3",
)


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model and its size."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    epochs: int = PROTOCOL["epochs"]


RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # The official Transolver(n_hidden=256, n_layers=8, n_head=8,
        # mlp_ratio=2, slice_num=32, unified_pos=1) with 9 inputs instead of 7:
        # 3,894,596 parameters, 1,024 (0.03%) more than the paper's model.
        RunConfig(
            "transolver",
            "transolver",
            dict(
                n_hidden=256,
                n_layers=8,
                n_head=8,
                mlp_ratio=2,
                slice_num=32,
                ref=8,
                layer_norm_epsilon=1e-5,
            ),
        ),
        # A 128 x 256 grid over the Transolver reference box, WNO width 28 with
        # 4 wavelet levels, and a 2-layer MLP decoder: 3,964,728 parameters,
        # 1.8% more than Transolver's.
        RunConfig(
            "wno",
            "wno",
            dict(
                H=128,
                W=256,
                hidden=28,
                n_layers=4,
                level=4,
                wavelet="db4",
                latent=64,
                dec_hidden=256,
                dec_layers=3,
            ),
        ),
    )
}


#: The published GINO run (doc/benchmarks/airfrans/results/comparative.md). GINO has its own
#: trainer, ``_gino.py``; ``train.py --run gino`` and ``infer.py`` pass it
#: these settings.
GINO_ARGS: list[str] = [
    "--epochs", "500",
    "--latent-sdf",
    "--decoder-point-features",
    "--out-gno-mlp", "512", "362",
    "--latent-res", "128", "64",
    "--loader-surf-mask",
]  # fmt: skip
#: The test case drawn in GINO's ``contour.png``.
GINO_CONTOUR_CASE = "airFoil2D_SST_59.56_-4.362_1.048_5.664_1.0_18.157"


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint."""
    arch = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw["arch"].items()
    }
    return RunConfig(
        name=raw["name"], model=raw["model"], arch=arch, epochs=raw["epochs"]
    )
