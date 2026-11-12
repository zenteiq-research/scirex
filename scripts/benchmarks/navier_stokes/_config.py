# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the Navier-Stokes benchmark, in one place.

* :data:`PROTOCOL` is the shared training recipe, the Transolver paper's own
  for this benchmark (``exp_ns.py`` with ``Transolver_NS.sh``), so
  that every architecture is trained the same way.
* :data:`RUNS` are the named runs reported under ``doc/benchmarks/``.
  ``train.py --run <name>`` trains one of them and ``infer.py`` evaluates
  whichever have a checkpoint.

Which architectures exist, and how each is built, lives with the models in
:data:`scripts.benchmarks.navier_stokes._models.SPECS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.benchmarks.navier_stokes._models import SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/navier_stokes"
# Each run writes to <out_dir>/navier_stokes_<run name>/.
DEFAULT_OUT_DIR = REPO_ROOT / "doc/assets/benchmarks"
# Directory holding the data file below.
DEFAULT_DATA_DIR = Path(os.environ.get("NS_DATA_DIR", "Dataset"))
DATA_FILE = "NavierStokes_V1e-5_N1200_T20.mat"

MODELS: tuple[str, ...] = tuple(SPECS)
# RunConfig.optimizer values; train.OPTIMIZERS maps each to its optax builder.
OPTIMIZERS: tuple[str, ...] = ("adamw", "adamw_cx")
MODEL_LABELS: dict[str, str] = {name: spec.label for name, spec in SPECS.items()}

# The Transolver paper's recipe (exp_ns.py + Transolver_NS.sh), shared by
# every run.
PROTOCOL: dict = dict(
    ntrain=1000,  # first 1000 samples
    ntest=200,  # last 200 samples
    t_in=10,  # frames given to the model
    t_out=10,  # frames predicted, one at a time
    adam_eps=1e-8,  # PyTorch's and optax.adamw's default (adamw_cx's is 1e-6)
    lr=1e-3,  # peak of the one-cycle schedule
    weight_decay=1e-5,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    max_grad_norm=None,
    epochs=500,
    batch_size=2,
    loss="relative L2 per sample and step, summed over the batch and the "
    "teacher-forced steps",
    seed=0,
)

# Transolver at the paper's size, the size every other architecture is
# matched to (doc/benchmarks/navier_stokes/results/comparative.md).
TRANSOLVER_PARAMS = 11_232_321

# Transolver's test relative L2 on this benchmark, as reported in the
# Transolver paper (Wu et al. 2024, ICML).
PAPER_REL_L2 = 0.0900


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model, its size, and its optimiser option.

    ``cycle_momentum`` cycles AdamW's beta1 from 0.95 down to 0.85 at the
    learning-rate peak and back, as PyTorch's ``OneCycleLR`` does by default
    (the paper's optimiser); otherwise beta1 stays at optax's 0.9.

    ``optimizer`` is ``"adamw"`` (``optax.adamw``) or ``"adamw_cx"``
    (``scirex.training.adamw_cx``), the complex-aware AdamW that
    every model with complex spectral weights needs: ``jax.grad`` returns the
    conjugate of the descent direction for a complex parameter, so plain
    ``optax.adamw`` moves its imaginary part the wrong way. ``adamw_cx``
    also places eps as ``sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps)``,
    an effective eps ``1 / sqrt(1 - b2^t)`` times optax's: 1.26x at step
    1,000, under 1.01x from step 5,000 on (epoch 10 of 500).
    """

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    cycle_momentum: bool = False
    optimizer: str = "adamw"
    epochs: int = PROTOCOL["epochs"]

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"{self.name}: optimizer {self.optimizer!r} is not one of {OPTIMIZERS}"
            )


RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # 11,232,321 parameters, the paper's Transolver_NS.sh, with the
        # PyTorch reference's numerics: exact GELU, LayerNorm eps 1e-5,
        # torch.nn.Conv2d init in the slice projections, and beta1 cycling.
        RunConfig(
            "transolver",
            "transolver",
            dict(
                n_layers=8,
                n_hidden=256,
                n_head=8,
                mlp_ratio=1,
                slice_num=32,
                ref=8,
                unified_pos=True,
                dropout_rate=0.0,
                gelu="exact",
                layer_norm_epsilon=1e-5,
                conv_init="torch",
            ),
            cycle_momentum=True,
        ),
        # 11,183,467 parameters, 0.4% below Transolver's.
        RunConfig(
            "fno",
            "fno",
            dict(hidden_channels=98, n_modes=(16, 16), n_layers=4),
            optimizer="adamw_cx",
        ),
        # 11,312,957 parameters, 0.7% above Transolver's.
        RunConfig(
            "wno",
            "wno",
            dict(hidden_channels=60, n_layers=4, level=3, wavelet="db4"),
        ),
        # 11,297,041 parameters, 0.6% above Transolver's.
        RunConfig(
            "gino",
            "gino",
            dict(
                in_gno_radius=0.033,
                out_gno_radius=0.033,
                fno_n_modes=(16, 16),
                fno_hidden_channels=97,
                fno_n_layers=4,
            ),
            optimizer="adamw_cx",
        ),
    )
}


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Rebuilds a :class:`RunConfig` from its JSON form in a checkpoint."""
    arch = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw["arch"].items()
    }
    return RunConfig(
        name=raw["name"],
        model=raw["model"],
        arch=arch,
        cycle_momentum=raw.get("cycle_momentum", False),
        optimizer=raw.get("optimizer", "adamw"),
        epochs=raw["epochs"],
    )
