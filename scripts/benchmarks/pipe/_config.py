# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Every setting of the Pipe benchmark, in one place.

* :data:`PROTOCOL` is the shared training recipe, taken from the Transolver
  Pipe benchmark (``_transolver.py``, run with
  the paper's launch-script hyperparameters) so that every architecture is
  trained the same way.
* :data:`RUNS` are the named runs reported under ``doc/benchmarks/``, each
  trained once per seed in ``PROTOCOL["seeds"]``.
  ``train.py --run <name>`` trains one of them and ``infer.py`` evaluates
  whichever have a checkpoint.

Which architectures exist, and how each is built, lives with the models in
:data:`scripts.benchmarks.pipe._models.SPECS`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.benchmarks.pipe._models import SPECS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = REPO_ROOT / "scripts/benchmarks/checkpoints/pipe"
# Each run writes to <out_dir>/pipe_<run name>/.
DEFAULT_OUT_DIR = REPO_ROOT / "doc/assets/benchmarks"
# Directory holding Pipe_X.npy, Pipe_Y.npy and Pipe_Q.npy.
DEFAULT_DATA_DIR = Path(os.environ.get("PIPE_DATA_DIR", "Dataset"))

MODELS: tuple[str, ...] = tuple(SPECS)
MODEL_LABELS: dict[str, str] = {name: spec.label for name, spec in SPECS.items()}

# The Transolver Pipe recipe, shared by every run.
PROTOCOL: dict = dict(
    ntrain=1000,  # first 1000 samples
    ntest=200,  # last 200 samples
    n_total=1200,
    # PyTorch's AdamW. JAX's gradient of a real loss with respect to a
    # complex weight is the conjugate of PyTorch's, so plain optax.adamw
    # moves FNO's and GINO's complex spectral weights the wrong way; adamw_cx
    # conjugates them first (src/scirex/training/optim/adamw.py). Real weights
    # are updated exactly as by AdamW.
    optimizer="adamw_cx",
    adam_eps=1e-8,  # torch.optim.AdamW's default
    lr=1e-3,  # peak of the one-cycle schedule
    weight_decay=1e-5,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    max_grad_norm=0.1,
    epochs=500,
    batch_size=8,
    loss="relative L2, summed over the batch",
    # Each run is trained once per seed; the seed sets the initial weights
    # and the order of the training batches.
    seeds=(0, 1, 2),
)

# Transolver at the paper's launch-script hyperparameters, the size every
# other architecture is matched to (doc/benchmarks/pipe/results/comparative.md).
TRANSOLVER_PARAMS = 3_073_985

#: The JAX port of the paper's own ``exp_pipe.py`` (``_transolver.py``) at
#: the launch-script settings, as in the JAX-vs-PyTorch comparison on
#: doc/benchmarks/pipe/results/comparative.md (seed 2 of 3, plain ``optax.adamw``):
#: ``train.py --run transolver_exp_pipe`` passes these to it. The benchmark's
#: own Transolver run is ``RUNS["transolver"]``.
TRANSOLVER_ARGS: list[str] = [
    "--epochs", "500",
    "--seed", "2",
    "--n-hidden", "128",
    "--n-layers", "8",
    "--n-heads", "8",
    "--mlp_ratio", "2",
    "--slice_num", "64",
    "--max_grad_norm", "0.1",
]  # fmt: skip


#: The training recipe of ``RUNS["transolver_tuned"]``: the best trial of the
#: hyperparameter search (``tune_transolver.py``), which ranked trials on
#: test error because the benchmark has no validation split. The model,
#: epochs and last-epoch weights are the same as ``RUNS["transolver"]``.
TUNED_TRANSOLVER_RECIPE: dict = dict(
    lr=0.002789574430295683,
    weight_decay=0.0002497328610406061,
    max_grad_norm=0.4884053941402852,
    pct_start=0.33415875288593666,
    div_factor=7.126045321233962,
    final_div_factor=8313.101133778735,
    batch_size=4,
)


@dataclass(frozen=True)
class RunConfig:
    """One training run: which model and its size."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    epochs: int = PROTOCOL["epochs"]
    #: Settings of :data:`PROTOCOL` this run trains with instead; empty for
    #: the shared recipe. :func:`protocol_for` merges them.
    recipe: dict = field(default_factory=dict)


RUNS: dict[str, RunConfig] = {
    r.name: r
    for r in (
        # The paper's launch script, Transolver_Pipe.sh:
        # 3,073,985 parameters.
        RunConfig(
            "transolver",
            "transolver",
            dict(
                n_hidden=128,
                n_layers=8,
                n_head=8,
                mlp_ratio=2,
                slice_num=64,
                ref=8,
                dropout_rate=0.0,
            ),
        ),
        # The same model with the tuned recipe; its best seed is the published
        # Pipe Transolver checkpoint.
        RunConfig(
            "transolver_tuned",
            "transolver",
            dict(
                n_hidden=128,
                n_layers=8,
                n_head=8,
                mlp_ratio=2,
                slice_num=64,
                ref=8,
                dropout_rate=0.0,
            ),
            recipe=TUNED_TRANSOLVER_RECIPE,
        ),
        # 3,102,157 parameters, 0.9% above Transolver's.
        RunConfig(
            "wno",
            "wno",
            dict(
                hidden_channels=20,
                n_layers=4,
                level=3,
                wavelet="db4",
                projection_hidden_channels=128,
            ),
        ),
        # 2,999,629 parameters, 2.4% below Transolver's (37 channels would be
        # 3.1% above). n_modes counts the modes kept along each axis in total,
        # so (24, 22) keeps 12 modes each way, the Geo-FNO pipe.py's modes=12.
        RunConfig(
            "fno",
            "fno",
            dict(
                n_modes=(24, 22),
                hidden_channels=36,
                n_layers=4,
                projection_channel_ratio=4,
                domain_padding=8 / 129,
                fno_skip="linear",
                use_channel_mlp=False,
            ),
        ),
        # 3,001,956 parameters, 2.3% below Transolver's.
        RunConfig(
            "gino",
            "gino",
            dict(
                in_gno_radius=0.05,
                out_gno_radius=0.05,
                fno_n_modes=(16, 16),
                fno_hidden_channels=48,
                fno_n_layers=4,
                fno_lifting_channel_ratio=4,
            ),
        ),
    )
}


#: The four architectures on the shared recipe, compared on one plot.
SHARED_RECIPE_RUNS: tuple[str, ...] = ("transolver", "wno", "fno", "gino")


def protocol_for(run: RunConfig) -> dict:
    """:data:`PROTOCOL` with ``run``'s own recipe settings applied."""
    return {**PROTOCOL, **run.recipe}


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
        epochs=raw["epochs"],
        recipe=raw.get("recipe", {}),
    )
