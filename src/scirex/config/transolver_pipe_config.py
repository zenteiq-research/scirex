# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Experiment configuration for the Transolver paper's own 2D Navier-Stokes
Pipe-flow benchmark
(https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/exp_pipe.py),
faithfully ported by `scripts/benchmarks/pipe/_transolver.py`.

Every field below is `exp_pipe.py`'s own `argparse` default -- see that
script's module docstring for the exact line-by-line correspondence and for
the one data-availability caveat (`Pipe_Q.npy` truncated in this repo's
copy of `Dataset/`, `Pipe_Q_interp_x.npy` used as its channel-0
equivalent instead).
"""

from zencfg import ConfigBase

from .models import ModelConfig, TransolverPipe
from .opt import OptimizationConfig


class TransolverPipeOptConfig(OptimizationConfig):
    """Matches `exp_pipe.py`'s `--lr`, `--epochs`, `--weight_decay`,
    `--batch-size` and its unoverridden ``torch.optim.lr_scheduler.
    OneCycleLR`` defaults (`pct_start=0.3`, `div_factor=25`,
    `final_div_factor=1e4`, `anneal_strategy='cos'`) -- reproduced here
    via `optax.cosine_onecycle_schedule` in the training script. Optimizer
    is `torch.optim.AdamW` (NOT plain Adam, unlike this repo's ShapeNetCar/
    AirfRANS Transolver recipes) -- `weight_decay` is genuinely used.
    """

    n_epochs: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 10_000.0
    # exp_pipe.py's --max_grad_norm default: no clipping.
    max_grad_norm: float | None = None
    # Checkpoint every ckpt_iter epochs (exp_pipe.py: `if ep % 100 == 0`),
    # plus always at the final epoch.
    ckpt_iter: int = 100


class TransolverPipeDatasetConfig(ConfigBase):
    """Mirrors `exp_pipe.py`'s hardcoded `ntrain=1000`, `ntest=200`,
    `N=1200` and its `--downsamplex`/`--downsampley` (both default 1,
    i.e. no downsampling of the raw 129x129 mesh -> `s1=s2=129`).
    """

    data_dir: str = "~/Zenteiq/Pipe/Dataset"
    ntrain: int = 1000
    ntest: int = 200
    n_total: int = 1200
    downsamplex: int = 1
    downsampley: int = 1


class Default(ConfigBase):
    """Top-level experiment configuration: model, optimization and dataset."""

    verbose: bool = True
    arch: str = "transolver"
    model: ModelConfig = TransolverPipe()
    opt: TransolverPipeOptConfig = TransolverPipeOptConfig()
    data: TransolverPipeDatasetConfig = TransolverPipeDatasetConfig()
