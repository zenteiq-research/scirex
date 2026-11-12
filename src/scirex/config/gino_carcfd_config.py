# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Experiment configuration for the Car-CFD benchmark using a GINO model."""

from typing import Any

from zencfg import ConfigBase

from scirex.experimental.distributed.config import DistributedConfig

from .models import GINOSmall3d, ModelConfig
from .opt import PatchingConfig


class CarCFDDatasetConfig(ConfigBase):
    """Dataset location, size and SDF query resolution of the Car-CFD benchmark."""

    root: str = "~/data/car-pressure-data/processed-car-pressure-data"
    sdf_query_resolution: int = 32
    n_train: int = 500  # 500
    n_test: int = 111  # 111
    download: bool = True


class CarCFDOptConfig(ConfigBase):
    """Optimizer and schedule settings of the Car-CFD GINO experiment."""

    n_epochs: int = 10  # 301
    learning_rate: float = 1e-3
    training_loss: str = "l2"
    testing_loss: str = "l2"
    weight_decay: float = 1e-4
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.5
    scheduler_patience: int = 10
    scheduler_t_max: int = 10


class Default(ConfigBase):
    """Top-level experiment configuration: GINO model, optimization, data, patching."""

    n_params_baseline: Any | None = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = GINOSmall3d()
    opt: ConfigBase = CarCFDOptConfig()
    data: CarCFDDatasetConfig = CarCFDDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
