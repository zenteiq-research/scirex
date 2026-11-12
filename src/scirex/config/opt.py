# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared optimization and multi-grid patching configuration objects."""

from typing import Literal

from zencfg import ConfigBase


class OptimizationConfig(ConfigBase):
    """Optimizer, learning-rate schedule and loss settings of a training run.

    The configuration only stores the values; the training script decides which
    of them it uses. The scheduler keys follow the usual meaning of the
    scheduler they are named after.

    Attributes
    ----------
    n_epochs : int
        Number of training epochs (required).
    training_loss : str or list of str
        `"h1"` or `"l2"`; a list of loss-term names for the physics-informed
        subclasses.
    testing_loss : str
        Loss used for evaluation.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        Weight decay of the optimizer.
    eval_interval : int
        Evaluate every this many epochs.
    scheduler : str
        Learning-rate schedule: `"StepLR"`, `"ReduceLROnPlateau"`,
        `"CosineAnnealingLR"` or `"OneCycleLR"`.
    scheduler_t_max : int
        Period of the cosine schedule (`CosineAnnealingLR`).
    scheduler_patience : int
        Epochs without improvement before the learning rate is reduced
        (`ReduceLROnPlateau`).
    step_size : int
        Epochs between learning-rate decays (`StepLR`).
    gamma : float
        Learning-rate decay factor (`StepLR` and `ReduceLROnPlateau`).
    max_grad_norm : float or None
        Global gradient-norm clipping threshold; `None` disables clipping.
    save_dir : str or None
        If set, the training script writes a checkpoint there after training;
        `None` writes none.
    """

    n_epochs: int
    # str for the standard "h1"/"l2" loss selection; list[str] for the
    # PINO subclasses, whose training_loss names a set of physics-loss terms.
    training_loss: Literal["h1", "l2"] | list[str] = "h1"
    testing_loss: str = "l2"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    eval_interval: int = 1
    scheduler: Literal[
        "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"
    ] = "StepLR"
    scheduler_t_max: int = 500
    scheduler_patience: int = 50
    step_size: int = 100
    gamma: float = 0.5
    max_grad_norm: float | None = None
    # Opt-in: if set, the training script saves a checkpoint (weights +
    # architecture hyperparameters) here after training completes. None
    # (default) means no checkpoint is written -- matches the existing
    # scripts/benchmarks/ convention of an explicit, gitignored checkpoint
    # path rather than always writing to disk.
    save_dir: str | None = None


class PatchingConfig(ConfigBase):
    """Settings for multigrid patching of the input domain.

    Attributes
    ----------
    levels : int
        Number of patching levels; 0 disables patching.
    padding : int
        Padding added around each patch.
    stitching : bool
        Whether the patch outputs are stitched back together.
    """

    levels: int = 0
    padding: int = 0
    stitching: bool = False
