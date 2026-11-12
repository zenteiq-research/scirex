# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Model and optimization configuration classes and presets.

Collects the `zencfg.ConfigBase` configuration objects describing the models,
optimizers, patching and distributed setup used by the training scripts.

Unlike the other ScireX packages, this one is imported eagerly on purpose: every
class registers itself with its `zencfg` base class when it is defined, and a
preset can only be selected by name (for example `--model FNOSmall2d` on a
command line) after the module that defines it has been imported. A lazy export
would not register the preset.

The experiment configurations `scirex.config.gino_carcfd_config` and
`scirex.config.transolver_pipe_config` are imported here so that their classes
are registered too, but they are not part of the public API.
"""

from . import (  # noqa: F401  (registers presets)
    gino_carcfd_config,
    transolver_pipe_config,
)
from .models import (
    FNOConfig,
    FNOHuge2d,
    FNOLarge2d,
    FNOMedium2d,
    FNOMedium3d,
    FNOSmall2d,
    GINOConfig,
    GINODarcy2d,
    GINOPoisson2d,
    GINOSmall3d,
    ModelConfig,
    SimpleFNOConfig,
    TransolverAirfRANS,
    TransolverConfig,
    TransolverDarcy2dAligned,
    TransolverKappaOperator2d,
    TransolverLaplace2d,
    TransolverPipe,
    TransolverPipeOfficial,
    TransolverShapeNetCar,
)
from .opt import OptimizationConfig, PatchingConfig

__all__ = [
    "FNOConfig",
    "FNOHuge2d",
    "FNOLarge2d",
    "FNOMedium2d",
    "FNOMedium3d",
    "FNOSmall2d",
    "GINOConfig",
    "GINODarcy2d",
    "GINOPoisson2d",
    "GINOSmall3d",
    "ModelConfig",
    "OptimizationConfig",
    "PatchingConfig",
    "SimpleFNOConfig",
    "TransolverAirfRANS",
    "TransolverConfig",
    "TransolverDarcy2dAligned",
    "TransolverKappaOperator2d",
    "TransolverLaplace2d",
    "TransolverPipe",
    "TransolverPipeOfficial",
    "TransolverShapeNetCar",
]
