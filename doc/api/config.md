<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.config

Configuration classes and named presets for the models.

```python
from scirex.config import FNOSmall2d
```

A preset is selected by class name and is not case-sensitive, so `FNOSmall2d` and
`fnosmall2d` are the same preset. This package is imported eagerly because each class
registers itself when it is defined.

## Base classes

::: scirex.config.ModelConfig

::: scirex.config.OptimizationConfig

## FNO

::: scirex.config.FNOConfig

::: scirex.config.SimpleFNOConfig

::: scirex.config.FNOSmall2d

::: scirex.config.FNOMedium2d

::: scirex.config.FNOLarge2d

::: scirex.config.FNOHuge2d

::: scirex.config.FNOMedium3d

## GINO

::: scirex.config.GINOConfig

::: scirex.config.GINOSmall3d

::: scirex.config.GINOPoisson2d

::: scirex.config.GINODarcy2d

## Transolver

::: scirex.config.TransolverConfig

::: scirex.config.TransolverDarcy2dAligned

::: scirex.config.TransolverLaplace2d

::: scirex.config.TransolverKappaOperator2d

::: scirex.config.TransolverAirfRANS

::: scirex.config.TransolverShapeNetCar

::: scirex.config.TransolverPipe

::: scirex.config.TransolverPipeOfficial

## Patching

::: scirex.config.PatchingConfig
