<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.data

Dataset loaders, containers and normalisers.

```python
from scirex.data import load_darcy_flow_small
```

| Group | Names |
| --- | --- |
| Loaders | `load_darcy_flow_small`, `load_darcy_pt`, `load_navier_stokes_pt`, `load_cavity_kepsilon`, `load_airfrans`, `load_shapenet_car` |
| Containers | `TensorDataset`, `DictDataset` |
| Processing | `DefaultDataProcessor`, `UnitGaussianNormalizer` |

`load_airfrans` and `load_shapenet_car` read `.vtk` meshes and need the `benchmark` extra.
The data of each benchmark is described in the [dataset guides](../dataset_guide/darcy_2d.md).

## Dataset loaders

`load_darcy_flow_small`, `load_darcy_pt` and `load_navier_stokes_pt` return a training loader, a dict of test loaders keyed by resolution and a data processor. The other loaders return a dict of arrays; see each docstring. `load_airfrans` and `load_shapenet_car` need the `benchmark` extra (they read `.vtk` meshes).

::: scirex.data.load_darcy_flow_small

::: scirex.data.load_darcy_pt

::: scirex.data.load_navier_stokes_pt

::: scirex.data.load_cavity_kepsilon

::: scirex.data.load_airfrans

::: scirex.data.load_shapenet_car

## Containers

::: scirex.data.TensorDataset

::: scirex.data.DictDataset

## Processors and normalisers

::: scirex.data.DefaultDataProcessor

::: scirex.data.UnitGaussianNormalizer
