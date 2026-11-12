<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.models

Neural-operator architectures, the `get_model_jax` factory and the `BaseModel` class
they share.

```python
from scirex.models import FNO, get_model_jax
```

| Name | Use it to |
| --- | --- |
| [`FNO`](#scirex.models.FNO) | learn maps between functions on regular grids (Fourier layers) |
| [`GINO`](#scirex.models.GINO) | learn on irregular meshes and geometries (graph and Fourier layers) |
| [`WNO`](#scirex.models.WNO) | learn with wavelet layers, good for sharp local features |
| [`Transolver`](#scirex.models.Transolver) | learn on meshes of any layout with physics-aware attention |
| [`get_model_jax`](#scirex.models.get_model_jax) | build a model from a configuration |
| [`BaseModel`](#scirex.models.BaseModel) | save and load checkpoints; subclass it to add a model |

`GIWO` is experimental; see [`scirex.experimental`](experimental.md).

## Architectures

::: scirex.models.FNO

::: scirex.models.FNOClassic2d

::: scirex.models.GINO

::: scirex.models.WNO

::: scirex.models.Transolver

## Factory and registry

`get_model_jax` builds a model from a config object whose `model_arch` names an architecture. `available_models` lists the accepted names.

::: scirex.models.get_model_jax

::: scirex.models.available_models

## Base class

Base class of every model. It provides checkpoint saving and loading (`save_checkpoint`, `load_checkpoint`, `from_checkpoint`). Subclass it to add an architecture.

::: scirex.models.BaseModel
