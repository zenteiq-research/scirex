<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# API reference

Everything you can import from ScireX and rely on, grouped by package.

```python
from scirex.models import FNO
from scirex.losses import LpLoss
```

Import from the package, not from the file that defines the name.

| Package | What it provides |
| --- | --- |
| [`scirex.models`](models.md) | `FNO`, `FNOClassic2d`, `GINO`, `WNO`, `Transolver`, `get_model_jax`, `available_models`, `BaseModel` |
| [`scirex.layers`](layers.md) | spectral, wavelet and graph blocks, embeddings, MLP |
| [`scirex.losses`](losses.md) | data losses, PDE residual terms, Galerkin loss |
| [`scirex.data`](data.md) | dataset loaders, containers, normalisers |
| [`scirex.training`](training.md) | complex-aware optimiser `adamw_cx` |
| [`scirex.config`](config.md) | model and optimiser configurations, presets |
| [`scirex.utils`](utils.md) | parameter counting |
| [`scirex.experimental`](experimental.md) | work in progress, no stability guarantee |

## Stability

| Tier | Where | What you can expect |
| --- | --- | --- |
| Stable | every package above except `scirex.experimental` | no breaking change within a major version; removals are deprecated first |
| Experimental | `scirex.experimental` | may change or disappear in any release |
| Private | names starting with `_`, `scirex._internal` | not for use |

See the [API policy](../dev_guide/api_policy.md) for what counts as a breaking change.

## Optional dependencies

A missing optional package never breaks `import scirex`. Using a name that needs one
raises an `ImportError` that tells you which extra to install, for example
`pip install "scirex[hpo]"` for hyperparameter tuning.
