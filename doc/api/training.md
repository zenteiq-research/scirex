<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.training

Optimisers that handle complex-valued weights.

```python
from scirex.training import adamw_cx
```

Use `adamw_cx` instead of `optax.adamw` when the model has complex weights (FNO and GINO
do; WNO has real weights only). Plain AdamW treats a complex weight incorrectly. The training loop is experimental:
see [`scirex.experimental.training`](experimental.md#experimental-training-loop).

## Optimisers

`scirex.training.optim` also exports [`scale_by_adam_cx`](#scirex.training.optim.scale_by_adam_cx), the gradient transformation `adamw_cx` is built from.

::: scirex.training.adamw_cx

::: scirex.training.optim.scale_by_adam_cx
