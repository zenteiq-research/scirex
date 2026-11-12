# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

from collections.abc import Callable
from typing import Any

import flax.linen as nn


class MLP(nn.Module):
    """Multi-layer perceptron.

    Parameters
    ----------
    features : list[int]
        Width of every layer; the last entry is the output width.
    activation : Callable, optional
        Activation between layers. Default: `nn.relu`.
    dropout_rate : float, optional
        Dropout rate after each layer. Default: 0.0.
    activation_last : bool, optional
        Whether to apply the activation after the last layer. Default: False.
    residual : bool, optional
        Whether to add skip connections between hidden layers. Default: False.
    """

    features: list[int]
    activation: Callable = nn.relu
    dropout_rate: float = 0.0
    activation_last: bool = False
    residual: bool = False

    @nn.compact
    def __call__(self, x: Any, deterministic: bool = True) -> Any:
        for i, feat in enumerate(self.features):
            x_in = x
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1 or self.activation_last:
                x = self.activation(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
            if self.residual:
                if i > 0 and i < len(self.features) - 1:
                    x = x + x_in
        return x
