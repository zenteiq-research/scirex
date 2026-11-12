# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Chunked final-eval prediction, shared across benchmark runners.

A benchmark's training loop already processes data in ``--batch-size``
chunks; the final zero-shot evaluation over the full test set should too,
rather than passing every test instance through ``predict_fn`` in one call
-- at high resolution that single large call can need transient memory well
beyond what training's smaller batches needed (observed directly: an eval
call over 500 instances OOM'd at s=85 under GPU contention even though
training itself completed fine).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def predict_in_batches(
    predict_fn: Callable[[object, np.ndarray], np.ndarray],
    params: object,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Apply ``predict_fn(params, batch)`` over ``inputs`` in ``batch_size``
    chunks and concatenate the results.

    Raises ``ValueError`` up front if ``inputs`` is empty, rather than
    letting ``np.concatenate`` fail on an empty list of chunks with a less
    informative error.
    """
    if inputs.shape[0] == 0:
        raise ValueError(
            "predict_in_batches: inputs is empty (0 instances) -- nothing to predict."
        )
    chunks = [
        np.asarray(predict_fn(params, inputs[start : start + batch_size]))
        for start in range(0, inputs.shape[0], batch_size)
    ]
    return np.concatenate(chunks, axis=0)
