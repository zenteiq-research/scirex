# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data split, normalisation and loss of the Pipe benchmark.

These reuse the Transolver Pipe benchmark's own code
(``_transolver.py``) rather than copies of
it, so every architecture sees exactly the data, normalisation and loss
that Transolver does:

* the target: channel 0 of the official ``Pipe_Q.npy``, which must be complete;
* the split: first 1000 samples for training, last 200 for testing, no
  downsampling of the 129x129 mesh;
* inputs: the raw ``(x, y)`` coordinates, normalised with a unit Gaussian
  fitted on the training inputs;
* targets: kept in physical units; the model's output is decoded with a unit
  Gaussian fitted on the training targets before the loss;
* loss and metric: relative L2 per sample, summed over the batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.benchmarks.pipe import _transolver

rel_l2_sum = _transolver._rel_l2_sum
UnitTransformer = _transolver._UnitTransformer


def official_output_file(data_dir: Path) -> Path:
    """``data_dir/Pipe_Q.npy``, after checking it is complete.

    The Transolver loader falls back to ``Pipe_Q_interp_x.npy``, with only a
    printed warning, when ``Pipe_Q.npy`` is missing or truncated. That would
    silently change the benchmark's target, so this benchmark refuses to run
    instead.
    """
    q_path = data_dir / "Pipe_Q.npy"
    if not q_path.exists():
        raise FileNotFoundError(f"{q_path} not found")
    shape, dtype = _transolver._npy_shape(q_path)
    if q_path.stat().st_size < int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError(f"{q_path} is truncated: its header declares {shape}")
    return q_path


def load_split(data_dir: Path, protocol: dict) -> dict:
    """Train/test split with fitted normalisers.

    Returns flat arrays, ``x_*`` as ``(n, H * W, 2)`` (normalised) and
    ``y_*`` as ``(n, H * W, 1)`` (physical units), the raw test coordinates
    for plotting, the ``grid`` shape and the two normalisers.
    """
    q_path = official_output_file(Path(data_dir))
    data = _transolver.load_pipe_data(
        Path(data_dir),
        q_path,
        ntrain=protocol["ntrain"],
        ntest=protocol["ntest"],
        n_total=protocol["n_total"],
        downsamplex=1,
        downsampley=1,
    )
    x_normalizer = UnitTransformer(data["x_train"])
    y_normalizer = UnitTransformer(data["y_train"])
    return dict(
        x_train=x_normalizer.encode(data["x_train"]).astype(np.float32),
        y_train=data["y_train"][..., None],
        x_test=x_normalizer.encode(data["x_test"]).astype(np.float32),
        y_test=data["y_test"][..., None],
        xy_test=data["x_test"],
        grid=(data["s1"], data["s2"]),
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
    )


def normalizer_from_stats(mean, std) -> UnitTransformer:
    """A :class:`UnitTransformer` with stored statistics, for inference.

    float32, like the statistics fitted on the float32 training data, so a
    reloaded normaliser encodes and decodes bit for bit as the fitted one.
    """
    normalizer = UnitTransformer.__new__(UnitTransformer)
    normalizer.mean = np.asarray(mean, dtype=np.float32)
    normalizer.std = np.asarray(std, dtype=np.float32)
    return normalizer
