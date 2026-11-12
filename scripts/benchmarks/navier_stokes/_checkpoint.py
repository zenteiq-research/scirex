# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   weights after the last epoch
                         (flax.serialization, via scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train loss, test relative L2, lr, seconds
        config.json      everything needed to rebuild and use the model:
                         architecture, training settings and provenance
                         (git commit, library versions)

The last epoch's weights are saved, not the best epoch's: the benchmark has
no validation split, and choosing an epoch by its test error would leak the
test set into the result.

``config.json`` is written last, so a directory without it is an unfinished
run. ``infer.py`` needs only this directory and the test data.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from scirex.utils import count_model_params
from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.navier_stokes._config import (
    RunConfig,
    run_config_from_dict,
)
from scripts.benchmarks.navier_stokes._models import get_spec
from scripts.benchmarks.pipe._checkpoint import _to_jsonable, provenance, sha256

__all__ = [
    "is_complete",
    "load_checkpoint",
    "save_checkpoint",
    "sha256",
]


def save_checkpoint(
    run_dir: Path,
    run: RunConfig,
    params: Any,
    grid: tuple[int, int],
    t_in: int,
    history: list[dict],
    training: dict,
    extra_provenance: dict | None = None,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    config = dict(
        run=asdict(run),
        grid=list(grid),
        t_in=t_in,
        n_params=int(count_model_params(params)),
        training=training,
        provenance={**provenance(), **(extra_provenance or {})},
    )
    (run_dir / "config.json").write_text(
        json.dumps(_to_jsonable(config), indent=2) + "\n"
    )


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "config.json").exists() and (
        (run_dir / "params.msgpack").exists()
        or (run_dir / "model.safetensors").exists()
    )


def load_checkpoint(run_dir: Path) -> dict:
    """Rebuilds the model and restores its weights from ``run_dir``.

    Returns ``model``, ``spec``, ``params``, the ``run`` config, the
    ``grid``, ``t_in``, ``history`` and the raw ``config`` dict.
    """
    config = json.loads((run_dir / "config.json").read_text())
    run = run_config_from_dict(config["run"])
    grid = tuple(config["grid"])
    t_in = int(config["t_in"])
    spec = get_spec(run.model)
    model = spec.build(run.arch, grid, t_in)
    fx = jnp.zeros((1, grid[0] * grid[1], t_in), jnp.float32)
    # Only the structure matters here; Flax's init is enough.
    template = model.init(jax.random.PRNGKey(0), **spec._inputs(model, fx, grid))
    params = load_params(template, resolve_weights_path(run_dir))
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    return dict(
        model=model,
        spec=spec,
        params=params,
        run=run,
        grid=grid,
        t_in=t_in,
        history=history,
        config=config,
    )
