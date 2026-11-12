# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run and seed is saved to its own directory::

    <ckpt_root>/<run>/seed<seed>/
        params.msgpack   weights after the last epoch
                         (flax.serialization, via scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train loss, test relative L2, lr, seconds
        config.json      everything needed to rebuild and use the model:
                         architecture, normalisation, training settings and
                         provenance (git commit, library versions)

The last epoch's weights are saved, not the best epoch's: the benchmark has
no validation split, and choosing an epoch by its test error would leak the
test set into the result.

``config.json`` is written last, so a directory without it is an unfinished
run. ``infer.py`` needs only this directory and the test data.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from scirex.utils import count_model_params
from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.pipe._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)
from scripts.benchmarks.pipe._data import UnitTransformer, normalizer_from_stats
from scripts.benchmarks.pipe._models import get_spec


def provenance() -> dict:
    """Where and with what a checkpoint was produced."""
    import flax
    import optax

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return dict(
        git_commit=commit,
        git_dirty=dirty,
        jax=jax.__version__,
        flax=flax.__version__,
        optax=optax.__version__,
        device=str(jax.devices()[0]),
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_checkpoint(
    run_dir: Path,
    run: RunConfig,
    params: Any,
    grid: tuple[int, int],
    x_normalizer: UnitTransformer,
    y_normalizer: UnitTransformer,
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
        n_params=int(count_model_params(params)),
        normalisation=dict(
            x_mean=x_normalizer.mean,
            x_std=x_normalizer.std,
            y_mean=y_normalizer.mean,
            y_std=y_normalizer.std,
        ),
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

    Returns ``model``, ``spec``, ``params``, the ``run`` config, the ``grid``,
    both normalisers, ``history`` and the raw ``config`` dict.
    """
    config = json.loads((run_dir / "config.json").read_text())
    run = run_config_from_dict(config["run"])
    grid = tuple(config["grid"])
    spec = get_spec(run.model)
    model = spec.build(run.arch, grid)
    inputs = spec.take(spec.prepare(run.arch, spec.template(grid), grid), np.arange(1))
    template = model.init(jax.random.PRNGKey(0), **inputs)
    params = load_params(template, resolve_weights_path(run_dir))
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    norm = config["normalisation"]
    return dict(
        model=model,
        spec=spec,
        params=params,
        run=run,
        grid=grid,
        x_normalizer=normalizer_from_stats(norm["x_mean"], norm["x_std"]),
        y_normalizer=normalizer_from_stats(norm["y_mean"], norm["y_std"]),
        history=history,
        config=config,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
