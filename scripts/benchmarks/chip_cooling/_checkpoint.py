# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   weights after the last epoch
                         (flax.serialization, via scripts/benchmarks/_ckpt.py)
        history.json     per-epoch training history (the published
                         metrics.json)
        config.json      the run, the normalisation statistics, the training
                         summary and provenance (git commit, library versions)

The last epoch's weights are saved: the benchmark has no validation split, so
no epoch is chosen by its test error.

``config.json`` is written last, so a directory without it is an unfinished
run.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from scirex.utils import count_model_params
from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.chip_cooling._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)


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
    normalisation: dict,
    history: dict,
    training: dict,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    config = dict(
        run=asdict(run),
        n_params=int(count_model_params(params)),
        normalisation=normalisation,
        training=training,
        provenance=provenance(),
    )
    (run_dir / "config.json").write_text(
        json.dumps(_to_jsonable(config), indent=2) + "\n"
    )


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "config.json").exists() and (
        (run_dir / "params.msgpack").exists()
        or (run_dir / "model.safetensors").exists()
    )


def load_config(run_dir: Path) -> tuple[RunConfig, dict]:
    """The run and the raw ``config.json`` of a checkpoint."""
    config = json.loads((Path(run_dir) / "config.json").read_text())
    return run_config_from_dict(config["run"]), config


def load_params_into(run_dir: Path, template: Any) -> Any:
    """The checkpoint's weights, restored into ``template``'s structure (the
    model's freshly initialised parameters)."""
    return load_params(template, resolve_weights_path(run_dir))


def load_history(run_dir: Path) -> dict:
    path = Path(run_dir) / "history.json"
    return json.loads(path.read_text()) if path.exists() else {}


def check_normalisation(stored: dict, computed: dict, name: str) -> None:
    """Fails if the data's normalisation differs from the checkpoint's: the
    model would then see inputs on a different scale than it was trained on."""
    for key, value in computed.items():
        if key == "coord_stats":
            check_normalisation(stored[key], value, name)
            continue
        if not np.allclose(np.asarray(stored[key]), np.asarray(value), rtol=1e-6):
            raise ValueError(
                f"{name}: normalisation {key!r} of the data differs from the "
                "checkpoint's; is this the dataset and split it was trained on?"
            )
