# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   trained weights (flax.serialization, via
                         scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train error, de-normalised test error,
                         learning rate and seconds
        config.json      the run config, parameter count, training summary
                         and provenance (git commit, library versions)

``config.json`` is written last, so a directory without it is an unfinished
run. ``train.py`` also rewrites ``params.msgpack``/``history.json`` every
``--ckpt-every`` epochs, so an interrupted run keeps its weights.

Unlike ``shapenetcar/_checkpoint.py``, no normalisation stats are stored:
``CarCFDDatasetjax`` fits its pressure normaliser on all loaded cars (train
and test together, ``mesh_datamodule.py``) deterministically, so ``infer.py``
rebuilds it from the same ``n_train`` and ``n_test``.
The params template needs a real sample too (GINO's latent-grid shape comes
from the data), so loading is done by ``infer.py`` against the rebuilt
dataset rather than here.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scripts.benchmarks._ckpt import save_params
from scripts.benchmarks.car_cfd._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)


def count_params(params: Any) -> int:
    """Number of trained real values in ``params`` (complex weights count
    twice: real and imaginary parts are each trained)."""
    return sum(
        int(np.prod(p.shape)) * (2 if jnp.iscomplexobj(p) else 1)
        for p in jax.tree_util.tree_leaves(params)
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


def save_progress(run_dir: Path, params: Any, history: list[dict]) -> None:
    """Weights and history so far, without marking the run complete."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")


def save_checkpoint(
    run_dir: Path,
    run: RunConfig,
    params: Any,
    history: list[dict],
    training: dict,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    save_progress(run_dir, params, history)
    config = dict(
        run=asdict(run),
        n_params=count_params(params),
        training=training,
        provenance=provenance(),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def is_complete(run_dir: Path) -> bool:
    return (run_dir / "config.json").exists() and (
        (run_dir / "params.msgpack").exists()
        or (run_dir / "model.safetensors").exists()
    )


def read_checkpoint(run_dir: Path) -> dict:
    """``run`` (a :class:`RunConfig`), ``history`` and the raw ``config``."""
    config = json.loads((run_dir / "config.json").read_text())
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    return dict(run=run_config_from_dict(config["run"]), history=history, config=config)
