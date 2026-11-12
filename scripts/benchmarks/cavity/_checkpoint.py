# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   weights with the best validation loss
                         (flax.serialization, via scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train loss, validation loss, lr, seconds
        config.json      everything needed to rebuild and use the model:
                         architecture, time windows, normalisation, training
                         settings and provenance (git commit, library versions)

``config.json`` is written last, so a directory without it is an unfinished
run. ``infer.py`` needs only this directory and the test case; it never
reloads the training data.
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
import jax.numpy as jnp
import numpy as np

from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.cavity._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)
from scripts.benchmarks.cavity._data import (
    FIELDS,
    N_CELL,
    normalizer_from_stats,
)
from scripts.benchmarks.cavity._models import build_model


def count_params(params: Any) -> int:
    """Number of trained real values in ``params``.

    A complex weight counts twice, because its real and imaginary parts are
    each trained. Without that, two models that hold the same number of
    trainable values would report different sizes purely because one stores
    its spectral weights as complex arrays and the other as real pairs, which
    is exactly the comparison ``gino_3to1`` and ``fno_3to1`` are set up to
    make.
    """
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
    mean: np.ndarray,
    std: np.ndarray,
    history: list[dict],
    training: dict,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    config = dict(
        run=asdict(run),
        n_params=count_params(params),
        normalisation=dict(
            fields=list(FIELDS),
            mean=np.asarray(mean, np.float32).reshape(-1),
            std=np.asarray(std, np.float32).reshape(-1),
        ),
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


def load_checkpoint(run_dir: Path) -> dict:
    """Rebuilds the model and restores its weights from ``run_dir``.

    Returns ``model``, ``params``, the ``run`` config, the ``normalizer``, its
    ``mean``/``std``, ``history`` and the raw ``config`` dict.
    """
    config = json.loads((run_dir / "config.json").read_text())
    run = run_config_from_dict(config["run"])
    model = build_model(run.model, n_out=run.n_out, **run.arch)
    # Parameter shapes do not depend on the batch or on the values, so a
    # zero input of the right spatial size gives a valid template.
    template = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, run.n_in, N_CELL, N_CELL, len(FIELDS)))
    )
    params = load_params(template, resolve_weights_path(run_dir))
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    mean = np.asarray(config["normalisation"]["mean"], np.float32)
    std = np.asarray(config["normalisation"]["std"], np.float32)
    return dict(
        model=model,
        params=params,
        run=run,
        normalizer=normalizer_from_stats(mean, std),
        mean=mean,
        std=std,
        history=history,
        config=config,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
