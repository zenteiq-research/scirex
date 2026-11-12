# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   EMA of the weights after the last epoch
                         (flax.serialization, via scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train and validation losses, lr, seconds
        config.json      everything needed to rebuild and use the model:
                         architecture, both normalisations, training settings
                         and provenance (git commit, library versions)

Nothing is chosen by a held-out score: the reported weights are the EMA after
the last epoch, as the official recipe reports its last weights.
``config.json`` is written last, so a directory without it is an unfinished
run. ``infer.py`` needs only this directory and the dataset.
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

from scirex.utils import count_model_params
from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.airfrans._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)
from scripts.benchmarks.airfrans._data import N_FEATURES, Stats
from scripts.benchmarks.airfrans._models import get_spec


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
    stats: Stats,
    official_out: tuple[np.ndarray, np.ndarray],
    history: list[dict],
    training: dict,
    extra_provenance: dict | None = None,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(
        json.dumps(_to_jsonable(history), indent=1) + "\n"
    )
    config = dict(
        run=asdict(run),
        n_params=int(count_model_params(params)),
        normalisation=dict(
            model=stats.to_dict(),
            official_output=dict(mean=official_out[0], std=official_out[1]),
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


def template_params(spec, model, arch, n_nodes: int = 256) -> Any:
    """A parameter pytree of the right structure, from dummy inputs."""
    x = jnp.zeros((n_nodes, N_FEATURES), jnp.float32)
    pos = jnp.linspace(-1.0, 1.0, 2 * n_nodes, dtype=jnp.float32).reshape(n_nodes, 2)
    corners = np.array(
        [[0, 0], [1, 0], [1, 0.1], [0, 0.1]], np.float32
    )  # dummy outline
    segments = np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)
    case = dict(pos=np.asarray(pos), surf=np.zeros(n_nodes, bool), segments=segments)
    return spec.init(model, jax.random.PRNGKey(0), x, pos, spec.extra(arch, case))


def load_checkpoint(run_dir: Path) -> dict:
    """Rebuilds the model and restores its weights from ``run_dir``.

    Returns ``model``, ``spec``, ``params``, the ``run`` config, the model
    ``stats``, the ``official_out`` (mean, std), ``history`` and the raw
    ``config`` dict.
    """
    config = json.loads((run_dir / "config.json").read_text())
    run = run_config_from_dict(config["run"])
    spec = get_spec(run.model)
    model = spec.build(run.arch)
    params = load_params(
        template_params(spec, model, run.arch), resolve_weights_path(run_dir)
    )
    norm = config["normalisation"]
    return dict(
        model=model,
        spec=spec,
        params=params,
        run=run,
        stats=Stats.from_dict(norm["model"]),
        official_out=(
            np.asarray(norm["official_output"]["mean"], np.float32),
            np.asarray(norm["official_output"]["std"], np.float32),
        ),
        history=json.loads((run_dir / "history.json").read_text()),
        config=config,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
