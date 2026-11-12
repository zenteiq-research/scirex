# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack   trained weights (flax.serialization, via
                         scripts/benchmarks/_ckpt.py)
        history.json     per-epoch train/val split-MSE and seconds
        config.json      everything needed to rebuild and use the model:
                         architecture, input/output normalization, training
                         settings and provenance (git commit, library versions)

``config.json`` is written last, so a directory without it is an unfinished
run. ``infer.py`` needs only this directory and the dataset's val split (for
the raw VTK geometry the drag-coefficient metric needs); it never re-derives
normalization stats from scratch (they are stored here, not recomputed).

Template init uses a small dummy point cloud (``N=4``, an arbitrary size --
see ``load_checkpoint``) rather than a real sample: Transolver's parameter
shapes come from ``n_hidden``/``n_layers``/``n_head``/``space_dim``/``out_dim``
only, not from the point count ``N``, the same way a Transformer's weights
don't depend on sequence length. This is verified in
``scripts/tests/test_shapenetcar.py``.
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
from scripts.benchmarks.shapenetcar._config import (
    REPO_ROOT,
    RunConfig,
    run_config_from_dict,
)
from scripts.benchmarks.shapenetcar._data import MODEL_KWARGS
from scripts.benchmarks.shapenetcar._models import build_model

#: Arbitrary >0 size for the template init -- see this module's docstring.
_TEMPLATE_N = 4


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
    coef_norm: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    history: list[dict],
    training: dict,
) -> None:
    """Writes ``params.msgpack``, ``history.json`` and, last, ``config.json``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_params(jax.device_get(params), run_dir / "params.msgpack")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    mean_in, std_in, mean_out, std_out = coef_norm
    config = dict(
        run=asdict(run),
        n_params=count_params(params),
        normalisation=dict(
            mean_in=np.asarray(mean_in, np.float32).tolist(),
            std_in=np.asarray(std_in, np.float32).tolist(),
            mean_out=np.asarray(mean_out, np.float32).tolist(),
            std_out=np.asarray(std_out, np.float32).tolist(),
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

    Returns ``model``, ``params``, the ``run`` config, ``coef_norm`` (the
    ``(mean_in, std_in, mean_out, std_out)`` tuple ``compute_drag_coefficient``/
    ``rel_l2_metrics`` need to denormalize predictions), ``history`` and the
    raw ``config`` dict.
    """
    config = json.loads((run_dir / "config.json").read_text())
    run = run_config_from_dict(config["run"])
    model = build_model(run.model, run.arch, MODEL_KWARGS)
    # Parameter shapes do not depend on the point count -- see this module's
    # docstring -- so a small dummy point cloud gives a valid template.
    x0 = jnp.zeros((1, _TEMPLATE_N, MODEL_KWARGS["space_dim"]))
    template = model.init(jax.random.PRNGKey(0), x=x0, fx=None)
    params = load_params(template, resolve_weights_path(run_dir))
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    norm = config["normalisation"]
    coef_keys = ("mean_in", "std_in", "mean_out", "std_out")
    coef_norm = tuple(np.asarray(norm[k], np.float32) for k in coef_keys)
    return dict(
        model=model,
        params=params,
        run=run,
        coef_norm=coef_norm,
        history=history,
        config=config,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
