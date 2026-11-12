# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The benchmark's checkpoint format, shared by ``train.py`` and ``infer.py``.

Each run is saved to its own directory::

    <ckpt_root>/<run>/
        params.msgpack       weights after the last epoch (the reported ones)
        best_params.msgpack  weights at the lowest test error seen, kept for
                             reference only; nothing reported is chosen by it
        history.json         test error, training loss and time per
                             evaluated epoch
        config.json          the run, the protocol, the normalisation
                             statistics, the training summary and provenance
        frames.npz           with ``train.py --animate`` only: the frames
                             ``animate.py`` renders

``config.json`` is written last, so a directory without it is an unfinished
run.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from scirex.utils import count_model_params
from scripts.benchmarks._ckpt import load_params, resolve_weights_path, save_params
from scripts.benchmarks.chip_cooling._checkpoint import provenance
from scripts.benchmarks.darcy._config import RunConfig, run_config_from_dict


def save_checkpoint(
    run_dir: Path,
    run: RunConfig,
    protocol: dict,
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
        protocol=protocol,
        n_params=int(count_model_params(params)),
        normalisation=normalisation,
        training=training,
        provenance=provenance(),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


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
    """The checkpoint's last-epoch weights, restored into ``template``."""
    return load_params(template, resolve_weights_path(run_dir))


def load_history(run_dir: Path) -> dict:
    path = Path(run_dir) / "history.json"
    return json.loads(path.read_text()) if path.exists() else {}


def check_normalisation(stored: dict, computed: dict, name: str) -> None:
    """Fails if the data's normalisation differs from the checkpoint's: the
    model would then see inputs on a different scale than it was trained on."""
    for key, value in computed.items():
        if not np.isclose(stored[key], value, rtol=1e-5):
            raise ValueError(
                f"{name}: normalisation {key!r} of the data differs from the "
                "checkpoint's; is this the dataset and split it was trained on?"
            )
