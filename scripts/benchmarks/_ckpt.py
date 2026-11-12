# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Checkpoint save/load for the benchmark runners.

Flax parameters are a pytree of arrays; we serialize them with
``flax.serialization`` (msgpack). Loading restores into a *template* params
pytree of the same structure, which the inference script obtains by rebuilding
the model and calling ``init`` once.

Checkpoints are binary build artifacts and are gitignored - only the resulting
PNGs and metrics.json are committed.

A run directory may instead (or additionally) hold ``model.safetensors`` -
the format the Hugging Face release exports to (see
``internal/hf_release/build.py``): the same params pytree, flattened with
``/`` separators. ``params.msgpack`` is preferred when both are present, since
it is the exact bytes the run produced; ``model.safetensors`` is what a
downloaded Hugging Face checkpoint ships.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization, traverse_util
from safetensors.numpy import load_file as load_safetensors_file


def save_params(params, path: str | Path) -> Path:
    """Serialize a Flax params pytree to ``path`` (msgpack bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))
    return path


def resolve_weights_path(run_dir: str | Path) -> Path:
    """The run's weights file: ``params.msgpack`` if present, else
    ``model.safetensors``; raises if the directory has neither."""
    run_dir = Path(run_dir)
    msgpack = run_dir / "params.msgpack"
    if msgpack.exists():
        return msgpack
    safetensors = run_dir / "model.safetensors"
    if safetensors.exists():
        return safetensors
    raise FileNotFoundError(f"no params.msgpack or model.safetensors in {run_dir}")


def load_params(template, path: str | Path):
    """Load params from ``path`` into the structure of ``template``.

    ``template`` is a params pytree of the correct shape (e.g. from a fresh
    ``model.init``); the stored values are restored into it. ``path`` may be
    either a ``params.msgpack`` file or a ``model.safetensors`` file.

    Leaves are mapped to ``jnp`` arrays either way (dtypes, including complex,
    are preserved); ``flax.serialization.from_bytes`` restores msgpack leaves
    as NumPy arrays, and safetensors leaves are NumPy arrays already.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        flat = load_safetensors_file(str(path))
        nested = traverse_util.unflatten_dict(flat, sep="/")
        restored = serialization.from_state_dict(template, nested)
    else:
        restored = serialization.from_bytes(template, path.read_bytes())
    return jax.tree_util.tree_map(jnp.asarray, restored)
