# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""

import logging
import warnings
from pathlib import Path
from typing import Any

from scirex._internal.serialization import load_msgpack, save_msgpack

logger = logging.getLogger(__name__)


def load_training_state(
    save_dir: str | Path,
    save_name: str,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    regularizer: Any = None,
    map_location: dict | None = None,
) -> tuple[Any, Any, Any, Any, int | None]:
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional
        optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    model : object, optional
        model object to load (Flax module or similar), by default None
    optimizer : object, optional
        optimizer object to load, by default None
    scheduler : object, optional
        scheduler object to load, by default None
    regularizer : object, optional
        regularizer object to load, by default None
    map_location : dict, optional
        mapping dictionary (for API compatibility), by default None

    Returns
    -------
    tuple of training state
        `model, optimizer, scheduler, regularizer, epoch`

    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # optionally load epoch
    epoch = None
    manifest_pth = save_dir / "manifest.msgpack"
    if manifest_pth.exists():
        manifest = load_msgpack(manifest_pth)
        epoch = manifest.get("epoch")

    # Load model if state exists
    if model is not None:
        save_pth = save_dir / f"{save_name}_state_dict.msgpack"
        if save_pth.exists():
            state_dict = load_msgpack(save_pth)
            # For Flax models, assign loaded state to model
            # This assumes the model has a method to load parameters, or
            # the caller handles parameter assignment
            if hasattr(model, "load_state_dict"):
                model.load_state_dict(state_dict)
            else:
                # If model is a dict of parameters, return the loaded state
                model = state_dict
        else:
            warnings.warn(
                f"requested to load model state, but no saved model"
                f" state exists in {save_dir}.",
                stacklevel=2,
            )

    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.msgpack"
        if optimizer_pth.exists():
            state_dict = load_msgpack(optimizer_pth)
            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(state_dict)
            else:
                optimizer = state_dict
        else:
            warnings.warn(
                f"requested to load optimizer state, but no saved"
                f" optimizer state exists in {save_dir}.",
                stacklevel=2,
            )

    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.msgpack"
        if scheduler_pth.exists():
            state_dict = load_msgpack(scheduler_pth)
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(state_dict)
            else:
                scheduler = state_dict
        else:
            warnings.warn(
                f"requested to load scheduler state, but no saved"
                f" scheduler state exists in {save_dir}.",
                stacklevel=2,
            )

    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.msgpack"
        if regularizer_pth.exists():
            state_dict = load_msgpack(regularizer_pth)
            if hasattr(regularizer, "load_state_dict"):
                regularizer.load_state_dict(state_dict)
            else:
                regularizer = state_dict
        else:
            warnings.warn(
                f"requested to load regularizer state, but no saved"
                f" regularizer state exists in {save_dir}.",
                stacklevel=2,
            )

    return model, optimizer, scheduler, regularizer, epoch


def save_training_state(
    save_dir: str | Path,
    save_name: str,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    regularizer: Any = None,
    epoch: int | None = None,
) -> None:
    """save_training_state saves model and optional other training modules
    for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory to save training state (model, optional optimizer,
        scheduler, regularizer)
    save_name : str
        name of model to save
    model : object, optional
        model object to save (Flax module or state dict), by default None
    optimizer : object, optional
        optimizer object to save, by default None
    scheduler : object, optional
        scheduler object to save, by default None
    regularizer : object, optional
        regularizer object to save, by default None
    epoch : int, optional
        epoch number to save, by default None
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True, parents=True)
    manifest: dict[str, str | int] = {}

    # Save model
    if model is not None:
        model_pth = save_dir / f"{save_name}_state_dict.msgpack"
        # Extract state dict if model has the method, otherwise save directly
        if hasattr(model, "state_dict"):
            state_to_save = model.state_dict()
        else:
            state_to_save = model

        save_msgpack(state_to_save, model_pth)
        manifest["model"] = f"{save_name}_state_dict.msgpack"

    # save optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.msgpack"
        # Extract state dict if optimizer has the method, otherwise save directly
        if hasattr(optimizer, "state_dict"):
            opt_state = optimizer.state_dict()
        else:
            opt_state = optimizer

        save_msgpack(opt_state, optimizer_pth)
        manifest["optimizer"] = "optimizer.msgpack"

    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.msgpack"
        # Extract state dict if scheduler has the method, otherwise save directly
        if hasattr(scheduler, "state_dict"):
            sched_state = scheduler.state_dict()
        else:
            sched_state = scheduler

        save_msgpack(sched_state, scheduler_pth)
        manifest["scheduler"] = "scheduler.msgpack"

    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.msgpack"
        # Extract state dict if regularizer has the method, otherwise save directly
        if hasattr(regularizer, "state_dict"):
            reg_state = regularizer.state_dict()
        else:
            reg_state = regularizer

        save_msgpack(reg_state, regularizer_pth)
        manifest["regularizer"] = "regularizer.msgpack"

    if epoch is not None:
        manifest["epoch"] = epoch

    manifest_pth = save_dir / "manifest.msgpack"
    save_msgpack(manifest, manifest_pth)
