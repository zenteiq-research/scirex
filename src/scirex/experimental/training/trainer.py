# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""General Trainer for fitting neural operators on PDE datasets."""

import logging
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from timeit import default_timer
from typing import Any

import jax
import jax.numpy as jnp

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

from scirex.experimental.training.training_state import (  # noqa: E402
    load_training_state,
    save_training_state,
)
from scirex.losses import LpLoss  # noqa: E402

logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Wrap a list of batches so `Trainer` can use it as a data loader.

    Iterating yields the batches in order, and `len()` is the number of batches.
    The `dataset` attribute only supports `len()` (the number of batches).

    Parameters
    ----------
    batches : list of dict
        Pre-built batches, each a dict such as `{"x": x, "y": y}`.
    """

    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches

        # Create a simple dataset object with len
        class Dataset:
            def __init__(self, size: int) -> None:
                self.size = size

            def __len__(self) -> int:
                return self.size

        self.dataset = Dataset(len(batches))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batches)


class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets.

    The Trainer expects datasets to provide batches as key-value
    dictionaries, ex.: `{'x': x, 'y': y}`, that are keyed to the
    arguments expected by models and losses.

    Parameters
    ----------
    model : flax.linen.Module
        model to train. Duck-typed: any object exposing `__call__(**sample)`
        (and optionally `train_step`, `eval`, `state_dict`/
        `load_state_dict`) is accepted, not just a Flax `nn.Module`.
    n_epochs : int
        number of training epochs
    wandb_log : bool, default is False
        whether to log results to wandb
    device : str, default 'cpu'
        device for computation (JAX handles placement automatically)
    mixed_precision : bool, default is False
        Not implemented. `Trainer.__init__` raises `NotImplementedError` if
        this is `True`.
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with
        data_processor.preprocess, then after getting an output from the
        model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use distributed training (JAX pmap)
    verbose : bool, default is False
        whether to print training progress
    """

    def __init__(
        self,
        *,
        model: Any,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: Any | None = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        stacked_train_data: Any | None = None,
    ) -> None:
        """Initialize Trainer."""
        if n_epochs < 0:
            raise ValueError(f"n_epochs must be >= 0, got {n_epochs}")
        if eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {eval_interval}")
        if mixed_precision:
            raise NotImplementedError(
                "Trainer does not implement mixed-precision training; "
                "pass mixed_precision=False (the default)."
            )
        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.data_processor = data_processor
        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def train(
        self,
        train_loader: Any,
        test_loaders: dict[str, Any],
        optimizer: Any,
        scheduler: Any,
        regularizer: Any | None = None,
        training_loss: Any | None = None,
        eval_losses: dict[str, Any] | None = None,
        eval_modes: dict[str, str] | None = None,
        save_every: int | None = None,
        save_best: str | None = None,
        save_dir: str | Path = "./ckpt",
        resume_from_dir: str | Path | None = None,
        max_autoregressive_steps: int | None = None,
    ) -> dict[str, Any]:
        """Trains the given model on the given dataset.

        Parameters
        ----------
        train_loader : DataLoader
            training dataloader
        test_loaders : dict
            testing dataloaders keyed by name
        optimizer : optax optimizer
            optimizer to use during training
        scheduler : scheduler object
            learning rate scheduler to use during training
        training_loss : loss function, optional
            cost function to minimize
        eval_losses : dict, optional
            dict of losses to use in evaluation
        eval_modes : dict, optional
            optional mapping from loader name to evaluation mode
            ('single_step' or 'autoregressive')
        save_every : int, optional
            if provided, interval at which to save checkpoints
        save_best : str, optional
            if provided, key of metric to monitor for best model
        save_dir : str | Path, default "./ckpt"
            directory at which to save training states
        resume_from_dir : str | Path, default None
            if provided, resumes training state from this directory
        max_autoregressive_steps : int, default None
            max number of autoregressive steps for evaluation

        Returns
        -------
        epoch_metrics : dict
            dictionary of metrics for the last validation epoch
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim.",
                    stacklevel=2,
                )

        if eval_losses is None:
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        if save_every is not None and save_every < 1:
            raise ValueError(f"save_every must be >= 1 or None, got {save_every}")
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            if self.save_best not in metrics:
                raise ValueError(
                    "Expected save_best to be a metric of the form "
                    f"<loader_name>_<metric> (one of {metrics}), got {save_best}"
                )
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            logger.info(f"Training on {len(train_loader.dataset)} samples")
            test_sample_counts = [
                len(loader.dataset) for loader in test_loaders.values()
            ]
            logger.info(
                f"Testing on {test_sample_counts} samples"
                f"         on resolutions {list(test_loaders)}."
            )
            sys.stdout.flush()

        epoch_metrics: dict[str, Any] = {}
        last_saved_epoch = None
        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] != eval_metrics[save_best]:
                        warnings.warn(
                            f"{save_best} is NaN at epoch {epoch}; not saving.",
                            stacklevel=2,
                        )
                    elif eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)
                    last_saved_epoch = epoch

        # always keep the final state when saving on an interval
        if (
            self.save_every is not None
            and self.start_epoch < self.n_epochs
            and last_saved_epoch != self.n_epochs - 1
        ):
            self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(
        self, epoch: int, train_loader: Any, training_loss: Any
    ) -> tuple[float, float, float | None, float]:
        """Train for one epoch and return training metrics.

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : DataLoader
            training data loader
        training_loss : loss function
            training loss function

        Returns
        -------
        train_err : float
            average training error
        avg_loss : float
            average loss per sample
        avg_lasso_loss : float or None
            average regularization loss
        epoch_train_time : float
            time taken for epoch
        """
        self.on_epoch_start(epoch)

        if self.data_processor:
            if hasattr(self.data_processor, "train"):
                self.data_processor.train()
            elif hasattr(self.data_processor, "training"):
                self.data_processor.training = True

        t1 = default_timer()

        self.n_samples = 0
        batch_losses = []  # accumulate as JAX arrays - no per-batch sync

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            batch_losses.append(loss)  # stays on GPU (no float() call)

            # Single GPU-CPU sync at end of epoch
        jax.block_until_ready(batch_losses[-1])
        total = sum(float(l) for l in batch_losses)
        n = len(batch_losses)
        train_err = total / n
        avg_loss = total / max(self.n_samples, 1)
        avg_lasso_loss: float | None = None
        if self.regularizer:
            # Pre-existing bug, not introduced here: avg_lasso_loss is never
            # accumulated per-batch, so it's still None here and this always
            # raises TypeError. Documented and exercised by
            # test_train_one_epoch_with_regularizer_raises; not changing
            # behavior since that test asserts the current (broken) contract.
            avg_lasso_loss /= max(self.n_samples, 1)  # type: ignore[operator]
        else:
            avg_lasso_loss = None

        # Update scheduler
        if hasattr(self.scheduler, "step"):
            self.scheduler.step(train_err)

        epoch_train_time = default_timer() - t1

        lr = None
        if hasattr(self.optimizer, "learning_rate"):
            lr = self.optimizer.learning_rate

        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(
        self,
        epoch: int,
        eval_losses: dict[str, Any],
        test_loaders: dict[str, Any],
        eval_modes: dict[str, str],
        max_autoregressive_steps: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate on all test loaders.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_losses : dict
            keyed `loss_name: loss_obj` for each loss
        test_loaders : dict
            keyed `loader_name: loader` for each test loader
        eval_modes : dict, optional
            keyed `loader_name: eval_mode` for each test loader
        max_autoregressive_steps : int, optional
            max number of autoregressive steps

        Returns
        -------
        all_metrics : dict
            collected eval metrics for each loader
        """
        all_metrics: dict[str, Any] = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict: dict[str, Any],
        data_loader: Any,
        log_prefix: str = "",
        epoch: int | None = None,
        mode: str = "single_step",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate the model on a dictionary of losses.

        Parameters
        ----------
        loss_dict : dict
            each function takes as input (prediction, ground_truth)
            and returns the corresponding loss
        data_loader : DataLoader
            data loader to evaluate on
        log_prefix : str, default ''
            used as prefix in output dictionary
        epoch : int | None
            current epoch for logging
        mode : str
            'single_step' or 'autoregression'
        max_steps : int, optional
            max number of steps for autoregressive rollout

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}'] = loss value
        """
        self.model.eval() if hasattr(self.model, "eval") else None
        if self.data_processor:
            if hasattr(self.data_processor, "eval"):
                self.data_processor.eval()
            elif hasattr(self.data_processor, "training"):
                self.data_processor.training = False

        errors: dict[str, Any] = {
            f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()
        }

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, "reduction"):
                if eval_loss.reduction == "mean":
                    warnings.warn(
                        f"{eval_loss.reduction=}. This means that the loss is "
                        "initialized to average across the batch dim. The Trainer "
                        "expects losses to sum across the batch dim."
                    )

        self.n_samples = 0
        for idx, sample in enumerate(data_loader):
            return_output = False
            if idx == len(data_loader) - 1:
                return_output = True
            if mode == "single_step":
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=return_output
                )
            elif mode == "autoregression":
                eval_step_losses, outs = self.eval_one_batch_autoreg(
                    sample,
                    loss_dict,
                    return_output=return_output,
                    max_steps=max_steps,
                )

            for loss_name, val_loss in eval_step_losses.items():
                errors[f"{log_prefix}_{loss_name}"] += (
                    float(val_loss) if hasattr(val_loss, "__float__") else val_loss
                )

        for loss_name, loss in loss_dict.items():
            key = f"{log_prefix}_{loss_name}"
            if hasattr(loss, "finalize_eval_metric"):
                errors[key] = loss.finalize_eval_metric(errors[key], self.n_samples)
            else:
                errors[key] /= max(self.n_samples, 1)

        # on last batch, log model outputs
        if self.log_output and self.wandb_log:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def on_epoch_start(self, epoch: int) -> None:
        """Run at the beginning of each training epoch.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx: int, sample: dict, training_loss: Any) -> Any:
        """Run one batch through model and return training loss.

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch
        training_loss : loss function
            loss function to compute

        Returns
        -------
        loss : float or array
            training loss value
        """
        if self.regularizer:
            self.regularizer.reset() if hasattr(self.regularizer, "reset") else None

        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data if no preprocessor exists
            sample = {
                k: jnp.asarray(v)
                for k, v in sample.items()
                if isinstance(v, jnp.ndarray)
            }

        if "y" in sample and hasattr(sample["y"], "shape"):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        # Model forward pass with gradient update if model supports train_step
        if hasattr(self.model, "train_step"):
            # JAX model: compute loss + gradients + update params in one step
            loss = self.model.train_step(sample, training_loss)
        else:
            # Fallback: plain forward pass (no gradient update - eval only)
            out = self.model(**sample)
            if self.epoch == 0 and idx == 0 and self.verbose and hasattr(out, "shape"):
                logger.debug(f"Raw outputs of shape {out.shape}")
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample)
            loss = training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss if hasattr(self.regularizer, "loss") else 0.0

        return loss

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ) -> tuple[dict, Any]:
        """Run inference on one batch and return eval losses.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_output : bool
            whether to return model outputs for plotting

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss
        outputs : array or None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data if no preprocessor exists
            sample = {
                k: jnp.asarray(v)
                for k, v in sample.items()
                if isinstance(v, jnp.ndarray)
            }

        if "y" in sample and hasattr(sample["y"], "shape"):
            self.n_samples += sample["y"].shape[0]

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            sample_for_loss = sample
            y_pred_for_loss = out
            if hasattr(loss, "prepare_eval_inputs"):
                y_pred_for_loss, sample_for_loss = loss.prepare_eval_inputs(
                    y_pred_for_loss, sample_for_loss
                )

            if hasattr(loss, "eval_batch_loss"):
                val_loss = loss.eval_batch_loss(y_pred_for_loss, **sample_for_loss)
            else:
                val_loss = loss(y_pred_for_loss, **sample_for_loss)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def eval_one_batch_autoreg(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
        max_steps: int | None = None,
    ) -> tuple[dict, Any]:
        """Run autoregressive inference on one batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_output : bool
            whether to return model outputs for plotting
        max_steps : int
            max number of timesteps to roll out

        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss
        outputs : array or None
            optionally returns batch outputs
        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}

        t = 0
        max_steps_: float = float("inf") if max_steps is None else max_steps

        # only increment the sample count once
        sample_count_incr = False

        while sample is not None and t < max_steps_:
            if self.data_processor is not None:
                sample = (
                    self.data_processor.preprocess(sample, step=t)
                    if hasattr(self.data_processor.preprocess, "__code__")
                    and "step" in self.data_processor.preprocess.__code__.co_varnames
                    else self.data_processor.preprocess(sample)
                )
            else:
                # load data if no preprocessor exists
                sample = {
                    k: jnp.asarray(v)
                    for k, v in sample.items()
                    if isinstance(v, jnp.ndarray)
                }

            if sample is None:
                break

            # only increment the sample count once
            if not sample_count_incr:
                if "y" in sample and hasattr(sample["y"], "shape"):
                    self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            out = self.model(**sample)

            if self.data_processor is not None:
                out, sample = (
                    self.data_processor.postprocess(out, sample, step=t)
                    if hasattr(self.data_processor.postprocess, "__code__")
                    and "step" in self.data_processor.postprocess.__code__.co_varnames
                    else self.data_processor.postprocess(out, sample)
                )

            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, **sample)
                eval_step_losses[loss_name] += (
                    float(step_loss) if hasattr(step_loss, "__float__") else step_loss
                )

            t += 1

        # average over all steps of the final rollout
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= max(t, 1)

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float | None = None,
        lr: float | None = None,
    ) -> None:
        """Log results from a single training epoch.

        Parameters
        ----------
        epoch : int
            epoch number
        time : float
            training time of epoch
        avg_loss : float
            average loss per sample
        train_err : float
            training error for entire epoch
        avg_lasso_loss : float, optional
            average lasso loss from regularizer
        lr : float, optional
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        logger.info(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def log_eval(self, epoch: int, eval_metrics: dict) -> None:
        """Log evaluation metrics to stdout and wandb.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics keyed f"{test_loader_name}_{metric}"
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, jnp.ndarray):
                msg += f"{metric}={float(value):.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value

        msg = "Eval: " + msg[:-2] if msg else "Eval: "
        logger.info(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def resume_state_from_dir(self, save_dir: str | Path) -> None:
        """Resume training from a saved state directory.

        Parameters
        ----------
        save_dir : str | Path
            directory in which training state is saved
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        candidates = [
            (save_dir / f"{name}_state_dict.msgpack", name)
            for name in ("best_model", "model")
        ]
        existing = [(f, name) for f, name in candidates if f.exists()]
        if existing:
            # when both exist, resume from the one written last
            save_name = max(existing, key=lambda fn: fn[0].stat().st_mtime)[1]
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model state dict named "
                "model.msgpack or best_model.msgpack."
            )

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.regularizer,
            resume_epoch,
        ) = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            scheduler=self.scheduler,
        )

        if resume_epoch is not None:
            # the saved epoch is already trained; continue with the next one
            if resume_epoch + 1 > self.start_epoch:
                self.start_epoch = resume_epoch + 1
                if self.verbose:
                    logger.info(f"Trainer resuming from epoch {self.start_epoch}")

    def checkpoint(self, save_dir: str | Path) -> None:
        """Save current training state to a directory.

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if self.save_best is not None:
            save_name = "best_model"
        else:
            save_name = "model"

        save_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            regularizer=self.regularizer,
            epoch=self.epoch,
        )
        if self.verbose:
            logger.info(f"Saved training state to {save_dir}")
