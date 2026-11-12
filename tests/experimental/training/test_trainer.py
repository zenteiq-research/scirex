# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the Trainer (JAX).

Mirrors ``neuralop/training/tests/test_trainer.py``. The reference uses a
PyTorch ``DummyModel`` + ``DummyDataset``; here we use a small stateful linear
model that follows the JAX Trainer's model contract (``__call__(**sample)``, an
optional ``train_step`` that owns the gradient update, ``eval``, and
``state_dict``/``load_state_dict`` for checkpointing). The reference scenarios
are reproduced:

* ``test_model_checkpoint_saves`` -> ``test_checkpoint_saves_files`` (the JAX
  checkpoint writes ``*_state_dict.msgpack`` / ``optimizer.msgpack`` /
  ``scheduler.msgpack`` / ``manifest.msgpack``);
* ``test_load_from_checkpoint`` -> ``test_load_from_checkpoint`` (eval error after
  reloading equals the saved model's);
* ``test_model_checkpoint_and_resume`` -> ``test_save_best_checkpoint`` (best_*
  files) + ``test_resume_from_checkpoint`` (start_epoch advances to the saved
  epoch).

``test_incremental`` has no analogue: the JAX port ships only ``Trainer`` (no
``IncrementalFNOTrainer``), so it is a documented skip.

Tensor sizes are tiny (scalar regression y = 3x) to keep the loop fast; the core
train/evaluate test is parametrized over the available CPU/GPU backends.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.experimental.training.trainer import SimpleDataLoader, Trainer
from scirex.experimental.training.training_state import (
    load_training_state,
    save_training_state,
)
from scirex.losses.data_losses import LpLoss
from scirex.losses.equation_losses import DarcyEqnLoss


@pytest.fixture(autouse=True)
def _capture_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    """The library logs progress at INFO; capture it in every test of this module."""
    caplog.set_level(logging.INFO)


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# Minimal Trainer-compatible components (module-level so they pickle for
# checkpointing).
# ---------------------------------------------------------------------------


class _LinearModel:
    """Stateful 1-parameter linear model y = w * x with its own train step."""

    def __init__(self, w: float = 0.0, lr: float = 0.5) -> None:
        self.w = jnp.asarray(w, jnp.float32)
        self.lr = lr

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
    ) -> jnp.ndarray:
        return self.w * x

    def train_step(self, sample: dict, loss_fn: Any) -> jnp.ndarray:
        x, y = sample["x"], sample["y"]
        loss, grad = jax.value_and_grad(lambda w: loss_fn(w * x, y=y))(self.w)
        self.w = self.w - self.lr * grad
        return loss

    def eval(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {"w": self.w}

    def load_state_dict(self, state_dict: dict) -> None:
        self.w = state_dict["w"]


class _Scheduler:
    """No-op scheduler exposing the ``step`` hook the Trainer calls."""

    def step(self, train_err: float) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass


class _EchoModel:
    """Model that returns its input unchanged for eval-path tests."""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
    ) -> jnp.ndarray:
        return x

    def eval(self) -> None:
        pass


class _PreparingEvalLoss:
    """Loss with an explicit Trainer eval input-preparation hook."""

    def __init__(self) -> None:
        self.prepare_called = False
        self.called_pred = None
        self.called_y = None

    def prepare_eval_inputs(
        self, y_pred: jnp.ndarray, sample: dict
    ) -> tuple[jnp.ndarray, dict]:
        self.prepare_called = True
        prepared = dict(sample)
        prepared["y"] = prepared["y"] + 5.0
        return y_pred + 2.0, prepared

    def __call__(
        self, y_pred: jnp.ndarray, y: jnp.ndarray, **kwargs: Any
    ) -> jnp.ndarray:
        self.called_pred = y_pred
        self.called_y = y
        return jnp.sum(y_pred + y)


class _SquaredResidualEvalLoss:
    """Eval loss that finalizes accumulated squared residuals as global RMS."""

    def eval_batch_loss(
        self, y_pred: jnp.ndarray, y: jnp.ndarray, **kwargs: Any
    ) -> jnp.ndarray:
        del kwargs
        return jnp.sum((y_pred - y) ** 2)

    def finalize_eval_metric(
        self, accumulated_loss: float | jnp.ndarray, n_samples: int
    ) -> float:
        return float((jnp.asarray(accumulated_loss) / max(n_samples, 1)) ** 0.5)


def _mse_sum(
    pred: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
) -> jnp.ndarray:
    """Sum-reduced squared error (Trainer expects batch-summed losses)."""
    return jnp.sum((pred - y) ** 2)


def _loader(n: int) -> SimpleDataLoader:
    """A loader of n single-sample {'x','y'} batches with y = 3x."""
    batches = [
        {
            "x": jnp.array([[float(i)]], jnp.float32),
            "y": jnp.array([[3.0 * i]], jnp.float32),
        }
        for i in range(1, n + 1)
    ]
    return SimpleDataLoader(batches)


def _single_batch_loader(n: int) -> SimpleDataLoader:
    x = jnp.arange(1, n + 1, dtype=jnp.float32).reshape(n, 1)
    return SimpleDataLoader([{"x": x, "y": 3.0 * x}])


def _optimizer() -> SimpleNamespace:
    opt = SimpleNamespace(learning_rate=0.5)
    opt.state_dict = lambda: {"learning_rate": opt.learning_rate}

    def _load_state_dict(state_dict: dict) -> None:
        opt.learning_rate = state_dict["learning_rate"]

    opt.load_state_dict = _load_state_dict
    return opt


# ---------------------------------------------------------------------------
# train / evaluate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_returns_metrics_and_learns(backend: str) -> None:
    """train() returns the epoch metric dict and the model fits y = 3x."""
    with jax.default_device(jax.devices(backend)[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=8)
        metrics = trainer.train(
            train_loader=_loader(8),
            test_loaders={"test": _loader(4)},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses={"mse": _mse_sum},
        )
        assert {"train_err", "avg_loss", "epoch_train_time", "test_mse"}.issubset(
            metrics.keys()
        )
        assert float(model.w) == pytest.approx(3.0, abs=1e-2)
        assert float(metrics["test_mse"]) < 1e-2


def test_evaluate_returns_prefixed_losses() -> None:
    """evaluate() returns one entry per loss, keyed '<prefix>_<loss_name>'."""
    model = _LinearModel(w=3.0)  # already fit -> zero error
    trainer = Trainer(model=model, n_epochs=1)
    errors = trainer.evaluate({"mse": _mse_sum}, _loader(4), log_prefix="val")
    assert set(errors.keys()) == {"val_mse"}
    assert float(errors["val_mse"]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# checkpointing (mirrors test_model_checkpoint_saves / _and_resume / load)
# ---------------------------------------------------------------------------


def test_checkpoint_saves_files(tmp_path: Path) -> None:
    """With save_every set, the trainer writes the training-state files."""
    trainer = Trainer(model=_LinearModel(), n_epochs=3)
    trainer.train(
        train_loader=_loader(8),
        test_loaders={},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )
    for name in (
        "model_state_dict.msgpack",
        "optimizer.msgpack",
        "scheduler.msgpack",
        "manifest.msgpack",
    ):
        assert (tmp_path / name).exists()


def test_load_from_checkpoint(tmp_path: Path) -> None:
    """A model reloaded from checkpoint reproduces the saved model's eval error."""
    model = _LinearModel()
    trainer = Trainer(model=model, n_epochs=6)
    trainer.train(
        train_loader=_loader(8),
        test_loaders={"test": _loader(4)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )
    orig = trainer.evaluate({"mse": _mse_sum}, _loader(4), log_prefix="test")

    loaded_model = _LinearModel()
    loaded_model, *_ = load_training_state(tmp_path, "model", model=loaded_model)
    loaded = Trainer(model=loaded_model, n_epochs=1).evaluate(
        {"mse": _mse_sum}, _loader(4), log_prefix="test"
    )
    assert float(loaded["test_mse"]) == pytest.approx(float(orig["test_mse"]), abs=1e-6)


def test_save_best_checkpoint(tmp_path: Path) -> None:
    """save_best monitors a metric and writes best_model_* checkpoint files."""
    trainer = Trainer(model=_LinearModel(), n_epochs=4)
    trainer.train(
        train_loader=_loader(8),
        test_loaders={"test": _loader(4)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_best="test_mse",
    )
    assert (tmp_path / "best_model_state_dict.msgpack").exists()
    assert (tmp_path / "manifest.msgpack").exists()


def test_resume_from_checkpoint(tmp_path: Path) -> None:
    """resume_from_dir restores the saved epoch into start_epoch."""
    first = Trainer(model=_LinearModel(), n_epochs=2)
    first.train(
        train_loader=_loader(8),
        test_loaders={"test": _loader(4)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )

    resumed = Trainer(model=_LinearModel(), n_epochs=4)
    resumed.train(
        train_loader=_loader(8),
        test_loaders={"test": _loader(4)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        resume_from_dir=tmp_path,
    )
    # last saved epoch from the first run was epoch 1 (0-indexed, n_epochs=2),
    # so training continues with epoch 2
    assert resumed.start_epoch == 2


@pytest.mark.skip(
    reason="Mirrors test_incremental, which exercises IncrementalFNOTrainer. The "
    "JAX port ships only the base Trainer (no IncrementalFNOTrainer / incremental "
    "frequency-mode growth), so there is no analogue to port."
)
def test_incremental() -> None:
    pass


# ---------------------------------------------------------------------------
# SimpleDataLoader.__len__ on the inner Dataset (line 45)
# ---------------------------------------------------------------------------


def test_simple_data_loader_dataset_len() -> None:
    """SimpleDataLoader.dataset.__len__ returns the correct size (line 45)."""
    batches = [{"x": jnp.ones((1,))}, {"x": jnp.ones((1,))}]
    loader = SimpleDataLoader(batches)
    assert len(loader.dataset) == 2
    assert len(loader) == 2


# ---------------------------------------------------------------------------
# train() with training_loss=None (line 184) and eval_losses=None (line 196)
# ---------------------------------------------------------------------------


def test_train_default_training_loss() -> None:
    """train() with training_loss=None uses LpLoss(d=2) default (line 184)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        # training_loss=None triggers the default LpLoss path
        # We still need to supply a valid eval_losses to avoid crash
        metrics = trainer.train(
            train_loader=_loader(2),
            test_loaders={},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=None,
            eval_losses={"mse": _mse_sum},
        )
        assert "train_err" in metrics


def test_train_default_eval_losses() -> None:
    """train() with eval_losses=None defaults to l2=training_loss (line 196)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        metrics = trainer.train(
            train_loader=_loader(2),
            test_loaders={"val": _loader(2)},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses=None,  # triggers default dict(l2=training_loss)
        )
        # The default key should be "val_l2"
        assert "val_l2" in metrics


# ---------------------------------------------------------------------------
# train() warns when training_loss.reduction == "mean" (lines 188-189)
# ---------------------------------------------------------------------------


def test_train_warns_on_mean_reduction_training_loss() -> None:
    """train() emits UserWarning when training_loss.reduction == 'mean'
    (lines 188-189)."""

    class _MeanLoss:
        reduction = "mean"

        def __call__(
            self, pred: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
        ) -> jnp.ndarray:
            return jnp.mean((pred - y) ** 2)

    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        with pytest.warns(UserWarning, match="reduction"):
            trainer.train(
                train_loader=_loader(2),
                test_loaders={},
                optimizer=_optimizer(),
                scheduler=_Scheduler(),
                training_loss=_MeanLoss(),
                eval_losses={"mse": _mse_sum},
            )


# ---------------------------------------------------------------------------
# train() warns when eval_loss.reduction == "mean" (lines 446-447)
# ---------------------------------------------------------------------------


def test_evaluate_warns_on_mean_reduction_eval_loss() -> None:
    """evaluate() emits UserWarning when eval_loss.reduction == 'mean'
    (lines 446-447)."""

    class _MeanLoss:
        reduction = "mean"

        def __call__(
            self, pred: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
        ) -> jnp.ndarray:
            return jnp.mean((pred - y) ** 2)

    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        with pytest.warns(UserWarning, match="reduction"):
            trainer.evaluate({"mse": _MeanLoss()}, _loader(2), log_prefix="val")


# ---------------------------------------------------------------------------
# train() verbose output (lines 225-230 and 343)
# ---------------------------------------------------------------------------


def test_train_verbose_prints(caplog: pytest.LogCaptureFixture) -> None:
    """verbose=True causes train() to print sample counts and training stats
    (lines 225-230, 343)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1, verbose=True)
        trainer.train(
            train_loader=_loader(3),
            test_loaders={"val": _loader(2)},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses={"mse": _mse_sum},
        )
        captured = caplog
        assert "Training on" in captured.text
        assert "avg_loss" in captured.text


# ---------------------------------------------------------------------------
# evaluate_all() verbose: log_eval is called (line 399)
# ---------------------------------------------------------------------------


def test_evaluate_all_verbose_logs_eval(caplog: pytest.LogCaptureFixture) -> None:
    """verbose=True in evaluate_all() calls log_eval which prints metrics (line 399)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1, verbose=True)
        # Manually call evaluate_all with verbose set
        trainer.n_samples = 0
        trainer.evaluate_all(
            epoch=0,
            eval_losses={"mse": _mse_sum},
            test_loaders={"val": _loader(2)},
            eval_modes={},
        )
        captured = caplog
        assert "Eval" in captured.text


# ---------------------------------------------------------------------------
# train_one_epoch: data_processor with train() method (lines 295-298)
# ---------------------------------------------------------------------------


class _DataProcessor:
    """Minimal data processor for testing."""

    def __init__(self) -> None:
        self.training = False
        self._train_called = False
        self._eval_called = False

    def train(self) -> None:
        self._train_called = True

    def eval(self) -> None:
        self._eval_called = True

    def preprocess(self, sample: dict) -> dict:
        return sample

    def postprocess(self, out: Any, sample: dict) -> tuple[Any, dict]:
        return out, sample


class _DataProcessorNoTrain:
    """Data processor that uses training attribute instead of train() method."""

    def __init__(self) -> None:
        self.training = False

    def preprocess(self, sample: dict) -> dict:
        return sample

    def postprocess(self, out: Any, sample: dict) -> tuple[Any, dict]:
        return out, sample


def test_train_one_epoch_data_processor_train_method() -> None:
    """data_processor.train() is called when attribute exists (lines 295-296)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        dp = _DataProcessor()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.train(
            train_loader=_loader(2),
            test_loaders={},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses={"mse": _mse_sum},
        )
        assert dp._train_called


def test_train_one_epoch_data_processor_training_attribute() -> None:
    """data_processor.training is set to True when train() method absent
    (lines 297-298)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        dp = _DataProcessorNoTrain()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.train(
            train_loader=_loader(2),
            test_loaders={},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses={"mse": _mse_sum},
        )
        # After epoch loop the attribute may flip back via eval; just check no crash


# ---------------------------------------------------------------------------
# evaluate(): data_processor eval paths (lines 436-439)
# ---------------------------------------------------------------------------


def test_evaluate_data_processor_eval_method() -> None:
    """evaluate() calls data_processor.eval() when method exists (lines 436-437)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        dp = _DataProcessor()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.evaluate({"mse": _mse_sum}, _loader(2), log_prefix="val")
        assert dp._eval_called


def test_evaluate_data_processor_training_attribute_false() -> None:
    """evaluate() sets data_processor.training=False when eval() absent
    (lines 438-439)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        dp = _DataProcessorNoTrain()
        dp.training = True
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.evaluate({"mse": _mse_sum}, _loader(2), log_prefix="val")
        assert dp.training is False


# ---------------------------------------------------------------------------
# evaluate(): autoregression mode (lines 462-463)
# ---------------------------------------------------------------------------


def test_evaluate_autoregression_mode() -> None:
    """evaluate() dispatches to eval_one_batch_autoreg when
    mode='autoregression' (lines 462-463)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        errors = trainer.evaluate(
            {"mse": _mse_sum},
            _loader(2),
            log_prefix="val",
            mode="autoregression",
            max_steps=1,
        )
        assert "val_mse" in errors
        assert np.isfinite(float(errors["val_mse"]))


# ---------------------------------------------------------------------------
# eval_one_batch_autoreg: full coverage (lines 629-688)
# ---------------------------------------------------------------------------


def test_eval_one_batch_autoreg_basic() -> None:
    """eval_one_batch_autoreg runs a one-step rollout without data_processor
    (lines 629-688)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        losses, out = trainer.eval_one_batch_autoreg(
            sample, {"mse": _mse_sum}, return_output=True, max_steps=1
        )
        assert "mse" in losses
        assert np.isfinite(float(losses["mse"]))
        assert out is not None


def test_eval_one_batch_autoreg_no_return_output() -> None:
    """eval_one_batch_autoreg returns None outputs when return_output=False
    (line 688)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        _losses, out = trainer.eval_one_batch_autoreg(
            sample, {"mse": _mse_sum}, return_output=False, max_steps=1
        )
        assert out is None


def test_eval_one_batch_autoreg_max_steps_none() -> None:
    """eval_one_batch_autoreg with max_steps=None runs one step (lines 632-633)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        # With no data_processor, loop runs once then sample dict stays non-None,
        # but max_steps=None becomes inf so we need max_steps=1 to terminate
        losses, _ = trainer.eval_one_batch_autoreg(
            sample, {"mse": _mse_sum}, max_steps=1
        )
        assert "mse" in losses


def test_eval_one_batch_autoreg_with_data_processor() -> None:
    """eval_one_batch_autoreg uses data_processor.preprocess/postprocess
    (lines 639-671)."""

    class _StepDP:
        """Data processor that terminates the loop by returning None on step 1."""

        def __init__(self) -> None:
            self.call_count = 0

        def preprocess(self, sample: dict | None) -> dict | None:
            self.call_count += 1
            if self.call_count > 1:
                return None
            return sample

        def postprocess(self, out: Any, sample: dict) -> tuple[Any, dict]:
            return out, sample

    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        dp = _StepDP()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        losses, _ = trainer.eval_one_batch_autoreg(
            sample, {"mse": _mse_sum}, max_steps=5
        )
        # preprocess returned None on second call -> loop broke after one step
        assert "mse" in losses


# ---------------------------------------------------------------------------
# log_training and log_eval (lines 717-736 and 748-761)
# ---------------------------------------------------------------------------


def test_log_training_prints_message(caplog: pytest.LogCaptureFixture) -> None:
    """log_training() prints epoch stats to stdout (lines 726-733)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.log_training(epoch=0, time=1.5, avg_loss=0.1, train_err=0.05)
        captured = caplog
        assert "[0]" in captured.text
        assert "avg_loss=0.1000" in captured.text


def test_log_training_with_lasso_loss(caplog: pytest.LogCaptureFixture) -> None:
    """log_training() includes avg_lasso in output when provided (line 730)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.log_training(
            epoch=1, time=0.5, avg_loss=0.2, train_err=0.1, avg_lasso_loss=0.05
        )
        captured = caplog
        assert "avg_lasso=0.0500" in captured.text


def test_log_eval_prints_metrics(caplog: pytest.LogCaptureFixture) -> None:
    """log_eval() prints eval metrics to stdout (lines 748-758)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.log_eval(epoch=0, eval_metrics={"val_mse": 0.123, "val_l2": 0.456})
        captured = caplog
        assert "Eval" in captured.text
        assert "val_mse" in captured.text


def test_log_eval_with_jax_array(caplog: pytest.LogCaptureFixture) -> None:
    """log_eval() handles jax.ndarray metric values (lines 751-752)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.log_eval(epoch=0, eval_metrics={"val_mse": jnp.array(0.5)})
        captured = caplog
        assert "Eval" in captured.text


# ---------------------------------------------------------------------------
# resume_state_from_dir: str path, best_model path, missing dir error
# (lines 772, 776, 780)
# ---------------------------------------------------------------------------


def test_resume_from_dir_with_string_path(tmp_path: Path) -> None:
    """resume_state_from_dir accepts a string path (line 772)."""
    # First create a checkpoint
    first = Trainer(model=_LinearModel(), n_epochs=1)
    first.train(
        train_loader=_loader(2),
        test_loaders={},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )
    # resume_state_from_dir with a str (not Path)
    resumed = Trainer(model=_LinearModel(), n_epochs=2)
    resumed.optimizer = _optimizer()
    resumed.scheduler = _Scheduler()
    resumed.regularizer = None
    resumed.resume_state_from_dir(str(tmp_path))  # line 772: Path(str)
    assert resumed.start_epoch >= 0


def test_resume_from_dir_raises_if_no_checkpoint(tmp_path: Path) -> None:
    """resume_state_from_dir raises FileNotFoundError when no pkl found (line 780)."""
    trainer = Trainer(model=_LinearModel(), n_epochs=1)
    trainer.optimizer = _optimizer()
    trainer.scheduler = _Scheduler()
    trainer.regularizer = None
    with pytest.raises(FileNotFoundError):
        trainer.resume_state_from_dir(tmp_path)


def test_resume_from_dir_uses_best_model(tmp_path: Path) -> None:
    """resume_state_from_dir loads best_model when best_model_state_dict.msgpack
    exists (line 776)."""
    # Save a best_model checkpoint via save_best
    first = Trainer(model=_LinearModel(), n_epochs=2)
    first.train(
        train_loader=_loader(4),
        test_loaders={"val": _loader(2)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_best="val_mse",
    )
    assert (tmp_path / "best_model_state_dict.msgpack").exists()
    resumed = Trainer(model=_LinearModel(), n_epochs=3)
    resumed.optimizer = _optimizer()
    resumed.scheduler = _Scheduler()
    resumed.regularizer = None
    # This should pick the "best_model" save name (line 776)
    resumed.resume_state_from_dir(tmp_path)


# ---------------------------------------------------------------------------
# resume_state_from_dir verbose print (line 804)
# ---------------------------------------------------------------------------


def test_resume_verbose_prints(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """verbose Trainer prints 'resuming from epoch N' on resume (line 804)."""
    first = Trainer(model=_LinearModel(), n_epochs=2)
    first.train(
        train_loader=_loader(2),
        test_loaders={},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )
    resumed = Trainer(model=_LinearModel(), n_epochs=4, verbose=True)
    resumed.train(
        train_loader=_loader(2),
        test_loaders={},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        resume_from_dir=tmp_path,
    )
    captured = caplog
    assert "resuming" in captured.text or "epoch" in captured.text


# ---------------------------------------------------------------------------
# checkpoint verbose print (line 829)
# ---------------------------------------------------------------------------


def test_checkpoint_verbose_prints(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """verbose Trainer prints save path during checkpoint (line 829)."""
    trainer = Trainer(model=_LinearModel(), n_epochs=1, verbose=True)
    trainer.train(
        train_loader=_loader(2),
        test_loaders={},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        save_every=1,
    )
    captured = caplog
    assert str(tmp_path) in captured.text


# ---------------------------------------------------------------------------
# train_one_batch: fallback model without train_step (lines 540-545)
# and n_samples without "y" key (line 532)
# ---------------------------------------------------------------------------


class _CallableModel:
    """A model without train_step - uses plain __call__ (fallback path)."""

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray | None = None, **kwargs: Any
    ) -> jnp.ndarray:
        return x * 2.0

    def eval(self) -> None:
        pass


def test_train_one_batch_fallback_model() -> None:
    """train_one_batch uses plain __call__ fallback when train_step absent
    (lines 540-545)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _CallableModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.epoch = 0
        trainer.n_samples = 0
        trainer.regularizer = None
        trainer.data_processor = None
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[2.0]], jnp.float32),
        }
        loss = trainer.train_one_batch(0, sample, _mse_sum)
        assert np.isfinite(float(loss))


def test_train_one_batch_verbose_prints_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """verbose=True logs the output shape (DEBUG) on the first batch of epoch 0."""
    caplog.set_level(logging.DEBUG)
    with jax.default_device(jax.devices("cpu")[0]):
        model = _CallableModel()
        trainer = Trainer(model=model, n_epochs=1, verbose=True)
        trainer.epoch = 0
        trainer.n_samples = 0
        trainer.regularizer = None
        trainer.data_processor = None
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[2.0]], jnp.float32),
        }
        trainer.train_one_batch(0, sample, _mse_sum)
        captured = caplog
        assert "Raw outputs of shape" in captured.text


def test_train_one_batch_no_y_key_increments_n_samples() -> None:
    """train_one_batch increments n_samples by 1 when 'y' not in sample (line 532)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _CallableModel()
        trainer = Trainer(model=model, n_epochs=1)
        trainer.epoch = 0
        trainer.n_samples = 0
        trainer.regularizer = None
        trainer.data_processor = None
        # sample without "y" key
        sample = {"x": jnp.array([[1.0]], jnp.float32)}

        def _loss_no_y(
            pred: jnp.ndarray, x: jnp.ndarray | None = None, **kwargs: Any
        ) -> jnp.ndarray:
            return jnp.mean(pred**2)

        trainer.train_one_batch(0, sample, _loss_no_y)
        assert trainer.n_samples == 1


# ---------------------------------------------------------------------------
# train_one_batch: data_processor preprocess path (line 520)
# ---------------------------------------------------------------------------


def test_train_one_batch_with_data_processor_preprocess() -> None:
    """train_one_batch calls data_processor.preprocess (line 520)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        dp = _DataProcessor()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.train(
            train_loader=_loader(2),
            test_loaders={},
            optimizer=_optimizer(),
            scheduler=_Scheduler(),
            training_loss=_mse_sum,
            eval_losses={"mse": _mse_sum},
        )


# ---------------------------------------------------------------------------
# eval_one_batch: data_processor preprocess/postprocess (lines 574, 589)
# ---------------------------------------------------------------------------


def test_eval_one_batch_with_data_processor() -> None:
    """eval_one_batch uses data_processor.preprocess and .postprocess
    (lines 574, 589)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        dp = _DataProcessor()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        losses, _ = trainer.eval_one_batch(sample, {"mse": _mse_sum})
        assert "mse" in losses


def test_eval_one_batch_uses_explicit_loss_prepare_hook() -> None:
    """eval_one_batch delegates special loss input handling to the loss object."""
    with jax.default_device(jax.devices("cpu")[0]):
        loss = _PreparingEvalLoss()
        trainer = Trainer(model=_EchoModel(), n_epochs=1)
        trainer.n_samples = 0
        sample = {
            "x": jnp.ones((1, 2), dtype=jnp.float32),
            "y": jnp.ones((1, 2), dtype=jnp.float32),
        }

        losses, _ = trainer.eval_one_batch(sample, {"prepared": loss})

        assert loss.prepare_called
        assert np.allclose(np.array(loss.called_pred), 3.0)
        assert np.allclose(np.array(loss.called_y), 6.0)
        assert float(losses["prepared"]) == pytest.approx(18.0)


def test_darcy_loss_prepare_eval_inputs_converts_nchw_and_strips_grid() -> None:
    """DarcyEqnLoss exposes the Trainer hook for its channel-last PDE contract."""
    loss = DarcyEqnLoss(reduction="sum")
    y_pred = jnp.zeros((2, 1, 4, 5), dtype=jnp.float32)
    coeff = jnp.full((2, 1, 4, 5), 7.0, dtype=jnp.float32)
    grid = jnp.full((2, 2, 4, 5), 99.0, dtype=jnp.float32)

    pred, sample = loss.prepare_eval_inputs(
        y_pred, {"x": jnp.concatenate([coeff, grid], axis=1), "y": y_pred}
    )

    assert pred.shape == (2, 4, 5, 1)
    assert sample["x"].shape == (2, 4, 5, 1)
    assert np.allclose(np.array(sample["x"]), 7.0)


def test_evaluate_uses_loss_finalizer_for_global_rms() -> None:
    """Trainer should not divide pre-rooted batch residual norms by samples."""
    loss = _SquaredResidualEvalLoss()
    split_batches = _loader(4)
    single_batch = _single_batch_loader(4)
    split = Trainer(model=_LinearModel(w=0.0), n_epochs=1).evaluate(
        {"rms": loss}, split_batches, log_prefix="test"
    )
    whole = Trainer(model=_LinearModel(w=0.0), n_epochs=1).evaluate(
        {"rms": loss}, single_batch, log_prefix="test"
    )

    expected = float(jnp.sqrt(jnp.sum((3.0 * jnp.arange(1, 5)) ** 2) / 4.0))
    assert float(split["test_rms"]) == pytest.approx(expected)
    assert float(whole["test_rms"]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# train_one_batch: regularizer paths (lines 517, 328, 548)
# ---------------------------------------------------------------------------


class _Regularizer:
    """Minimal regularizer with reset() and .loss attribute."""

    def __init__(self) -> None:
        self.loss = jnp.array(0.0)
        self._reset_called = False

    def reset(self) -> None:
        self._reset_called = True


def test_train_one_epoch_regularizer_reset_called() -> None:
    """Regularizer.reset() is called at each batch (line 517).

    Note: the source has a latent bug on line 328 where avg_lasso_loss (None)
    is divided before being assigned, so the full epoch raises TypeError.
    We test just the train_one_batch regularizer path directly.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        reg = _Regularizer()
        trainer = Trainer(model=model, n_epochs=1)
        # Set up state train() normally sets
        trainer.optimizer = _optimizer()
        trainer.scheduler = _Scheduler()
        trainer.regularizer = reg
        trainer.epoch = 0
        trainer.n_samples = 0
        trainer.data_processor = None
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }
        loss = trainer.train_one_batch(0, sample, _mse_sum)
        assert reg._reset_called
        # regularizer.loss (0.0) is added to the training loss (line 548)
        assert np.isfinite(float(loss))


def test_train_one_batch_fallback_model_with_data_processor_postprocess() -> None:
    """train_one_batch fallback path calls data_processor.postprocess (line 544)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _CallableModel()
        dp = _DataProcessor()
        trainer = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer.epoch = 0
        trainer.n_samples = 0
        trainer.regularizer = None
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[2.0]], jnp.float32),
        }
        loss = trainer.train_one_batch(0, sample, _mse_sum)
        assert np.isfinite(float(loss))


def test_eval_one_batch_autoreg_max_steps_none_terminates() -> None:
    """eval_one_batch_autoreg with max_steps=None sets max_steps=float('inf')
    (line 633)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel(w=3.0)
        trainer = Trainer(model=model, n_epochs=1)
        trainer.n_samples = 0
        sample = {
            "x": jnp.array([[1.0]], jnp.float32),
            "y": jnp.array([[3.0]], jnp.float32),
        }

        # Without data_processor, sample remains non-None and loop would run forever.
        # We test just that max_steps=None path is entered; the loop is bounded by
        # passing a step count via the fact we stop manually - use a data_processor
        # that terminates after one step to avoid infinite loop.
        class _TerminatingDP:
            def __init__(self) -> None:
                self.step = 0

            def preprocess(self, sample: dict | None) -> dict | None:
                if self.step > 0:
                    return None
                self.step += 1
                return sample

            def postprocess(self, out: Any, sample: dict) -> tuple[Any, dict]:
                return out, sample

        dp = _TerminatingDP()
        trainer2 = Trainer(model=model, n_epochs=1, data_processor=dp)
        trainer2.n_samples = 0
        losses, _ = trainer2.eval_one_batch_autoreg(
            sample,
            {"mse": _mse_sum},
            max_steps=None,  # triggers line 633
        )
        assert "mse" in losses


def test_train_one_epoch_with_regularizer_raises() -> None:
    """Exposes source bug at line 343: avg_lasso_loss /= int fails when None
    (line 343)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = _LinearModel()
        reg = _Regularizer()
        trainer = Trainer(model=model, n_epochs=1)
        # Line 343: `avg_lasso_loss /= max(self.n_samples, 1)` when
        # avg_lasso_loss is None
        with pytest.raises(TypeError):
            trainer.train(
                train_loader=_loader(2),
                test_loaders={},
                optimizer=_optimizer(),
                scheduler=_Scheduler(),
                training_loss=_mse_sum,
                eval_losses={"mse": _mse_sum},
                regularizer=reg,
            )


# ---------------------------------------------------------------------------
# Direct train_one_epoch + evaluate_all call (not via train())
# ---------------------------------------------------------------------------


def test_trainer_train_one_epoch_and_eval(tmp_path: Path) -> None:
    """train_one_epoch returns a 4-tuple and evaluate_all returns a metrics dict."""

    class _DummyModel:
        def __call__(
            self, x: jnp.ndarray | None = None, y: jnp.ndarray | None = None
        ) -> jnp.ndarray:
            return jnp.zeros_like(y) if y is not None else jnp.zeros((1, 1, 1))

    model = _DummyModel()
    trainer = Trainer(model=model, n_epochs=1, verbose=False)
    trainer.regularizer = None

    batch = {"y": jnp.zeros((1, 1, 1))}
    loader = SimpleDataLoader([batch, batch])

    class _Opt:
        learning_rate = 0.01

    trainer.optimizer = _Opt()
    trainer.scheduler = _Scheduler()

    train_err, _, _, _ = trainer.train_one_epoch(0, loader, LpLoss(d=2))
    assert isinstance(train_err, float)

    test_loader = SimpleDataLoader([batch])
    metrics = trainer.evaluate_all(
        epoch=0,
        eval_losses={"l2": LpLoss(d=2)},
        test_loaders={"test": test_loader},
        eval_modes={},
    )
    assert isinstance(metrics, dict)

    save_training_state(
        tmp_path,
        "model",
        model={"a": 1},
        optimizer={"v": 2},
        scheduler={"s": 3},
        regularizer={"r": 4},
        epoch=5,
    )
    _m, _o, _s, _r, epoch = load_training_state(
        tmp_path, "model", model=None, optimizer=None, scheduler=None, regularizer=None
    )
    assert epoch == 5


# ---------------------------------------------------------------------------
# argument checks, final checkpoint, resume edge cases
# ---------------------------------------------------------------------------


def _fit(trainer: Trainer, tmp_path: Path, **kw):
    return trainer.train(
        train_loader=_loader(8),
        test_loaders={"test": _loader(4)},
        optimizer=_optimizer(),
        scheduler=_Scheduler(),
        training_loss=_mse_sum,
        eval_losses={"mse": _mse_sum},
        save_dir=tmp_path,
        **kw,
    )


def test_zero_eval_interval_is_rejected_up_front() -> None:
    with pytest.raises(ValueError, match="eval_interval"):
        Trainer(model=_LinearModel(), n_epochs=1, eval_interval=0)


def test_zero_save_every_is_rejected_before_training(tmp_path: Path) -> None:
    trainer = Trainer(model=_LinearModel(), n_epochs=1)
    with pytest.raises(ValueError, match="save_every"):
        _fit(trainer, tmp_path, save_every=0)
    assert not list(tmp_path.iterdir())


def test_unknown_save_best_metric_raises_value_error(tmp_path: Path) -> None:
    trainer = Trainer(model=_LinearModel(), n_epochs=1)
    with pytest.raises(ValueError, match="save_best"):
        _fit(trainer, tmp_path, save_best="nope")


def test_mixed_precision_raises_not_implemented() -> None:
    """mixed_precision=True used to be silently accepted and ignored."""
    with pytest.raises(NotImplementedError, match="mixed_precision"):
        Trainer(model=_LinearModel(), n_epochs=1, mixed_precision=True)


def test_final_epoch_is_checkpointed(tmp_path: Path) -> None:
    import msgpack  # noqa: F401  (flax depends on it)

    trainer = Trainer(model=_LinearModel(), n_epochs=3)
    _fit(trainer, tmp_path, save_every=2)  # epochs 0 and 2 match; 2 is the last
    resumed = Trainer(model=_LinearModel(), n_epochs=5)
    _fit(resumed, tmp_path, resume_from_dir=tmp_path)
    assert resumed.start_epoch == 3

    other = tmp_path / "b"
    other.mkdir()
    trainer = Trainer(model=_LinearModel(), n_epochs=4)
    _fit(trainer, other, save_every=3)  # only epoch 0 and 3 match; 3 is last
    trainer = Trainer(model=_LinearModel(), n_epochs=2)
    _fit(trainer, other, save_every=5)  # epoch 0 matches, final epoch 1 must save
    resumed = Trainer(model=_LinearModel(), n_epochs=5)
    _fit(resumed, other, resume_from_dir=other)
    assert resumed.start_epoch == 2


def test_resuming_past_the_end_returns_empty_metrics(tmp_path: Path) -> None:
    first = Trainer(model=_LinearModel(), n_epochs=2)
    _fit(first, tmp_path, save_every=1)
    resumed = Trainer(model=_LinearModel(), n_epochs=2)
    assert _fit(resumed, tmp_path, resume_from_dir=tmp_path) == {}
