# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Optuna hyperparameter tuning for any model + training loop, recorded in MLflow.

The user supplies three things:

* `build_model(params) -> model`      -- constructs a fresh model for a trial
* `train_fn(model, params, ctx) -> float | None`
                                          -- their normal training loop; it calls
                                             `ctx.report(value, step)` once per
                                             epoch (or any unit of "resource")
* `search_space`                       -- a dict spec or a `(trial) -> dict`
                                          callable

Multi-fidelity: every `ctx.report` goes to Optuna's pruner (Hyperband by
default), so weak configurations are stopped after a few epochs and the budget
is spent on the promising ones. Each trial is an MLflow child run nested under
one parent run for the whole study.

Parallelism: run several worker *processes* that share one `storage` and
`study_name`. Threads are not supported, because MLflow's tracking URI and
active-run stack are process-global.
"""

from __future__ import annotations

import logging
import math
import socket
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import optuna
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

SearchSpace = Callable[[optuna.Trial], dict[str, Any]] | Mapping[str, Any]
BuildModel = Callable[[dict[str, Any]], Any]
TrainFn = Callable[[Any, dict[str, Any], "TrialContext"], "float | None"]


class TrialContext:
    """The handle a training function uses to talk to the tuner.

    The only required call is `report`, once per epoch/step.
    """

    def __init__(
        self,
        trial: optuna.Trial | optuna.trial.FixedTrial,
        metric_name: str,
        max_resource: int | None,
    ) -> None:
        self.trial = trial
        self.metric_name = metric_name
        self.max_resource = max_resource
        self.last_value: float | None = None
        self.last_step: int | None = None

    @property
    def number(self) -> int:
        """Index of this trial within its Optuna study."""
        return self.trial.number

    def report(self, value: float, step: int, **extra_metrics: float) -> None:
        """Record the objective at `step`.

        Raises `optuna.TrialPruned` if the pruner decides this trial should
        stop, or if `value` is not finite. Let that exception propagate.
        """
        value = float(value)
        if not math.isfinite(value):
            # A diverged run (NaN/inf loss) has nothing more to tell us.
            raise optuna.TrialPruned(f"non-finite {self.metric_name} at step {step}")

        mlflow.log_metrics(
            {
                self.metric_name: value,
                **{k: float(v) for k, v in extra_metrics.items()},
            },
            step=step,
        )
        self.trial.report(value, step)
        self.last_value, self.last_step = value, step
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"pruned at step {step} ({self.metric_name}={value:.5g})"
            )

    def log_metrics(
        self, metrics: Mapping[str, float], step: int | None = None
    ) -> None:
        """Log extra metrics to this trial's MLflow run (the pruner is not told).

        Parameters
        ----------
        metrics : mapping of str to float
            Metric names and values.
        step : int, optional
            Step to attach the values to.
        """
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log additional parameters to this trial's MLflow run."""
        mlflow.log_params(dict(params))

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on this trial's MLflow run."""
        mlflow.set_tag(key, value)

    def log_artifact(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        """Upload a local file to this trial's MLflow run.

        Parameters
        ----------
        local_path : str or Path
            File to upload.
        artifact_path : str, optional
            Sub-directory of the run's artifact store to put it in.
        """
        mlflow.log_artifact(str(local_path), artifact_path)


def suggest_params(
    trial: optuna.Trial | optuna.trial.FixedTrial, search_space: SearchSpace
) -> dict[str, Any]:
    """Turn a search-space spec into concrete params for one trial.

    Dict spec: each value is either a constant (passed through unchanged) or a
    dict with a `type` of `float`, `int` or `categorical`::

        {
            "lr":         {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "hidden":     {"type": "int", "low": 16, "high": 256, "log": True},
            "optimizer":  {"type": "categorical", "choices": ["adam", "sgd"]},
            "epochs":     30,
        }

    For conditional spaces, pass a callable `(trial) -> dict` instead.
    """
    if callable(search_space):
        return dict(search_space(trial))  # type: ignore[arg-type]

    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, Mapping) or "type" not in spec:
            params[name] = spec
            continue
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step"),
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step", 1),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unknown search-space type {kind!r} for {name!r}")
    return params


def tune(
    build_model: BuildModel,
    train_fn: TrainFn,
    search_space: SearchSpace,
    *,
    n_trials: int | None = 50,
    timeout: float | None = None,
    direction: str = "minimize",
    metric_name: str = "val_loss",
    max_resource: int | None = None,
    min_resource: int = 1,
    reduction_factor: int = 3,
    pruner: optuna.pruners.BasePruner | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,
    experiment_name: str = "hpo",
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    seed: int | None = None,
    catch: Sequence[type[Exception]] = (),
    run_tags: Mapping[str, Any] | None = None,
    enabled: bool = True,
    default_params: Mapping[str, Any] | None = None,
) -> optuna.Study:
    """Run an Optuna study over `train_fn` and record everything in MLflow.

    `max_resource` is the most steps (usually epochs) a trial may run; it sets
    Hyperband's budget. `storage` (e.g. `"sqlite:///optuna.db"`) makes the
    study resumable: calling `tune` again with the same `study_name` adds
    trials to it. Returns the `optuna.Study`.

    With `enabled=False` there is no search: `train_fn` runs once, with no
    pruning, on `default_params` (every tunable key in `search_space` must be
    given; constants pass through). It logs one plain MLflow run and returns a
    one-trial in-memory study, so `study.best_params` / `best_value` still
    work. The HPO-only arguments (pruner, sampler, storage, ...) are ignored.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        try:
            client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        except MlflowException:  # another worker created it first
            pass
    mlflow.set_experiment(experiment_name)

    if not enabled:
        return _train_once(
            build_model,
            train_fn,
            search_space,
            default_params or {},
            direction=direction,
            metric_name=metric_name,
            run_tags=run_tags,
        )

    if pruner is None:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource if max_resource is not None else "auto",
            reduction_factor=reduction_factor,
        )
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,  # type: ignore[arg-type]
        sampler=sampler,
        pruner=pruner,
        load_if_exists=study_name is not None,
    )

    parent = mlflow.start_run(run_name=f"study-{study.study_name}")
    parent_id = parent.info.run_id
    # Several workers may share one study (same storage + study_name); each gets
    # its own parent run, tagged with the host it ran on.
    mlflow.set_tags(
        {
            "optuna.study": study.study_name,
            "role": "study",
            "worker": socket.gethostname(),
            **(run_tags or {}),
        }
    )
    mlflow.log_params(
        {
            "direction": direction,
            "metric": metric_name,
            "n_trials": n_trials,
            "sampler": type(sampler).__name__,
            "pruner": type(pruner).__name__,
            "max_resource": max_resource,
            "min_resource": min_resource,
            "reduction_factor": reduction_factor,
        }
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, search_space)
        run = mlflow.start_run(
            run_name=f"trial-{trial.number:03d}", nested=True, parent_run_id=parent_id
        )
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        status = "FINISHED"
        # Bound before the try so the handlers below can read it even when the
        # failure (or a prune raised from build_model) happens before it exists.
        ctx = TrialContext(trial, metric_name, max_resource)
        try:
            mlflow.set_tags(
                {
                    "optuna.study": study.study_name,
                    "optuna.trial": trial.number,
                    "role": "trial",
                }
            )
            mlflow.log_params(params)

            model = build_model(params)
            result = train_fn(model, params, ctx)
            value = ctx.last_value if result is None else float(result)
            if value is None:
                raise RuntimeError(
                    "train_fn returned None and never called ctx.report()"
                )

            mlflow.log_metric(f"final_{metric_name}", value)
            mlflow.set_tags(
                {"optuna.state": "COMPLETE", "optuna.steps_run": ctx.last_step}
            )
            return value
        except optuna.TrialPruned as exc:
            status = "KILLED"  # shows up as "Killed" in the MLflow UI
            mlflow.set_tags(
                {
                    "optuna.state": "PRUNED",
                    "optuna.steps_run": ctx.last_step,
                    "optuna.prune_reason": str(exc),
                }
            )
            raise
        except BaseException as exc:
            status = "FAILED"
            mlflow.set_tags({"optuna.state": "FAIL", "error": repr(exc)[:5000]})
            raise
        finally:
            mlflow.end_run(status)

    def log_progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        states = [t.state for t in study.trials]
        metrics: dict[str, float] = {
            "trials_complete": states.count(optuna.trial.TrialState.COMPLETE),
            "trials_pruned": states.count(optuna.trial.TrialState.PRUNED),
            "trials_failed": states.count(optuna.trial.TrialState.FAIL),
        }
        try:
            metrics[f"best_{metric_name}"] = study.best_value
        except ValueError:  # no completed trial yet
            pass
        for key, val in metrics.items():
            client.log_metric(parent_id, key, val, step=trial.number)

    status = "FINISHED"
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=tuple(catch),
            callbacks=[log_progress],
        )
    except KeyboardInterrupt:
        status = "KILLED"  # still write the summary for what finished
    except BaseException:
        status = "FAILED"
        raise
    finally:
        _log_study_summary(study, metric_name)
        mlflow.end_run(status)
    return study


def _train_once(
    build_model: BuildModel,
    train_fn: TrainFn,
    search_space: SearchSpace,
    default_params: Mapping[str, Any],
    *,
    direction: str,
    metric_name: str,
    run_tags: Mapping[str, Any] | None,
) -> optuna.Study:
    """The `enabled=False` path: one ordinary training run, no Optuna search."""
    # A FixedTrial answers every suggest_* call from default_params (and raises
    # if one is missing); its report() is a no-op and it never prunes, so the
    # user's train_fn and ctx calls work unchanged.
    fixed = optuna.trial.FixedTrial(dict(default_params))
    params = suggest_params(fixed, search_space)

    with mlflow.start_run(run_name="train") as run:
        mlflow.set_tags({"role": "train", "hpo": "disabled", **(run_tags or {})})
        mlflow.log_params(params)
        ctx = TrialContext(fixed, metric_name, None)
        try:
            result = train_fn(build_model(params), params, ctx)
        except optuna.TrialPruned as exc:  # only raised for a NaN/inf metric here
            raise RuntimeError(f"training diverged: {exc}") from exc
        value = ctx.last_value if result is None else float(result)
        if value is None:
            raise RuntimeError("train_fn returned None and never called ctx.report()")
        mlflow.log_metric(f"final_{metric_name}", value)

    study = optuna.create_study(direction=direction)  # type: ignore[arg-type]
    study.add_trial(
        optuna.trial.create_trial(
            params={k: params[k] for k in fixed.distributions},
            distributions=fixed.distributions,
            value=value,
            user_attrs={"mlflow_run_id": run.info.run_id},
        )
    )
    return study


def _log_study_summary(study: optuna.Study, metric_name: str) -> None:
    """Best trial, a CSV of all trials, and Optuna's plots on the parent run."""
    try:
        best = study.best_trial
    except ValueError:
        best = None
    if best is not None:
        mlflow.log_params({f"best.{k}": v for k, v in best.params.items()})
        mlflow.log_metric(f"best_{metric_name}", best.value)  # type: ignore[arg-type]
        mlflow.set_tags(
            {
                "best_trial": best.number,
                "best_trial_run_id": best.user_attrs.get("mlflow_run_id", ""),
            }
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            study.trials_dataframe().to_csv(tmp_path / "trials.csv", index=False)
        except Exception as exc:
            logger.warning("could not write trials.csv: %r", exc)

        plots = {
            "optimization_history": optuna.visualization.plot_optimization_history,
            # shows where trials were pruned
            "intermediate_values": optuna.visualization.plot_intermediate_values,
            "param_importances": optuna.visualization.plot_param_importances,
            "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
        }
        for name, plot in plots.items():
            try:  # needs plotly, and some plots need >1 completed trial
                plot(study).write_html(tmp_path / f"{name}.html")
            except Exception as exc:
                logger.warning("skipped %s plot: %r", name, exc)

        if any(tmp_path.iterdir()):
            mlflow.log_artifacts(str(tmp_path), artifact_path="study")
