# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the MLflow-tracked Optuna tuner (``scirex.experimental.tuning``).

MLflow and Optuna both use SQLite under ``tmp_path``: no server, no network, no GPU.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pytest
from mlflow.tracking import MlflowClient

from scirex.experimental.tuning import TrialContext, suggest_params, tune

pytestmark = pytest.mark.unit

EXPERIMENT = "hpo-test"
SPACE = {
    "x": {"type": "float", "low": -3.0, "high": 3.0},
    "const": 7,
}


@pytest.fixture(autouse=True)
def _quiet_optuna() -> Iterator[None]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    while mlflow.active_run() is not None:  # a failing test must not leak a run
        mlflow.end_run()


@pytest.fixture
def mlflow_kwargs(tmp_path: Path) -> dict[str, str]:
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_location": str(tmp_path / "artifacts"),
        "experiment_name": EXPERIMENT,
    }


def build(params: dict[str, Any]) -> Any:
    return params["x"]


def train_quadratic(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
    """Two reports of (x - 1)^2, flat in the step so no pruner is ever tempted."""
    for step in (1, 2):
        ctx.report((x - 1.0) ** 2, step=step)


def runs_by_role(kwargs: dict[str, str]) -> dict[str | None, list[Any]]:
    client = MlflowClient(kwargs["tracking_uri"])
    exp = client.get_experiment_by_name(kwargs["experiment_name"])
    runs = client.search_runs([exp.experiment_id])
    roles = {}
    for run in runs:
        roles.setdefault(run.data.tags.get("role"), []).append(run)
    return roles


# --------------------------------------------------------------------------- #
# suggest_params
# --------------------------------------------------------------------------- #
class TestSuggestParams:
    def test_dict_spec_types_and_constants(self) -> None:
        fixed = optuna.trial.FixedTrial({"lr": 0.01, "hidden": 32, "opt": "adam"})
        space = {
            "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
            "hidden": {"type": "int", "low": 16, "high": 256, "log": True},
            "opt": {"type": "categorical", "choices": ["adam", "sgd"]},
            "epochs": 30,
        }
        assert suggest_params(fixed, space) == {
            "lr": 0.01,
            "hidden": 32,
            "opt": "adam",
            "epochs": 30,
        }

    def test_unknown_type_raises(self) -> None:
        fixed = optuna.trial.FixedTrial({})
        with pytest.raises(ValueError, match="unknown search-space type"):
            suggest_params(fixed, {"p": {"type": "bogus"}})

    def test_callable_space_supports_conditionals(self) -> None:
        def space(trial: Any) -> dict[str, Any]:
            kind = trial.suggest_categorical("kind", ["a", "b"])
            return (
                {"kind": kind, "depth": trial.suggest_int("depth", 1, 4)}
                if kind == "a"
                else {"kind": kind}
            )

        assert suggest_params(optuna.trial.FixedTrial({"kind": "b"}), space) == {
            "kind": "b"
        }
        assert suggest_params(
            optuna.trial.FixedTrial({"kind": "a", "depth": 3}), space
        ) == {"kind": "a", "depth": 3}


# --------------------------------------------------------------------------- #
# TrialContext
# --------------------------------------------------------------------------- #
class TestTrialContext:
    def test_non_finite_value_prunes(self) -> None:
        ctx = TrialContext(optuna.trial.FixedTrial({}), "val_loss", None)
        for bad in (math.nan, math.inf):
            with pytest.raises(optuna.TrialPruned, match="non-finite"):
                ctx.report(bad, step=1)
        assert ctx.last_value is None


# --------------------------------------------------------------------------- #
# tune(): the search
# --------------------------------------------------------------------------- #
class TestTune:
    def test_study_and_mlflow_runs(self, mlflow_kwargs: dict[str, str]) -> None:
        # NopPruner: this test is about the MLflow bookkeeping of completed
        # trials, so Hyperband must not prune any of them.
        study = tune(
            build,
            train_quadratic,
            SPACE,
            n_trials=4,
            seed=0,
            pruner=optuna.pruners.NopPruner(),
            **mlflow_kwargs,
        )

        assert len(study.trials) == 4
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        assert study.best_params.keys() == {"x"}  # constants are not Optuna parameters

        roles = runs_by_role(mlflow_kwargs)
        assert len(roles["study"]) == 1
        assert len(roles["trial"]) == 4
        parent = roles["study"][0]
        assert {r.data.tags["mlflow.parentRunId"] for r in roles["trial"]} == {
            parent.info.run_id
        }
        assert (
            parent.data.tags["best_trial_run_id"]
            == study.best_trial.user_attrs["mlflow_run_id"]
        )
        assert parent.data.metrics["best_val_loss"] == pytest.approx(study.best_value)
        for run in roles["trial"]:
            assert run.info.status == "FINISHED"
            assert run.data.params["const"] == "7"
            assert "final_val_loss" in run.data.metrics

    def test_study_artifacts_are_logged(self, mlflow_kwargs: dict[str, str]) -> None:
        tune(
            build,
            train_quadratic,
            SPACE,
            n_trials=3,
            seed=0,
            max_resource=2,
            **mlflow_kwargs,
        )
        parent = runs_by_role(mlflow_kwargs)["study"][0]
        files = {
            a.path
            for a in MlflowClient(mlflow_kwargs["tracking_uri"]).list_artifacts(
                parent.info.run_id, "study"
            )
        }
        assert "study/trials.csv" in files

    def test_pruned_trial_is_marked_killed(self, mlflow_kwargs: dict[str, str]) -> None:
        def always_bad(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            ctx.report(100.0, step=1)
            raise AssertionError(
                "unreachable: the pruner should have stopped the trial"
            )

        study = tune(
            build,
            always_bad,
            SPACE,
            n_trials=2,
            pruner=optuna.pruners.ThresholdPruner(upper=1.0),
            **mlflow_kwargs,
        )
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED] * 2
        for run in runs_by_role(mlflow_kwargs)["trial"]:
            assert run.info.status == "KILLED"
            assert run.data.tags["optuna.state"] == "PRUNED"

    def test_prune_before_training_does_not_crash(
        self, mlflow_kwargs: dict[str, str]
    ) -> None:
        """A TrialPruned raised from build_model must be handled like any prune."""

        def prune_in_build(params: dict[str, Any]) -> Any:
            raise optuna.TrialPruned("bad architecture")

        study = tune(
            prune_in_build, train_quadratic, SPACE, n_trials=1, **mlflow_kwargs
        )
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
        run = runs_by_role(mlflow_kwargs)["trial"][0]
        assert run.data.tags["optuna.prune_reason"] == "bad architecture"

    def test_failure_is_recorded_and_propagates(
        self, mlflow_kwargs: dict[str, str]
    ) -> None:
        def boom(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            tune(build, boom, SPACE, n_trials=3, **mlflow_kwargs)

        roles = runs_by_role(mlflow_kwargs)
        assert roles["study"][0].info.status == "FAILED"
        assert roles["trial"][0].info.status == "FAILED"
        assert "kaboom" in roles["trial"][0].data.tags["error"]

    def test_catch_lets_the_study_continue(self, mlflow_kwargs: dict[str, str]) -> None:
        def boom(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            raise RuntimeError("kaboom")

        study = tune(
            build, boom, SPACE, n_trials=3, catch=(RuntimeError,), **mlflow_kwargs
        )
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.FAIL] * 3

    def test_train_fn_without_report_or_return_fails(
        self, mlflow_kwargs: dict[str, str]
    ) -> None:
        study = tune(
            build,
            lambda x, p, c: None,
            SPACE,
            n_trials=1,
            catch=(RuntimeError,),
            **mlflow_kwargs,
        )
        assert study.trials[0].state == optuna.trial.TrialState.FAIL

    def test_return_value_overrides_last_report(
        self, mlflow_kwargs: dict[str, str]
    ) -> None:
        def train(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            ctx.report(5.0, step=1)
            return 0.5

        study = tune(build, train, SPACE, n_trials=1, **mlflow_kwargs)
        assert study.best_value == 0.5

    def test_resume_adds_trials_to_the_same_study(
        self, mlflow_kwargs: dict[str, str], tmp_path: Path
    ) -> None:
        common = {
            "storage": f"sqlite:///{tmp_path / 'optuna.db'}",
            "study_name": "resumable",
            "max_resource": 2,
            **mlflow_kwargs,
        }
        tune(build, train_quadratic, SPACE, n_trials=2, seed=0, **common)
        study = tune(build, train_quadratic, SPACE, n_trials=2, seed=0, **common)
        assert len(study.trials) == 4
        assert len(runs_by_role(mlflow_kwargs)["trial"]) == 4

    def test_maximize_direction(self, mlflow_kwargs: dict[str, str]) -> None:
        def train(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            ctx.report(x, step=1)

        study = tune(
            build,
            train,
            SPACE,
            n_trials=6,
            seed=0,
            direction="maximize",
            **mlflow_kwargs,
        )
        assert study.best_value == max(t.value for t in study.trials)


# --------------------------------------------------------------------------- #
# tune(enabled=False): one plain run
# --------------------------------------------------------------------------- #
class TestDisabled:
    def test_single_run_on_default_params(self, mlflow_kwargs: dict[str, str]) -> None:
        study = tune(
            build,
            train_quadratic,
            SPACE,
            enabled=False,
            default_params={"x": 2.0},
            **mlflow_kwargs,
        )

        assert len(study.trials) == 1
        assert study.best_params == {"x": 2.0}
        assert study.best_value == pytest.approx(1.0)
        roles = runs_by_role(mlflow_kwargs)
        assert set(roles) == {"train"}
        assert roles["train"][0].data.tags["hpo"] == "disabled"

    def test_missing_default_param_raises(self, mlflow_kwargs: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="x"):
            tune(
                build,
                train_quadratic,
                SPACE,
                enabled=False,
                default_params={},
                **mlflow_kwargs,
            )

    def test_divergence_raises_runtime_error(
        self, mlflow_kwargs: dict[str, str]
    ) -> None:
        def diverge(x: Any, params: dict[str, Any], ctx: TrialContext) -> None:
            ctx.report(math.nan, step=1)

        with pytest.raises(RuntimeError, match="diverged"):
            tune(
                build,
                diverge,
                SPACE,
                enabled=False,
                default_params={"x": 0.0},
                **mlflow_kwargs,
            )
