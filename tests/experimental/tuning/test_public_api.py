# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The package-level exports of ``scirex.experimental.tuning`` and the missing-extra message."""

import importlib

import pytest

import scirex.experimental.tuning
from tests.public_api_helpers import (
    assert_all_is_well_formed,
    assert_exports_resolve,
    assert_run_ok,
)


def test_all_is_the_decided_surface() -> None:
    assert_all_is_well_formed(scirex.experimental.tuning)
    assert scirex.experimental.tuning.__all__ == [
        "TrialContext",
        "suggest_params",
        "tune",
    ]


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        scirex.experimental.tuning.does_not_exist


def test_every_export_resolves_and_is_documented_when_the_extra_is_installed() -> None:
    pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    assert_exports_resolve(scirex.experimental.tuning)
    from scirex.experimental.tuning.tuner import TrialContext, suggest_params, tune

    assert scirex.experimental.tuning.TrialContext is TrialContext
    assert scirex.experimental.tuning.suggest_params is suggest_params
    assert scirex.experimental.tuning.tune is tune


def test_public_trial_context_methods_are_documented() -> None:
    pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    for method in (
        "number",
        "report",
        "log_metrics",
        "log_params",
        "set_tag",
        "log_artifact",
    ):
        assert getattr(scirex.experimental.tuning.TrialContext, method).__doc__, method


@pytest.mark.parametrize("name", ["TrialContext", "suggest_params", "tune"])
def test_missing_extra_gives_an_actionable_import_error(name: str) -> None:
    out = assert_run_ok(
        f"""
        import sys
        sys.modules["mlflow"] = None   # `import mlflow` now fails
        sys.modules["optuna"] = None
        import scirex.experimental.tuning           # importing the package still works
        try:
            getattr(scirex.experimental.tuning, {name!r})
        except ImportError as err:
            print(err)
            assert isinstance(err.__cause__, ModuleNotFoundError)
        else:
            raise SystemExit("no ImportError")
        try:
            exec("from scirex.experimental.tuning import {name}")
        except ImportError as err:
            assert "hpo" in str(err)
        else:
            raise SystemExit("no ImportError from the from-import")
        """
    )
    assert 'pip install "scirex[hpo]"' in out
    assert "'mlflow'" in out or "'optuna'" in out
    assert f"scirex.experimental.tuning.{name}" in out


def test_import_error_for_an_unrelated_missing_module_is_not_disguised() -> None:
    assert_run_ok(
        """
        import importlib
        import scirex.experimental.tuning
        original = importlib.import_module

        def fake(name, *args, **kwargs):
            if name == "scirex.experimental.tuning.tuner":
                raise ModuleNotFoundError("No module named 'foo'", name="foo")
            return original(name, *args, **kwargs)

        importlib.import_module = fake
        try:
            scirex.experimental.tuning.tune
        except ModuleNotFoundError as err:
            assert err.name == "foo"
            assert "hpo" not in str(err)
        else:
            raise SystemExit("no ModuleNotFoundError")
        """
    )


def test_importing_the_package_does_not_import_the_extra() -> None:
    assert_run_ok(
        """
        import sys
        import scirex.experimental.tuning
        assert "mlflow" not in sys.modules
        assert "optuna" not in sys.modules
        assert "scirex.experimental.tuning.tuner" not in sys.modules
        """
    )


def test_tuning_package_is_importable_without_the_extra_in_this_process() -> None:
    assert (
        importlib.import_module("scirex.experimental.tuning")
        is scirex.experimental.tuning
    )
