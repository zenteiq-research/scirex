# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the ``scirex.losses`` public API: exports, laziness, deprecated names."""

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scirex
import scirex.losses

_SRC_DIR = Path(scirex.__file__).resolve().parent.parent


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter that has imported nothing yet."""
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR), "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_all_has_the_stable_names() -> None:
    assert len(scirex.losses.__all__) == 8
    assert len(set(scirex.losses.__all__)) == 8
    assert set(scirex.losses.__all__) == {
        "AxisSpec",
        "DarcyEqnLoss",
        "DiffusionTerm",
        "FiniteDiff",
        "H1Loss",
        "LpLoss",
        "Q4Galerkin",
        "SourceTerm",
    }
    for experimental in (
        "ADDiff",
        "PDEResidualLoss",
        "T3Galerkin",
        "T6Galerkin",
        "DarcyHybridLossGINO",
        "NavierStokesEqnLoss",
        "BatchedVariableCoefficientGalerkinResidualLossT6",
        "darcy_piecewise_coefficient",
    ):
        assert experimental not in scirex.losses.__all__


def test_experimental_losses_are_exported_by_the_experimental_package() -> None:
    import scirex.experimental.losses as exp

    assert sorted(exp.__all__) == sorted(exp.EXPERIMENTAL_EXPORTS)
    assert not set(exp.__all__) & set(scirex.losses.__all__)
    for name in exp.__all__:
        assert getattr(exp, name) is not None, name


@pytest.mark.parametrize("name", scirex.losses.__all__)
def test_every_exported_name_resolves(name: str) -> None:
    assert getattr(scirex.losses, name) is not None


@pytest.mark.parametrize("name", scirex.losses.__all__)
def test_exported_names_have_a_real_docstring(name: str) -> None:
    doc = inspect.getdoc(getattr(scirex.losses, name)) or ""
    assert doc, f"{name} has no docstring"
    assert not doc.startswith(f"{name}("), f"{name} only has the auto-generated doc"


def test_lazy_table_matches_all_and_dir() -> None:
    assert set(scirex.losses._LAZY_EXPORTS) == set(scirex.losses.__all__)
    assert set(scirex.losses.__all__) <= set(dir(scirex.losses))
    assert not set(scirex.losses._DEPRECATED_EXPORTS) & set(dir(scirex.losses))


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        scirex.losses.does_not_exist


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("DarcyHybridLossGINO", "scirex.experimental.losses.darcy_hybrid_loss"),
        ("NavierStokesEqnLoss", "scirex.losses.equation_losses"),
        (
            "BatchedVariableCoefficientGalerkinResidualLossT6",
            "scirex.experimental.losses.galerkin_t6",
        ),
    ],
)
def test_demoted_names_still_resolve_with_a_deprecation_warning(
    name: str, module: str
) -> None:
    with pytest.warns(DeprecationWarning, match=f"scirex.experimental.losses.{name}"):
        obj = getattr(scirex.losses, name)
    assert obj is getattr(importlib.import_module(module), name)


def test_import_scirex_losses_imports_no_loss_module() -> None:
    result = _run_fresh(
        """
        import sys
        import scirex.losses

        assert "jax" not in sys.modules
        assert not any(m.startswith("scirex.losses.") for m in sys.modules)
        scirex.losses.LpLoss
        assert "scirex.losses.data_losses" in sys.modules
        assert "scirex.losses.galerkin" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr
