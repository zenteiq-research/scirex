# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the ``scirex.models`` public API: exports, laziness, registry."""

import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scirex
import scirex.models

_SRC_DIR = Path(scirex.__file__).resolve().parent.parent

_BUILTIN = ["fno", "fnoclassic2d", "giwo", "gino", "transolver", "wno"]

# Constructor kwargs per architecture, small enough to build instantly.
_CONFIG_MODELS = {
    "fno": {"data_channels": 1, "out_channels": 1, "hidden_channels": 4},
    "fnoclassic2d": {
        "data_channels": 1,
        "out_channels": 1,
        "n_modes1": 2,
        "n_modes2": 2,
        "hidden_channels": 4,
    },
    "giwo": {
        "data_channels": 3,
        "out_channels": 2,
        "gno_use_open3d": False,
        "wno_size": (6, 6, 6),
        "wno_hidden_channels": 8,
        "wno_n_layers": 2,
        "wno_level": 1,
    },
    "gino": {"data_channels": 1, "out_channels": 1, "gno_use_open3d": False},
    "transolver": {"n_layers": 1, "n_hidden": 8, "n_head": 2, "slice_num": 4},
    "wno": {
        "data_channels": 1,
        "out_channels": 1,
        "hidden_channels": 4,
        "n_layers": 1,
        "level": 1,
        "size": (8, 8),
    },
}


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter that has imported nothing yet."""
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR), "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_all_lists_the_stable_surface() -> None:
    assert scirex.models.__all__ == [
        "FNO",
        "GINO",
        "WNO",
        "BaseModel",
        "FNOClassic2d",
        "Transolver",
        "available_models",
        "get_model_jax",
    ]
    assert "GIWO" not in scirex.models.__all__


@pytest.mark.parametrize("name", scirex.models.__all__)
def test_every_exported_name_resolves(name: str) -> None:
    assert getattr(scirex.models, name) is not None


@pytest.mark.parametrize("name", scirex.models.__all__)
def test_exported_names_have_a_real_docstring(name: str) -> None:
    doc = inspect.getdoc(getattr(scirex.models, name)) or ""
    assert doc, f"{name} has no docstring"
    # Flax dataclasses get an auto-generated "Name(field: type = ...)" docstring.
    assert not doc.startswith(f"{name}("), f"{name} only has the auto-generated doc"


def test_lazy_table_matches_all_and_dir() -> None:
    assert set(scirex.models._LAZY_EXPORTS) == set(scirex.models.__all__)
    assert set(scirex.models.__all__) <= set(dir(scirex.models))


def test_giwo_is_not_exported_from_models() -> None:
    with pytest.raises(AttributeError):
        scirex.models.GIWO


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        scirex.models.does_not_exist


def test_available_models_lists_every_builtin_architecture() -> None:
    assert set(_BUILTIN) <= set(scirex.models.available_models())


def test_available_models_is_sorted() -> None:
    names = scirex.models.available_models()
    assert names == sorted(names)


def test_import_scirex_models_is_light() -> None:
    result = _run_fresh(
        """
        import sys
        import scirex.models

        heavy = {"jax", "flax", "jaxwt", "open3d", "torch", "tensorly"}
        assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))
        assert not any(m.startswith("scirex.models.") for m in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr


def test_available_models_in_fresh_process_imports_no_architecture() -> None:
    result = _run_fresh(
        """
        import sys
        from scirex.models import available_models

        names = available_models()
        expected = ["fno", "fnoclassic2d", "gino", "giwo", "transolver", "wno"]
        assert names == expected, names
        for module in ("fno", "fno_classic", "gino", "transolver", "wno"):
            assert f"scirex.models.{module}" not in sys.modules, module
        assert "scirex.experimental.giwo" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arch", _BUILTIN)
def test_get_model_jax_builds_every_builtin_in_a_fresh_process(arch: str) -> None:
    """No prior import of the architecture's class or module is needed."""
    code = f"""
        import sys
        from scirex.models.base import BaseModel, get_model_jax

        assert not BaseModel._models

        class Config(dict):
            model = dict(model_arch={arch.upper()!r}, **{_CONFIG_MODELS[arch]!r})

        model = get_model_jax(Config())
        assert type(model).__name__.lower() == {arch!r}, type(model)
        assert BaseModel._models[{arch!r}] is type(model)
        """
    result = _run_fresh(code)
    assert result.returncode == 0, result.stderr


def test_get_model_jax_from_top_level_without_importing_models_first() -> None:
    result = _run_fresh(
        """
        import scirex

        class Config(dict):
            model = {"model_arch": "transolver", "n_layers": 1, "n_hidden": 8,
                     "n_head": 2, "slice_num": 4}

        model = scirex.get_model_jax(Config())
        assert type(model) is scirex.Transolver
        """
    )
    assert result.returncode == 0, result.stderr


def test_get_model_jax_rejects_unknown_architecture() -> None:
    class Config(dict):
        model = {"model_arch": "no_such_arch"}

    with pytest.raises(ValueError, match="expected one of"):
        scirex.models.get_model_jax(Config())


def test_missing_jaxwt_raises_import_error_naming_it_only_when_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WNO needs jaxwt; without it ``import scirex.models`` still works."""
    for module in ("scirex.models.wno", "scirex.layers.blocks.wno_block"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.delitem(sys.modules, "scirex.layers.blocks.wavelet_conv", raising=False)
    monkeypatch.delitem(vars(scirex.models), "WNO", raising=False)
    monkeypatch.setitem(sys.modules, "jaxwt", None)

    assert scirex.models.FNO is not None  # other architectures are unaffected
    with pytest.raises(ImportError, match=r"jaxwt.*pip install jaxwt"):
        scirex.models.WNO

    class Config(dict):
        model = {
            "model_arch": "wno",
            "data_channels": 1,
            "out_channels": 1,
            "hidden_channels": 4,
            "n_layers": 1,
            "level": 1,
            "size": (8, 8),
        }

    monkeypatch.delitem(scirex.models.BaseModel._models, "wno", raising=False)
    with pytest.raises(ImportError, match="jaxwt"):
        scirex.models.get_model_jax(Config())
