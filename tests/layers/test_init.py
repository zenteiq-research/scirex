# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the ``scirex.layers`` public API (flat, ``blocks`` and ``common``)."""

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scirex
import scirex.layers

_SRC_DIR = Path(scirex.__file__).resolve().parent.parent

_PACKAGES = ["scirex.layers", "scirex.layers.blocks", "scirex.layers.common"]
_CASES = [
    (pkg, name) for pkg in _PACKAGES for name in importlib.import_module(pkg).__all__
]


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter that has imported nothing yet."""
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR), "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(("package", "name"), _CASES)
def test_every_exported_name_resolves(package: str, name: str) -> None:
    assert getattr(importlib.import_module(package), name) is not None


@pytest.mark.parametrize(("package", "name"), _CASES)
def test_exported_names_have_a_real_docstring(package: str, name: str) -> None:
    doc = inspect.getdoc(getattr(importlib.import_module(package), name)) or ""
    assert doc, f"{package}.{name} has no docstring"
    assert not doc.startswith(f"{name}("), f"{name} only has the auto-generated doc"


@pytest.mark.parametrize("package", _PACKAGES)
def test_lazy_table_matches_all_and_dir(package: str) -> None:
    module = importlib.import_module(package)
    assert set(module._LAZY_EXPORTS) == set(module.__all__)
    assert set(module.__all__) <= set(dir(module))
    assert module.__all__ == sorted(module.__all__, key=_ruff_natural_key)


def _ruff_natural_key(name: str) -> tuple[int, str]:
    """RUF022 order: SCREAMING_CASE, then CamelCase, then everything else."""
    if name.isupper():
        return (0, name)
    if name[0].isupper():
        return (1, name)
    return (2, name)


def test_flat_namespace_is_the_union_of_blocks_and_common() -> None:
    import scirex.layers.blocks
    import scirex.layers.common

    union = {*scirex.layers.blocks.__all__, *scirex.layers.common.__all__}
    assert set(scirex.layers.__all__) == union
    assert len(scirex.layers.__all__) == len(scirex.layers.blocks.__all__) + len(
        scirex.layers.common.__all__
    )
    for name in union:
        assert getattr(scirex.layers, name) is getattr(
            importlib.import_module(scirex.layers._LAZY_EXPORTS[name]), name
        )


@pytest.mark.parametrize("package", _PACKAGES)
def test_unknown_attribute_raises_attribute_error(package: str) -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        importlib.import_module(package).does_not_exist


def test_importing_layer_packages_is_light() -> None:
    result = _run_fresh(
        """
        import sys
        import scirex.layers, scirex.layers.blocks, scirex.layers.common

        heavy = {"jax", "flax", "jaxwt", "open3d", "torch", "tensorly", "scipy"}
        assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))
        scirex.layers.ChannelMLP
        assert "jax" in sys.modules
        assert "jaxwt" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_missing_jaxwt_raises_import_error_naming_it_only_when_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scirex.layers.blocks

    for module in ("wavelet_conv", "wno_block"):
        monkeypatch.delitem(
            sys.modules, f"scirex.layers.blocks.{module}", raising=False
        )
    for namespace in (vars(scirex.layers), vars(scirex.layers.blocks)):
        for name in ("WaveletConv", "WNOBlock"):
            monkeypatch.delitem(namespace, name, raising=False)
    monkeypatch.setitem(sys.modules, "jaxwt", None)

    assert scirex.layers.blocks.FNOBlocks is not None  # unaffected
    with pytest.raises(ImportError, match=r"WaveletConv.*'jaxwt'.*pip install jaxwt"):
        scirex.layers.blocks.WaveletConv
    with pytest.raises(ImportError, match="jaxwt"):
        scirex.layers.WNOBlock
