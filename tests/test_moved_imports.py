# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Modules that moved keep their old import path (``scirex._internal.aliases``).

A module is cached after its first import, so the real package is checked in a fresh
interpreter.
"""

import subprocess
import sys

import pytest

from scirex._internal import aliases


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


@pytest.mark.parametrize("old,new", sorted(aliases.MOVED.items()))
def test_old_path_is_the_same_module_and_warns(old: str, new: str) -> None:
    out = _run(
        "import warnings, importlib\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        f"    old = importlib.import_module({old!r})\n"
        f"new = importlib.import_module({new!r})\n"
        "assert old is new, 'not the same module object'\n"
        "msgs = [str(x.message) for x in w if x.category is DeprecationWarning]\n"
        f"assert any({old!r} in m and {new!r} in m for m in msgs), msgs\n"
        "print('ok')\n"
    )
    assert "ok" in out


def test_importing_scirex_does_not_import_jax() -> None:
    out = _run("import sys, scirex; print('jax' in sys.modules)")
    assert out.strip().endswith("False")


def test_checkpoint_remap_uses_the_same_table() -> None:
    from scirex.models._legacy_refs import MOVED_MODULES, remap_legacy_module

    assert MOVED_MODULES is aliases.MOVED
    assert remap_legacy_module("scirex.data.web_utils") == "scirex._internal.web_utils"
    assert (
        remap_legacy_module("scirex.layers.common.einsum_utils")
        == "scirex._internal.einsum_utils"
    )


def test_every_target_exists() -> None:
    import importlib.util

    for new in aliases.MOVED.values():
        assert importlib.util.find_spec(new) is not None, new
