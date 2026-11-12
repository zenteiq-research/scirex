# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared helpers for the tests that pin the package-level public API."""

import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
REPO = SRC.parent

# The stable public API: every name in ``__all__`` of these packages. This is the
# single definition used by the cross-cutting tests (snapshot, docstrings,
# isolation, documentation coverage). ``scirex.experimental`` and
# ``scirex._internal`` are deliberately absent.
PUBLIC_PACKAGES: tuple[str, ...] = (
    "scirex",
    "scirex.models",
    "scirex.layers",
    "scirex.layers.blocks",
    "scirex.layers.common",
    "scirex.losses",
    "scirex.data",
    "scirex.data.containers",
    "scirex.data.transforms",
    "scirex.data.datasets",
    "scirex.training",
    "scirex.training.optim",
    "scirex.config",
    "scirex.utils",
)


def live_surface() -> dict[str, list[str]]:
    """Return ``{package: sorted(__all__)}`` for :data:`PUBLIC_PACKAGES`.

    Only ``__all__`` is read, so no exported symbol (and hence no optional
    dependency) is imported.
    """
    return {
        package: sorted(importlib.import_module(package).__all__)
        for package in PUBLIC_PACKAGES
    }


SNAPSHOT_PATH = Path(__file__).resolve().parent / "api_snapshot.json"


def render_snapshot(surface: dict[str, list[str]]) -> str:
    """Serialise a surface deterministically (sorted keys, trailing newline)."""
    return json.dumps(surface, indent=2, sort_keys=True) + "\n"


def public_items() -> list[tuple[str, str]]:
    """Return every ``(package, name)`` pair of the public API."""
    return [
        (package, name) for package, names in live_surface().items() for name in names
    ]


@cache
def resolve_public(package: str, name: str) -> Any:
    """Import and return ``package.name``.

    Raises
    ------
    ImportError
        If an optional third-party dependency of the symbol is not installed.
    """
    return getattr(importlib.import_module(package), name)


def ruff_order_key(name: str) -> tuple[int, str]:
    """Sort key of ruff's RUF022: SCREAMING_CASE, then CamelCase, then the rest."""
    if len(name) > 1 and name.isupper():
        return (0, name)
    if name[:1].isupper():
        return (1, name)
    return (2, name)


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that imports ``scirex`` from ``src``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    return subprocess.run(
        [sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


def assert_run_ok(code: str) -> str:
    """Run ``code`` in a fresh interpreter, assert it succeeds and return stdout."""
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    return result.stdout


def assert_all_is_well_formed(module: ModuleType) -> None:
    """``__all__`` is a duplicate-free list in RUF022 order."""
    names = list(module.__all__)
    assert names, f"{module.__name__}.__all__ is empty"
    assert len(names) == len(set(names)), f"duplicates in {module.__name__}.__all__"
    assert names == sorted(
        names, key=ruff_order_key
    ), f"{module.__name__}.__all__ is not sorted"


def assert_exports_resolve(module: ModuleType) -> None:
    """Every name in ``__all__`` resolves to a documented, non-None object."""
    assert_all_is_well_formed(module)
    for name in module.__all__:
        obj = getattr(module, name)
        assert obj is not None, f"{module.__name__}.{name} is None"
        assert name in dir(module)
        doc = inspect.getdoc(obj) or ""
        # Flax/dataclass-generated docs look like "Name(field: type, ...)".
        assert doc and not doc.startswith(
            f"{name}("
        ), f"{module.__name__}.{name} has no docstring"


def assert_lazy_table_matches(module: ModuleType) -> None:
    """The lazy table lists exactly ``__all__`` and points at the defining modules."""
    table: dict[str, str] = module._LAZY_EXPORTS
    assert set(table) == set(module.__all__)
    for name, target in table.items():
        assert getattr(module, name).__module__ == target, name
        assert importlib.import_module(target) is not None
