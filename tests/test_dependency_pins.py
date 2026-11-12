# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""``requirements.txt`` (the exact lock) must satisfy the ranges in ``pyproject.toml``.

The package declares ranges so it can be installed next to other libraries;
``requirements.txt`` records the exact versions it is tested with. This test keeps the
two from drifting apart.
"""

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent


def _lock() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if m:
            pins[canonicalize_name(m.group(1))] = m.group(2)
    return pins


def _list(key: str) -> list[str]:
    """Return the strings of ``key = [ ... ]`` in pyproject.toml (no toml parser:
    ``tomllib`` is missing on Python 3.10)."""
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(rf"^{key}\s*=\s*\[(.*?)\]\s*$", text, re.M | re.S)
    assert m, f"{key} not found in pyproject.toml"
    return re.findall(r'"([^"]+)"', re.sub(r"#.*", "", m.group(1)))


def _declared() -> list[Requirement]:
    return [
        Requirement(r) for r in _list("dependencies") + _list("wandb") + _list("open3d")
    ]


def test_runtime_dependencies_are_ranges_not_exact_pins() -> None:
    for text in _list("dependencies"):
        spec = str(Requirement(text).specifier)
        assert "==" not in spec, f"{text}: use a range in pyproject.toml"
        assert "<" in spec, f"{text}: give the range an upper bound"


@pytest.mark.parametrize("req", _declared(), ids=lambda r: r.name)
def test_locked_version_satisfies_the_declared_range(req: Requirement) -> None:
    lock = _lock()
    name = canonicalize_name(req.name)
    assert name in lock, f"{req.name} is declared but not pinned in requirements.txt"
    assert req.specifier.contains(
        lock[name]
    ), f"requirements.txt pins {req.name}=={lock[name]}, outside {req.specifier}"
