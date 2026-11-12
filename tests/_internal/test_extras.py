# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the optional-extra install hints."""

import pytest

from scirex._internal.extras import missing_extra_error, requires_extra


def test_missing_extra_error_names_package_and_extra() -> None:
    err = ModuleNotFoundError("No module named 'vtk.util'", name="vtk.util")
    result = missing_extra_error(err, packages=("vtk",), extra="benchmark", feature="X")
    assert isinstance(result, ImportError)
    assert "'vtk'" in str(result)
    assert 'pip install "scirex[benchmark]"' in str(result)
    assert str(result).startswith("X requires")


def test_missing_extra_error_ignores_other_packages() -> None:
    err = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    assert (
        missing_extra_error(err, packages=("vtk",), extra="benchmark", feature="X")
        is None
    )


def test_requires_extra_translates_only_the_listed_packages() -> None:
    @requires_extra("definitely_not_installed_pkg", extra="benchmark")
    def needs_listed() -> None:
        import definitely_not_installed_pkg  # noqa: F401

    @requires_extra("vtk", extra="benchmark")
    def needs_other() -> None:
        import another_missing_pkg  # noqa: F401

    with pytest.raises(ImportError, match=r"scirex\[benchmark\]") as info:
        needs_listed()
    assert isinstance(info.value.__cause__, ModuleNotFoundError)

    with pytest.raises(ModuleNotFoundError) as other:
        needs_other()
    assert "benchmark" not in str(other.value)


def test_requires_extra_keeps_signature_docstring_and_result() -> None:
    @requires_extra("vtk", extra="benchmark")
    def add(a: int, b: int = 1) -> int:
        """Add."""
        return a + b

    assert add(2) == 3
    assert add.__doc__ == "Add."
    assert add.__wrapped__(1, 1) == 2
