# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the deprecation helpers."""

import pytest

from scirex._internal.deprecation import deprecated, warn_deprecated


def test_warn_deprecated_message_and_location() -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"old\.name .*since ScireX 1\.0\.0.*1\.2\.0.*Use new\.name",
    ) as rec:
        warn_deprecated(
            "old.name", since="1.0.0", removal="1.2.0", alternative="new.name"
        )
    assert rec[0].filename == __file__


def test_warn_deprecated_without_alternative() -> None:
    with pytest.warns(DeprecationWarning) as rec:
        warn_deprecated("old.name", since="1.0.0", removal="1.2.0")
    assert "Use" not in str(rec[0].message)


def test_deprecated_function_warns_and_returns() -> None:
    @deprecated(since="1.0.0", removal="1.2.0", alternative="g")
    def f(x: int) -> int:
        """Add one."""
        return x + 1

    with pytest.warns(DeprecationWarning, match="Use g") as rec:
        assert f(1) == 2
    assert rec[0].filename == __file__
    assert f.__doc__ == "Add one."


def test_deprecated_class_warns_on_construction() -> None:
    @deprecated(since="1.0.0", removal="1.2.0")
    class C:
        def __init__(self, v: int) -> None:
            self.v = v

    with pytest.warns(DeprecationWarning) as rec:
        c = C(3)
    assert c.v == 3
    assert rec[0].filename == __file__
