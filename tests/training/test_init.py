# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the deprecated re-exports of ``scirex.training``."""

import pytest

import scirex._internal.wno_train_utils as wno_train_utils
import scirex.training

_NAMES = [
    "append_grid_nhwc",
    "append_grid_nwc",
    "create_train_state",
    "make_schedule",
    "per_frame_relative_l2",
    "plot_loss_curves",
    "relative_l2_sum",
    "resolve_dtype",
    "write_metrics",
]


@pytest.mark.parametrize("name", _NAMES)
def test_old_wno_train_utils_names_still_resolve_with_a_warning(name: str) -> None:
    with pytest.warns(DeprecationWarning, match="scirex._internal.wno_train_utils"):
        obj = getattr(scirex.training, name)
    assert obj is getattr(wno_train_utils, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        scirex.training.does_not_exist
