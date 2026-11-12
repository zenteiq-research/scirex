# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Config keys are snake_case: the renamed keys exist and their old forms don't."""

from scirex.config import gino_carcfd_config, opt


def test_scheduler_t_max_is_snake_case() -> None:
    assert "scheduler_t_max" in opt.OptimizationConfig.__annotations__
    assert "scheduler_T_max" not in opt.OptimizationConfig.__annotations__
    carcfd = gino_carcfd_config.Default().to_dict()["opt"]
    assert carcfd["scheduler_t_max"] == 10
    assert "scheduler_T_max" not in carcfd
