# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The package-level exports of ``scirex.experimental.training``."""

import scirex.experimental.training as training


def test_all_matches_the_lazy_table_and_every_name_resolves() -> None:
    assert sorted(training.__all__) == sorted(training.EXPERIMENTAL_EXPORTS)
    for name in training.__all__:
        assert getattr(training, name) is not None, name
