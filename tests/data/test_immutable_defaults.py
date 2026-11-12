# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Public data functions must not share a mutable default between calls."""

import inspect

import pytest

from scirex.data.datasets import car_cfd, darcy, navier_stokes
from scirex.data.transforms import data_processors

TARGETS = [
    (darcy.DarcyDataset.__init__, "test_resolutions"),
    (darcy.load_darcy_flow_small, "test_resolutions"),
    (darcy.load_darcy_pt, "test_resolutions"),
    (navier_stokes.NavierStokesDataset.__init__, "test_resolutions"),
    (navier_stokes.load_navier_stokes_pt, "test_resolutions"),
    (car_cfd.CarCFDDatasetjax.__init__, "query_res"),
    (data_processors.IncrementalDataProcessor.__init__, "subsampling_rates"),
    (data_processors.IncrementalDataProcessor.__init__, "dataset_indices"),
]


@pytest.mark.parametrize(
    ("func", "name"), TARGETS, ids=lambda v: getattr(v, "__qualname__", v)
)
def test_default_is_a_tuple(func, name) -> None:
    default = inspect.signature(func).parameters[name].default
    assert isinstance(
        default, tuple
    ), f"{func.__qualname__}({name}) default is {default!r}"


def test_no_list_or_dict_defaults_in_the_data_package() -> None:
    """A list, dict or set default in scirex.data would be shared between calls."""
    import ast
    from pathlib import Path

    import scirex.data

    bad = []
    for path in Path(scirex.data.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                defaults = node.args.defaults + [
                    d for d in node.args.kw_defaults if d is not None
                ]
                for d in defaults:
                    if isinstance(d, ast.List | ast.Dict | ast.Set):
                        bad.append(f"{path.name}:{d.lineno} in {node.name}")
    assert bad == []
