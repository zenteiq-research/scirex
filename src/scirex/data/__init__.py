# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Datasets, data processors and benchmark dataset loaders.

Everything listed in `__all__` is public API. The names are resolved lazily
(PEP 562): `import scirex.data` imports nothing heavy, and each name is imported
on first access. Loaders that read VTK files (`load_airfrans`,
`load_shapenet_car`) need the `benchmark` extra
(`pip install "scirex[benchmark]"`); this is reported when they are called.

The same names are also available from the sub-packages `scirex.data.containers`,
`scirex.data.transforms` and `scirex.data.datasets`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .containers.dict_dataset import DictDataset
    from .containers.tensor_dataset import TensorDataset
    from .datasets.airfrans import load_airfrans
    from .datasets.cavity_kepsilon import load_cavity_kepsilon
    from .datasets.darcy import load_darcy_flow_small, load_darcy_pt
    from .datasets.navier_stokes import load_navier_stokes_pt
    from .datasets.shapenet_car import load_shapenet_car
    from .transforms.data_processors import DefaultDataProcessor
    from .transforms.normalizers import UnitGaussianNormalizer

__all__ = [
    "DefaultDataProcessor",
    "DictDataset",
    "TensorDataset",
    "UnitGaussianNormalizer",
    "load_airfrans",
    "load_cavity_kepsilon",
    "load_darcy_flow_small",
    "load_darcy_pt",
    "load_navier_stokes_pt",
    "load_shapenet_car",
]

_LAZY_EXPORTS: dict[str, str] = {
    "DefaultDataProcessor": "scirex.data.transforms.data_processors",
    "DictDataset": "scirex.data.containers.dict_dataset",
    "TensorDataset": "scirex.data.containers.tensor_dataset",
    "UnitGaussianNormalizer": "scirex.data.transforms.normalizers",
    "load_airfrans": "scirex.data.datasets.airfrans",
    "load_cavity_kepsilon": "scirex.data.datasets.cavity_kepsilon",
    "load_darcy_flow_small": "scirex.data.datasets.darcy",
    "load_darcy_pt": "scirex.data.datasets.darcy",
    "load_navier_stokes_pt": "scirex.data.datasets.navier_stokes",
    "load_shapenet_car": "scirex.data.datasets.shapenet_car",
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
