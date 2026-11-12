# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Dataset loaders for the neural-operator benchmarks (lazy exports).

See `scirex.data` for the lazy-loading behaviour and the `benchmark` extra
that `load_airfrans` and `load_shapenet_car` need. Other loaders in
this package (chip cooling, Car-CFD, ...) are not part of the public API; import
them from their modules if needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scirex._internal.lazy import resolve_lazy_attr

if TYPE_CHECKING:
    from .airfrans import load_airfrans
    from .cavity_kepsilon import load_cavity_kepsilon
    from .darcy import load_darcy_flow_small, load_darcy_pt
    from .navier_stokes import load_navier_stokes_pt
    from .shapenet_car import load_shapenet_car

__all__ = [
    "load_airfrans",
    "load_cavity_kepsilon",
    "load_darcy_flow_small",
    "load_darcy_pt",
    "load_navier_stokes_pt",
    "load_shapenet_car",
]

_LAZY_EXPORTS: dict[str, str] = {
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
