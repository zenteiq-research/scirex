# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data loading for the Car-Design-ShapeNetCar benchmark.

Thin re-export of :func:`scirex.data.datasets.shapenet_car.
load_shapenet_car` -- see that module's docstring for the on-disk layout,
caching and normalization it performs. Kept as its own file (rather than
calling the loader directly from ``train.py``/``infer.py``) only to match this
benchmark package's file-naming convention; there is no windowing/tiling here
the way ``lid_driven_cavity/_data.py`` has, since each sample is already an
irregular per-mesh point cloud (see this package's ``__init__.py``).

``MODEL_KWARGS`` is the geometry contract :class:`scirex.models.
transolver.Transolver` must be built with for this dataset (point
features, output fields, "irregular" geometry) -- fixed by the data, not a
sweepable architecture knob, so it lives here rather than in
``_config.RunConfig.arch``. ``load_shapenet_car`` returns the same values via
its ``model_kwargs`` key; this constant exists so ``_checkpoint.py`` can
rebuild a model without re-loading the dataset first.
"""

from __future__ import annotations

from pathlib import Path

from scirex.data import load_shapenet_car

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = Path("~/data/mlcfd_data/training_data")
DEFAULT_CACHE_DIR = Path("~/data/mlcfd_data/preprocessed_data")

#: space_dim=7 ([pos_x,pos_y,pos_z,sdf,normal_x,normal_y,normal_z]),
#: out_dim=4 ([velo_x,velo_y,velo_z,press]) -- see load_shapenet_car's
#: docstring. Matches scirex.models.Transolver's kwargs.
MODEL_KWARGS: dict = dict(
    space_dim=7, fun_dim=0, out_dim=4, unified_pos=False, geometry="irregular"
)

__all__ = ["DEFAULT_CACHE_DIR", "DEFAULT_DATA_DIR", "MODEL_KWARGS", "load_split"]


def load_split(
    data_dir: str | Path,
    fold_id: int,
    cache_dir: str | Path | None,
    use_cache: bool = True,
) -> dict:
    """Loads the train/val split for one fold. See ``load_shapenet_car``'s
    docstring for the returned dict's ``train``/``val``/``coef_norm``/
    ``model_kwargs`` keys."""
    return load_shapenet_car(
        data_dir, fold_id=fold_id, cache_dir=cache_dir, use_cache=use_cache
    )
