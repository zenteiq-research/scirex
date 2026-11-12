# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data split and normalisation of the chip cooling benchmark.

The dataset itself is read by ``scirex.data.datasets.chipcooling``.
Two views of it are used:

* :func:`grid_split` (FNO, WNO): every field interpolated onto a regular
  128x256 grid; 7 input channels (the six parameters broadcast over the grid
  and the in-domain mask), z-scored with training statistics; the target
  z-scored with the mean and std over the in-domain cells of the training
  fields. Metrics count the in-domain cells only.
* :func:`mesh_split` (GINO, Transolver): the 11,110 mesh nodes as they are;
  the parameters z-scored per parameter and the temperature with one mean
  and std, both from the training samples.

Both take the first ``ntrain`` samples for training and the last ``ntest``
for testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scirex.data.datasets.chipcooling import (
    build_grid_cache,
    build_input_channels,
    load_data,
    normalize_inputs,
    normalize_targets,
)


def load_raw(data_dir: Path | None) -> dict:
    """Parameters ``(n, 6)``, temperature fields ``(n, n_nodes)``, mesh
    coordinates ``(n_nodes, 3)`` and the parameter names."""
    parameters, fields, coords, names = load_data(data_dir)
    return dict(parameters=parameters, fields=fields, coords=coords, names=names)


def grid_split(raw: dict, protocol: dict, cache_dir: Path) -> dict:
    """The grid view, for FNO and WNO; arrays are channels-first ``(n, C, H, W)``.

    ``Y_test`` is the test temperature on the grid in kelvin (0 outside the
    domain). The interpolated grid is cached in ``cache_dir``.
    """
    parameters, fields, coords = raw["parameters"], raw["fields"], raw["coords"]
    h, w = protocol["grid"]
    cache = Path(cache_dir) / f"grid_{len(fields)}x{h}x{w}.npz"
    _, grid_fields, mask, y_grid, z_grid = build_grid_cache(
        coords, fields, cache, grid_h=h, grid_w=w
    )
    X = build_input_channels(parameters, mask, H=h, W=w)
    Y = grid_fields[:, None, :, :]
    ntrain, ntest = protocol["ntrain"], protocol["ntest"]
    X_train_n, X_test_n, x_mean, x_std = normalize_inputs(X[:ntrain], X[-ntest:])
    Y_train_n, Y_test_n, y_mean, y_std = normalize_targets(Y[:ntrain], Y[-ntest:], mask)
    return dict(
        kind="grid",
        X_train=X_train_n,
        Y_train=Y_train_n,
        X_test=X_test_n,
        Y_test_n=Y_test_n,
        Y_test=Y[-ntest:],
        mask=mask,
        y_grid=y_grid,
        z_grid=z_grid,
        normalisation=dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std),
    )


def mesh_split(raw: dict, protocol: dict) -> dict:
    """The mesh view, for GINO and Transolver.

    ``params_n`` holds the normalised parameters of every sample (train and
    test); ``fields_test`` is the test temperature on the mesh in kelvin.
    """
    parameters, fields, coords = raw["parameters"], raw["fields"], raw["coords"]
    ntrain, ntest = protocol["ntrain"], protocol["ntest"]
    p_mean = parameters[:ntrain].mean(axis=0).astype(np.float32)
    p_std = (parameters[:ntrain].std(axis=0) + 1e-8).astype(np.float32)
    t_mean = float(fields[:ntrain].mean())
    t_std = float(fields[:ntrain].std()) + 1e-8
    y, z = coords[:, 1], coords[:, 2]
    coord_stats = dict(
        y_min=float(y.min()),
        y_max=float(y.max()),
        z_min=float(z.min()),
        z_max=float(z.max()),
    )
    return dict(
        kind="mesh",
        params_n=((parameters - p_mean) / p_std).astype(np.float32),
        fields=fields,
        fields_test=fields[-ntest:],
        coords=coords,
        ntrain=ntrain,
        ntest=ntest,
        normalisation=dict(
            p_mean=p_mean,
            p_std=p_std,
            t_mean=t_mean,
            t_std=t_std,
            coord_stats=coord_stats,
        ),
    )
