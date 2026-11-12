# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The four architectures of the Darcy benchmark.

Each :class:`ModelSpec` turns the ``(n, H, W)`` arrays of ``_data.load_split``
into what its model takes, normalised as its own reference code does, and
returns a :class:`Prepared` bundle: the model, its encoded inputs, ``init``
and a ``predict`` that always returns physical-unit predictions of shape
``(B, H * W)``. ``train.py`` and ``infer.py`` only ever go through it.

=========== ============================== =========================
Model       Input layout                   Normalisation
=========== ============================== =========================
FNO         ``(n, 1, H, W)``               per pixel
WNO         ``(n, H, W, 1)``, (x, y) added per pixel
GINO        ``(n, H * W)`` on a 2-D point  per pixel
            cloud with fixed neighbours
Transolver  ``(n, H * W, 1)``              one scalar per field
=========== ============================== =========================
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from zencfg import ConfigBase

from scirex import get_model_jax
from scirex.config import GINODarcy2d, ModelConfig
from scirex.layers import kdtree_neighbor_search
from scirex.models import FNO, WNO, Transolver
from scripts.benchmarks.darcy._data import GlobalNormalizer, PixelNormalizer


@dataclass
class Prepared:
    """One architecture, built and wired to the data."""

    model: nn.Module
    train_x: np.ndarray
    test_x: np.ndarray
    init: Callable[[jax.Array], Any]
    predict: Callable[[Any, jnp.ndarray], jnp.ndarray]
    normalisation: dict


def _pixel_summary(a_norm: PixelNormalizer, u_norm: PixelNormalizer) -> dict:
    """A few scalars that identify the per-pixel statistics of a split."""
    return dict(
        a_mean=float(a_norm.mean.mean()),
        a_std=float(a_norm.std.mean()),
        u_mean=float(u_norm.mean.mean()),
        u_std=float(u_norm.std.mean()),
    )


def _prepare_fno(arch: dict, data: dict) -> Prepared:
    train_a, train_u = data["train_a"][:, None], data["train_u"][:, None]
    test_a = data["test_a"][:, None]
    n_side = train_a.shape[-1]
    n_points = n_side * n_side
    a_norm, u_norm = PixelNormalizer(train_a), PixelNormalizer(train_u)
    train_x, test_x = a_norm.encode(train_a), a_norm.encode(test_a)
    u_mean = jnp.asarray(u_norm.mean)
    u_std_eps = jnp.asarray(u_norm.std + u_norm.eps)

    model = FNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=arch["hidden_channels"],
        n_modes=(arch["modes"], arch["modes"]),
        n_layers=arch["n_layers"],
    )

    def predict(p: Any, x: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(p, x) * u_std_eps + u_mean
        return out[:, 0, :, :].reshape(x.shape[0], n_points)

    return Prepared(
        model=model,
        train_x=train_x,
        test_x=test_x,
        init=lambda key: model.init(key, jnp.asarray(train_x[:1])),
        predict=predict,
        normalisation=_pixel_summary(a_norm, u_norm),
    )


def _prepare_wno(arch: dict, data: dict) -> Prepared:
    train_a, train_u = data["train_a"][..., None], data["train_u"][..., None]
    test_a = data["test_a"][..., None]
    n_side = train_a.shape[1]
    n_points = n_side * n_side
    a_norm, u_norm = PixelNormalizer(train_a), PixelNormalizer(train_u)
    train_x, test_x = a_norm.encode(train_a), a_norm.encode(test_a)
    u_mean = jnp.asarray(u_norm.mean)
    u_std_eps = jnp.asarray(u_norm.std + u_norm.eps)

    model = WNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=arch["hidden_channels"],
        n_layers=arch["n_layers"],
        level=arch["level"],
        size=(n_side, n_side),
        wavelet=arch["wavelet"],
        padding=arch["padding"],
        use_grid=True,  # appends (x, y), the reference code's in_channel=3
        # The reference recipe is PyTorch float32; WNO's own default is bf16.
        compute_dtype=jnp.float32,
    )

    def predict(p: Any, x: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(p, x) * u_std_eps + u_mean
        return out[..., 0].reshape(x.shape[0], n_points)

    return Prepared(
        model=model,
        train_x=train_x,
        test_x=test_x,
        init=lambda key: model.init(key, jnp.asarray(train_x[:1])),
        predict=predict,
        normalisation=_pixel_summary(a_norm, u_norm),
    )


class _GINODarcyConfig(ConfigBase):
    model: ModelConfig = GINODarcy2d()


def build_grid_geometry(n_side: int) -> np.ndarray:
    """The uniform ``n_side x n_side`` grid over [0, 1]^2 as ``(n_side**2, 2)``
    points, flattened in the same (C) order as ``.reshape(n, n_side**2)`` of
    the data, so data and geometry need no reindexing."""
    xs = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, n_side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def _build_neighbors(data: np.ndarray, queries: np.ndarray, radius: float) -> dict:
    """Fixed-radius neighbours in the segment form GINO's GNO blocks take."""
    raw = kdtree_neighbor_search(data=data, queries=queries, radius=radius)
    idx = np.asarray(raw["neighbors_index"])
    splits = np.asarray(raw["neighbors_row_splits"])
    counts = (splits[1:] - splits[:-1]).astype(np.int32)
    segment_ids = np.repeat(np.arange(counts.shape[0], dtype=np.int32), counts)
    return {
        "neighbors_index": jnp.asarray(idx),
        "segment_ids": jnp.asarray(segment_ids),
        "counts": jnp.asarray(counts),
    }


def gno_radii(arch: dict, n_side: int) -> tuple[float, float]:
    """The input and output GNO radii: ``0.16 * 16 / (s - 1)`` and
    ``0.175 * 16 / (s - 1)`` unless ``arch`` sets them."""
    resolution = n_side - 1
    in_radius = arch.get("in_gno_radius") or 0.16 * 16.0 / resolution
    out_radius = arch.get("out_gno_radius") or 0.175 * 16.0 / resolution
    return in_radius, out_radius


def _prepare_gino(arch: dict, data: dict) -> Prepared:
    n_train, n_side = data["train_a"].shape[0], data["train_a"].shape[1]
    n_points = n_side * n_side
    train_a = data["train_a"].reshape(n_train, n_points)
    train_u = data["train_u"].reshape(n_train, n_points)
    test_a = data["test_a"].reshape(data["test_a"].shape[0], n_points)
    a_norm, u_norm = PixelNormalizer(train_a), PixelNormalizer(train_u)
    train_x, test_x = a_norm.encode(train_a), a_norm.encode(test_a)

    in_radius, out_radius = gno_radii(arch, n_side)
    xy = build_grid_geometry(n_side)
    input_geom = xy[None, ...]
    output_queries = xy[None, ...]
    latent_queries = xy.reshape(n_side, n_side, 2)[None, ...]
    neighbors_in = _build_neighbors(xy, xy, in_radius)
    neighbors_out = _build_neighbors(xy, xy, out_radius)

    config = _GINODarcyConfig().to_dict()
    config["model"]["in_gno_radius"] = in_radius
    config["model"]["out_gno_radius"] = out_radius
    config["model"]["fno_hidden_channels"] = arch["hidden_channels"]
    config["model"]["fno_n_layers"] = arch["n_layers"]
    config["model"]["fno_n_modes"] = [arch["modes"], arch["modes"]]
    model = get_model_jax(config)
    geometry = dict(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        neighbors_in=neighbors_in,
        neighbors_out=neighbors_out,
    )

    def predict(p: Any, x: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(p, x=x[..., None], **geometry)  # (B, N, 1)
        return u_norm.decode(jnp.asarray(out)[..., 0])

    return Prepared(
        model=model,
        train_x=train_x,
        test_x=test_x,
        init=lambda key: model.init(
            key, x=jnp.asarray(train_x[:1])[..., None], **geometry
        ),
        predict=predict,
        normalisation=_pixel_summary(a_norm, u_norm),
    )


def _prepare_transolver(arch: dict, data: dict) -> Prepared:
    train_a, train_u = data["train_a"][..., None], data["train_u"][..., None]
    test_a = data["test_a"][..., None]
    n_side = train_a.shape[1]
    n_points = n_side * n_side
    a_norm, u_norm = GlobalNormalizer(train_a), GlobalNormalizer(train_u)
    train_x = a_norm.encode(train_a).reshape(train_a.shape[0], n_points, 1)
    test_x = a_norm.encode(test_a).reshape(test_a.shape[0], n_points, 1)
    u_mean = jnp.asarray(u_norm.mean.reshape(1, 1))
    u_std = jnp.asarray(u_norm.std.reshape(1, 1))

    model = Transolver(
        space_dim=2,
        n_layers=arch["n_layers"],
        n_hidden=arch["n_hidden"],
        n_head=arch["n_head"],
        mlp_ratio=arch["mlp_ratio"],
        fun_dim=1,
        out_dim=1,
        slice_num=arch["slice_num"],
        ref=arch["ref"],
        unified_pos=True,
        geometry="structured_2d",
        H=n_side,
        W=n_side,
        kernel=arch["kernel"],
        dropout_rate=0.0,
        time_input=False,
    )
    # unified_pos recomputes the positions from H and W; this placeholder
    # only carries the batch size.
    x_placeholder = jnp.zeros((1, n_points, 2), dtype=jnp.float32)

    def predict(p: Any, x: jnp.ndarray) -> jnp.ndarray:
        x_b = jnp.broadcast_to(x_placeholder, (x.shape[0], n_points, 2))
        out = model.apply(p, x=x_b, fx=x) * u_std + u_mean
        return out[:, :, 0]

    return Prepared(
        model=model,
        train_x=train_x,
        test_x=test_x,
        init=lambda key: model.init(key, x=x_placeholder, fx=jnp.asarray(train_x[:1])),
        predict=predict,
        normalisation=dict(
            a_mean=float(a_norm.mean.ravel()[0]),
            a_std=float(a_norm.std.ravel()[0]),
            u_mean=float(u_norm.mean.ravel()[0]),
            u_std=float(u_norm.std.ravel()[0]),
        ),
    )


SPECS: dict[str, Callable[[dict, dict], Prepared]] = {
    "fno": _prepare_fno,
    "wno": _prepare_wno,
    "gino": _prepare_gino,
    "transolver": _prepare_transolver,
}


def prepare(model: str, arch: dict, data: dict) -> Prepared:
    """The architecture ``model`` at the size ``arch``, wired to ``data``."""
    if model not in SPECS:
        raise ValueError(f"unknown model {model!r}; choose from {list(SPECS)}")
    return SPECS[model](arch, data)
