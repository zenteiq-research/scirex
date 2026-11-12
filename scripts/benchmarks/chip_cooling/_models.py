# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architectures of the chip cooling benchmark, built from this repo's
operators.

Nothing here is a network. Each :class:`ModelSpec` builds a model that
already lives in ``src/scirex/`` and adapts the benchmark's data to it:

* :meth:`~ModelSpec.prepare` turns the split from ``_data.py`` into the
  model's inputs and normalised targets, for the training and test samples;
* :meth:`~ModelSpec.init` initialises the parameters (also the template a
  checkpoint is restored into);
* :meth:`~ModelSpec.predict` returns the test predictions in kelvin, on the
  grid ``(n, 1, H, W)`` for FNO and WNO or on the mesh ``(n, n_nodes)`` for
  GINO and Transolver.

:data:`SPECS` maps each name to its spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModelSpec(ABC):
    """One architecture: how to build it and how to feed it.

    ``kind`` is ``"grid"`` (works on the interpolated 128x256 grid) or
    ``"mesh"`` (works on the mesh nodes).
    """

    name: str
    label: str
    kind: str

    @abstractmethod
    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        """The model at the size given by ``arch``."""

    @abstractmethod
    def prepare(self, arch: Mapping[str, Any], data: dict) -> dict:
        """Model inputs and normalised targets for the training and test samples."""

    @abstractmethod
    def init(
        self, model: nn.Module, arch: Mapping[str, Any], prepared: dict, seed: int
    ) -> Any:
        """Initial parameters, as the training run starts from them."""

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        params: Any,
        prepared: dict,
        data: dict,
        idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predictions in kelvin for the test samples ``idx`` (all by default)."""


_JITTED: dict[int, Any] = {}


def _jitted(model: nn.Module, fn) -> Any:
    """``jax.jit(fn)``, compiled once per model, so repeated predictions (and
    their timing) do not recompile."""
    key = id(model)
    if key not in _JITTED:
        _JITTED[key] = (model, jax.jit(fn))
    return _JITTED[key][1]


def _test_idx(n: int, idx) -> np.ndarray:
    return np.arange(n) if idx is None else np.asarray(idx).reshape(-1)


# ---------------------------------------------------------------------------
# Grid models
# ---------------------------------------------------------------------------


def _grid_predict(apply, params, x, data, batch_size):
    preds = [
        np.asarray(apply(params, jnp.asarray(x[i : i + batch_size], dtype=jnp.float32)))
        for i in range(0, x.shape[0], batch_size)
    ]
    norm = data["normalisation"]
    return np.concatenate(preds, axis=0) * norm["y_std"] + norm["y_mean"]


@dataclass(frozen=True)
class FNOSpec(ModelSpec):
    """:class:`scirex.models.FNO`, channels-first, 7 input channels."""

    name: str = "fno"
    label: str = "FNO"
    kind: str = "grid"

    def build(self, arch, grid):
        from scirex.models import FNO

        return FNO(
            n_modes=tuple(arch["n_modes"]),
            hidden_channels=arch["hidden_channels"],
            in_channels=7,
            out_channels=1,
        )

    def prepare(self, arch, data):
        return dict(
            x_train=data["X_train"],
            y_train=data["Y_train"],
            x_test=data["X_test"],
            y_test=data["Y_test_n"],
        )

    def init(self, model, arch, prepared, seed):
        _, key = jax.random.split(jax.random.PRNGKey(seed))
        x = prepared["x_train"]
        return model.init(key, x=jnp.zeros((1, *x.shape[1:]), dtype=jnp.float32))

    def predict(self, model, params, prepared, data, idx=None, batch_size=8):
        apply = _jitted(model, lambda p, x: model.apply(p, x=x))
        x = prepared["x_test"][_test_idx(len(prepared["x_test"]), idx)]
        return _grid_predict(apply, params, x, data, batch_size)


def _nchw_to_nhwc(x: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(x, dtype=np.float32), (0, 2, 3, 1)).astype(
        np.float32
    )


@dataclass(frozen=True)
class WNOSpec(ModelSpec):
    """:class:`scirex.models.WNO`, channels-last.

    The grid coordinates are appended to the 7 input channels here
    (``use_grid=False`` in the model), so it sees 9.
    """

    name: str = "wno"
    label: str = "WNO"
    kind: str = "grid"

    def build(self, arch, grid):
        from scirex._internal.wno_train_utils import resolve_dtype
        from scirex.models import WNO

        return WNO(
            in_channels=9,
            out_channels=1,
            hidden_channels=arch["hidden_channels"],
            n_layers=arch["n_layers"],
            level=arch["level"],
            size=tuple(grid),
            wavelet=arch["wavelet"],
            mode="symmetric",
            wavelet_precision="fastest",
            use_grid=False,
            padding=arch["padding"],
            compute_dtype=resolve_dtype(arch["compute_dtype"]),
            param_dtype=resolve_dtype(arch["param_dtype"]),
            padding_side="end",
            projection_hidden_channels=arch["projection_hidden_channels"],
        )

    def prepare(self, arch, data):
        from scirex._internal.wno_train_utils import append_grid_nhwc

        def with_grid(x):
            x = jnp.asarray(_nchw_to_nhwc(x))
            return np.asarray(
                append_grid_nhwc(x, tuple(arch["grid_range"])), dtype=np.float32
            )

        return dict(
            x_train=with_grid(data["X_train"]),
            y_train=_nchw_to_nhwc(data["Y_train"]),
            x_test=with_grid(data["X_test"]),
            y_test=_nchw_to_nhwc(data["Y_test_n"]),
        )

    def init(self, model, arch, prepared, seed):
        _, key = jax.random.split(jax.random.PRNGKey(seed))
        x = prepared["x_train"]
        return model.init(key, jnp.zeros((1, *x.shape[1:]), dtype=jnp.float32))[
            "params"
        ]

    def predict(self, model, params, prepared, data, idx=None, batch_size=8):
        # Not jitted: in bfloat16, jit's fusion changes the result by up to a
        # few kelvin at single cells; the published predictions are eager.
        def apply(p, x):
            return model.apply({"params": p}, x)

        x = prepared["x_test"][_test_idx(len(prepared["x_test"]), idx)]
        pred = _grid_predict(apply, params, x, data, batch_size)
        return np.transpose(pred, (0, 3, 1, 2))


# ---------------------------------------------------------------------------
# Mesh models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GINOSpec(ModelSpec):
    """:class:`scirex.models.GINO` on the mesh nodes.

    The (y, z) coordinates are min-max scaled to [0, 1]. The parameters
    enter as latent features, broadcast over a ``latent_h x latent_w`` grid
    in that unit square. The radius graphs between the mesh and the latent
    grid are the same for every sample (one mesh), so they are built once.
    One sample per step.
    """

    name: str = "gino"
    label: str = "GINO"
    kind: str = "mesh"

    def build(self, arch, grid):
        from scirex.models import GINO

        return GINO(
            in_channels=0,
            out_channels=1,
            latent_feature_channels=arch["latent_feature_channels"],
            gno_coord_dim=arch["gno_coord_dim"],
            gno_coord_embed_dim=arch["gno_coord_embed_dim"],
            in_gno_radius=arch["in_gno_radius"],
            out_gno_radius=arch["out_gno_radius"],
            in_gno_transform_type=arch["in_gno_transform_type"],
            out_gno_transform_type=arch["out_gno_transform_type"],
            in_gno_pos_embed_type=arch["in_gno_pos_embed_type"],
            out_gno_pos_embed_type=arch["out_gno_pos_embed_type"],
            gno_embed_channels=arch["gno_embed_channels"],
            gno_embed_max_positions=arch["gno_embed_max_positions"],
            gno_use_open3d=False,
            gno_use_torch_scatter=False,
            in_gno_channel_mlp_hidden_layers=list(
                arch["in_gno_channel_mlp_hidden_layers"]
            ),
            out_gno_channel_mlp_hidden_layers=list(
                arch["out_gno_channel_mlp_hidden_layers"]
            ),
            fno_n_modes=tuple(arch["fno_n_modes"]),
            fno_hidden_channels=arch["fno_hidden_channels"],
            fno_lifting_channel_ratio=arch["fno_lifting_channel_ratio"],
            fno_n_layers=arch["fno_n_layers"],
            fno_use_channel_mlp=arch["fno_use_channel_mlp"],
            fno_channel_mlp_expansion=arch["fno_channel_mlp_expansion"],
            fno_norm=arch["fno_norm"],
            fno_ada_in_features=arch["fno_ada_in_features"],
            fno_factorization=arch["fno_factorization"],
            fno_rank=arch["fno_rank"],
            fno_domain_padding=arch["fno_domain_padding"],
            projection_channel_ratio=arch["projection_channel_ratio"],
        )

    def prepare(self, arch, data):
        from scirex.layers import kdtree_neighbor_search

        coords, stats = data["coords"], data["normalisation"]["coord_stats"]
        y = coords[:, 1].astype(np.float32)
        z = coords[:, 2].astype(np.float32)
        y_n = (y - stats["y_min"]) / (stats["y_max"] - stats["y_min"] + 1e-8)
        z_n = (z - stats["z_min"]) / (stats["z_max"] - stats["z_min"] + 1e-8)
        vertices = np.stack([y_n, z_n], axis=-1)[None].astype(np.float32)

        h, w = arch["latent_h"], arch["latent_w"]
        Yg, Zg = np.meshgrid(
            np.linspace(0, 1, h, dtype=np.float32),
            np.linspace(0, 1, w, dtype=np.float32),
            indexing="ij",
        )
        latent_queries = np.stack([Yg, Zg], axis=-1)[None]
        lq_flat = latent_queries[0].reshape(-1, 2)

        def graph(points, queries, radius):
            nbrs = kdtree_neighbor_search(
                data=points, queries=queries, radius=radius, return_norm=False
            )
            idx = np.asarray(nbrs["neighbors_index"])
            counts = np.diff(np.asarray(nbrs["neighbors_row_splits"]))
            return {
                "neighbors_index": idx,
                "segment_ids": np.repeat(np.arange(len(counts)), counts),
                "counts": counts,
            }

        n_feat = arch["latent_feature_channels"]
        params_n = data["params_n"]
        latent_features = np.zeros((len(params_n), h, w, n_feat), dtype=np.float32)
        for i in range(n_feat):
            latent_features[:, :, :, i] = params_n[:, i, None, None]

        norm = data["normalisation"]
        t_mean, t_std = norm["t_mean"], norm["t_std"]
        fields = data["fields"]
        ntrain, ntest = data["ntrain"], data["ntest"]
        return dict(
            geometry=dict(
                input_geom=vertices,
                latent_queries=latent_queries,
                output_queries=vertices,
                neighbors_in=graph(vertices[0], lq_flat, arch["in_gno_radius"]),
                neighbors_out=graph(lq_flat, vertices[0], arch["out_gno_radius"]),
            ),
            lf_train=latent_features[:ntrain],
            lf_test=latent_features[-ntest:],
            y_train=[
                ((f[:, None][None] - t_mean) / t_std).astype(np.float32)
                for f in fields[:ntrain]
            ],
            y_test=[
                ((f[:, None][None] - t_mean) / t_std).astype(np.float32)
                for f in fields[-ntest:]
            ],
        )

    @staticmethod
    def inputs(prepared: dict, lf: np.ndarray) -> dict:
        """The model's keyword arguments for one sample's latent features."""
        return dict(prepared["geometry"], latent_features=lf)

    def init(self, model, arch, prepared, seed):
        return model.init(
            jax.random.PRNGKey(seed), **self.inputs(prepared, prepared["lf_train"][:1])
        )

    def predict(self, model, params, prepared, data, idx=None):
        # The training run's evaluation step (prediction and MAE), compiled
        # as one function as it was for the published predictions.
        def evaluate(p, inputs, y):
            out = model.apply(p, **inputs)
            return out, jnp.mean(jnp.abs(out - y))

        apply = _jitted(model, evaluate)
        norm = data["normalisation"]
        lf, y = prepared["lf_test"], prepared["y_test"]
        return np.stack(
            [
                np.asarray(
                    apply(params, self.inputs(prepared, lf[i : i + 1]), y[i])[0]
                )[0, :, 0]
                * norm["t_std"]
                + norm["t_mean"]
                for i in _test_idx(len(lf), idx)
            ]
        )


@dataclass(frozen=True)
class TransolverSpec(ModelSpec):
    """:class:`scirex.models.Transolver` on the mesh nodes.

    Each node's input is its (y, z), min-max scaled to [0, 1], and the six
    normalised parameters.
    """

    name: str = "transolver"
    label: str = "Transolver"
    kind: str = "mesh"

    def build(self, arch, grid):
        from scirex.models import Transolver

        return Transolver(**arch)

    def prepare(self, arch, data):
        stats = data["normalisation"]["coord_stats"]
        lo = np.array([stats["y_min"], stats["z_min"]], dtype=np.float32)
        hi = np.array([stats["y_max"], stats["z_max"]], dtype=np.float32)
        yz = np.asarray(data["coords"][:, 1:3], dtype=np.float32)
        x = ((yz - lo) / (hi - lo + 1e-8)).astype(np.float32)

        norm = data["normalisation"]
        fields_n = ((data["fields"] - norm["t_mean"]) / norm["t_std"]).astype(
            np.float32
        )
        params_n = data["params_n"]
        n_nodes = x.shape[0]

        def arrays(rows):
            X = np.broadcast_to(x, (len(rows), *x.shape)).astype(np.float32)
            FX = np.stack(
                [params_n[i][None, :].repeat(n_nodes, axis=0) for i in rows]
            ).astype(np.float32)
            Y = np.stack([fields_n[i][:, None] for i in rows]).astype(np.float32)
            return X, FX, Y

        n = len(params_n)
        X_tr, FX_tr, Y_tr = arrays(range(data["ntrain"]))
        X_te, FX_te, Y_te = arrays(range(n - data["ntest"], n))
        return dict(
            x_train=X_tr,
            fx_train=FX_tr,
            y_train=Y_tr,
            x_test=X_te,
            fx_test=FX_te,
            y_test=Y_te,
        )

    def init(self, model, arch, prepared, seed):
        _, key = jax.random.split(jax.random.PRNGKey(seed))
        return model.init(
            key,
            x=jnp.asarray(prepared["x_train"][:1], dtype=jnp.float32),
            fx=jnp.asarray(prepared["fx_train"][:1], dtype=jnp.float32),
        )

    def predict(self, model, params, prepared, data, idx=None):
        # Not jitted, as the published predictions were made.
        def apply(p, x, fx):
            return model.apply(p, x=x, fx=fx)

        norm = data["normalisation"]
        x, fx = prepared["x_test"], prepared["fx_test"]
        return np.stack(
            [
                np.asarray(apply(params, x[i : i + 1], fx[i : i + 1]))[0, :, 0]
                * norm["t_std"]
                + norm["t_mean"]
                for i in _test_idx(len(x), idx)
            ]
        )


#: Every architecture the benchmark knows, keyed by ``RunConfig.model``.
SPECS: Mapping[str, ModelSpec] = {
    spec.name: spec for spec in (FNOSpec(), WNOSpec(), GINOSpec(), TransolverSpec())
}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(SPECS)}"
        ) from None
