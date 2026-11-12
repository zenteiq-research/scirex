# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architectures of the Navier-Stokes benchmark, built from this repo's
operators.

Nothing here is a network. Each :class:`ModelSpec` builds a model that
already lives in ``src/scirex/`` and adapts the benchmark's data to it. Every
model sees the same input, the last ``t_in`` vorticity frames as
``(batch, H * W, t_in)``, and returns the next frame as ``(batch, H * W, 1)``:

* :meth:`~ModelSpec.build` makes the model at the size given by ``arch``;
* :meth:`~ModelSpec.init` initialises its parameters (Transolver also applies
  the reference's own weight initialisation);
* :meth:`~ModelSpec.apply` runs one step.

so ``train.py`` and ``infer.py`` handle every architecture the same way.
:data:`SPECS` maps each name to its spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from scirex.layers import build_neighbor_graph, edge_index_to_neighbors
from scirex.models import FNO, GINO, WNO, Transolver
from scripts.benchmarks.pipe._transolver import _match_official_weight_init


@dataclass(frozen=True)
class ModelSpec(ABC):
    """One architecture: how to build it and how to feed it."""

    name: str
    label: str

    @abstractmethod
    def build(
        self, arch: Mapping[str, Any], grid: tuple[int, int], t_in: int
    ) -> nn.Module:
        """The model at the size given by ``arch``, for an ``H x W`` grid."""

    @abstractmethod
    def apply(
        self, model: nn.Module, params: Any, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> jnp.ndarray:
        """Next frame for the history ``fx``, ``(batch, H * W, t_in)`` ->
        ``(batch, H * W, 1)``."""

    def init(
        self,
        model: nn.Module,
        key: jax.Array,
        fx: jnp.ndarray,
        grid: tuple[int, int],
        seed: int = 0,
    ) -> Any:
        """Fresh parameters for ``model``, given one batch ``fx``."""
        return model.init(key, **self._inputs(model, fx, grid))

    @abstractmethod
    def _inputs(
        self, model: nn.Module, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        """The model's keyword arguments for the history ``fx``."""


def grid_points(grid: tuple[int, int]) -> np.ndarray:
    """``(H * W, 2)`` coordinates in [0, 1]^2, in the data's flattening order
    (``u[:, i, j]`` -> point ``i * W + j``)."""
    xs = np.linspace(0.0, 1.0, grid[0], dtype=np.float32)
    ys = np.linspace(0.0, 1.0, grid[1], dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


@dataclass(frozen=True)
class TransolverSpec(ModelSpec):
    """:class:`scirex.models.Transolver` on the structured
    grid, as in the paper's ``exp_ns.py``.

    ``unified_pos=True`` replaces the ``(x, y)`` input with a learned
    positional feature, so the coordinates passed in only give the batch
    size. ``arch["gelu"]`` selects the activation: ``"exact"`` (PyTorch's
    ``nn.GELU``) or ``"tanh"`` (Flax's default approximation).
    """

    name: str = "transolver"
    label: str = "Transolver"

    def build(
        self, arch: Mapping[str, Any], grid: tuple[int, int], t_in: int
    ) -> nn.Module:
        arch = dict(arch)
        gelu = arch.pop("gelu", "tanh")
        if gelu not in ("exact", "tanh"):
            raise ValueError(f"gelu must be 'exact' or 'tanh', got {gelu!r}")
        return Transolver(
            space_dim=2,
            fun_dim=t_in,
            out_dim=1,
            geometry="structured_2d",
            H=grid[0],
            W=grid[1],
            act=partial(jax.nn.gelu, approximate=gelu == "tanh"),
            **arch,
        )

    def _inputs(
        self, model: nn.Module, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        pts = jnp.asarray(grid_points(grid))
        pos = jnp.broadcast_to(pts, (fx.shape[0], *pts.shape))
        return dict(x=pos, fx=fx)

    def init(
        self,
        model: nn.Module,
        key: jax.Array,
        fx: jnp.ndarray,
        grid: tuple[int, int],
        seed: int = 0,
    ) -> Any:
        """Flax init, then the reference's ``Model.initialize_weights``:
        every Linear kernel ~ N(0, 0.02^2) except ``in_project_slice``."""
        params = model.init(key, **self._inputs(model, fx, grid))
        return _match_official_weight_init(
            params, seed=seed, n_layers=getattr(model, "n_layers", None)
        )

    def apply(
        self, model: nn.Module, params: Any, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> jnp.ndarray:
        return model.apply(params, **self._inputs(model, fx, grid))


@dataclass(frozen=True)
class FNOSpec(ModelSpec):
    """:class:`scirex.models.FNO`, which is channels-first and
    appends its own grid channels (``positional_embedding="grid"``)."""

    name: str = "fno"
    label: str = "FNO"

    def build(
        self, arch: Mapping[str, Any], grid: tuple[int, int], t_in: int
    ) -> nn.Module:
        return FNO(in_channels=t_in, out_channels=1, **arch)

    def _inputs(
        self, model: nn.Module, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        x = fx.reshape(fx.shape[0], *grid, fx.shape[-1])
        return dict(x=jnp.transpose(x, (0, 3, 1, 2)))

    def apply(
        self, model: nn.Module, params: Any, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> jnp.ndarray:
        out = model.apply(params, **self._inputs(model, fx, grid))  # (batch, 1, H, W)
        return jnp.transpose(out, (0, 2, 3, 1)).reshape(fx.shape[0], -1, 1)


@dataclass(frozen=True)
class WNOSpec(ModelSpec):
    """:class:`scirex.models.WNO`, which is channels-last and
    appends its own grid channels (``use_grid=True``). Computes in float32,
    like the other three, not WNO's default bfloat16."""

    name: str = "wno"
    label: str = "WNO"

    def build(
        self, arch: Mapping[str, Any], grid: tuple[int, int], t_in: int
    ) -> nn.Module:
        return WNO(
            in_channels=t_in,
            out_channels=1,
            size=grid,
            compute_dtype=jnp.float32,
            **arch,
        )

    def _inputs(
        self, model: nn.Module, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        return dict(x=fx.reshape(fx.shape[0], *grid, fx.shape[-1]))

    def apply(
        self, model: nn.Module, params: Any, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> jnp.ndarray:
        out = model.apply(params, **self._inputs(model, fx, grid))  # (batch, H, W, 1)
        return out.reshape(fx.shape[0], -1, 1)


@lru_cache(maxsize=4)
def _grid_graph(grid: tuple[int, int], radius: float) -> dict[str, jnp.ndarray]:
    """The radius graph of the grid to itself, built once per grid."""
    pts = grid_points(grid)
    edges = build_neighbor_graph(pts, pts, radius, "radius")
    return edge_index_to_neighbors(jnp.asarray(edges), num_query=pts.shape[0])


@dataclass(frozen=True)
class GINOSpec(ModelSpec):
    """:class:`scirex.models.GINO` on the fixed grid.

    Every sample lives on the same 64x64 grid, so one radius graph from the
    grid to itself serves both GNO layers for every sample and batch. Input,
    latent and output points are the grid points, and the ``t_in`` history
    frames are the per-point input features. ``fno_in_channels`` equals
    ``t_in`` because the input GNO's linear kernel multiplies the features
    channel by channel.
    """

    name: str = "gino"
    label: str = "GINO"

    def build(
        self, arch: Mapping[str, Any], grid: tuple[int, int], t_in: int
    ) -> nn.Module:
        model = GINO(
            in_channels=t_in,
            out_channels=1,
            gno_coord_dim=2,
            fno_in_channels=t_in,
            **arch,
        )
        if model.in_gno_radius != model.out_gno_radius:
            raise ValueError(
                "GINOSpec builds one graph for both GNO layers and needs "
                f"in_gno_radius == out_gno_radius, got {model.in_gno_radius} "
                f"and {model.out_gno_radius}"
            )
        return model

    def _inputs(
        self, model: nn.Module, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        pts = jnp.asarray(grid_points(grid))[None]
        nbrs = _grid_graph(tuple(grid), float(model.in_gno_radius))
        return dict(
            input_geom=pts,
            latent_queries=pts.reshape(1, *grid, 2),
            output_queries=pts,
            x=fx,
            neighbors_in=nbrs,
            neighbors_out=nbrs,
        )

    def apply(
        self, model: nn.Module, params: Any, fx: jnp.ndarray, grid: tuple[int, int]
    ) -> jnp.ndarray:
        out = model.apply(params, **self._inputs(model, fx, grid))
        return out.reshape(fx.shape[0], -1, 1)


#: Every architecture the benchmark knows, keyed by the name used in
#: ``RunConfig.model``.
SPECS: Mapping[str, ModelSpec] = {
    spec.name: spec for spec in (TransolverSpec(), FNOSpec(), WNOSpec(), GINOSpec())
}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(SPECS)}"
        ) from None
