# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architectures of the Pipe benchmark, built from this repo's operators.

Nothing here is a network. Each :class:`ModelSpec` builds a model that
already lives in ``src/scirex/`` and adapts the benchmark's data to it:

* :meth:`~ModelSpec.prepare` turns normalised coordinates, ``(n, H * W, 2)``,
  into whatever the model consumes for those samples (a grid-shaped array,
  per-sample neighbour graphs, ...);
* :meth:`~ModelSpec.take` picks the model's keyword arguments for a batch of
  sample indices from that;
* :meth:`~ModelSpec.apply` runs the model on them and returns the
  normalised prediction as ``(batch, H * W, 1)``.

so ``train.py`` and ``infer.py`` handle every architecture the same way.
:data:`SPECS` maps each name to its spec: adding an architecture means adding
one entry and touching no ``if``/``elif`` anywhere in the benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from scirex.layers import build_neighbor_graph, edge_index_to_neighbors
from scirex.models import FNO, GINO, WNO, Transolver


@dataclass(frozen=True)
class ModelSpec(ABC):
    """One architecture: how to build it and how to feed it.

    ``batch_size`` is how many samples an architecture with no batch axis
    takes at a time; ``train.py`` sums their gradients over the protocol's
    batch before each optimiser step. ``None`` takes the protocol's batch.
    """

    name: str
    label: str
    batch_size: int | None = None

    @abstractmethod
    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        """The model at the size given by ``arch``, for an ``H x W`` mesh."""

    @abstractmethod
    def prepare(
        self, arch: Mapping[str, Any], x: np.ndarray, grid: tuple[int, int]
    ) -> Any:
        """Model inputs for the samples in ``x``, normalised ``(n, H * W, 2)``."""

    @abstractmethod
    def take(self, prepared: Any, idx: np.ndarray) -> dict[str, jnp.ndarray]:
        """The model's keyword arguments for the samples ``idx``."""

    @abstractmethod
    def apply(
        self, model: nn.Module, params: Any, inputs: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Normalised prediction for ``inputs``, as ``(batch, H * W, 1)``."""

    def template(self, grid: tuple[int, int]) -> np.ndarray:
        """One sample of coordinates for building a parameter template."""
        return template_inputs(grid)

    def post_init(self, params: Any, arch: Mapping[str, Any], seed: int) -> Any:
        """Adjusts freshly initialised weights; the default keeps Flax's."""
        return params


@dataclass(frozen=True)
class TransolverSpec(ModelSpec):
    """:class:`scirex.models.Transolver` on the structured
    129x129 mesh, as in the paper's ``exp_pipe.py``.

    The input is the raw ``(x, y)`` coordinates (``fun_dim=0``,
    ``unified_pos=False``). The numerics follow the PyTorch reference: exact
    (erf) GELU, LayerNorm eps 1e-5 and ``torch.nn.Conv2d``'s default init for
    the slice convolutions; :meth:`post_init` applies the reference's
    ``Model.initialize_weights`` (``_transolver._match_official_weight_init``).
    """

    name: str = "transolver"
    label: str = "Transolver"

    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        return Transolver(
            space_dim=2,
            fun_dim=0,
            out_dim=1,
            unified_pos=False,
            geometry="structured_2d",
            H=grid[0],
            W=grid[1],
            act=partial(jax.nn.gelu, approximate=False),
            layer_norm_epsilon=1e-5,
            conv_init="torch",
            **arch,
        )

    def prepare(
        self, arch: Mapping[str, Any], x: np.ndarray, grid: tuple[int, int]
    ) -> np.ndarray:
        return x

    def take(self, prepared: np.ndarray, idx: np.ndarray) -> dict[str, jnp.ndarray]:
        return dict(x=jnp.asarray(prepared[idx]))

    def apply(
        self, model: nn.Module, params: Any, inputs: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        return model.apply(params, x=inputs["x"], fx=None)  # (batch, H * W, 1)

    def post_init(self, params: Any, arch: Mapping[str, Any], seed: int) -> Any:
        from scripts.benchmarks.pipe._transolver import _match_official_weight_init

        return _match_official_weight_init(params, seed=seed, n_layers=arch["n_layers"])


@dataclass(frozen=True)
class WNOSpec(ModelSpec):
    """:class:`scirex.models.WNO`, which is channels-last.

    ``use_grid=False``: like every run in this benchmark, the input is the
    raw ``(x, y)`` coordinates only, with no extra grid channels.
    ``compute_dtype=float32``: every model in the benchmark computes in
    float32 (WNO's own default is bfloat16).
    """

    name: str = "wno"
    label: str = "WNO"

    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        return WNO(
            in_channels=2,
            out_channels=1,
            size=grid,
            use_grid=False,
            compute_dtype=jnp.float32,
            **arch,
        )

    def prepare(
        self, arch: Mapping[str, Any], x: np.ndarray, grid: tuple[int, int]
    ) -> np.ndarray:
        return x.reshape(x.shape[0], *grid, x.shape[-1])

    def take(self, prepared: np.ndarray, idx: np.ndarray) -> dict[str, jnp.ndarray]:
        return dict(x=jnp.asarray(prepared[idx]))

    def apply(
        self, model: nn.Module, params: Any, inputs: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        out = model.apply(params, inputs["x"])  # (batch, H, W, 1)
        return out.reshape(out.shape[0], -1, out.shape[-1])


@dataclass(frozen=True)
class FNOSpec(ModelSpec):
    """:class:`scirex.models.FNO`, which is channels-first.

    ``positional_embedding=None``: like every run in this benchmark, the
    input is the raw ``(x, y)`` coordinates only, with no extra grid
    channels.
    """

    name: str = "fno"
    label: str = "FNO"

    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        return FNO(in_channels=2, out_channels=1, positional_embedding=None, **arch)

    def prepare(
        self, arch: Mapping[str, Any], x: np.ndarray, grid: tuple[int, int]
    ) -> np.ndarray:
        return x.reshape(x.shape[0], *grid, x.shape[-1]).transpose(0, 3, 1, 2)

    def take(self, prepared: np.ndarray, idx: np.ndarray) -> dict[str, jnp.ndarray]:
        return dict(x=jnp.asarray(prepared[idx]))

    def apply(
        self, model: nn.Module, params: Any, inputs: dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        out = model.apply(params, inputs["x"])  # (batch, 1, H, W)
        return out.reshape(out.shape[0], out.shape[1], -1).transpose(0, 2, 1)


@dataclass(frozen=True)
class GINOSpec(ModelSpec):
    """:class:`scirex.models.GINO`, one sample at a time.

    Every Pipe sample has its own deformed mesh, so each needs its own
    neighbour graph. One radius graph from the mesh to itself serves both
    GNO layers, and the latent grid of the FNO processor is the sample's own
    129x129 mesh. :meth:`prepare` builds the graphs once per sample, padded
    to ``max_edges`` so that every sample has the same shapes and the jitted
    step compiles once.

    The geometry enters only through the node coordinates; the node feature
    is a constant 1.
    """

    name: str = "gino"
    label: str = "GINO"
    batch_size: int | None = 1
    # At radius 0.05 the graphs have about 306,000 to 311,000 edges on Pipe
    # samples, so 320,000 pads all of them.
    max_edges: int = 320_000
    max_num_neighbors: int = 32

    def build(self, arch: Mapping[str, Any], grid: tuple[int, int]) -> nn.Module:
        return GINO(in_channels=1, out_channels=1, gno_coord_dim=2, **arch)

    def prepare(
        self, arch: Mapping[str, Any], x: np.ndarray, grid: tuple[int, int]
    ) -> dict[str, Any]:
        model = self.build(arch, grid)
        if model.in_gno_radius != model.out_gno_radius:
            # One graph serves both GNO layers, so it fits only one radius.
            raise ValueError(
                f"GINOSpec builds one graph for both GNO layers and needs "
                f"in_gno_radius == out_gno_radius, got {model.in_gno_radius} "
                f"and {model.out_gno_radius}"
            )
        n, n_pts, _ = x.shape
        index = np.empty((n, self.max_edges), dtype=np.int32)
        segment = np.empty((n, self.max_edges), dtype=np.int32)
        counts = np.empty((n, n_pts), dtype=np.int32)
        for i in range(n):
            edges = build_neighbor_graph(
                x[i],
                x[i],
                model.in_gno_radius,
                "radius",
                max_edges=self.max_edges,
                max_num_neighbors=self.max_num_neighbors,
            )
            # build_neighbor_graph truncates silently past max_edges, and pads
            # with the query count, so a graph with no padding left may have
            # lost edges.
            if np.count_nonzero(edges[1] < n_pts) >= self.max_edges:
                raise ValueError(
                    f"sample {i}: neighbour graph fills max_edges="
                    f"{self.max_edges} and may be truncated; raise it"
                )
            nbrs = edge_index_to_neighbors(jnp.asarray(edges), num_query=n_pts)
            index[i] = np.asarray(nbrs["neighbors_index"])
            segment[i] = np.asarray(nbrs["segment_ids"])
            counts[i] = np.asarray(nbrs["counts"])
        return dict(
            pos=x.astype(np.float32),
            grid=grid,
            neighbors_index=index,
            segment_ids=segment,
            counts=counts,
        )

    def take(self, prepared: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
        (i,) = np.asarray(idx).reshape(-1)  # one sample per step
        pos = jnp.asarray(prepared["pos"][i])[None]  # (1, H * W, 2)
        nbrs = {
            key: jnp.asarray(prepared[key][i])
            for key in ("neighbors_index", "segment_ids", "counts")
        }
        return dict(
            input_geom=pos,
            latent_queries=pos.reshape(1, *prepared["grid"], 2),
            output_queries=pos,
            x=jnp.ones((1, pos.shape[1], 1), dtype=jnp.float32),
            neighbors_in=nbrs,
            neighbors_out=nbrs,
        )

    def apply(
        self, model: nn.Module, params: Any, inputs: dict[str, Any]
    ) -> jnp.ndarray:
        return model.apply(params, **inputs)  # (1, H * W, 1)

    def template(self, grid: tuple[int, int]) -> np.ndarray:
        """A regular grid over the normalised Pipe coordinate range.

        The radius graph's size depends on the point density: on the
        generic [-1, 1]^2 grid nearly every node would reach
        ``max_num_neighbors`` and the graph would overflow ``max_edges``.
        """
        x = template_inputs(grid)
        return (x * np.array([1.75, 3.15], np.float32)).astype(np.float32)


#: Every architecture the benchmark knows, keyed by the name used in
#: ``RunConfig.model``.
SPECS: Mapping[str, ModelSpec] = {
    spec.name: spec for spec in (TransolverSpec(), WNOSpec(), FNOSpec(), GINOSpec())
}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(SPECS)}"
        ) from None


def template_inputs(grid: tuple[int, int]) -> np.ndarray:
    """One sample of normalised coordinates on a regular grid in [-1, 1]^2.

    Parameter shapes depend only on the architecture and the mesh, so this
    is enough to build a parameter template for loading a checkpoint.
    """
    gx, gy = np.meshgrid(
        np.linspace(-1.0, 1.0, grid[0]), np.linspace(-1.0, 1.0, grid[1]), indexing="ij"
    )
    return np.stack([gx, gy], axis=-1).reshape(1, -1, 2).astype(np.float32)
