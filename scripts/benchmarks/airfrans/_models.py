# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architectures of the AirfRANS benchmark and how each is fed.

A model sees one case at a time: ``x`` ``(N, 9)`` normalised node features,
``pos`` ``(N, 2)`` raw node coordinates (chords), and an optional per-case
``extra`` input that does not depend on which nodes are sampled. It returns
``(N, 4)`` normalised scaled targets.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from scirex.models import WNO, Transolver
from scripts.benchmarks.airfrans._data import N_FEATURES

# The box of Transolver's unified-position reference grid (official
# models/Transolver.py get_grid): x in [-2, 4], y in [-1.5, 1.5] chords.
GRID_RANGE = ((-2.0, 4.0), (-1.5, 1.5))


class ModelSpec(ABC):
    """One architecture: how to build, initialise and apply it."""

    name: str
    label: str

    @abstractmethod
    def build(self, arch: Mapping[str, Any]) -> nn.Module:
        """The model at the size given by ``arch``."""

    @abstractmethod
    def init(
        self,
        model: nn.Module,
        key: jax.Array,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        extra: Any,
    ) -> Any:
        """Initial parameters."""

    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        params: Any,
        x: jnp.ndarray,
        pos: jnp.ndarray,
        extra: Any,
    ) -> jnp.ndarray:
        """``(N, 4)`` normalised prediction for the nodes ``x``, ``pos``."""

    def extra(self, arch: Mapping[str, Any], case: Mapping[str, np.ndarray]) -> Any:
        """Per-case input independent of the sampled nodes (none by default)."""
        return jnp.zeros((1,), jnp.float32)


def official_transolver_init(params: Any, key: jax.Array) -> Any:
    """The official ``Transolver._init_weights``: every Linear kernel
    ``trunc_normal_(std=0.02)``, biases 0.

    It runs after the blocks are built, so it also overwrites the orthogonal
    init of ``in_project_slice``. timm's truncation bounds are +-2 in absolute
    value, which is 100 std away at std 0.02; the clip keeps that rule.
    """
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = jax.random.split(key, len(leaves))
    out = []
    for (path, leaf), k in zip(leaves, keys, strict=True):
        if jax.tree_util.keystr(path).endswith("['kernel']") and leaf.ndim == 2:
            leaf = jnp.clip(
                0.02 * jax.random.normal(k, leaf.shape, leaf.dtype), -2.0, 2.0
            )
        out.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


@dataclass(frozen=True)
class TransolverSpec(ModelSpec):
    """:class:`scirex.models.Transolver` as in the official
    AirfRANS ``main.py``: unified-position grid (8 x 8 reference points over
    :data:`GRID_RANGE`, distances concatenated to the features), exact GELU as
    ``torch.nn.GELU``, and the official weight initialisation."""

    name: str = "transolver"
    label: str = "Transolver"

    def build(self, arch: Mapping[str, Any]) -> nn.Module:
        return Transolver(
            space_dim=N_FEATURES,
            fun_dim=0,
            out_dim=4,
            unified_pos=True,
            concat_pos=True,
            geometry="irregular",
            grid_range=GRID_RANGE,
            dropout_rate=0.0,
            act=functools.partial(jax.nn.gelu, approximate=False),
            **arch,
        )

    def init(self, model, key, x, pos, extra):
        k_init, k_official = jax.random.split(key)
        params = model.init(k_init, x=x[None], fx=None, pos=pos[None])
        return official_transolver_init(params, k_official)

    def apply(self, model, params, x, pos, extra):
        return model.apply(params, x=x[None], fx=None, pos=pos[None])[0]


# ---------------------------------------------------------------------------
# WNO on the irregular AirfRANS mesh
# ---------------------------------------------------------------------------

# Airfoil outlines have 843 to 1,217 segments; padding to one size keeps a
# single compiled geometry function.
MAX_SEGMENTS = 1280


def grid_points(H: int, W: int) -> jnp.ndarray:
    """``(H * W, 2)`` cell centres of the H x W grid over :data:`GRID_RANGE`,
    row-major in ``(y, x)``."""
    xs = jnp.linspace(GRID_RANGE[0][0], GRID_RANGE[0][1], W)
    ys = jnp.linspace(GRID_RANGE[1][0], GRID_RANGE[1][1], H)
    gy, gx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([gx, gy], -1).reshape(-1, 2)


def signed_distance(
    points: jnp.ndarray, seg: jnp.ndarray, n_seg: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Distance of ``points`` ``(P, 2)`` to the outline ``seg`` ``(S, 2, 2)``
    (padded, ``n_seg`` real segments), negative inside; inside means an odd
    number of crossings of a ray in +x, ignoring padded segments."""
    a, b = seg[:, 0], seg[:, 1]
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = jnp.clip(
        jnp.sum(ap * ab[None], -1) / jnp.maximum(jnp.sum(ab * ab, -1), 1e-12)[None],
        0.0,
        1.0,
    )
    d = jnp.sqrt(jnp.min(jnp.sum((ap - t[..., None] * ab[None]) ** 2, -1), axis=1))
    py, px = points[:, 1:2], points[:, 0:1]
    ay, by, ax, bx = a[None, :, 1], b[None, :, 1], a[None, :, 0], b[None, :, 0]
    straddle = (ay > py) != (by > py)
    xcross = ax + (py - ay) * (bx - ax) / jnp.where(by == ay, 1e-12, by - ay)
    real = jnp.arange(seg.shape[0])[None] < n_seg
    inside = (jnp.sum(straddle & (px < xcross) & real, axis=1) % 2) == 1
    return jnp.where(inside, -d, d), inside


@functools.partial(jax.jit, static_argnums=(2, 3))
def grid_geometry(seg: jnp.ndarray, n_seg: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """``(H, W, 2)`` geometry channels: tanh(signed distance / 0.5) and the
    inside mask."""
    sdf, inside = signed_distance(grid_points(H, W), seg, n_seg)
    return jnp.stack([jnp.tanh(sdf / 0.5), inside.astype(jnp.float32)], -1).reshape(
        H, W, 2
    )


def bilinear(grid: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """Sample ``grid`` ``(H, W, C)`` over :data:`GRID_RANGE` at ``pos``
    ``(N, 2)``; positions outside the box are clamped to its edge."""
    H, W, _ = grid.shape
    (x0b, x1b), (y0b, y1b) = GRID_RANGE
    fx = (jnp.clip(pos[:, 0], x0b, x1b) - x0b) / (x1b - x0b) * (W - 1)
    fy = (jnp.clip(pos[:, 1], y0b, y1b) - y0b) / (y1b - y0b) * (H - 1)
    x0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, W - 2)
    y0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, H - 2)
    wx, wy = (fx - x0)[:, None], (fy - y0)[:, None]
    return (1 - wy) * ((1 - wx) * grid[y0, x0] + wx * grid[y0, x0 + 1]) + wy * (
        (1 - wx) * grid[y0 + 1, x0] + wx * grid[y0 + 1, x0 + 1]
    )


class WNOGridOperator(nn.Module):
    """:class:`scirex.models.WNO` between a grid encoder and a node
    decoder, for the irregular AirfRANS mesh.

    * Encoder: an ``H x W`` grid over :data:`GRID_RANGE` with the case's
      geometry (tanh signed distance and inside mask, from
      :func:`grid_geometry`) and the normalised inflow; WNO appends the grid
      coordinates.
    * WNO: lift, ``n_layers`` wavelet blocks, projection to ``latent``
      channels, in float32.
    * Decoder: for every queried node, the latent field sampled bilinearly at
      the node, concatenated with the node's own features, through an MLP to
      the 4 outputs.
    """

    H: int = 128
    W: int = 256
    hidden: int = 28
    n_layers: int = 4
    level: int = 4
    wavelet: str = "db4"
    latent: int = 64
    dec_hidden: int = 256
    dec_layers: int = 3

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, pos: jnp.ndarray, geo: jnp.ndarray
    ) -> jnp.ndarray:
        inflow = x[0, 2:4]  # the normalised inflow, the same at every node
        g = jnp.concatenate([geo, jnp.broadcast_to(inflow, (self.H, self.W, 2))], -1)
        g = jax.lax.stop_gradient(g)[None]
        lat = WNO(
            in_channels=4,
            out_channels=self.latent,
            hidden_channels=self.hidden,
            n_layers=self.n_layers,
            level=self.level,
            size=(self.H, self.W),
            wavelet=self.wavelet,
            use_grid=True,
            grid_boundaries=(GRID_RANGE[1], GRID_RANGE[0]),
            compute_dtype=jnp.float32,
            param_dtype=jnp.float32,
            projection_hidden_channels=128,
        )(g)[0]
        h = jnp.concatenate([bilinear(lat, pos), x], -1)
        for _ in range(self.dec_layers - 1):
            h = nn.gelu(nn.Dense(self.dec_hidden)(h), approximate=False)
        return nn.Dense(4)(h)


@dataclass(frozen=True)
class WNOSpec(ModelSpec):
    """:class:`WNOGridOperator`; its per-case input is the grid geometry."""

    name: str = "wno"
    label: str = "WNO"

    def build(self, arch: Mapping[str, Any]) -> nn.Module:
        return WNOGridOperator(**arch)

    def extra(self, arch: Mapping[str, Any], case: Mapping[str, np.ndarray]) -> Any:
        seg = np.asarray(case["segments"], np.float32)
        if len(seg) > MAX_SEGMENTS:
            raise ValueError(
                f"{len(seg)} outline segments > MAX_SEGMENTS={MAX_SEGMENTS}"
            )
        padded = np.concatenate([seg, np.repeat(seg[-1:], MAX_SEGMENTS - len(seg), 0)])
        H = arch.get("H", WNOGridOperator.H)
        W = arch.get("W", WNOGridOperator.W)
        return grid_geometry(jnp.asarray(padded), jnp.asarray(len(seg)), H, W)

    def init(self, model, key, x, pos, extra):
        return model.init(key, x, pos, extra)

    def apply(self, model, params, x, pos, extra):
        return model.apply(params, x, pos, extra)


SPECS: dict[str, ModelSpec] = {s.name: s for s in (TransolverSpec(), WNOSpec())}


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {', '.join(SPECS)}"
        ) from None
