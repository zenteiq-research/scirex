# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Validation step 1 of the physics-loss roadmap's Phase 3b: cross-check
:class:`~scirex.experimental.losses.ad_differentiation.ADDiff` (the O(1) sum
trick) against
:class:`~scirex.experimental.losses.exact_ad_differentiation.ExactPointwiseADDiff`
(the O(n) exact fallback) on a real
:class:`~scirex.layers.blocks.gno_block.GNOBlock` configured the way
GINO's own ``gno_out`` decode step uses it -- not a synthetic
block-diagonal toy function, an actual architecture.

Since GINO is block-diagonal in the output query coordinates (see
:mod:`~scirex.experimental.losses.ad_differentiation`'s module docstring), both
differentiators must agree exactly here. If they didn't, either ``ADDiff``'s
sum trick would be broken, or the block-diagonality claim about GINO's
decode step would be wrong -- this test is the empirical check for both,
independent of the analytic-function tests each differentiator already has
in its own test module.

Why neighbors must be precomputed outside the differentiated closure
-----------------------------------------------------------------------
:class:`~scirex.layers.blocks.neighbor_search.NeighborSearch`'s native
fallback (used here since ``coord_dim=2`` -- open3d requires 3D) calls
``scipy.spatial.cKDTree`` on concrete NumPy arrays, which cannot run inside
a ``jax.grad``-traced function at all (a KDTree query has no VJP rule, and
can't accept a tracer). Passing ``neighbors=None`` in the closure would
therefore either error or force the search itself to be non-differentiable
in a way that couldn't participate in the AD graph. The fix used here
matches how :class:`~scirex.models.gino.GINO` itself handles this in
practice (:meth:`GINO.precompute_neighbors`): compute the neighbor
CRS-format dict once, from the *same* query coordinates the derivative will
be requested at, entirely outside any traced function, then pass it in as
the (fixed) ``neighbors=`` argument. Differentiating w.r.t. ``coords``
still reaches every place ``x`` enters
:class:`~scirex.layers.blocks.integral_transform.IntegralTransform`
(``self_features = x[safe_seg_ids]``, gathered per query row) -- only the
*which-points-are-neighbors* structure is held fixed, which is also exactly
why block-diagonality holds: each output row only ever gathers from its own
query's coordinate, never another row's.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from scirex.experimental.losses.ad_differentiation import ADDiff
from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from scirex.layers.blocks.gno_block import GNOBlock
from scirex.layers.blocks.neighbor_search import NeighborSearch
from tests import tolerances

_COORD_DIM = 2
_N_SOURCE = 6
_N_QUERY = 9
_RADIUS = 10.0  # larger than the domain diameter -> every query sees every source point


def _build_gno_closure() -> tuple[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]:
    """Build a small GNOBlock (GINO's gno_out decode step, standalone),
    precompute its neighbor structure, and return an unbatched closure
    ``f(coords) -> (n_query,)`` plus the query coordinates it's valid at.
    """
    key_y, key_fy, key_x, key_params = jax.random.split(jax.random.PRNGKey(0), 4)
    y = jax.random.uniform(key_y, (_N_SOURCE, _COORD_DIM), minval=-1.0, maxval=1.0)
    f_y = jax.random.normal(key_fy, (_N_SOURCE, 3))
    x0 = jax.random.uniform(key_x, (_N_QUERY, _COORD_DIM), minval=-1.0, maxval=1.0)

    raw = cast(
        dict[str, jnp.ndarray],
        NeighborSearch(use_open3d=False).apply({}, data=y, queries=x0, radius=_RADIUS),
    )
    row_splits = raw["neighbors_row_splits"]
    counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
    segment_ids = jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)
    neighbors = {
        "neighbors_index": raw["neighbors_index"],
        "segment_ids": segment_ids,
        "counts": counts,
    }
    # sanity: radius chosen so every query has all _N_SOURCE points as neighbors
    assert bool(jnp.all(counts == _N_SOURCE))

    gno_out = GNOBlock(
        in_channels=3,
        out_channels=1,
        coord_dim=_COORD_DIM,
        radius=_RADIUS,
        transform_type="linear",
        pos_embedding_type=None,  # keep x's path into the kernel MLP direct
        channel_mlp_layers=[16],
        reduction="sum",
        use_open3d_neighbor_search=False,
    )
    params = gno_out.init(key_params, y=y, x=x0, f_y=f_y, neighbors=neighbors)

    def f(coords: jnp.ndarray) -> jnp.ndarray:
        raw_out = gno_out.apply(params, y=y, x=coords, f_y=f_y, neighbors=neighbors)
        return cast(jnp.ndarray, raw_out)[:, 0]

    return f, x0


def test_addiff_matches_exactpointwiseaddiff_on_gno_decode_step() -> None:
    f, coords = _build_gno_closure()

    addiff_grad = ADDiff().derivative(f, coords, dim=0, order=1)
    exact_grad = ExactPointwiseADDiff().derivative(f, coords, dim=0, order=1)

    assert jnp.allclose(addiff_grad, exact_grad, atol=tolerances.atol(1e-5))
    # not a degenerate all-zero agreement
    assert jnp.max(jnp.abs(exact_grad)) > 1e-6
