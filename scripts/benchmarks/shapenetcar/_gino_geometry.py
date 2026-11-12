# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""GINO's point-cloud geometry for this benchmark: a fixed latent grid, and
the per-sample neighbor search that connects each car's mesh to it.

Unlike ``lid_driven_cavity``'s ``gino_geometry`` (one 128x128 grid, shared and
cached across every sample because the cavity mesh never changes), every
ShapeNetCar sample is a *different* car, so the input-to-latent and
latent-to-output neighbor lists differ per sample and cannot be cached by
shape alone. What *is* shared across every sample is the coordinate frame:
``load_shapenet_car`` normalizes ``x`` (including the leading 3 position
channels) to zero mean / unit std over the whole training set before any
model sees it, so one fixed box safely covers every car -- there is no
per-sample bounding box to track. :data:`_DOMAIN_MIN`/:data:`_DOMAIN_MAX` are
that box, computed once from the real fold-0 split (train+val, 789+100
samples) with a 10% margin:

    pos min [-1.8280482 -2.2077749 -2.2663684]
    pos max [1.8559335  5.6664624  2.4517229]

:func:`build_latent_queries` builds the regular grid over that box.
:func:`attach_neighbors` runs the one-time (not per-epoch) KD-tree search
linking each sample's 32,186 mesh points to it. Called once after
``_data.load_split`` and before training starts, over *every* sample that
will be used (train and val together -- see its own docstring for why);
the result is plain numpy-backed CSR dicts attached to each sample dict,
consumed by ``train.py``/``_physics.py`` only when the active run's model is
``"gino"``.

Every sample's neighbor lists are then padded to one fixed length (the
longest list seen across the whole split). This is not a convenience: each
car's point cloud finds a different number of neighbors, and
``train.py``'s ``train_step``/``eval_step`` are ``jax.jit``-compiled, so
without padding, every sample with a new total edge count forces a fresh
XLA compile. Measured on this GPU, that cost ~6.5s *per sample* (not just
the first one), which made a 200-epoch, 789-sample run run for weeks rather
than hours. Padding every sample to the same shape means ``jax.jit``
compiles ``train_step``/``eval_step`` exactly once for the whole run.
``scirex._internal.segment_csr.segment_csr``'s own docstring already
describes this contract (padded edges map to a dropped ``n_out``'th
segment, so they contribute nothing) -- :func:`_pad_to` below is the one
place that contract is produced; the model otherwise returns identical
results whether it sees a padded or an unpadded neighbor dict.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from scirex.layers import kdtree_neighbor_search, neighbors_to_segments

#: 10% margin beyond the observed fold-0 train+val extent (module docstring).
_DOMAIN_MIN = np.array([-2.0, -2.5, -2.5], dtype=np.float32)
_DOMAIN_MAX = np.array([2.0, 6.2, 2.7], dtype=np.float32)


@lru_cache(maxsize=4)
def build_latent_queries(latent_grid: int) -> np.ndarray:
    """A ``(latent_grid, latent_grid, latent_grid, 3)`` regular grid over the
    fixed domain, row-major in (x, y, z) -- matching ``np.meshgrid``'s default
    ``indexing="xy"`` swapped explicitly to ``"ij"`` so axis order is literal.

    Cached on ``latent_grid`` alone (a handful of distinct sizes at most, one
    per run): both :class:`~scripts.benchmarks.shapenetcar._models.
    GINOCarWrapper` (every forward pass) and :func:`attach_neighbors` (once
    per split) call this and must get the identical array, or the
    precomputed neighbor indices would point at different grid points than
    the ones the model actually builds.
    """
    axes = [
        np.linspace(lo, hi, latent_grid, dtype=np.float32)
        for lo, hi in zip(_DOMAIN_MIN, _DOMAIN_MAX, strict=True)
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack(grids, axis=-1)


def _pad_to(neighbors: dict, n_segments: int, max_edges: int) -> dict:
    """Extends ``neighbors``' edge-indexed arrays to ``max_edges``, with every
    padding edge mapped to segment ``n_segments`` -- one past the last real
    query -- so ``segment_csr`` (``num_segments=n_segments + 1``, then
    ``out[:n_segments]``) drops it. ``counts`` (per-query, not per-edge) is
    left untouched: it already holds the true neighbor counts, which is what
    a mean reduction must divide by.
    """
    index = np.asarray(neighbors["neighbors_index"])
    segment_ids = np.asarray(neighbors["segment_ids"])
    n_pad = max_edges - index.shape[0]
    return {
        "neighbors_index": np.concatenate([index, np.zeros(n_pad, dtype=index.dtype)]),
        "segment_ids": np.concatenate(
            [segment_ids, np.full(n_pad, n_segments, dtype=segment_ids.dtype)]
        ),
        "counts": neighbors["counts"],
    }


def attach_neighbors(
    samples: list[dict],
    latent_grid: int,
    in_radius: float,
    out_radius: float,
) -> None:
    """Adds ``neighbors_in``/``neighbors_out`` to every sample dict, in place,
    padded to one fixed shape across ``samples`` (this module's docstring).

    ``neighbors_in``: each of the latent grid's points <- the sample's mesh
    points within ``in_radius`` (what the input GNO integrates over).
    ``neighbors_out``: each mesh point <- latent points within ``out_radius``
    (what the output GNO integrates over to decode back onto the mesh).

    Pass every sample that will be used in the same ``jax.jit``-compiled
    ``train_step``/``eval_step`` (train and val together) in one call, so
    the padded length -- the longest list found across all of them -- is the
    same for both; calling this separately per split would still leave two
    distinct shapes (one compile each), half the problem this exists to
    avoid.

    Calls ``kdtree_neighbor_search``/``neighbors_to_segments`` directly
    rather than ``precompute_neighbors`` (the cavity benchmark's choice):
    that wrapper raises if *any* query point has no neighbor, which is the
    right safety check for one grid shared by every sample (the cavity),
    but not here -- the latent grid is fixed over the whole dataset's
    bounding box (this module's docstring) while any one car only occupies
    part of it, so some latent points legitimately have no nearby mesh point
    for a given sample (and vice versa for a car's own points against a
    coarse grid); such a point contributing nothing to that sample's
    integral is the expected behaviour, not a misconfiguration. This is the
    same lower-level pair GINO's own ``precompute_neighbors`` convenience
    method uses, for the same reason.
    """
    latent_flat = build_latent_queries(latent_grid).reshape(-1, 3)
    n_latent = latent_flat.shape[0]

    raw: list[tuple[dict, dict, int]] = []
    for sample in samples:
        pos = np.asarray(sample["x"][:, :3], dtype=np.float32)
        n_in = neighbors_to_segments(
            kdtree_neighbor_search(data=pos, queries=latent_flat, radius=in_radius)
        )
        n_out = neighbors_to_segments(
            kdtree_neighbor_search(data=latent_flat, queries=pos, radius=out_radius)
        )
        raw.append((n_in, n_out, pos.shape[0]))

    max_in = max(int(n_in["neighbors_index"].shape[0]) for n_in, _, _ in raw)
    max_out = max(int(n_out["neighbors_index"].shape[0]) for _, n_out, _ in raw)
    for sample, (n_in, n_out, n_car) in zip(samples, raw, strict=True):
        sample["neighbors_in"] = _pad_to(n_in, n_latent, max_in)
        sample["neighbors_out"] = _pad_to(n_out, n_car, max_out)
