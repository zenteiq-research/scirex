# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Neighborhood search between arbitrary coordinate meshes (native and KD-tree
implementations).
"""

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree


@functools.lru_cache(maxsize=1)
def _fixed_radius_search_class() -> type | None:
    """Return open3d's `FixedRadiusSearch` layer, or `None` if unusable.

    open3d is optional and importing it pulls in PyTorch, so it is only probed
    the first time a `NeighborSearch` is set up with `use_open3d=True`.
    Catches more than ImportError: `open3d.ml.torch` raises a plain Exception on
    a torch/open3d version mismatch, which must also fall back to the native
    implementation.
    """
    try:
        from open3d.ml.torch.layers import FixedRadiusSearch
    except Exception:
        return None
    return FixedRadiusSearch


# Uses open3d by default which, as of October 2024, requires torch 2.0 and cuda11.*
class NeighborSearch(nn.Module):
    """Neighborhood search between two arbitrary coordinate meshes.

    For each point `x` in `queries`, returns a set of the indices of all points `y`
    in `data` within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool, optional
        Whether to use open3d or native JAX implementation, by default True
        NOTE: open3d implementation requires 3d data
    return_norm : bool, optional
        Whether to return normalized distances, by default False
    """

    use_open3d: bool = True
    return_norm: bool = False

    def setup(self) -> None:
        fixed_radius_search = _fixed_radius_search_class() if self.use_open3d else None
        if fixed_radius_search is not None:  # slightly faster, works on GPU in 3d only
            self.search_fn = fixed_radius_search()
            self._use_open3d = True
        else:  # slower fallback, works on GPU and CPU
            self.search_fn = None
            self._use_open3d = False

    def __call__(
        self, data: jnp.ndarray, queries: jnp.ndarray, radius: float
    ) -> dict[str, jnp.ndarray]:
        """
        Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : jnp.ndarray of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : jnp.ndarray of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Returns
        -------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: jnp.ndarray with dtype=int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: jnp.ndarray of shape [m+1] with dtype=int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self._use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict["neighbors_index"] = jnp.array(
                search_return.neighbors_index, dtype=jnp.int64
            )
            return_dict["neighbors_row_splits"] = jnp.array(
                search_return.neighbors_row_splits, dtype=jnp.int64
            )
        else:
            return_dict = native_neighbor_search_chunked(
                data, queries, radius, self.return_norm
            )

        return return_dict


def native_neighbor_search(
    data: jnp.ndarray, queries: jnp.ndarray, radius: float, return_norm: bool = False
) -> dict[str, jnp.ndarray]:
    """
    Native JAX implementation of a neighborhood search
    between two arbitrary coordinate meshes.

    Parameters
    ----------
    data : jnp.ndarray
        vector of data points from which to find neighbors
    queries : jnp.ndarray
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """
    nbr_dict = {}

    dists = jnp.linalg.norm(queries[:, None, :] - data[None, :, :], axis=-1)

    # compute pairwise distances

    subtraction = queries[:, None, :] - data[None, :, :]

    all_dists = jnp.linalg.norm(subtraction, axis=-1)

    # keep zero-distance points
    eps = 1e-7
    all_dists = jnp.where(all_dists == 0.0, eps, all_dists)

    dists = jnp.where(
        all_dists <= radius, all_dists, 0.0
    )  # i,j is nonzero if j is i's neighbor

    nbr_indices = jnp.nonzero(dists, size=dists.size)[1]  # only keep the column indices

    # filter to only actual nonzero entries
    mask = dists.reshape(-1) > 0
    nbr_indices = nbr_indices[mask]

    if return_norm:
        weights = dists[dists > 0]
        nbr_dict["weights"] = weights**2  # weighting function computed on squared norms

    in_nbr = jnp.where(dists > 0, 1.0, 0.0)
    nbrhd_sizes = jnp.cumsum(
        jnp.sum(in_nbr, axis=1), axis=0
    )  # cumulative neighborhood sizes
    splits = jnp.concatenate((jnp.array([0.0]), nbrhd_sizes))

    nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
    nbr_dict["neighbors_row_splits"] = splits.astype(jnp.int64)

    return nbr_dict


def native_neighbor_search_chunked(
    data: jnp.ndarray,
    queries: jnp.ndarray,
    radius: float,
    return_norm: bool = False,
    batch_size: int = 1024,
) -> dict[str, jnp.ndarray]:
    """
    Memory-efficient chunked neighbor search.

    Processes queries in small batches to avoid allocating the full [m, n, d]
    intermediate tensor. Uses vectorized JAX operations on each batch.

    Parameters
    ----------
    data : jnp.ndarray of shape [n, d]
        Search space of possible neighbors
    queries : jnp.ndarray of shape [m, d]
        Points for which to find neighbors
    radius : float
        Radius of each ball: B(queries[j], radius)
    return_norm : bool, optional
        Whether to return normalized distances, by default False
    batch_size : int, optional
        Number of queries to process at once, by default 1024

    Returns
    -------
    dict
        Dictionary with keys: neighbors_index, neighbors_row_splits, and optional
        weights
    """
    nbr_dict = {}
    nbr_indices_list = []
    counts = []
    weights_list = []

    # Process queries in chunks
    for chunk_idx in range(0, len(queries), batch_size):
        end_idx = min(chunk_idx + batch_size, len(queries))
        q_batch = queries[chunk_idx:end_idx]  # [batch_size, d]

        # Compute distances for this batch: [batch_size, n]
        # Intermediate: [batch_size, n, d] which is much smaller than [m, n, d]
        batch_dists = jnp.linalg.norm(q_batch[:, None, :] - data[None, :, :], axis=-1)

        # keep zero-distance points
        eps = 1e-7
        batch_dists = jnp.where(batch_dists == 0.0, eps, batch_dists)

        # Find neighbors within radius
        mask = batch_dists <= radius  # [batch_size, n]

        # Process each query in the batch
        for i in range(len(q_batch)):
            query_mask = mask[i]  # [n]
            query_dists = batch_dists[i]  # [n]

            # Find neighbor indices
            nbr_idx = jnp.nonzero(query_mask, size=len(data))[0]

            # Count actual neighbors
            actual_count = jnp.sum(query_mask)
            nbr_idx = nbr_idx[:actual_count]  # Trim to actual count

            nbr_indices_list.append(nbr_idx.astype(jnp.int64))
            counts.append(int(actual_count))

            if return_norm:
                weights = query_dists[query_mask]
                weights_list.append(weights**2)

    # Concatenate all results
    if nbr_indices_list:
        nbr_indices = jnp.concatenate(nbr_indices_list)
    else:
        nbr_indices = jnp.array([], dtype=jnp.int64)

    splits = jnp.concatenate([jnp.array([0]), jnp.array(counts, dtype=jnp.int64)])
    splits = jnp.cumsum(splits).astype(jnp.int64)

    nbr_dict["neighbors_index"] = nbr_indices.astype(jnp.int64)
    nbr_dict["neighbors_row_splits"] = splits

    if return_norm:
        if weights_list:
            weights = jnp.concatenate(weights_list)
        else:
            weights = jnp.array([], dtype=jnp.float32)
        nbr_dict["weights"] = weights

    return nbr_dict


def native_neighbor_search_chunked_optimized(
    data: jnp.ndarray,
    queries: jnp.ndarray,
    radius: float,
    return_norm: bool = False,
    batch_size: int = 2048,
) -> dict[str, jnp.ndarray]:
    """Find the neighbours of every query point within `radius`, processing the queries in chunks.

    Parameters
    ----------
    data : jnp.ndarray
        Points to search in, shape `(n_data, dim)`.
    queries : jnp.ndarray
        Query points, shape `(n_queries, dim)`.
    radius : float
        Search radius.
    return_norm : bool, optional
        Whether to also return the distances. Default: False.
    batch_size : int, optional
        Number of queries handled at once. Default: 2048.

    Returns
    -------
    dict[str, jnp.ndarray]
        The neighbour indices of every query and the row splits that delimit them.
    """
    nbr_indices_list = []
    weights_list = []

    # We will build the row splits dynamically
    row_splits = [0]
    current_split = 0
    eps = 1e-7

    for chunk_idx in range(0, len(queries), batch_size):
        end_idx = min(chunk_idx + batch_size, len(queries))
        q_batch = queries[chunk_idx:end_idx]

        # 1. Vectorized distance calculation
        batch_dists = jnp.linalg.norm(q_batch[:, None, :] - data[None, :, :], axis=-1)
        batch_dists = jnp.where(batch_dists == 0.0, eps, batch_dists)

        # 2. Vectorized mask
        mask = batch_dists <= radius

        # --- NO INNER LOOP BEYOND THIS POINT ---

        # 3. Get all valid indices for the ENTIRE batch in one GPU sweep
        # row_idx is the query index within the batch, data_idx is the neighbor index
        row_idx, data_idx = jnp.nonzero(mask)

        nbr_indices_list.append(data_idx.astype(jnp.int64))

        if return_norm:
            # Extract weights for the whole batch in one sweep
            weights = batch_dists[row_idx, data_idx]
            weights_list.append(weights**2)

        # 4. Get counts for the whole batch in one sweep
        batch_counts = jnp.sum(mask, axis=1)

        # We only sync `batch_counts` to CPU here, which is 1024x less syncing than
        # per-query syncing would require.
        for count in np.array(batch_counts):
            current_split += int(count)
            row_splits.append(current_split)

    # Concatenate chunked results
    nbr_dict = {
        "neighbors_index": jnp.concatenate(nbr_indices_list)
        if nbr_indices_list
        else jnp.array([], dtype=jnp.int64),
        "neighbors_row_splits": jnp.array(row_splits, dtype=jnp.int64),
    }

    if return_norm:
        nbr_dict["weights"] = (
            jnp.concatenate(weights_list)
            if weights_list
            else jnp.array([], dtype=jnp.float32)
        )

    return nbr_dict


def native_neighbor_search_numpy(
    data: np.ndarray | jnp.ndarray,
    queries: np.ndarray | jnp.ndarray,
    radius: float,
    return_norm: bool = False,
) -> dict[str, jnp.ndarray]:
    """Delegates to native_neighbor_search_chunked for memory efficiency.

    This function is kept for backwards compatibility with the original API.
    """
    # Convert inputs to JAX arrays if needed
    if not isinstance(data, jnp.ndarray):
        data = jnp.array(data)
    if not isinstance(queries, jnp.ndarray):
        queries = jnp.array(queries)

    return native_neighbor_search_chunked_optimized(
        data, queries, radius, return_norm, batch_size=2048
    )


def kdtree_neighbor_search(
    data: np.ndarray, queries: np.ndarray, radius: float, return_norm: bool = False
) -> dict[str, np.ndarray]:
    """
    Lightning-fast O(N log M) neighbor search using Scipy's C++ KD-Tree.
    Run this eagerly on the CPU inside your data loader.
    """
    # 1. Build the KD-Tree on the search space (data)
    tree = cKDTree(data)

    # 2. Query the tree for points within the radius
    # Returns a list of lists containing neighbor indices for each query
    neighbor_lists = tree.query_ball_point(queries, r=radius)

    # 3. Format the outputs to match your exact dictionary structure
    all_indices = []
    row_splits = [0]
    current_split = 0
    all_weights = []

    for i, nbrs in enumerate(neighbor_lists):
        n_count = len(nbrs)
        current_split += n_count
        row_splits.append(current_split)

        if n_count > 0:
            all_indices.extend(nbrs)

            if return_norm:
                # Calculate distances only for actual neighbors, saving massive
                # computation
                q_point = queries[i]
                d_points = data[nbrs]
                sq_dists = np.sum((d_points - q_point) ** 2, axis=-1)
                all_weights.extend(sq_dists)

    nbr_dict = {
        "neighbors_index": np.array(all_indices, dtype=np.int32),
        "neighbors_row_splits": np.array(row_splits, dtype=np.int32),
    }

    if return_norm:
        nbr_dict["weights"] = np.array(all_weights, dtype=np.float32)

    return nbr_dict


def neighbors_to_segments(raw: dict) -> dict[str, jnp.ndarray]:
    """CRS neighbor-search output -> the segment format the GNO layers consume.

    `NeighborSearch`, `native_neighbor_search` and
    `kdtree_neighbor_search` all return compressed row storage
    (`neighbors_index` plus `neighbors_row_splits`), while
    `IntegralTransform` indexes
    edges by `segment_ids` and `counts`.
    `__call__` performs exactly
    this conversion when it runs the search itself; this function is the same
    conversion, exposed so that a caller with a *fixed* geometry can run the
    search once up front and pass the result in as `neighbors` on every step
    instead of repeating it.

    Parameters
    ----------
    raw : dict
        Must contain `neighbors_index` and `neighbors_row_splits`. An
        optional `weights` entry is passed through unchanged, so a search run
        with `return_norm=True` keeps its squared distances.

    Returns
    -------
    dict
        `neighbors_index`, `segment_ids` and `counts`, as JAX arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from scirex.layers.blocks.neighbor_search import (
    ...     kdtree_neighbor_search,
    ...     neighbors_to_segments,
    ... )
    >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    >>> neighbors = neighbors_to_segments(
    ...     kdtree_neighbor_search(data=points, queries=points, radius=1.5)
    ... )
    >>> [int(c) for c in neighbors["counts"]]
    [2, 3, 2]
    >>> [int(s) for s in neighbors["segment_ids"]]
    [0, 0, 1, 1, 1, 2, 2]
    """
    index = jnp.asarray(raw["neighbors_index"])
    row_splits = jnp.asarray(raw["neighbors_row_splits"])
    counts = (row_splits[1:] - row_splits[:-1]).astype(jnp.int32)
    segment_ids = jnp.repeat(
        jnp.arange(counts.shape[0], dtype=jnp.int32),
        counts,
        total_repeat_length=int(index.shape[0]),
    )
    out = {
        "neighbors_index": index,
        "segment_ids": segment_ids,
        "counts": counts,
    }
    if "weights" in raw:
        out["weights"] = jnp.asarray(raw["weights"])
    return out


def precompute_neighbors(
    data: np.ndarray | jnp.ndarray,
    queries: np.ndarray | jnp.ndarray,
    radius: float,
    return_norm: bool = False,
) -> dict[str, jnp.ndarray]:
    """One fixed-radius search, ready to hand to a GNO layer as `neighbors`.

    `kdtree_neighbor_search` followed by `neighbors_to_segments`.
    Use this when the geometry does not change between samples: the KD-tree is
    built on the CPU once, and every subsequent forward pass reuses the result
    rather than repeating an O(n m) search inside the training step.

    Raises
    ------
    ValueError
        If some query point has no neighbor within `radius`. Such a point
        contributes nothing to the kernel integral, which is nearly always a
        radius that is too small for the query spacing rather than an intended
        result, so it is reported here instead of silently producing zeros.
    """
    raw = kdtree_neighbor_search(
        data=np.asarray(data),
        queries=np.asarray(queries),
        radius=radius,
        return_norm=return_norm,
    )
    row_splits = np.asarray(raw["neighbors_row_splits"])
    counts = row_splits[1:] - row_splits[:-1]
    n_empty = int((counts == 0).sum())
    if n_empty:
        raise ValueError(
            f"{n_empty} of {counts.shape[0]} query points have no neighbor "
            f"within radius {radius}; increase the radius or refine the data "
            "points"
        )
    return neighbors_to_segments(raw)


# ---------------------------------------------------------------------------
# Padded edge_index graphs  (pure numpy/scipy, runs outside jax.jit)
# ---------------------------------------------------------------------------


def build_neighbor_graph(
    source_pos: np.ndarray,  # (N_src, D)
    query_pos: np.ndarray,  # (N_qry, D)
    radius: float,
    strategy: str,  # 'radius' | 'knn' | 'bidirectional' | 'reverse'
    k_neighbors: int = 1,
    max_edges: int | None = None,
    max_num_neighbors: int = 32,
) -> np.ndarray:  # (2, max_edges) padded int32 edge_index
    """Build a bipartite graph and return a padded edge_index array.

    Edge direction: source → query  (info flows from source to query).
    Padding uses query index = N_qry (one past the last valid node).

    Parameters
    ----------
    source_pos : (N_src, D)  coordinates of source nodes
    query_pos  : (N_qry, D)  coordinates of query nodes
    radius     : search radius for radius / bidirectional strategies
    strategy   : one of 'radius', 'knn', 'bidirectional', 'reverse'
    k_neighbors: k for knn strategy
    max_edges  : pad / truncate to this number of edges.
                 If None, use the actual number of edges (no padding).
    max_num_neighbors : cap on the number of source neighbours returned per
                 query node in the radius search (the nearest `max_num_neighbors`
                 within `radius` are kept).  Without this cap a dense mesh
                 yields millions of edges per sample; the default of 32 keeps the
                 per-query neighbourhood bounded.

    Returns
    -------
    edge_index : (2, max_edges) int32  - row 0 = src, row 1 = qry
    """
    N_src, N_qry = len(source_pos), len(query_pos)

    if strategy == "reverse":
        # Reverse of the bidirectional encoder graph (latent→phys becomes
        # phys→latent flipped). For a decoder (source=latent, query=phys),
        # 'reverse' means the encoder's edges, flipped.
        enc_edges = build_neighbor_graph(
            query_pos,
            source_pos,
            radius,
            "bidirectional",
            k_neighbors,
            max_num_neighbors=max_num_neighbors,
        )
        # enc_edges[0]=phys_idx, enc_edges[1]=latent_idx → flip: src=latent, qry=phys
        src_all = enc_edges[1]
        qry_all = enc_edges[0]
        keep = src_all < N_src  # drop padding edges (src == N_qry_of_enc)
        src_arr = src_all[keep].astype(np.int32)
        qry_arr = qry_all[keep].astype(np.int32)
    elif N_src == 0 or N_qry == 0:
        src_arr = np.empty(0, dtype=np.int32)
        qry_arr = np.empty(0, dtype=np.int32)
    else:
        src_parts, qry_parts = [], []

        if strategy in ("radius", "bidirectional"):
            # For each query node, the nearest source nodes within `radius`,
            # capped at `max_num_neighbors`.  cKDTree.query with
            # distance_upper_bound runs in C and parallelises over cores - no
            # per-edge Python loop.
            src_tree = cKDTree(source_pos)
            k = min(max_num_neighbors, N_src)
            _, s_idx = src_tree.query(
                query_pos, k=k, distance_upper_bound=radius, workers=-1
            )
            s_idx = np.asarray(s_idx).reshape(N_qry, k)
            q_rep = np.repeat(np.arange(N_qry, dtype=np.int32), k)
            s_flat = s_idx.reshape(-1)
            # Misses are marked with index == N_src (and distance inf); drop them.
            valid = s_flat < N_src
            src_parts.append(s_flat[valid].astype(np.int32))
            qry_parts.append(q_rep[valid])

        if strategy in ("knn", "bidirectional"):
            # For each source node, its k nearest query nodes (vectorised).
            qry_tree = cKDTree(query_pos)
            k = min(k_neighbors, N_qry)
            _, q_indices = qry_tree.query(source_pos, k=k, workers=-1)
            q_indices = np.asarray(q_indices).reshape(N_src, k)
            src_parts.append(np.repeat(np.arange(N_src, dtype=np.int32), k))
            qry_parts.append(q_indices.reshape(-1).astype(np.int32))

        if src_parts:
            src_arr = np.concatenate(src_parts)
            qry_arr = np.concatenate(qry_parts)
        else:
            src_arr = np.empty(0, dtype=np.int32)
            qry_arr = np.empty(0, dtype=np.int32)

    # Remove duplicates (important for bidirectional)
    if len(src_arr) > 0:
        pairs = np.unique(np.stack([src_arr, qry_arr], axis=1), axis=0)
        src_arr = pairs[:, 0]
        qry_arr = pairs[:, 1]

    n_real = len(src_arr)
    if max_edges is None:
        max_edges = n_real

    # Pad / truncate to max_edges; dummy edges use qry = N_qry
    edge_index = np.full((2, max_edges), fill_value=N_qry, dtype=np.int32)
    n_keep = min(n_real, max_edges)
    edge_index[0, :n_keep] = src_arr[:n_keep]
    edge_index[1, :n_keep] = qry_arr[:n_keep]
    return edge_index


# ---------------------------------------------------------------------------
# JAX helper: edge_index → IntegralTransform neighbors dict
# ---------------------------------------------------------------------------


def edge_index_to_neighbors(
    edge_index: jnp.ndarray,  # (2, max_edges)
    num_query: int,
) -> dict:
    """Convert a padded edge_index to the dict expected by IntegralTransform.

    Padding edges must have edge_index[1] == num_query (dummy segment).

    Returns
    -------
    dict with keys:
        neighbors_index : (max_edges,) int  - source indices per edge
        segment_ids     : (max_edges,) int  - query  indices per edge
        counts          : (num_query,) int  - valid edges per query node
    """
    src_idx = edge_index[0]
    qry_idx = edge_index[1]

    # Count valid edges per query (segment_sum ignores the dummy segment)
    ones = jnp.ones(src_idx.shape[0], dtype=jnp.float32)
    counts_full = jax.ops.segment_sum(ones, qry_idx, num_segments=num_query + 1)
    counts = counts_full[:num_query].astype(jnp.int32)

    return {
        "neighbors_index": src_idx,
        "segment_ids": qry_idx,
        "counts": counts,
    }
