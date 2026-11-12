# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Debug logger for GNO neighbor search results."""

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class NeighborSearchLogger:
    """Logger for GNO neighbor search results."""

    def __init__(self, log_dir: str = "./neighbor_logs") -> None:
        """
        Initialize the logger.

        Parameters
        ----------
        log_dir : str, optional
            Directory to store log files, by default "./neighbor_logs"
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_gno_neighbors(
        self,
        neighbors_dict: dict[str, jnp.ndarray],
        query_points: jnp.ndarray,
        data_points: jnp.ndarray,
        epoch: int,
        stage: str,  # "train" or "test"
        gno_type: str,  # "gno_in" or "gno_out"
    ) -> None:
        """
        Log neighbor search results to file.

        Parameters
        ----------
        neighbors_dict : Dict[str, jax.Array]
            Dictionary from neighbor_search with keys:

            - 'neighbors_index': [n_neighbors] flat list of neighbor indices
            - 'neighbors_row_splits': [n_queries + 1] cumulative indices
            - 'weights' (optional): [n_neighbors] squared distances
        query_points : jax.Array
            Query points of shape [n_queries, d] or [batch, n_queries, d]
        data_points : jax.Array
            Data points of shape [n_data, d] or [batch, n_data, d]
        epoch : int
            Epoch number
        stage : str
            "train" or "test"
        gno_type : str
            "gno_in" or "gno_out"
        """
        # Handle batched inputs - use first batch if batched
        if query_points.ndim > 2:
            query_points = query_points[0]
        if data_points.ndim > 2:
            data_points = data_points[0]

        # Convert JAX arrays to numpy for file I/O
        neighbors_index = np.array(neighbors_dict["neighbors_index"])
        neighbors_row_splits = np.array(neighbors_dict["neighbors_row_splits"])
        weights_arr = neighbors_dict.get("weights")
        weights = np.array(weights_arr) if weights_arr is not None else None

        query_points_np = np.array(query_points)
        data_points_np = np.array(data_points)

        n_queries = len(neighbors_row_splits) - 1

        # Create filename
        filename = self.log_dir / f"{gno_type}_{stage}_epoch_{epoch}.txt"

        with open(filename, "w") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Neighbor Search Log: {gno_type.upper()} ({stage.upper()})\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("Summary Statistics:\n")
            f.write(f"  Total query points: {n_queries}\n")
            f.write(f"  Total data points: {len(data_points)}\n")
            f.write(f"  Total neighbors found: {len(neighbors_index)}\n")
            avg_neighbors = len(neighbors_index) / n_queries
            f.write(f"  Average neighbors per query: {avg_neighbors:.2f}\n")
            f.write(f"\n{'=' * 80}\n\n")

            # Log details for each query point
            for q_idx in range(n_queries):
                start_idx = int(neighbors_row_splits[q_idx])
                end_idx = int(neighbors_row_splits[q_idx + 1])
                n_neighbors = end_idx - start_idx

                f.write(f"Query Point {q_idx}:\n")
                f.write(f"  Coordinates: {query_points_np[q_idx]}\n")
                f.write(f"  Number of neighbors: {n_neighbors}\n")

                if n_neighbors > 0:
                    neighbor_indices = neighbors_index[start_idx:end_idx]
                    f.write(f"  Neighbor indices: {neighbor_indices.tolist()}\n")

                    # Write neighbor details
                    for i, nbr_idx in enumerate(neighbor_indices):
                        nbr_idx = int(nbr_idx)
                        distance = None
                        if weights is not None:
                            distance = weights[start_idx + i]

                        f.write(f"    [{i}] data_idx={nbr_idx}, ")
                        f.write(f"coords={data_points_np[nbr_idx]}, ")
                        if distance is not None:
                            f.write(f"sq_distance={distance:.6f}\n")
                        else:
                            f.write("\n")
                else:
                    f.write("  No neighbors found within radius\n")

                f.write("\n")

        logger.info(f"✓ Logged {gno_type} {stage} neighbors to: {filename}")

    def log_epoch(
        self,
        gno_in_neighbors_train: dict[str, jnp.ndarray],
        gno_in_query_points_train: jnp.ndarray,
        gno_in_data_points_train: jnp.ndarray,
        gno_out_neighbors_train: dict[str, jnp.ndarray],
        gno_out_query_points_train: jnp.ndarray,
        gno_out_data_points_train: jnp.ndarray,
        gno_in_neighbors_test: dict[str, jnp.ndarray] | None = None,
        gno_in_query_points_test: jnp.ndarray | None = None,
        gno_in_data_points_test: jnp.ndarray | None = None,
        gno_out_neighbors_test: dict[str, jnp.ndarray] | None = None,
        gno_out_query_points_test: jnp.ndarray | None = None,
        gno_out_data_points_test: jnp.ndarray | None = None,
        epoch: int = 0,
    ) -> None:
        """
        Log all neighbor search results for an epoch.

        Parameters
        ----------
        gno_in_neighbors_train : Dict
            Neighbors dict from gno_in during training
        gno_in_query_points_train : jax.Array
            Query points for gno_in during training
        gno_in_data_points_train : jax.Array
            Data points for gno_in during training
        gno_out_neighbors_train : Dict
            Neighbors dict from gno_out during training
        gno_out_query_points_train : jax.Array
            Query points for gno_out during training
        gno_out_data_points_train : jax.Array
            Data points for gno_out during training
        gno_in_neighbors_test : Dict, optional
            Neighbors dict from gno_in during testing
        gno_in_query_points_test : jax.Array, optional
            Query points for gno_in during testing
        gno_in_data_points_test : jax.Array, optional
            Data points for gno_in during testing
        gno_out_neighbors_test : Dict, optional
            Neighbors dict from gno_out during testing
        gno_out_query_points_test : jax.Array, optional
            Query points for gno_out during testing
        gno_out_data_points_test : jax.Array, optional
            Data points for gno_out during testing
        epoch : int, optional
            Epoch number, by default 0
        """
        # Log training
        if gno_in_neighbors_train is not None:
            self.log_gno_neighbors(
                gno_in_neighbors_train,
                gno_in_query_points_train,
                gno_in_data_points_train,
                epoch=epoch,
                stage="train",
                gno_type="gno_in",
            )

        if gno_out_neighbors_train is not None:
            self.log_gno_neighbors(
                gno_out_neighbors_train,
                gno_out_query_points_train,
                gno_out_data_points_train,
                epoch=epoch,
                stage="train",
                gno_type="gno_out",
            )

        # Log testing
        if gno_in_neighbors_test is not None:
            # gno_in_neighbors_test/query_points/data_points are always passed
            # together by every call site; asserted rather than silently
            # accepting a partial triple that would crash inside log_gno_neighbors.
            assert gno_in_query_points_test is not None
            assert gno_in_data_points_test is not None
            self.log_gno_neighbors(
                gno_in_neighbors_test,
                gno_in_query_points_test,
                gno_in_data_points_test,
                epoch=epoch,
                stage="test",
                gno_type="gno_in",
            )

        if gno_out_neighbors_test is not None:
            assert gno_out_query_points_test is not None
            assert gno_out_data_points_test is not None
            self.log_gno_neighbors(
                gno_out_neighbors_test,
                gno_out_query_points_test,
                gno_out_data_points_test,
                epoch=epoch,
                stage="test",
                gno_type="gno_out",
            )
