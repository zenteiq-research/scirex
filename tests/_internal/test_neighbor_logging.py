# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Unit tests for ``NeighborSearchLogger``."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from scirex._internal.neighbor_logging import NeighborSearchLogger

# ---------------------------------------------------------------------------
# NeighborSearchLogger (source lines 252-253, 286-353, 404-436)
# ---------------------------------------------------------------------------


def _make_nb_dict(n_queries: int = 3, with_weights: bool = True) -> dict:
    nb_dict = {
        "neighbors_index": jnp.array([0, 1, 2], dtype=jnp.int32),
        "neighbors_row_splits": jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    }
    if with_weights:
        nb_dict["weights"] = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
    return nb_dict


def test_neighbor_search_logger_init_creates_dir(tmp_path: Path) -> None:
    """NeighborSearchLogger creates the log directory in __init__ (lines 252-253)."""
    log_dir = tmp_path / "logs"
    NeighborSearchLogger(log_dir=str(log_dir))
    assert log_dir.exists()


def test_log_gno_neighbors_with_weights(tmp_path: Path) -> None:
    """log_gno_neighbors writes a file and includes sq_distance (lines 286-353)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    logger.log_gno_neighbors(
        _make_nb_dict(with_weights=True),
        qp,
        dp,
        epoch=0,
        stage="train",
        gno_type="gno_in",
    )
    log_file = tmp_path / "gno_in_train_epoch_0.txt"
    assert log_file.exists()
    content = log_file.read_text()
    assert "sq_distance" in content


def test_log_gno_neighbors_without_weights(tmp_path: Path) -> None:
    """log_gno_neighbors without weights omits sq_distance line (lines 346-347)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    logger.log_gno_neighbors(
        _make_nb_dict(with_weights=False),
        qp,
        dp,
        epoch=1,
        stage="test",
        gno_type="gno_out",
    )
    log_file = tmp_path / "gno_out_test_epoch_1.txt"
    assert log_file.exists()
    content = log_file.read_text()
    assert "sq_distance" not in content


def test_log_gno_neighbors_empty_neighbors(tmp_path: Path) -> None:
    """log_gno_neighbors with no neighbors hits the 'No neighbors found' branch
    (line 349)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    empty_dict = {
        "neighbors_index": jnp.array([], dtype=jnp.int32),
        "neighbors_row_splits": jnp.zeros(4, dtype=jnp.int32),
    }
    logger.log_gno_neighbors(
        empty_dict, qp, dp, epoch=2, stage="train", gno_type="gno_in"
    )
    content = (tmp_path / "gno_in_train_epoch_2.txt").read_text()
    assert "No neighbors found" in content


def test_log_gno_neighbors_batched_inputs(tmp_path: Path) -> None:
    """Batched inputs (ndim > 2) are sliced to the first element (lines 286-289)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((2, 3, 2), dtype=jnp.float32)  # batch dimension
    dp = jnp.ones((2, 5, 2), dtype=jnp.float32)
    logger.log_gno_neighbors(
        _make_nb_dict(with_weights=False),
        qp,
        dp,
        epoch=3,
        stage="train",
        gno_type="gno_in",
    )
    assert (tmp_path / "gno_in_train_epoch_3.txt").exists()


def test_log_epoch_train_only(tmp_path: Path) -> None:
    """log_epoch with only train args writes train files (lines 404-422)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    nd = _make_nb_dict(with_weights=False)
    logger.log_epoch(
        gno_in_neighbors_train=nd,
        gno_in_query_points_train=qp,
        gno_in_data_points_train=dp,
        gno_out_neighbors_train=nd,
        gno_out_query_points_train=qp,
        gno_out_data_points_train=dp,
        epoch=10,
    )
    assert (tmp_path / "gno_in_train_epoch_10.txt").exists()
    assert (tmp_path / "gno_out_train_epoch_10.txt").exists()


def test_log_epoch_with_test_data(tmp_path: Path) -> None:
    """log_epoch with test args also writes test files (lines 424-436)."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    nd = _make_nb_dict(with_weights=True)
    logger.log_epoch(
        gno_in_neighbors_train=nd,
        gno_in_query_points_train=qp,
        gno_in_data_points_train=dp,
        gno_out_neighbors_train=nd,
        gno_out_query_points_train=qp,
        gno_out_data_points_train=dp,
        gno_in_neighbors_test=nd,
        gno_in_query_points_test=qp,
        gno_in_data_points_test=dp,
        gno_out_neighbors_test=nd,
        gno_out_query_points_test=qp,
        gno_out_data_points_test=dp,
        epoch=5,
    )
    assert (tmp_path / "gno_in_test_epoch_5.txt").exists()
    assert (tmp_path / "gno_out_test_epoch_5.txt").exists()


def test_log_epoch_partial_test_args_raises_assertion(tmp_path: Path) -> None:
    """Partial test args assert instead of raising AttributeError downstream."""
    logger = NeighborSearchLogger(log_dir=str(tmp_path))
    qp = jnp.ones((3, 2), dtype=jnp.float32)
    dp = jnp.ones((5, 2), dtype=jnp.float32)
    nd = _make_nb_dict(with_weights=False)

    with pytest.raises(AssertionError):
        logger.log_epoch(
            gno_in_neighbors_train=nd,
            gno_in_query_points_train=qp,
            gno_in_data_points_train=dp,
            gno_out_neighbors_train=nd,
            gno_out_query_points_train=qp,
            gno_out_data_points_train=dp,
            gno_in_neighbors_test=nd,
            gno_in_query_points_test=None,
            gno_in_data_points_test=None,
        )

    with pytest.raises(AssertionError):
        logger.log_epoch(
            gno_in_neighbors_train=nd,
            gno_in_query_points_train=qp,
            gno_in_data_points_train=dp,
            gno_out_neighbors_train=nd,
            gno_out_query_points_train=qp,
            gno_out_data_points_train=dp,
            gno_out_neighbors_test=nd,
            gno_out_query_points_test=None,
            gno_out_data_points_test=None,
        )
