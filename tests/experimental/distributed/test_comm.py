# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the MPU communication helpers (JAX).

neuraloperator's ``mpu`` package has no test suite, so coverage follows the
Phase 1 ``layers/tests`` structure. The distributed code paths need a
multi-process JAX environment that is unavailable in CI; here we cover the
single-process (uninitialized) behaviour that the rest of the codebase relies on
when ``use_distributed=False``: the rank/size getters return their safe
single-process defaults, the group accessors guard against use before
initialization, ``DisableLogging`` toggles logging, and ``init`` is a no-op on a
single device.
"""

import logging

import pytest

import scirex.experimental.distributed.comm as comm

# ---------------------------------------------------------------------------
# rank / size getters (uninitialized -> single-process defaults)
# ---------------------------------------------------------------------------


def test_world_and_rank_defaults() -> None:
    """Uninitialized world size is 1 and all ranks are 0."""
    assert comm.get_world_size() == 1
    assert comm.get_local_rank() == 0
    assert comm.get_global_rank() == 0


def test_data_parallel_defaults() -> None:
    """Data-parallel size defaults to 1 and rank to 0 before init."""
    assert comm.get_data_parallel_size() == 1
    assert comm.get_data_parallel_rank() == 0


def test_model_parallel_defaults() -> None:
    """Model-parallel size defaults to 1 and rank to 0 before init."""
    assert comm.get_model_parallel_size() == 1
    assert comm.get_model_parallel_rank() == 0


# ---------------------------------------------------------------------------
# group accessors guard against uninitialized use
# ---------------------------------------------------------------------------


def test_group_accessors_require_initialization() -> None:
    """The data/model parallel group accessors assert until init() runs."""
    with pytest.raises(AssertionError):
        comm.get_data_parallel_group()
    with pytest.raises(AssertionError):
        comm.get_model_parallel_group()


# ---------------------------------------------------------------------------
# DisableLogging context manager
# ---------------------------------------------------------------------------


def test_disable_logging_restores_logging() -> None:
    """DisableLogging suppresses records inside the block and restores after."""
    logger = logging.getLogger("comm_test_logger")
    with comm.DisableLogging():
        # logging is globally disabled up to ERROR inside the block
        assert not logger.isEnabledFor(logging.ERROR)
    # restored to normal afterwards
    assert logger.isEnabledFor(logging.ERROR)


# ---------------------------------------------------------------------------
# init() on a single device
# ---------------------------------------------------------------------------


def test_init_single_device_is_noop() -> None:
    """init() on one device returns None and leaves the module uninitialized."""
    assert comm.init(model_parallel_size=1) is None
    # world_size == 1 means the distributed wireup is skipped
    assert comm.get_world_size() == 1
    assert comm.get_model_parallel_size() == 1


# ---------------------------------------------------------------------------
# rank / size / group getters in the initialized state
# ---------------------------------------------------------------------------


def test_getters_in_initialized_state(monkeypatch) -> None:
    """Once initialized, the getters report the live JAX process topology.

    The multi-host wireup itself needs a real distributed launch, so the
    initialized branches are exercised by patching the module globals; in a
    single-process test ``jax.process_count``/``process_index`` are 1/0.
    """
    monkeypatch.setattr(comm, "_initialized", True)
    monkeypatch.setattr(comm, "_DATA_PARALLEL_GROUP", [0])
    monkeypatch.setattr(comm, "_MODEL_PARALLEL_GROUP", [0])

    # world / rank getters take their initialized (else) branch
    assert comm.get_world_size() == 1
    assert comm.get_local_rank() == 0
    assert comm.get_global_rank() == 0

    # data-parallel getters and group accessor
    assert comm.get_data_parallel_size() == 1
    assert comm.get_data_parallel_rank() == 0
    assert comm.get_data_parallel_group() == [0]

    # model-parallel getters and group accessor
    assert comm.get_model_parallel_size() == 1
    assert comm.get_model_parallel_rank() == 0
    assert comm.get_model_parallel_group() == [0]


# ---------------------------------------------------------------------------
# init() multi-host wireup (distributed runtime mocked)
# ---------------------------------------------------------------------------


def test_init_multi_host_wireup_builds_groups(monkeypatch) -> None:
    """init() builds model/data parallel groups for a >1 world size.

    A real multi-host launch is unavailable, so the distributed runtime is
    mocked: world size 4 with model_parallel_size 2 yields a 2 x 2 decomposition.
    init() mutates module globals via ``global``, so they are saved and restored.
    """
    saved = (
        comm._initialized,
        comm._DATA_PARALLEL_GROUP,
        comm._MODEL_PARALLEL_GROUP,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(comm.jax, "local_device_count", lambda: 4)
    monkeypatch.setattr(comm.jax, "effects_barrier", lambda: None)
    monkeypatch.setattr(comm, "get_world_size", lambda: 4)
    try:
        comm.init(model_parallel_size=2, verbose=True)
        # process 0 belongs to model group [0, 1] and data group [0, 2]
        assert comm._MODEL_PARALLEL_GROUP == [0, 1]
        assert comm._DATA_PARALLEL_GROUP == [0, 2]
        assert comm._initialized is True
    finally:
        (
            comm._initialized,
            comm._DATA_PARALLEL_GROUP,
            comm._MODEL_PARALLEL_GROUP,
        ) = saved


def test_init_single_model_group_wireup(monkeypatch) -> None:
    """With model_parallel_size=1 and world>1, the else-branch sets default groups."""
    saved = (
        comm._initialized,
        comm._DATA_PARALLEL_GROUP,
        comm._MODEL_PARALLEL_GROUP,
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(comm.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(comm.jax, "effects_barrier", lambda: None)
    monkeypatch.setattr(comm, "get_world_size", lambda: 2)
    try:
        comm.init(model_parallel_size=1)
        assert comm._MODEL_PARALLEL_GROUP == [0]
        assert comm._DATA_PARALLEL_GROUP == [0, 1]
    finally:
        (
            comm._initialized,
            comm._DATA_PARALLEL_GROUP,
            comm._MODEL_PARALLEL_GROUP,
        ) = saved
