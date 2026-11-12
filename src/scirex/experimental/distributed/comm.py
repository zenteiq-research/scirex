# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Communication helpers: world/rank/group queries and process-group initialization."""

import logging
import os
from types import TracebackType

import jax

logger = logging.getLogger(__name__)


class DisableLogging:
    """Context manager that switches logging off inside its block and restores it on exit."""

    def __init__(self, level: int = logging.ERROR) -> None:
        logging.disable(level=level)

    def __enter__(self) -> "DisableLogging":
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        logging.disable(level=logging.NOTSET)


# dummy placeholders
# In JAX, process groups are represented as lists of process-index integers
# rather than torch.distributed group objects.
_initialized = False
_DATA_PARALLEL_GROUP: list[int] | None = None
_MODEL_PARALLEL_GROUP: list[int] | None = None


# world comm
def get_world_size() -> int:
    """Return the total number of processes."""
    if not _initialized:
        return 1
    else:
        return jax.process_count()


def get_local_rank() -> int:
    """Return the rank of this process on its node."""
    if not _initialized:
        return 0
    else:
        return jax.process_index()


def get_global_rank() -> int:
    """Return the rank of this process among all processes."""
    if not _initialized:
        return 0
    else:
        return jax.process_index()


# data parallel
def get_data_parallel_size() -> int:
    """Return the number of data-parallel processes."""
    if not _initialized:
        return 1
    else:
        return (
            len(_DATA_PARALLEL_GROUP)
            if _DATA_PARALLEL_GROUP is not None
            else jax.process_count()
        )


def get_data_parallel_rank() -> int:
    """Return the rank of this process in the data-parallel group."""
    if not _initialized:
        return 0
    else:
        return jax.process_index()


def get_data_parallel_group() -> list[int] | None:
    """Return the ranks of the data-parallel group, or None if not set."""
    assert _initialized, "Error, initialize JAX distributed first"
    return _DATA_PARALLEL_GROUP


# model parallel
def get_model_parallel_size() -> int:
    """Return the number of model-parallel processes."""
    if not _initialized or (_MODEL_PARALLEL_GROUP is None):
        return 1
    else:
        return len(_MODEL_PARALLEL_GROUP)


def get_model_parallel_rank() -> int:
    """Return the rank of this process in the model-parallel group."""
    if not _initialized or (_MODEL_PARALLEL_GROUP is None):
        return 0
    else:
        return jax.process_index()


def get_model_parallel_group() -> list[int] | None:
    """Return the ranks of the model-parallel group, or None if not set."""
    assert _initialized, "Error, initialize JAX distributed first"
    return _MODEL_PARALLEL_GROUP


def init(model_parallel_size: int = 1, verbose: bool = False) -> None:
    """
    Set up global and local communicator.
    JAX distributed must be initialized externally via jax.distributed.initialize()
    before calling this function. Process groups are represented as lists of
    process-index integers (JAX has no group-handle objects).
    """

    global_rank = int(os.getenv("RANK", 0))
    world_size = jax.local_device_count()

    global _initialized
    global _DATA_PARALLEL_GROUP
    global _MODEL_PARALLEL_GROUP

    if world_size > 1:
        with DisableLogging():
            _initialized = True

            # once initialized, get true values for rank and size using jax
            world_size = get_world_size()
            get_local_rank()

            # set a barrier until all processes reach this point
            jax.effects_barrier()

    # process 0 is logger
    is_logger = get_local_rank() == 0

    # get model groups
    model_group_size = model_parallel_size

    # compute data parallel size
    data_group_size = world_size // model_group_size

    if is_logger:
        logger.info(
            f"Using {world_size} in {model_group_size} x {data_group_size} "
            "decomposition (#model-ranks x #data-ranks)"
        )

    assert (model_group_size <= world_size) and (world_size % model_group_size == 0), (
        "Error, please make sure matmul_parallel_size * spatial_parallel_size <= "
        "world size and that world size is evenly divisible by "
        "matmul_parallel_size * spatial_parallel_size"
    )

    # number of model groups
    num_model_groups = world_size // model_group_size

    if is_logger:
        logger.info("Starting Wireup")

    if world_size > 1:
        if model_group_size > 1:
            model_groups = []
            for i in range(num_model_groups):
                start = i * model_group_size
                end = start + model_group_size
                model_groups.append(list(range(start, end)))

            data_groups = [sorted(list(i)) for i in zip(*model_groups)]

            if verbose and is_logger:
                logger.info("Model Parallel Groups w/ respect to world rank:")
                for grp in model_groups:
                    logger.info(grp)

            if verbose and is_logger:
                logger.info("Data Parallel Groups w/ respect to world rank:")
                for grp in data_groups:
                    logger.info(grp)

            # initialize groups - stored as lists of process-index integers
            with DisableLogging():
                # data groups
                for grp in data_groups:
                    if global_rank in grp:
                        _DATA_PARALLEL_GROUP = grp
                # model groups
                for grp in model_groups:
                    if global_rank in grp:
                        _MODEL_PARALLEL_GROUP = grp

        else:
            # technically unnecessary but we do it to be clean
            with DisableLogging():
                _MODEL_PARALLEL_GROUP = [global_rank]
                _DATA_PARALLEL_GROUP = list(range(world_size))

    # barrier
    if _initialized:
        jax.effects_barrier()

    if is_logger:
        logger.info("Finished Wireup")

    return
