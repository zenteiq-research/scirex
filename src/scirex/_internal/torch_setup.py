# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Device and runtime setup helpers for JAX training."""

from typing import Any

import jax


def setup(config: Any) -> tuple[jax.Device, bool]:
    """A convenience function to intialize the device, setup JAX settings and
    check multi-grid and other values. It sets up distributed communication, if used.

    Parameters
    ----------
    config : dict
        this function checks:
        * scirex.config.distributed (use_distributed, seed)
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)

    Returns
    -------
    device, is_logger
        device : jax.Device
        is_logger : bool
    """
    seed = config.distributed.seed

    if config.distributed.use_distributed:
        import scirex.experimental.distributed.comm as comm

        comm.init(
            model_parallel_size=config.distributed.model_parallel_size,
            verbose=config.verbose,
        )

        # Set process 0 to log screen and wandb
        is_logger = comm.get_local_rank() == 0

        # Set device for local rank
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        if gpus:
            device = gpus[comm.get_local_rank() % len(gpus)]
        else:
            device = jax.devices("cpu")[0]

        if seed is not None:
            seed = seed + comm.get_data_parallel_rank()

        # Ensure batch can be evenly split among the model-parallel group
        if config.patching.levels > 0:
            model_parallel_size = comm.get_model_parallel_size()
            patching_factor = 2 ** (2 * config.patching.levels)
            total_batch_size = config.data.batch_size * patching_factor
            assert total_batch_size % model_parallel_size == 0, (
                f"With MG patching, total batch-size of {total_batch_size}"
                f" ({config.data.batch_size} times {patching_factor})."
                f" However, this total batch-size cannot be evenly split among"
                f" the {model_parallel_size} model-parallel groups."
            )
            for j, b_size in enumerate(config.data.test_batch_sizes):
                test_total_batch_size = b_size * patching_factor
                assert test_total_batch_size % model_parallel_size == 0, (
                    f"With MG patching, for test resolution of"
                    f" {config.data.test_resolutions[j]}"
                    f" the total batch-size is {test_total_batch_size}"
                    f" ({b_size} times {patching_factor})."
                    f" However, this total batch-size cannot be evenly split among"
                    f" the {model_parallel_size} model-parallel groups."
                )

    else:
        is_logger = True
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        if gpus:
            device = gpus[0]
        else:
            device = jax.devices("cpu")[0]

    # Set device, optimization settings
    # JAX manages random state via explicit PRNG keys rather than global seeds;
    # seed is retained here for downstream construction of jax.random.PRNGKey(seed).
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    if has_gpu:
        try:
            jax.config.update("jax_default_device", device)
        except Exception:
            pass

        increase_l2_fetch_granularity()
        try:
            jax.config.update("jax_default_matmul_precision", "high")
        except AttributeError:
            pass

    return device, is_logger


def increase_l2_fetch_granularity() -> None:
    try:
        import ctypes

        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device
        # limit 0x05 is cudaLimitMaxL2FetchGranularity
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except Exception:
        return
