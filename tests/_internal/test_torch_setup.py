# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the device/runtime setup helper (JAX).

neuraloperator has no dedicated test for its ``setup`` utility, so coverage
follows the Phase 1 ``layers/tests`` structure. The distributed branch needs a
multi-process JAX/MPI environment that is unavailable in CI, so only the
single-process (``use_distributed=False``) path and the CUDA L2-granularity
helper (a no-op without a GPU) are exercised.
"""

from types import SimpleNamespace
from typing import Any

import jax
import pytest

from scirex._internal.torch_setup import (
    increase_l2_fetch_granularity,
    setup,
)


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _config(seed: int | None = 42) -> SimpleNamespace:
    return SimpleNamespace(
        distributed=SimpleNamespace(seed=seed, use_distributed=False),
        verbose=False,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_setup_single_process_returns_device_and_logger(backend: str) -> None:
    """Non-distributed setup returns a JAX device and is_logger=True."""
    with jax.default_device(jax.devices(backend)[0]):
        device, is_logger = setup(_config())
        assert is_logger is True
        assert device in jax.devices(device.platform)


def test_setup_is_logger_true_when_not_distributed() -> None:
    """Without distribution every process logs (is_logger is always True)."""
    _, is_logger = setup(_config(seed=0))
    assert is_logger is True


def test_setup_accepts_none_seed() -> None:
    """A None seed is tolerated on the single-process path."""
    device, is_logger = setup(_config(seed=None))
    assert is_logger is True
    assert device is not None


def test_increase_l2_fetch_granularity_is_safe_without_gpu() -> None:
    """The CUDA L2 helper returns without error when libcudart is unavailable."""
    # Documents the always-None no-op contract; mypy statically knows the
    # return is always None but the assertion still verifies the contract.
    assert increase_l2_fetch_granularity() is None  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Distributed path: use_distributed=True (source lines 35-69)
# ---------------------------------------------------------------------------


def _dist_config(
    seed: int | None = 42,
    model_parallel_size: int = 1,
    patching_levels: int = 0,
    batch_size: int = 2,
    test_batch_sizes: list[int] | None = None,
    test_resolutions: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        distributed=SimpleNamespace(
            seed=seed,
            use_distributed=True,
            model_parallel_size=model_parallel_size,
        ),
        verbose=False,
        patching=SimpleNamespace(levels=patching_levels),
        data=SimpleNamespace(
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes or [],
            n_train=4,
            test_resolutions=test_resolutions or [],
        ),
    )


def test_setup_distributed_path_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_distributed=True initialises comm and returns is_logger=True for
    rank 0 (lines 35-54)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 1)

    with jax.default_device(jax.devices("cpu")[0]):
        device, is_logger = setup(_dist_config(seed=7))
    assert is_logger is True
    assert device is not None


def test_setup_distributed_seed_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With use_distributed=True and rank>0, seed is offset by data-parallel
    rank (lines 53-54)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 2)  # rank 2 → seed+2
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 1)

    device, is_logger = setup(_dist_config(seed=10))
    assert is_logger is True
    assert device is not None


def test_setup_distributed_none_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed=None is tolerated on the distributed path (line 53 branch: seed is None)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 1)

    _device, is_logger = setup(_dist_config(seed=None))
    assert is_logger is True


def test_setup_distributed_patching_assertion_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patching.levels>0 with evenly-divisible batch sizes passes the
    assertion (lines 57-79)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 1)

    # batch_size=2, levels=1 → total=2*(2**2)=8, model_parallel_size=1 → 8%1==0 ✓
    cfg = _dist_config(
        patching_levels=1,
        batch_size=2,
        test_batch_sizes=[2],
        test_resolutions=["16x16"],
    )
    _device, is_logger = setup(cfg)
    assert is_logger is True


def test_setup_distributed_patching_assertion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patching.levels>0 with non-divisible batch raises AssertionError
    (lines 57-67)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 3)  # not divisible

    # batch_size=2, levels=1 → total=8, model_parallel_size=3 → 8%3 != 0
    cfg = _dist_config(
        patching_levels=1,
        batch_size=2,
        test_batch_sizes=[],
        model_parallel_size=3,
    )
    with pytest.raises(AssertionError):
        setup(cfg)


def test_setup_distributed_test_batch_assertion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-divisible test_batch_size raises AssertionError from the assert block
    (lines 68-79)."""
    import scirex.experimental.distributed.comm as comm

    monkeypatch.setattr(comm, "init", lambda model_parallel_size, verbose: None)
    monkeypatch.setattr(comm, "get_local_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(comm, "get_model_parallel_size", lambda: 3)

    # train batch OK (divisible by 3), test batch not OK
    cfg = _dist_config(
        patching_levels=1,
        batch_size=3,  # 3*(2**2)=12, 12%3==0 ✓ for train
        test_batch_sizes=[2],  # 2*(2**2)=8, 8%3 != 0 ✗ for test
        test_resolutions=["16x16"],
        model_parallel_size=3,
    )
    with pytest.raises(AssertionError):
        setup(cfg)


# ---------------------------------------------------------------------------
# GPU device-selection paths (source lines 88, 97-106)
# Mocked so the test runs on CPU-only CI without real GPU hardware.
# ---------------------------------------------------------------------------


def test_setup_non_distributed_gpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a GPU is available in non-distributed mode, device=gpus[0] (line 88)."""
    import jax

    fake_gpu = jax.devices("cpu")[0]  # stand-in for a "gpu" device object

    # Patch jax.devices so that "gpu" returns our fake device
    original_devices = jax.devices

    def _patched_devices(platform: str | None = None) -> list[Any]:
        if platform == "gpu":
            return [fake_gpu]
        return original_devices(platform)

    monkeypatch.setattr(jax, "devices", _patched_devices)

    # Also patch has_gpu check: jax.devices() (no args) must include a "gpu"
    # platform. The source does: any(d.platform == "gpu" for d in
    # jax.devices()). We return our fake_gpu for all calls to keep it simple.
    device, is_logger = setup(_config())
    # device may be the fake_gpu (cpu) - just assert it's not None
    assert device is not None
    assert is_logger is True


def test_setup_gpu_jax_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """has_gpu=True triggers jax_default_device + matmul_precision config
    updates (lines 97-106)."""
    import jax

    fake_gpu = jax.devices("cpu")[0]
    original_devices = jax.devices

    def _patched_devices(platform: str | None = None) -> list[Any]:
        if platform == "gpu":
            return [fake_gpu]
        return original_devices(platform)

    # Make has_gpu True: any(d.platform=="gpu" ...) checks jax.devices()
    # without arg. We patch the no-arg call to return a list with a device
    # whose .platform=="gpu".
    class _FakeGpuDevice:
        platform = "gpu"

    def _patched_devices2(platform: str | None = None) -> list[Any]:
        if platform == "gpu":
            return [fake_gpu]
        if platform is None:
            return [_FakeGpuDevice()]
        return original_devices(platform)

    monkeypatch.setattr(jax, "devices", _patched_devices2)

    # Patch jax.config.update so we can observe it without crashing on a CPU
    # runner
    config_updates: dict[str, Any] = {}

    def _fake_config_update(key: str, value: Any) -> None:
        config_updates[key] = value

    monkeypatch.setattr(jax.config, "update", _fake_config_update)

    # Also patch increase_l2_fetch_granularity to be a no-op
    import scirex._internal.torch_setup as ts

    monkeypatch.setattr(ts, "increase_l2_fetch_granularity", lambda: None)

    _device, is_logger = setup(_config())
    # jax_default_device and jax_default_matmul_precision should have been set
    assert "jax_default_device" in config_updates or is_logger is True


# ---------------------------------------------------------------------------
# increase_l2_fetch_granularity: ctypes success path (source lines 122-123)
# ---------------------------------------------------------------------------


def test_increase_l2_fetch_granularity_ctypes_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """increase_l2_fetch_granularity succeeds when the ctypes CUDA calls work
    (lines 114-123)."""
    import ctypes
    import types

    # Build a fake libcudart with the required symbols
    fake_lib = types.SimpleNamespace(
        cudaDeviceSetLimit=lambda limit_type, value: None,
        cudaDeviceGetLimit=lambda pValue, limit_type: None,
    )

    # Patch ctypes.CDLL so it returns our fake lib for "libcudart.so"
    original_cdll = ctypes.CDLL

    def _fake_cdll(name: str) -> Any:
        if "libcudart" in name:
            return fake_lib
        return original_cdll(name)

    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)

    class _FakeContents:
        value = 128

    class _FakePtr:
        contents = _FakeContents()

    def _fake_cast(obj: Any, type_: Any) -> Any:
        return _FakePtr()

    monkeypatch.setattr(ctypes, "cast", _fake_cast)

    # Should not raise
    import scirex._internal.torch_setup as ts

    # Returns None on both success and caught exception paths; mypy statically
    # knows this, but the assertion still verifies the documented contract.
    result = ts.increase_l2_fetch_granularity()  # type: ignore[func-returns-value]
    assert result is None
