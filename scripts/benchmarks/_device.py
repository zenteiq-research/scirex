# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Device checks shared by the benchmark runners."""

from __future__ import annotations

import logging

import jax

# scirex reports progress through ``logging`` (INFO) and prints nothing itself. Every
# benchmark runner imports this module, so one call here makes their console output
# look as it did before: plain messages on stderr. ``basicConfig`` does nothing if the
# caller has already configured logging.
logging.basicConfig(level=logging.INFO, format="%(message)s")


def require_gpu(allow_cpu: bool = False) -> None:
    """Aborts unless JAX is actually on the GPU.

    JAX falls back to CPU silently when the CUDA plugin cannot initialise, and
    a training run that quietly takes that path would waste hours, so the
    fallback is turned into a hard failure unless ``allow_cpu`` is set.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else "none"
    if platform != "gpu" and not allow_cpu:
        raise SystemExit(
            f"GPU required but JAX is on {platform!r} (devices={devices}). "
            "Check that /dev/nvidia* is reachable and JAX_PLATFORMS is not set "
            "to cpu, or pass --allow-cpu."
        )
    print(f"[device] {devices}")
