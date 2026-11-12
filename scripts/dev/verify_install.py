#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Check that ScireX and JAX work on this machine, and say what is wrong if not.

Prints the versions, the JAX backend and devices, the float64 setting, the CUDA-related
environment and installed JAX/NVIDIA wheels, then runs a small jitted computation on
the default device. With ``--require gpu|tpu|cpu`` it fails unless JAX is using that
kind of device; hints for the usual failures point to the troubleshooting section of
``doc/install/troubleshooting.md``.

Usage::

    python scripts/dev/verify_install.py [--require gpu|tpu|cpu] [--x64]

``--x64`` turns on float64 first (the same as ``JAX_ENABLE_X64=1``) and checks that a
float64 array really is float64. Exit status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

HINTS = {
    "cpu_only": (
        "JAX only sees the CPU. If you expected a GPU: check `nvidia-smi`, that the CUDA "
        'wheel is installed (`pip install "scirex[cuda12]"`), and that no other '
        "jaxlib is installed (see 'Wrong CUDA wheel' on the troubleshooting page)."
    ),
    "ld_library_path": (
        "LD_LIBRARY_PATH is set and mentions CUDA. A system CUDA or cuDNN found there can "
        "override the libraries of the pip wheel; try `env -u LD_LIBRARY_PATH python ...`."
    ),
}


def nvidia_smi() -> str:
    """Driver version and GPU names from nvidia-smi, or why it is not available."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found: no NVIDIA driver on this machine, or not on PATH"
    done = subprocess.run(
        [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    text = (done.stdout or done.stderr).strip()
    return text.replace("\n", " | ") or f"nvidia-smi exited with {done.returncode}"


def cuda_failure(jax) -> str:
    """Ask JAX for the CUDA backend and return the reason it gives for failing."""
    try:
        devices = jax.devices("cuda")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return f"CUDA devices found ({len(devices)}) but not the default backend"


def wheels() -> list[str]:
    names = []
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name.startswith(("jax", "flax", "optax", "nvidia-", "libtpu", "scirex")):
            names.append(f"{name}=={dist.version}")
    return sorted(set(names))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--require", choices=["gpu", "tpu", "cpu"])
    ap.add_argument("--x64", action="store_true", help="enable and check float64")
    args = ap.parse_args(argv)

    failures: list[str] = []
    print(f"python   {platform.python_version()} on {platform.platform()}")
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        failures.append("Python must be 3.10, 3.11 or 3.12")

    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        print(f"FAIL     cannot import jax: {exc}")
        return 1
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    try:
        import scirex

        print(f"scirex   {scirex.__version__}  ({scirex.__file__})")
    except Exception as exc:
        failures.append(f"cannot import scirex: {exc}")

    backend = jax.default_backend()
    devices = jax.devices()
    print(
        f"jax      {jax.__version__}, backend {backend}, x64 {jax.config.jax_enable_x64}"
    )
    for d in devices:
        print(f"device   {d}")
    cuda_env = {
        k: v
        for k, v in os.environ.items()
        if k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS")
        or k.startswith(("XLA_PYTHON", "JAX_"))
    }
    for k, v in cuda_env.items():
        print(f"env      {k}={v}")
    installed = wheels()
    print("wheels   " + ", ".join(installed))
    if backend == "cpu" and (
        args.require == "gpu" or any("jax-cuda" in w for w in installed)
    ):
        print(f"nvidia   {nvidia_smi()}")
        print(f"reason   {cuda_failure(jax)}")
        if os.environ.get("JAX_PLATFORMS", "") not in ("", "cuda", "gpu"):
            print(
                "hint     JAX_PLATFORMS is set and excludes the GPU; run `unset JAX_PLATFORMS`."
            )
        if os.environ.get("JAX_PLATFORM_NAME") == "cpu":
            print(
                "hint     JAX_PLATFORM_NAME=cpu forces the CPU; run `unset JAX_PLATFORM_NAME`."
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            print(
                "hint     CUDA_VISIBLE_DEVICES is empty, which hides every GPU; unset it."
            )

    try:
        x = jnp.arange(6.0).reshape(2, 3)
        y = jax.jit(lambda a: (a @ a.T).sum())(x)
        y.block_until_ready()
        print(f"compute  jit matmul on {y.devices()} -> {float(y)}")
        if float(y) != 83.0:
            failures.append(f"jit matmul gave {float(y)}, expected 83.0")
    except Exception as exc:
        failures.append(f"computation failed: {exc}")

    if args.x64 and jnp.zeros(1).dtype != jnp.float64:
        failures.append(
            "float64 requested but arrays are float32 (see x64 in the guide)"
        )

    if args.require and backend != args.require:
        failures.append(f"expected a {args.require} backend, JAX is using {backend}")
        if backend == "cpu":
            print("hint     " + HINTS["cpu_only"])
    if (
        "cuda" in os.environ.get("LD_LIBRARY_PATH", "").lower()
        and args.require == "gpu"
    ):
        print("hint     " + HINTS["ld_library_path"])

    for f in failures:
        print("FAIL    ", f)
    print("verify_install:", "FAILED" if failures else "ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
