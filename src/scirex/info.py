# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Environment report: `python -m scirex.info`."""

import importlib.metadata as md
import platform
import re

import jax

import scirex

_NVIDIA_DISTS = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cudnn-cu13",
)


def _dist_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _clean(text: str) -> str:
    """Drop the machine name and a stray quote from the JAX environment text."""
    text = re.sub(r"node='[^']*'", "node='<host>'", text)
    return re.sub(r'(local devices?)"', r"\1", text)


def environment_report() -> str:
    """Return a text report of the environment.

    It covers scirex, Python, OS, JAX and jaxlib, backend, devices, x64 flag, CUDA
    and cuDNN.

    Returns
    -------
    str
        The report, as printed by `python -m scirex.info`.
    """
    lines = [
        f"ScireX:  {scirex.__version__}",
        f"OS:      {platform.system()} {platform.release()} {platform.machine()}",
        f"backend: {jax.default_backend()}",
        f"devices: {jax.devices()}",
        f"x64:     {bool(jax.config.read('jax_enable_x64'))}",
    ]
    for name in ("flax", "optax", *_NVIDIA_DISTS):
        if (v := _dist_version(name)) is not None:
            lines.append(f"{name}: {v}")
    lines.append(_clean(jax.print_environment_info(return_string=True) or ""))
    return "\n".join(lines)


def main() -> None:
    """Print the environment report. This is the entry point of `python -m scirex.info`."""
    print(environment_report())  # noqa: T201 - a command-line report


if __name__ == "__main__":
    main()
