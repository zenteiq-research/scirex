<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Troubleshooting

First run `python -m scirex.info`. It prints the ScireX and JAX versions, the backend and the
devices, which is what to include in a bug report. From a source checkout,
`python scripts/dev/verify_install.py --require gpu` also prints the CUDA-related environment and the
installed JAX and NVIDIA wheels.

## JAX uses the CPU although a GPU is present, or "CUDA backend failed to initialize"

Look at the messages JAX prints at start-up and work through the causes below.

## `LD_LIBRARY_PATH` override

The pip CUDA wheels find their own libraries. If `LD_LIBRARY_PATH` lists a system CUDA
or cuDNN (for example `/usr/local/cuda-11.8/lib64`), the dynamic linker loads those
first and JAX fails with errors such as "Found cuDNN version ... but JAX was built
against ..." or "Unable to load cuDNN". Run once without it:

```bash
env -u LD_LIBRARY_PATH python -m scirex.info
```

If that works, remove the CUDA entries from `LD_LIBRARY_PATH` in your shell profile, or
unset it in the job script before starting Python.

## Old driver

An error such as "CUDA driver version is insufficient for CUDA runtime version" means the
NVIDIA driver is older than the CUDA 12 wheels need. Check `nvidia-smi` (top right shows
the driver) and update to driver 525 or newer. A driver is a system component; it is not
installed by pip. Do not try to get around it with an older JAX: this release needs
JAX 0.6.x.

## Wrong CUDA wheel

Symptoms: `jax.devices()` shows only `CpuDevice`, or pip reports conflicting
`jax-cuda*` packages. Typical causes are a CPU-only `jaxlib` installed after the CUDA
one, a plugin for another CUDA version (`jax-cuda13-plugin`), or two plugins at once.
List what is installed, remove all JAX wheels, and install again:

```bash
pip list | grep -i -E "jax|nvidia"
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt
pip install "scirex[cuda12]"
```

## Float64 results are float32 (x64 flag)

JAX computes in float32 by default. ScireX does not switch float64 on for you.
Enable it before the first JAX call, either in the code
(`jax.config.update("jax_enable_x64", True)`) or in the environment
(`export JAX_ENABLE_X64=1`). Without it, requests for float64 silently give float32 and
tight numerical tolerances can fail. Float64 is much slower on most GPUs and needs more
memory; benchmarks in this repository run in float32 unless their page says otherwise.

## Windows: `WinError 3` while installing

`pip install` stops with `WinError 3` on a file inside `orbax-checkpoint`, and the virtual environment
is left without `pip`. A path in that package is longer than the 260 characters Windows allows by
default. Switch on long paths and install again, see step 1 in
the Windows tab of [Install with pip](index.md#install-with-pip).

## Windows: `No module named 'scirex.cli'`

This release has no `scirex` command, so the error comes from a launcher left by an older
install. Reinstall (`python -m pip install --force-reinstall --no-deps scirex`, or
`python -m pip install -e . --no-deps` in a source checkout) and use the Python API and the
scripts in the repository.

## WSL2

GPU access in WSL2 works with the NVIDIA driver installed **on Windows**; do not install
a Linux NVIDIA driver inside WSL. Check that `nvidia-smi` works inside WSL, then install
the CUDA 12 wheels as above. Keep the project files in the Linux file system
(`~/...`), not under `/mnt/c`, for speed, and run `wsl --update` if the GPU is not
found. WSL2 GPU is supported by JAX as an experimental platform; for production runs use
native Linux.

## macOS: no GPU

There is no GPU backend for Apple silicon in this release. On a Mac ScireX runs
on the CPU only; the platform check passes there (see the [macOS log](logs.md#macos-27)). The
`jax-metal` plugin has not been tried. Use a Linux machine or the Docker image on a remote GPU host for real training.

## `open3d` does not load: "Library not loaded ... libusb"

The `open3d` 0.20 wheel links the system library `libusb`. Without it, importing `open3d` (and the
modules that use it, `scirex.data.datasets.car_cfd` and `mesh_datamodule`) fails with
`Library not loaded: /opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib` on macOS, or a missing
`libusb-1.0.so.0` on Linux.

On macOS this package keeps `open3d` at 0.19.x, whose wheel needs nothing from the system, so a fresh
`pip install -e ".[all]"` works without Homebrew. If you already have 0.20 installed, replace it:

```bash
pip install "open3d>=0.19,<0.20"        # macOS: no system library needed
```

On Linux install the library instead (the Docker image already does):

```bash
sudo apt-get install libusb-1.0-0       # Debian and Ubuntu
```

`open3d` is optional: the rest of ScireX works without it. The platform check reports this case as a
warning, not a failure.

## Out of memory

JAX reserves most of the GPU memory at start-up. Set the variables in
[Environment variables](index.md#environment-variables) and, if needed, reduce the batch size.
