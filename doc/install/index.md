<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Install ScireX

Install ScireX with `pip` into a virtual environment. The package runs on the CPU, on
NVIDIA GPUs (CUDA 12) and on TPUs.

## System requirements

| | Supported |
| --- | --- |
| Python | 3.10, 3.11 or 3.12 (64-bit) |
| Ubuntu | 22.04, 24.04 and 26.04 LTS, x86_64, CPU or NVIDIA GPU |
| macOS | 14 or later on Apple silicon, CPU only |
| Windows | Windows 11 on the CPU, see the Windows tab under [Install with pip](#install-with-pip) |
| NVIDIA GPU | Driver 525 or newer. No CUDA toolkit is needed: pip installs the CUDA 12 libraries |
| TPU | A Cloud TPU VM |
| `pip` | 23 or newer |

The package pins JAX to 0.6.x. Do not install a newer JAX by hand.

## Install with pip

=== "Linux"

    ```bash
    python3 --version                        # 3.10, 3.11 or 3.12
    sudo apt-get install python3-venv        # Ubuntu and Debian only, once

    python3 -m venv scirex-venv
    source scirex-venv/bin/activate
    python -m pip install --upgrade pip

    pip install scirex               # CPU
    pip install "scirex[cuda12]"     # or: NVIDIA GPU
    ```

    **Check.** `python -m scirex.info` lists the backend and the devices, and `python -m pip check` must say
    `No broken requirements found`. The Ubuntu 24.04 log shows both.

    **What does not work on Linux.** A GPU needs the NVIDIA driver and the `cuda12` extra; see
    [NVIDIA GPU](#nvidia-gpu) below. CUDA 13 is not supported.

=== "macOS"

    Use Python from python.org or `pyenv`. Homebrew is not required. Models run on the CPU, because
    JAX has no GPU backend for macOS.

    ```bash
    python3 --version                        # 3.10, 3.11 or 3.12

    python3 -m venv scirex-venv
    source scirex-venv/bin/activate
    python -m pip install --upgrade pip

    pip install scirex
    ```

    **Check.** `python -m scirex.info` lists the backend and the devices, and `python -m pip check` must say
    `No broken requirements found`. The macOS log shows both.

    **What does not work on macOS.** The GPU: JAX has no GPU backend for macOS, so the `cuda12` and
    `tpu` extras are not available. The optional `open3d` needs a system library; see
    [Troubleshooting](troubleshooting.md).

=== "Windows"

    ScireX runs on Windows 11 on the CPU with Python 3.10, 3.11 or 3.12 (64-bit). It was checked
    by hand on Windows 11 with Python 3.12.10: the install and `pip check`. The
    [log](logs.md#windows-11) shows the steps and their output. Windows is not part of CI.

    The install fails on a default Windows 11 for one reason: a dependency, `orbax-checkpoint`, has files
    whose paths are longer than the 260 characters Windows allows, unless long paths are switched on.
    Step 1 does that. It needs Administrator rights and a restart, once. All commands are for PowerShell.

    **1. Enable long paths.** In an Administrator PowerShell:

    ```powershell
    reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
    ```

    Restart Windows, then check:

    ```powershell
    reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled
    ```

    The value must be `0x1`. This is a setting for the whole PC.

    **2. Install Python.** Install Python 3.12 (3.10 and 3.11 also work). Python 3.13 does not: the
    package needs a version below 3.13.

    ```powershell
    winget install Python.Python.3.12 --source winget --scope user
    ```

    `--source winget` avoids a certificate error on the Microsoft Store source. With the python.org
    installer, tick "Disable path length limit".

    **3. Create a virtual environment.**

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
    once. Use `python -m pip` from now on, so that the pip of this environment is used and not another
    Python's.

    **4. Install.** From PyPI:

    ```powershell
    python -m pip install --upgrade pip
    python -m pip install scirex
    ```

    From a source checkout:

    ```powershell
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e . --no-deps
    ```

    For the tests add the dev tools: `python -m pip install -e ".[dev]"`.

    **5. Check.**

    ```powershell
    python -m pip check
    python -m scirex.info
    ```

    `pip check` must say `No broken requirements found`, and `scirex.info` must list a CPU device.

    **What does not work on Windows**

    - **GPU.** JAX has no CUDA build for native Windows, so the `cuda12` extra is not available here.
      Training runs on the CPU and is slow at benchmark sizes.
    - **Optional extras.** `mlflow` and `optuna` (the `hpo` extra), `ray` (the `tune` extra), `vtk` and `pandas` (the
      `benchmark` extra) are not installed by default; the tests that need them are skipped. [`tests/experimental/tuning`](../../tests/experimental/tuning) cannot
      be collected without `mlflow`: install the `hpo` extra or run
      `pytest tests --ignore=tests/experimental/tuning`.
    - **Shell scripts.** `*.sh` scripts run from Git Bash.

    For problems during the install see [Troubleshooting](troubleshooting.md).

=== "TPU"

    On a Cloud TPU VM (Linux), with Python 3.10, 3.11 or 3.12:

    ```bash
    python3 -m venv scirex-venv
    source scirex-venv/bin/activate
    python -m pip install --upgrade pip

    pip install "scirex[tpu]"
    ```

    If pip cannot find `libtpu`, add the JAX release index:
    `pip install "scirex[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`.

    **Check.** `python -m scirex.info` must list `TpuDevice`s, and `python -m pip check` must say
    `No broken requirements found`. The TPU v6e-8 log shows the platform check on a TPU VM.

    **What does not work on TPU.** The tests and benchmarks have run on CPU and NVIDIA GPU only; the
    platform check passes on a TPU, which has no float64 FFT.

**Linux install logs:** the Ubuntu 24.04, Ubuntu 24.04 GPU and Ubuntu 26.04 tabs of [Installation Logs](logs.md#ubuntu-2404) show the Linux
commands above and the checks run on each, with their full output.

**Windows install log:** the Windows 11 tab of [Installation Logs](logs.md#windows-11) shows the steps of the Windows tab above
and the checks run on Windows 11, with their output.

**macOS install log:** the macOS 27 tab of [Installation Logs](logs.md#macos-27) shows the macOS commands above and the checks run on macOS 27 (Apple M1), with their output.

**TPU log:** the TPU v6e-8 tab of [Installation Logs](logs.md#tpu-v6e-8) shows the platform check on a TPU VM.

Check the install:

```bash
python -c "import scirex; print(scirex.__version__)"
python -m scirex.info          # versions, operating system and the devices JAX found
```

A GPU install lists a `cuda` device. If it lists only `cpu`, see [Troubleshooting](troubleshooting.md).

## NVIDIA GPU

`pip install "scirex[cuda12]"` adds the CUDA 12 plugin and the NVIDIA runtime libraries as pip
packages, so only the NVIDIA driver must be installed on the machine. A driver built for CUDA 13 also runs
the CUDA 12 libraries, so a machine with a newer driver needs nothing different. The extra `cuda` is the
older name for `cuda12`.

!!! warning "CUDA 13 wheels are not supported in 1.0"
    The JAX versions that ScireX 1.0 is tested with (0.6.x) have no CUDA 13 wheels. Those start
    with JAX 0.7, outside the supported range. Install the CUDA 12 wheels on a CUDA 13 machine.

## TPU

The steps for a Cloud TPU VM are in the TPU tab of [Install with pip](#install-with-pip). JAX needs
`libtpu`, which the `tpu` extra installs; the TPU has no float64 FFT, so ScireX keeps float32 there.

## Environment variables

To keep GPU memory use predictable and to pick the GPU a run uses, set these before training or
running the benchmarks (`CUDA_VISIBLE_DEVICES` selects the GPU; the two `XLA` settings stop JAX from
reserving most of its memory at start-up):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0
```

## Other ways to install

- [Docker](docker.md): ready `dev` and `prod` images, with the GPU, your data and checkpoints mounted.
- [Build from source](source.md): for development, the tests and the benchmark scripts.

## Next

- [Verify the install](verify.md) on your system, CPU, GPU or TPU.
- [Quickstart](../user_guide/quickstart.md) trains your first operator.
- [Troubleshooting](troubleshooting.md) for GPU, driver, WSL2 and `open3d` problems.
