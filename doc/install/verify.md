<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Verify the install

```bash
python -m scirex.info    # ScireX and JAX versions, backend, devices, float64 flag
python -m pip check      # must say: No broken requirements found
```

On a GPU machine `backend` must say `gpu` and `devices` must list a `CudaDevice`. From a source
checkout, [`scripts/dev/verify_install.py`](../../scripts/dev/verify_install.py) also runs a jitted
computation and has options: `--require gpu` fails unless JAX uses a GPU, and `--x64` also checks that
float64 works.

## Check the whole workflow on your system

From a source checkout, one script identifies the operating system, checks it against the
supported set (Linux, macOS, Windows), imports every module, and for Transolver, FNO, GINO
and WNO trains two steps on dummy data, saves and loads a checkpoint and runs inference:

```bash
bash scripts/dev/check_platform.sh --backend cpu    # or gpu, tpu, auto; add --light on a busy machine
```

`RESULT: OK` with `--backend gpu` means the four models trained, saved, loaded and ran on the GPU. On Windows run it from Git Bash. It prints each step and ends with `RESULT: OK` or
the failing step.

| Option | What it does |
| --- | --- |
| `--backend auto\|cpu\|gpu\|tpu` | Device to test. `auto` (default) uses what JAX picks and says which. `cpu`, `gpu` and `tpu` are strict: the check fails, with a hint, if that device is not available, and also fails if the model arrays are not on it. |
| `--cpu` | The same as `--backend cpu`. |
| `--only fno,wno,gino,transolver` | Runs the workflow for these models only. Default: all four. |
| `--steps N` | Number of training steps per model. Default: 2. |
| `--python PATH` | Interpreter that has the library installed. Default: the first of `python3`, `python`, `py -3` on the path. |
| `--python-version X.Y` | Uses `pythonX.Y` (or the `py -X.Y` launcher on Windows) from the path when `--python` is not given, and fails unless the script runs on that version, for example `--python-version 3.11`. The supported versions are 3.10, 3.11 and 3.12. |
| `--keep DIR` | Writes the checkpoints to `DIR` and keeps them. Default: a temporary folder that is deleted. |
| `--skip-imports` | Skips the import of every module and the public API check; runs the workflow only. |
| `--light` | Uses 2 CPU threads and the lowest priority, for a busy or small machine. |
| `--help` | Prints the usage and exits. |

The exit status is 0 when nothing failed, 1 when a check failed and 2 when the script could not
start (no Python, or JAX missing).

### What a pass means

`RESULT: OK` shows that the system is supported, the library imports, and the train, save, load and
inference path runs on the device you asked for. It does **not** show that real training will work or
be practical there. It uses models of a few thousand parameters and 2 steps, so it does not test:

- memory at real model sizes, or how long training takes;
- float64 behaviour and numerical tolerances on that device;
- loading the real datasets, the benchmark scripts, `reproduce.py` or the physics-informed losses;
- multi-GPU or distributed runs.

The check has been run on the CPU on Linux (Ubuntu 24.04, Python 3.12, [log](logs.md#ubuntu-2404), and Ubuntu 26.04, Python 3.12, [log](logs.md#ubuntu-2604)), on two NVIDIA RTX PRO 6000 GPUs (Ubuntu 24.04, Python 3.12, CUDA 12, [log](logs.md#ubuntu-2404-gpu)), on macOS 27 with an Apple M1 (Python 3.11, [log](logs.md#macos-27)) and on a TPU v6e-8 VM ([log](logs.md#tpu-v6e-8)). Windows 11 (Python 3.12) is covered by the install steps and the checks in its [log](logs.md#windows-11), not by this script. To cover the other versions run the script once for each interpreter: `--python-version 3.10`, `3.11`, `3.12`. macOS runs on the CPU only.
To say that a platform can train, also pass the complete test suites on it (`make test`), then run a tutorial in `--smoke` mode on real
data, for example `python scripts/tutorials/darcy_fno.py --data-dir <data> --smoke`.

## Run the test suites

```bash
make test-unit          # tests/ with JIT disabled
make test-integration   # tests/ with JIT on
make test               # tests/ with coverage, then scripts/tests/
```

The suites run on the CPU. To check that JAX sees a GPU, run `python -m scirex.info`; `devices`
must show `CudaDevice`, not `CpuDevice`.

## If it fails

See [Troubleshooting](troubleshooting.md).
