<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Build from source

To use the library, [install it with pip](index.md). Build from source only to change the library, run
the test suite, or use the benchmark scripts and the reproduction driver. The system requirements are the
same as for pip. On Windows, follow the Windows tab of [Install with pip](index.md#install-with-pip) first.

## Get the code

```bash
git clone {{ repo_url }}
cd scirex
python3 -m venv scirex-venv
source scirex-venv/bin/activate
python -m pip install --upgrade pip
```

## Install

Install in editable mode, so that changes to the code take effect without reinstalling. Choose one line.

| Command | Installs |
| --- | --- |
| `pip install -e .` | The library, for the CPU |
| `pip install -e ".[dev]"` | The library, the tests, the linters and pre-commit |
| `pip install -e ".[all]"` | Everything: tests, benchmarks, tuning, meshing, docs and the release checks |
| `pip install -e ".[all,cuda12]"` | Everything, with the NVIDIA GPU (CUDA 12) wheels |
| `pip install -e ".[all,tpu]"` | Everything, with the TPU wheels |

The extras `hpo`, `mesh`, `wandb`, `open3d`, `benchmark`, `tune` and `docs` are listed in
`pyproject.toml`. `all` includes every one of them except the hardware wheels.

To develop, enable the git hooks once:

```bash
pre-commit install
```

## Exact tested versions

`requirements.txt` has the exact versions the project is tested with. To install them:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Check the build

```bash
bash scripts/dev/check_platform.sh --backend cpu    # or gpu; imports everything, trains, saves, loads, infers
```

## Next

- [Verify the install](verify.md) explains the output and the options of the check.
- [Contributing](../dev_guide/contributing.md) shows how to run the tests and open a pull request.
