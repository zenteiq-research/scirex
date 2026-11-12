<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Reproduce our benchmarks

Re-evaluate the published checkpoints with
[`scripts/reproduce.py`](../../scripts/reproduce.py) and compare each number with the published
reference. Nothing is trained. The published numbers come from Linux with NVIDIA GPUs.

## 1. Install

The scripts are in the source checkout ([Build from source](../install/source.md)).

=== "Linux / macOS"

    ```bash
    git clone {{ repo_url }} && cd scirex
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -e ".[benchmark]" huggingface_hub   # NVIDIA GPU: ".[benchmark,cuda12]"
    ```

=== "Windows"

    ```powershell
    git clone {{ repo_url }}; cd scirex
    python -m venv .venv; .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[benchmark]" huggingface_hub
    ```

## 2. Download the checkpoints

Each checkpoint is its own repository on Hugging Face, named `zenteiq-ai/scirex-<model>-<problem>`,
for example [`zenteiq-ai/scirex-fno-pipe`](https://huggingface.co/zenteiq-ai/scirex-fno-pipe).
Download it into `ckpt/<model>/<problem>`, the layout `reproduce.py` reads:

```bash
hf download zenteiq-ai/scirex-fno-pipe --local-dir ckpt/fno/pipe        # one checkpoint
```

All 33, one line per repository group (`<model>` is `fno`, `gino`, `transolver` or `wno`; AirfRANS has no
FNO, and the two car problems have `gino`, `giwo` and `transolver`):

```bash
dl() { hf download "zenteiq-ai/scirex-$1-$2" --local-dir "ckpt/$1/$3"; }
for m in fno gino transolver wno; do
  dl $m chip-cooling chip_cooling
  dl $m darcy darcy_s85/data_driven
  dl $m darcy-physics darcy_s85/physics_only_1000
  dl $m darcy-physics-extra-fields darcy_s85/physics_only_best
  dl $m navier-stokes navier_stokes
  dl $m pipe pipe
done
for m in gino transolver wno; do dl $m airfrans airfrans; done
for m in gino giwo transolver; do dl $m car-cfd car_cfd; dl $m shapenetcar shapenetcar; done
```

`darcy-physics` holds the runs on the 1,000 released fields and `darcy-physics-extra-fields` the best
run with extra unlabelled fields. Cavity has no repository, because its data is not public.

## 3. Get the data

Put each dataset in `data/<problem>/`, from its [dataset guide](../dataset_guide/index.md). Three use a
subfolder: `airfrans/Dataset/`, `car_cfd/processed-car-pressure-data/` and `shapenetcar/training_data/`.
Darcy needs no data; Cavity is not public.

## 4. Run

```bash
# one checkpoint
python scripts/reproduce.py --mode infer --model fno --problem pipe --ckpt-root ckpt --data-root data

# everything
python scripts/reproduce.py --mode infer --all --allow-archived-metrics --ckpt-root ckpt --data-root data
```

Add `--device cpu` without an NVIDIA GPU (always on macOS and Windows). `--list` shows every cell.

## 5. Read the output

Results go to `reproduce_out/`: `results.csv` (one row per cell) and `results_table.csv` and
`results_table.tex` (one column per model). The numbers are the ones measured from the checkpoints. A cell
that cannot be evaluated reads `MISSING` with the reason; add `--strict-missing` to make the run fail on it.

The checkpoint behind each number is on the [checkpoint cards](cards/index.md); maintainer details are in
[Checkpoints and Reproducibility](../user_guide/checkpoints_and_reproducibility.md#reproducing-the-published-tables).
