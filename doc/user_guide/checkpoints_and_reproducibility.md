<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Checkpoints and Reproducibility

A trained model is only useful if it can be loaded again, evaluated on the
same data and give the same number. This page covers the checkpoint formats,
how the benchmark `infer.py` scripts evaluate a checkpoint, what they write
to `metrics.json`, and how far a result can be reproduced, including the one
model whose GPU results are not bit-reproducible by default: GINO.

## Checkpoints

ScireX uses two checkpoint formats.

### The library format

Every model has `save_checkpoint` and `from_checkpoint`
(see [Workflow](workflow.md#6-checkpoint)):

```python
model.save_checkpoint("runs/poisson", "fno", params)
# runs/poisson/fno_params.bytes       the weights
# runs/poisson/fno_metadata.msgpack   the constructor arguments

model, params = FNO.from_checkpoint("runs/poisson", "fno", template)
```

`template` is a parameter pytree of the right structure, from a fresh
`model.init`. `from_checkpoint` rebuilds the model from the saved constructor
arguments and returns the weights as JAX arrays, ready for `apply`. The
metadata is msgpack, not pickle, so loading a checkpoint does not run code
from it.

This format holds the model only. Save everything else you need to evaluate
it, such as the config, the normalisation statistics and the seed, next to it.

### The benchmark run directory

The benchmark `train.py` scripts write one directory per run. For Darcy
([`scripts/benchmarks/darcy/_checkpoint.py`](../../scripts/benchmarks/darcy/_checkpoint.py)):

| File | Content |
|---|---|
| `params.msgpack` | The weights after the last epoch. Every reported number comes from these. |
| `best_params.msgpack` | The weights at the lowest test error seen during training, kept for reference only; no reported number is chosen by them. |
| `history.json` | Test error, training loss and time per evaluated epoch. |
| `config.json` | The run (model, size, recipe, seed), the data protocol, the normalisation statistics, the training summary and the provenance of the training run. |
| `frames.npz` | With `train.py --animate` only: the frames `animate.py` renders. |

`config.json` is written last, so a directory without it is an unfinished
run. The other benchmarks follow the same layout with their own extras; the
cavity, for instance, saves the weights at the lowest *validation* error.

Checkpoints are binary build artefacts: they are gitignored and kept in the
checkpoint archive, not in the repository. Only the figures and
`metrics.json` are committed.

!!! note "Legacy checkpoints"
    The archived Darcy runs and the AirfRANS GINO have no `config.json`. The
    `infer.py` scripts cannot load them; their published `metrics.json` files
    are the record. See [Reproduce our benchmarks](../benchmarks/reproducibility.md).

## Inference

Every benchmark is trained by `train.py` and evaluated by `infer.py`; no
training happens in `infer.py`.

```bash
# every run found under --ckpt-root
python scripts/benchmarks/darcy/infer.py

# one run directory
python scripts/benchmarks/darcy/infer.py --checkpoint /path/to/checkpoints/darcy/fno

# a directory whose name is not a run name, written somewhere else
python scripts/benchmarks/darcy/infer.py --checkpoint /tmp/my_fno --run fno --out /tmp/eval
```

| Flag | Meaning |
|---|---|
| `--checkpoint DIR` | Evaluate only this run directory. Without it, every run under `--ckpt-root` is evaluated. |
| `--run NAME` | The name in `_config.RUNS` of the `--checkpoint` run, when its directory name is not one. |
| `--out DIR` | Where the figures and `metrics.json` are written (older spellings: `--out-dir`, `--out-root`). |
| `--data-dir DIR` | The dataset root. |

Two checks run before any number is computed:

- **The run is known.** A directory that matches no run in `_config.RUNS`
  is refused, with the list of valid names.
- **The data match the checkpoint.** `infer.py` recomputes the normalisation
  statistics from the data and compares them with those stored in
  `config.json`; a difference is an error ("is this the dataset and split it
  was trained on?"). Without this check, a model evaluated on a different
  split or a re-generated dataset would report a wrong number without any sign.

## `metrics.json`

`infer.py` writes one `metrics.json` per run next to its figures, under
`doc/assets/benchmarks/<problem>_<run>/`. The benchmark pages quote these
files. What the current Darcy `infer.py` writes (the committed Darcy files
come from the archived legacy runs and use older keys):

| Key | Content |
|---|---|
| `problem`, `run`, `model`, `seed` | What was evaluated. |
| `arch`, `opt`, `protocol` | The model size, the training recipe and the data split, copied from `config.json`. |
| `n_params` | Parameter count, a complex weight counted as two. |
| `test_rel_l2_mean`, `test_rel_l2_std` | The result: relative L2 over the test samples. |
| `steps_run`, `train_seconds` | The training summary. |
| `history` | The per-epoch history, as in `history.json`. |
| `paper` | The published reference value, where there is one. |
| `inference_provenance` | What produced this file (below). |

The metric keys differ between benchmarks (Car-CFD reports pressure errors,
AirfRANS volume, surface and force errors, chip cooling R², RMSE and MAE);
each benchmark page names the keys it quotes.

### Provenance

`inference_provenance` records what produced the numbers:

| Key | Content |
|---|---|
| `commit` | `sha`, `branch` and `dirty` of the repository. `dirty: true` means uncommitted changes were present, so the SHA alone does not identify the code. |
| `checkpoints` | For each checkpoint: its path, the SHA-256 of its weights and `config.json`, and `checksum_verified` against the archive's recorded checksums. |
| `environment` | Python, JAX, jaxlib, Flax, NumPy and ScireX versions, the devices and the host. |
| `command`, `timestamp_utc`, `schema_version` | The exact command, its time and the block's format version. |

`checksum_verified` is `true` if every hash matches a recorded checksum, `false`
if one does not (a warning is logged and the evaluation still completes), and
`null` when no checksum is recorded, as for checkpoints downloaded from Hugging
Face (use their `SHA256SUMS` instead). The archive's own
`Checkpoints_Benchmarks/CHECKSUMS.sha256` takes precedence over the
[checkpoint inventory](#maintaining-the-checkpoints). The helper is
[`scripts/benchmarks/_provenance.py`](../../scripts/benchmarks/_provenance.py). AirfRANS' older
`provenance` key describes the training run.

## Reproducibility

To get the same number again you need the same code, data, weights,
software and hardware. `metrics.json` records the first four; compare them
first when a number differs.

| What | Fixed by |
|---|---|
| Model initialisation | `jax.random.PRNGKey(seed)`, the run's `seed` |
| Order of the training batches | A key split from the same seed, a new permutation every epoch |
| Code | The commit in `inference_provenance.commit`, with `dirty: false` |
| Weights | The SHA-256 in `inference_provenance.checkpoints` |
| Libraries | The versions in `inference_provenance.environment`; the pins in `pyproject.toml` |

With all of these equal:

- **On a CPU**, training is bit-reproducible: the Pipe runners train
  bit-identical weights from the same seed (checked for every model; see the
  [Pipe FNO page](../benchmarks/pipe/results/comparative.md#fno)).
- **On a GPU**, FNO, WNO and Transolver evaluate to bit-identical outputs
  and gradients from run to run on the same machine. GINO does not (next
  section).
- **Across devices** (CPU against GPU, or two GPU models) float32 results
  differ in the last digits: the devices use different kernels and summation
  orders. Compare such numbers to a relative tolerance, not exactly.

### Determinism: GINO's scatter-add

GINO's graph layers sum the contributions of each point's neighbours with a
scatter-add (`jax.ops.segment_sum`). On a GPU the additions run in parallel
and their order is not fixed, and float32 addition is not associative, so
the sum differs in the last bits from one call to the next. The differences
carry through the network: re-running `infer.py` on the same checkpoint can
change a metric in its third significant figure (on the cavity, 8.17e-3 and
8.18e-3 have both come out of one checkpoint).

XLA's deterministic mode fixes the order:

```bash
XLA_FLAGS=--xla_gpu_deterministic_ops=true python scripts/benchmarks/cavity/infer.py
```

Set the variable in the shell, or in Python before `import jax`, as the
Navier-Stokes `infer.py` does; it is read once, when JAX starts its GPU
backend. The Navier-Stokes script is the only one that sets it.

Measured on an RTX PRO 6000 (Blackwell) with JAX 0.6.2: the Darcy benchmark's
GINO (2.8 M parameters, 85 × 85 points, batch 8, random inputs), the same jitted
function called repeatedly on the same inputs:

| | Without the flag | With the flag |
|---|---|---|
| Distinct forward outputs in 20 calls | 20 | 1 |
| Largest forward difference (outputs up to 1.36) | 4.4e-4 | 0 |
| Distinct gradients in 20 calls | 20 | 1 |
| GINO forward time | 9.3 ms | 56 ms (6x) |
| GINO gradient time | 29 ms | 303 ms (10x) |

Under the same test FNO, WNO and Transolver gave identical results with and
without the flag; the flag slowed their gradient by 1.4x (WNO) to 3.5x (FNO).

So:

- **For evaluation** that must give the same digits every time, such as a
  regression test or a number compared at three significant figures, set the
  flag. Inference is cheap, so the slowdown rarely matters.
- **For training**, leave it off unless you need bit-identical runs: a 10x
  slower GINO gradient is a high price, and seed-to-seed variation is far
  larger than this noise.
- **When quoting a GINO number**, quote the committed `metrics.json`; the
  benchmark pages do.

## Reproducing the published tables

[`scripts/reproduce.py`](../../scripts/reproduce.py) re-evaluates the archived checkpoints with each
benchmark's `infer.py` and writes the numbers it measures: `results.csv` (one
row per cell) and `results_table.csv` and `.tex` (one column per model). The
numbers are not compared with any other table. No training is done. The steps
and the options are on [Reproduce our benchmarks](../benchmarks/reproducibility.md).
`--mode train` is not implemented in `reproduce.py`; each cell's training entry
point is `scripts/benchmarks/<problem>/train.py`.

### Maintaining the checkpoints

- **Cell map.** [`scripts/reproduce_paper_map.py`](../../scripts/reproduce_paper_map.py) is the single
  source of truth: per cell the benchmark, regime, model, run, checkpoint path,
  metric key and unit conversion. `reproduce_paper_map.hf_repo_of` gives the Hugging
  Face repository (`zenteiq-ai/scirex-<model>-<problem>`) and `hf_path` the folder it is downloaded
  into (`<model>/<problem>`, Darcy `<model>/darcy_s85/<regime>`);
  `--layout archive` reads the archive's folder names instead.
- **Inventory.** [`scripts/inventory_checkpoints.py`](../../scripts/inventory_checkpoints.py) writes
  `doc/benchmarks/checkpoint_inventory.csv` (weights, SHA-256, size, parameters,
  training commit, loadable or `legacy`) and `checkpoint_inventory.sha256`;
  `--verify` compares the archive with it and fails on any difference.
- **Cards.** The [checkpoint cards](../benchmarks/cards/index.md) are generated:
  `python scripts/make_checkpoint_cards.py --results reproduce_out/results.csv`
  (`--check` fails if they are stale).

To add or replace a checkpoint:

1. Put the weights in the archive, update its `README.md`, run
   `python scripts/inventory_checkpoints.py`, and add the files to its Hugging Face repository
   (`zenteiq-ai/scirex-<model>-<problem>`).
2. Edit the cell in [`scripts/reproduce_paper_map.py`](../../scripts/reproduce_paper_map.py) (`ckpt`, `run`, `key`,
   `scale`).
3. Run `python scripts/reproduce.py --only "<row>"` and
   `pytest scripts/tests/test_reproduce.py`.
4. Regenerate the cards.

## See also

- [Workflow](workflow.md): save and load a checkpoint in your own loop.
- [Training Scripts](training.md): the benchmark `train.py` and `infer.py`.
- [Reproduce our benchmarks](../benchmarks/reproducibility.md): the steps to
  re-evaluate the published checkpoints.
