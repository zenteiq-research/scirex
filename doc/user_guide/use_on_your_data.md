<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Use It on Your Data

!!! abstract "In short"

    You have input and output fields from a solver or an experiment, and you
    want a model that predicts the output from the input. Save the arrays in one
    `.npz` file, run one command, and get a trained model, its test error and a
    script-free way to predict new cases. No tutorial is needed first.

    ```bash
    python scripts/train.py --model fno --data my_data.npz
    python scripts/predict.py --run runs/fno --data new_inputs.npz
    ```

The four steps:

1. [Pick a model](#1-pick-a-model)
2. [Save your data](#2-save-your-data)
3. [Train](#3-train)
4. [Predict](#4-predict)

You need ScireX installed ([Install](../install/index.md)) and a clone
of the repository for the two scripts.

## 1. Pick a model

| Your data | Start with | Also try |
|---|---|---|
| Fields on a regular grid, smooth solution (diffusion, Darcy, heat) | `fno` | `transolver` |
| Fields on a regular grid with sharp fronts, local features or non-periodic boundaries | `wno` | `fno` |
| Values at the points of a mesh or a point cloud, same mesh for every sample | `transolver` | `gino` |
| A geometry that changes from sample to sample, e.g. a 3D surface | `transolver` | `gino` |

If you are unsure, start with `fno` on a grid and `transolver` on a mesh:
they train reliably with their default settings. The
[theory page](../theory_guide/operator_learning.md) explains the differences,
and every model's results on eight benchmarks are in the
[Benchmarks](../benchmarks/index.md).

## 2. Save your data

Save the inputs as `x` and the outputs as `y` in one `.npz` file, **samples
first and channels last**, the layout of an image. All models use this file
format; the script converts it to what each model expects.

=== "Regular grid"

    ```python
    import numpy as np

    # 1000 samples on a 64 x 64 grid, 1 input field, 1 output field
    x = ...  # shape (1000, 64, 64, 1)   e.g. the permeability
    y = ...  # shape (1000, 64, 64, 1)   e.g. the pressure
    np.savez("my_data.npz", x=x, y=y)
    ```

    A 1D problem is `(samples, n, channels)`, a 3D one
    `(samples, nx, ny, nz, channels)`. The grid is taken to be uniform on
    `[0, 1]` in every direction; pass `coords` (below) if it is not.

=== "Mesh or point cloud"

    ```python
    import numpy as np

    # 500 samples, 3586 points each, 3D coordinates
    coords = ...  # shape (500, 3586, 3), or (3586, 3) if the mesh is shared
    x = ...       # shape (500, 3586, 2)   inputs at the points
    y = ...       # shape (500, 3586, 1)   outputs at the points
    np.savez("my_data.npz", coords=coords, x=x, y=y)
    ```

    If the geometry itself is the input (no input field), leave out `x`.

| Key | Required | Shape | Notes |
|---|---|---|---|
| `x` | Yes, unless the geometry is the only input | `(samples, *grid, c_in)` or `(samples, points, c_in)` | float |
| `y` | Yes | Same as `x` with `c_out` channels | float |
| `coords` | For meshes | `(samples, points, d)` or `(points, d)` | `d` is 2 or 3 |
| `x_test`, `y_test` | No | As `x`, `y` | A fixed test set; otherwise one is split off |

Rules:

- **Every sample has the same shape.** Meshes with different numbers of
  points must be padded to one size first.
- **Use float data in physical units.** The script normalises the inputs and
  outputs with statistics of the training set and reports errors in your
  units.
- **No NaN or infinite values.** The script checks and stops if it finds any.
- **A fixed test set** (`x_test`, `y_test`) needs one mesh for all samples;
  with a mesh per sample, the script splits off the test set itself
  (`--test-fraction`).

## 3. Train

```bash
python scripts/train.py --model fno --data my_data.npz
```

The script splits off a test set (20%, unless `x_test` is given), normalises
the data, builds the model, trains it, and prints the test error every few
epochs:

```text
Device: 1 x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (gpu)
Data:   my_data.npz  grid (64, 64)  in 1  out 1  train 800  test 200
Model:  fno  size medium  1.58 M parameters
epoch    10   train 5.87e-02   test rel. L2 5.87%
epoch    20   train 5.05e-02   test rel. L2 4.49%
...
epoch   190   train 4.79e-03   test rel. L2 0.63%
epoch   200   train 4.75e-03   test rel. L2 0.63%
Saved runs/fno  (test rel. L2 0.63%)
```

The `train` column is the relative L2 error on the training batches, the
`test` column on the held-out samples, both in the units of your data.

Check the setup first on a CPU in under a minute: `--smoke` trains a tiny
model for 2 epochs on the first 32 samples. Its error stays near 100%; it
only shows that the data, the model and the saving work.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | `fno`, `wno`, `gino` or `transolver` |
| `--data` | required | The `.npz` file |
| `--size` | `medium` | `small`, `medium` or `large`: the width and depth of the model ([below](#model-sizes)) |
| `--epochs` | `200` | Passes over the training set |
| `--batch-size` | `16` | Samples per step, summed over all devices |
| `--lr` | `1e-3` | Peak learning rate (AdamW with cosine decay) |
| `--test-fraction` | `0.2` | Share of samples held out when there is no `x_test` |
| `--seed` | `0` | Seed of the split, the initialisation and the batch order |
| `--out` | `runs/<model>` | Where the run is saved |
| `--set KEY=VALUE` | – | Any model argument, e.g. `--set n_modes=24,24` or `--set slice_num=64` (see [Models and Losses](models_and_losses.md#key-settings)); repeat for several |
| `--smoke` | off | Tiny model, few steps: checks that everything runs |

The defaults are a reasonable first run, not a tuned one. If the test error
is much higher than the training error, the model overfits: use more data or
`--size small`. If both are high, try `--size large`, more epochs or the other
model in the table above.

### Model sizes

Parameters per `--size` on a 64 × 64 grid (FNO, WNO) and on 1,024 mesh
points in 2D (Transolver, GINO). WNO's weights grow with the grid; the
others do not depend on it.

| Model | `small` | `medium` | `large` | What `--size` changes |
|---|---|---|---|---|
| FNO | 0.19 M | 1.58 M | 10.7 M | `hidden_channels` 24 / 48 / 96, `n_modes` 8 / 12 / 16 |
| WNO | 1.81 M | 7.24 M | 29.0 M | `hidden_channels` 24 / 48 / 96 |
| Transolver | 0.10 M | 0.54 M | 2.82 M | `n_hidden` 64 / 128 / 256, `n_layers` 4 / 6 / 8 |
| GINO | 0.50 M | 1.89 M | 5.09 M | latent grid 16 / 24 / 32 per side, `fno_hidden_channels` 24 / 48 / 64 |

### What to expect

With every option at its default (`--size medium`, 200 epochs), on two
synthetic problems of 1,000 samples (800 to train, 200 to test), on one
NVIDIA RTX PRO 6000 GPU:

| Model | Data | Parameters | Test relative L2 | Training time |
|---|---|---|---|---|
| FNO | 2D Poisson on a 64 × 64 grid | 1.58 M | 0.63% | 40 s |
| WNO | 2D Poisson on a 64 × 64 grid | 7.24 M | 1.87% | 2.0 min |
| Transolver | A 2D field at 1,024 random points per sample | 0.54 M | 0.78% | 1.1 min |
| GINO | A 2D field at 1,024 random points per sample | 1.89 M | 0.98% | 2.2 min |

These problems are smooth and easy; they show that the defaults train, not
how accurate a model will be on your data.

### CPU, GPU, TPU and several devices

Nothing to configure: the script uses the best device it finds (TPU, then
GPU, then CPU) and prints it on the first line. Which devices exist depends
on the install (`ScireX`, `scirex[cuda12]` or
`scirex[tpu]`).

| You have | What happens |
|---|---|
| A CPU only | Training runs on all cores. Fine for `--smoke` and small grids; slow for large models |
| One GPU or TPU | Training runs on it |
| Several GPUs or TPU cores | Each batch is split across them (data parallel). The batch size is rounded to a multiple of the device count |

| Option | Use |
|---|---|
| `--device cpu\|gpu\|tpu` | Force a backend, e.g. stay on the CPU on a machine with a GPU |
| `--gpus 0,1` | Use only these GPUs (shared machines) |
| `--no-preallocate` | Do not reserve 75% of the GPU memory at start (shared GPUs) |
| `--deterministic` | Bit-identical runs. GINO on a GPU is otherwise not bit-reproducible, and becomes about 10x slower with this option; see [Checkpoints and Reproducibility](checkpoints_and_reproducibility.md#determinism-ginos-scatter-add) |

Tested on CPUs, on NVIDIA GPUs and on 4 simulated devices; TPUs are
supported by JAX but the script has not been run on one yet. Not supported:
one model split across several GPUs, and training on several machines.

### What is saved

```text
runs/fno/
├── model_params.bytes      the trained weights
├── model_metadata.msgpack  the model's constructor arguments
├── config.json             the options, the data shapes and the normalisation statistics
├── history.json            training loss and test error per evaluated epoch
└── metrics.json            the final test errors and what produced them
```

`config.json` is written last: a directory without it is an unfinished run.

## 4. Predict

From the command line, for new inputs saved as `x` (and `coords` for a mesh)
in the format of step 2:

```bash
python scripts/predict.py --run runs/fno --data new_inputs.npz --out predictions.npz
```

`predictions.npz` holds `y_pred` in your units and layout. If the file also
has `y`, the relative L2 error is printed.

From Python:

```python
import sys

sys.path.insert(0, "scripts")  # the repository's scripts folder
from predict import load_run  # reads the run directory of step 3

predict = load_run("runs/fno")
y_pred = predict(x_new)  # x_new: (samples, 64, 64, 1) -> y_pred: (samples, 64, 64, 1)
```

The run normalises the inputs and de-normalises the predictions itself. On a
grid model the new inputs may have a different resolution from the training
data (FNO only; see the [theory page](../theory_guide/operator_learning.md#fno)).

## Problems

| Message or symptom | Cause and fix |
|---|---|
| `RESOURCE_EXHAUSTED` / out of memory | Lower `--batch-size`, or `--size small` |
| `x and y have different numbers of samples` | Check the first axis of both arrays |
| `expected channels last` | The last axis looks like space, not channels: move the channel axis to the end |
| Test error stuck near 100% | The model predicts the mean: check that `x` and `y` belong together, then raise `--lr` or the epochs |
| Very slow on a GPU | Check the first line says `gpu`; if it says `cpu`, install `scirex[cuda12]` |
| GINO results change from run to run | Expected on a GPU; use `--deterministic` |

## Going further

- Write your own training loop, loss or optimiser: [Workflow](workflow.md).
- Train without labelled outputs, with the physics as the loss:
  [Models and Losses](models_and_losses.md#the-ritz-energy).
- See each model trained step by step on a real benchmark:
  [Tutorials](../tutorials/index.md).
- Add your problem as a tested example in the repository:
  [Add a New Example](../tutorials/NewExample/Overview.md).
