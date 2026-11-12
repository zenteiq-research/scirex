<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Pipe Flow Dataset

*Internal flow, deformed geometry*

[All datasets](index.md){ .md-button } [Results](../benchmarks/pipe/results/comparative.md){ .md-button .md-button--primary }

Steady flow through a 2D pipe of length 10 and width 1. In every sample the centreline of the pipe is a different smooth random curve. The model is given the shape of the pipe, as the coordinates of a mesh that follows its walls, and predicts the horizontal velocity at every node. The data is the Geo-FNO benchmark [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | Pipe (Geo-FNO pipe-flow benchmark) |
| Task | The $(x, y)$ coordinates of the 129 × 129 mesh nodes → the horizontal velocity at every node |
| Samples | 2,310 in the files; the benchmark uses the first 1,200 (1,000 for training, 200 for testing) |
| Metric | Relative L2 error on the 200 test samples, mean over 3 seeds |
| Source | [[1]](#ref1); reused unchanged by [[2]](#ref2) |
| Used by | [FNO](../benchmarks/pipe/results/comparative.md#fno), [WNO](../benchmarks/pipe/results/comparative.md#wno), [GINO](../benchmarks/pipe/results/comparative.md#gino), [Transolver](../benchmarks/pipe/results/comparative.md#transolver) |

## Get the data

| | |
| --- | --- |
| Download | The Geo-FNO data release [[1]](#ref1); the Transolver repository [[2]](#ref2) links the same files |
| Files | `Pipe_X.npy`, `Pipe_Y.npy`, `Pipe_Q.npy` |
| Size | About 1.5 GB |
| Folder | `PIPE_DATA_DIR` or `--data-dir` |

Put the three files in one folder, written `<DATA_DIR>` here, and train:

```bash
export PIPE_DATA_DIR=<DATA_DIR>   # or pass --data-dir <DATA_DIR>
python scripts/benchmarks/pipe/train.py --run all
```

## How the benchmark reads the data

- The loader is `load_split()` in [`scripts/benchmarks/pipe/_data.py`](../../scripts/benchmarks/pipe/_data.py). It stops with an error if `Pipe_Q.npy` is truncated. (The Transolver script only warns and falls back to an older file.)
- The velocity is channel 0 of `Pipe_Q.npy`. Inputs are normalised with statistics from the training samples, and predictions are converted back to physical units before the loss and the metric.

## What changes between samples

Only the shape of the centreline changes. The inlet profile (a parabola with peak 0.25), the fluid and the no-slip walls are fixed, and the pipe is exactly 1 wide everywhere. Only one velocity component is learned, not the pressure or the vertical velocity. A few hard test shapes account for most of the mean error.

## Splits

| Split | Samples |
| --- | --- |
| Train | first 1,000 |
| Test | last 200 of the first 1,200 |

Every sample is a different shape, so the test shapes are unseen. There is no validation split, and the weights after the last epoch are reported.

## Results

See the [Pipe Flow results page](../benchmarks/pipe/results/comparative.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries* (Geo-FNO), Journal of Machine Learning Research 24(388), 2023. <https://arxiv.org/abs/2207.05209>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
