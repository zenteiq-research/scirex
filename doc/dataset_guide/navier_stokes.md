<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Navier-Stokes 2D Dataset

*Incompressible fluid flow*

[All datasets](index.md){ .md-button } [Results](../benchmarks/navier_stokes/results/comparative.md){ .md-button .md-button--primary }

The swirling of a 2D fluid on a periodic unit square, at viscosity $\nu=10^{-5}$, stored as 20 snapshots of vorticity. The model reads the first 10 snapshots and predicts the next 10 one at a time, feeding its own predictions back in. At this viscosity the flow is close to turbulent, so small errors grow quickly. The data is the benchmark of [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | Navier-Stokes, viscosity 1e-5 (`NavierStokes_V1e-5_N1200_T20.mat`) |
| Task | Vorticity at times 1 to 10 → vorticity at times 11 to 20, on a 64 × 64 grid |
| Samples | 1,200 (the first 1,000 for training, the last 200 for testing) |
| Metric | Relative L2 error over the 10 predicted frames of each test sample |
| Source | [[1]](#ref1); reused unchanged by [[2]](#ref2) |
| Used by | [FNO](../benchmarks/navier_stokes/results/comparative.md#fno), [WNO](../benchmarks/navier_stokes/results/comparative.md#wno), [GINO](../benchmarks/navier_stokes/results/comparative.md#gino), [Transolver](../benchmarks/navier_stokes/results/comparative.md#transolver) |

## Get the data

| | |
| --- | --- |
| Download | Linked from the "Navier-Stokes" row of the Transolver benchmark [[2]](#ref2); the file belongs to the FNO data release [[1]](#ref1) |
| Files | `NavierStokes_V1e-5_N1200_T20.mat` |
| Size | About 400 MB |
| Folder | `NS_DATA_DIR` or `--data-dir` |

Put the file in one folder, written `<DATA_DIR>` here, and train:

```bash
export NS_DATA_DIR=<DATA_DIR>   # or pass --data-dir <DATA_DIR>
python scripts/benchmarks/navier_stokes/train.py --run all
```

## How the benchmark reads the data

- It reads one variable, `u`, of shape `(1200, 64, 64, 20)`, and uses the frames as stored, with no normalisation, as the Transolver reference does.
- Every model receives the 10 history frames as input channels.
- The loader is `load_split()` in [`scripts/benchmarks/navier_stokes/_data.py`](../../scripts/benchmarks/navier_stokes/_data.py).

## What changes between samples

Only the starting vorticity changes, a random Gaussian field. The viscosity, the forcing and the grid are fixed, so the results say nothing about other viscosities or resolutions.

## Splits

| Split | Samples |
| --- | --- |
| Train | first 1,000 |
| Test | last 200 |

There is no validation split, and the weights after the last epoch are reported. Training gives the model the true previous frames (teacher forcing), while testing feeds back its own predictions, as the reference implementation does. So errors grow along the rollout: for the best model (GINO), the tenth step has about 6 times the error of the first, and for the Transolver about 14 times.

## Results

See the [Navier-Stokes results page](../benchmarks/navier_stokes/results/comparative.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
