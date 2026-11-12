<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Lid-Driven Cavity Dataset

*Turbulent internal flow*

[All datasets](index.md){ .md-button } [Results](../benchmarks/cavity/results/comparative.md){ .md-button .md-button--primary }

Turbulent flow in a 0.1 m × 0.1 m square box whose lid slides sideways at a fixed speed, solved over time in OpenFOAM with a k-epsilon turbulence model. The model reads a short window of past snapshots (pressure and the two velocity components) and forecasts the next one. It is tested on a Reynolds number that was left out of training. The data is the benchmark of Costa [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | Lid-Driven Cavity, transient k-epsilon (OpenFOAM benchmark) |
| Task | The previous 3 snapshots of $(p, U_x, U_y)$ → the next snapshot, on a 128 × 128 grid |
| Cases | 9 of the 11 simulated Reynolds numbers, 101 time steps each |
| Metric | RMSE over $p$, $U_x$ and $U_y$ together, on the held-out Re 10,000 case |
| Source | Costa [[1]](#ref1) |
| Used by | [FNO](../benchmarks/cavity/results/comparative.md#fno), [WNO](../benchmarks/cavity/results/comparative.md#wno), [GINO](../benchmarks/cavity/results/comparative.md#gino), [Transolver](../benchmarks/cavity/results/comparative.md#transolver) |

The raw CSV export is not public and is not redistributed, so there is no download. The licence of the data is stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai).

## Get the data

| | |
| --- | --- |
| Download | Not distributed with this repository. The OpenFOAM runs are those of Costa [[1]](#ref1); the export used here is the team's private copy, and whether it matches the data linked in the paper is not confirmed |
| Files | One folder per case (`01k`, `02k`, `04k`, `06k`, `10k`, `14k`, `16k`, `18k`, `20k`), each with one `Data<case>0.<step>.csv` per snapshot |
| Size | About 160 MB once converted |
| Folder | `CAVITY_DATA_DIR` or `--data-dir`. Converted cases are cached in the folder given by `--cache-dir` |

Put the case folders in one folder, written `<DATA_DIR>` here, and train:

```bash
export CAVITY_DATA_DIR=<DATA_DIR>   # or pass --data-dir <DATA_DIR>
python scripts/benchmarks/cavity/train.py --run all
```

## How the benchmark reads the data

- The export samples the 129 × 129 mesh vertices. They are averaged onto the 128 × 128 cell centres, not cropped, because cropping would drop the row next to the moving lid that drives the flow.
- It reads steps 0 to 100 of each case and cuts sliding windows of 3 input frames and 1 output frame.
- Velocities are in m/s. No source states the unit of $p$; the values fit a kinematic pressure (m²/s²), but that is not confirmed.
- Each channel is normalised with statistics from the training windows only.

## What changes between cases

Only the lid speed changes, from 0.1 to 2.0 m/s, which gives Reynolds numbers from 1,000 to 20,000 ($L=0.1$ m, $\nu=10^{-5}$ m²/s). The geometry, the mesh and the boundary conditions are fixed. The benchmark tests interpolation to one Reynolds number in one 2D geometry, not extrapolation.

## Splits

| Split | Reynolds numbers |
| --- | --- |
| Train and validation | 1,000, 2,000, 4,000, 6,000, 14,000, 16,000, 18,000, 20,000 |
| Test | 10,000 |

The training windows are split 90% / 10% at random (seed 0); the 10% is used only for the learning-rate schedule and early stopping. The test case never enters training. Reynolds numbers 8,000 and 12,000 are left out, following the protocol of [[1]](#ref1), "to leave a reasonable gap".

## Results

See the [Lid-Driven Cavity results page](../benchmarks/cavity/results/comparative.md).

## References

1. <span id="ref1"></span>Costa Rocha et al., *Deep Neural Network Modeling for CFD Simulations: Benchmarking the Fourier Neural Operator on the Lid-Driven Cavity Case*, Applied Sciences 13(5), 3165 (2023). <https://doi.org/10.3390/app13053165>
