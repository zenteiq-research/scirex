<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Darcy Flow Dataset

*Subsurface flow*

[All datasets](index.md){ .md-button } [Results](../benchmarks/darcy/results/comparative.md){ .md-button .md-button--primary }

Fluid flowing through a porous medium made of two materials. The model is given a map of where each material sits (the permeability $a$) and predicts the pressure $u$ on the unit square, where $-\nabla\cdot(a\nabla u)=1$ and $u=0$ at the edge. The data is the benchmark of [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | Darcy Flow 2D (operator-learning benchmark) |
| Task | Permeability $a$ → pressure $u$, both on an 85 × 85 grid (the original is 421 × 421; every 5th point is kept) |
| Samples | 1,000 for training, 200 for testing |
| Metric | Relative L2 error of the pressure on the test samples |
| Source | [[1]](#ref1); [Zenodo record 12784353](https://zenodo.org/records/12784353) |
| Used by | [FNO](../benchmarks/darcy/results/comparative.md#fno), [WNO](../benchmarks/darcy/results/comparative.md#wno), [GINO](../benchmarks/darcy/results/comparative.md#gino), [Transolver](../benchmarks/darcy/results/comparative.md#transolver), and the [physics-only benchmark](../benchmarks/darcy_physics/results/setup.md) |

## Get the data

| | |
| --- | --- |
| Download | The Fourier Neural Operator data release of [[1]](#ref1), also linked as "Dataset-2" in the [WNO repository](https://github.com/TapasTripura/WNO). The `.pt` files of the [Zenodo record](https://zenodo.org/records/12784353) are a different release and are not used |
| Files | `piececonst_r421_N1024_smooth1.mat` (training) and `piececonst_r421_N1024_smooth2.mat` (testing) |
| Size | About 1.7 GB per file |
| Folder | `DARCY_DATA_DIR` or `--data-dir` |

Put both files in one folder, written `<DATA_DIR>` here, and train:

```bash
export DARCY_DATA_DIR=<DATA_DIR>   # or pass --data-dir <DATA_DIR>
python scripts/benchmarks/darcy/train.py --run all
```

## How the benchmark reads the data

- Each file holds 1,024 samples of `coeff` (the permeability) and `sol` (the pressure), each 421 × 421. The loader keeps every 5th point, as the code of [[1]](#ref1) does. `coeff` takes only the values 3 and 12.
- Each model is normalised the way its reference code does it, and predictions are converted back to physical units before the loss and the metric.
- The benchmark has its own loader; it does not use [`src/scirex/data/datasets/darcy.py`](../../src/scirex/data/datasets/darcy.py).

??? note "Technical details"

    - **Grid points.** The 421 values on each axis are cell centres, $(2k-1)/842$, not solver nodes. The kept 85 sit at $(10m+1)/842$, spaced $5/421$ apart.
    - **The edge of `sol` is small but not zero.** The solver interpolates `coeff` to a node grid, solves with $u=0$ on the boundary nodes, and interpolates back. On the first 1,000 samples of `smooth1` the mean $|u|$ on the outer ring is $5.2 \times 10^{-5}$ (largest $2.2 \times 10^{-3}$), against $5.8 \times 10^{-3}$ inside.
    - **Value ranges.** `sol` runs from $-1.8 \times 10^{-5}$ to $0.0178$ in `smooth1`, and from $-1.7 \times 10^{-3}$ to $0.0175$ in `smooth2`.
    - **How it was solved.** In MATLAB (the file headers read "MATLAB 5.0 MAT-file, Platform: MACI64"). `sol` matches a 5-point finite-difference solve to a relative L2 difference of about 4e-13.

## What changes between samples

Only the permeability field changes. It takes only two values, 3 and 12, and the source term ($f=1$) and the boundary condition ($u=0$) are the same every time. So the data covers one family of two-material fields, not general coefficients.

## Splits

| Split | Samples |
| --- | --- |
| Train | 1,000 |
| Test | 200 |

The two sets come from separate files with independently drawn fields, so no sample is in both. There is no validation split, and the normalisation uses the training set only.

## Results

See the [Darcy Flow results page](../benchmarks/darcy/results/comparative.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>
