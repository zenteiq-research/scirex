<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Lid-Driven Cavity

Predict the next flow snapshot (pressure and the two velocity components) of a turbulent lid-driven cavity from the previous three, and test on a lid speed (Re 10,000) that was held out of training. It benchmarks short-horizon forecasting of a CFD field on a regular grid, following [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Input | Three past frames of pressure and velocity on a 128 × 128 grid |
| Output | The next frame |
| Data | 8 training cases and one held-out test case, Re 10,000 ([dataset guide](../../dataset_guide/cavity.md)) |
| Metric | RMSE over pressure and both velocity components, in m/s (lower is better) |
| Best result | **GINO, 1.95e-3 m/s** (paper's FNO: 6.43e-3) |

| Model | One-step RMSE (m/s) |
|---|---|
| GINO | 1.95e-3 |
| FNO | 1.97e-3 |
| WNO | 2.04e-3 |
| Transolver | 2.04e-3 |
| Copy the last frame (baseline) | 4.40e-3 |

All models have about 1.05 M parameters. Only the FNO is in the source paper. A 10-step forecast, which feeds each prediction back as input, is on the [comparison page](results/comparative.md).

## Training and evaluation

Inputs and targets are normalised per channel with statistics from the training windows only. The training loss is the mean squared error of the normalised fields plus a penalty (weight 0.1) on the divergence of the predicted velocity, which should be zero for an incompressible flow.

The metric is the root-mean-square error in physical units, over all test windows, cells and the three channels:

$$
\mathrm{RMSE} = \sqrt{\overline{\left(\hat{q}_c - q_c\right)^2}}
$$

## Reproduce

You need a GPU and the data ([dataset guide](../../dataset_guide/cavity.md)). Run from the repository root:

```bash
export CAVITY_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/cavity/train.py --run fno_3to1
python scripts/benchmarks/cavity/infer.py --out-root <OUT_DIR>
```

`--run` also accepts `wno_3to1`, `gino_3to1`, `transolver_3to1` or `all`. Training takes from about 6 minutes (FNO) to about 90 minutes (Transolver) on one NVIDIA GB10 GPU.

## References

1. <span id="ref1"></span>Costa Rocha et al., *Deep Neural Network Modeling for CFD Simulations: Benchmarking the Fourier Neural Operator on the Lid-Driven Cavity Case*, Applied Sciences 13(5), 3165 (2023). <https://doi.org/10.3390/app13053165>
