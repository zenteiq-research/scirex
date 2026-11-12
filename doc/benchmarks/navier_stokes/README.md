<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Navier-Stokes 2D

Predict the next 10 vorticity snapshots of a 2-D incompressible flow on the unit torus (viscosity $10^{-5}$, 64 × 64 grid) from the previous 10. At this viscosity the flow is close to turbulent, so errors grow quickly over the rollout. It is the Navier-Stokes problem of [[1]](#ref1), with the same data and recipe as [[2]](#ref2).

## At a glance

| | |
| --- | --- |
| Input | Vorticity at 10 time steps on a 64 × 64 grid |
| Output | Vorticity at the next 10 time steps |
| Data | 1,000 training and 200 test flows ([dataset guide](../../dataset_guide/navier_stokes.md)) |
| Metric | Relative L2 error over the 10 predicted frames, on the test flows (lower is better) |
| Best result | **GINO, 0.0666** (published Transolver: 0.0900) |

| Model | Test error |
|---|---|
| GINO | 0.0666 |
| FNO | 0.0705 |
| Transolver | 0.0872 |
| WNO | 0.3198 |
| Copy the last frame (baseline) | 0.7481 |

All models have about 11.2 M parameters, matched to Transolver at the size used in its paper.

## Training and evaluation

Each model maps the last 10 frames to the next frame. During training it is given the true previous frames (teacher forcing); the loss is the relative L2 error of each predicted frame, summed over the 10 steps. No normalisation is applied.

At test time the model is rolled out on its own predictions for 10 steps, and the metric is the relative L2 error over all 10 frames of a flow, averaged over the 200 test flows:

$$
E = \frac{1}{N_\mathrm{test}} \sum_{s=1}^{N_\mathrm{test}} \frac{\lVert \hat W^{(s)} - W^{(s)} \rVert_2}{\lVert W^{(s)} \rVert_2}
$$

## Reproduce

You need a GPU and the data ([Get the data](../../dataset_guide/navier_stokes.md#get-the-data)). Run from the repository root:

```bash
export NS_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/navier_stokes/train.py --run gino
python scripts/benchmarks/navier_stokes/infer.py --out-dir <OUT_DIR>
```

`--run` also accepts `transolver`, `fno`, `wno` or `all`. Training takes from about 2.5 hours (FNO) to about 9 hours (Transolver, alone on a GPU). The Transolver and WNO checkpoints were trained by an earlier script with the same recipe and converted to the current format without retraining.

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>
2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
