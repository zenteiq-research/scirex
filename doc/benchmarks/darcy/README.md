<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Darcy Flow

[FNO](../../api/models.md#scirex.models.FNO) [[1]](#ref1), [WNO](../../api/models.md#scirex.models.WNO) [[2]](#ref2), [GINO](../../api/models.md#scirex.models.GINO) [[3]](#ref3) and [Transolver](../../api/models.md#scirex.models.Transolver) [[4]](#ref4) learn the map from a permeability field to the steady pressure field of 2D Darcy flow, on the data of [[1]](#ref1). The four models are sized to about 2.83 M parameters and trained on the same data and split.

## At a glance

| | |
| --- | --- |
| Input | The permeability field $a$ on an 85 × 85 grid |
| Output | The pressure field $u$ on the same grid |
| Data | 1,000 training and 200 test samples ([dataset guide](../../dataset_guide/darcy_2d.md)) |
| Metric | Relative L2 error of the pressure on the test samples (lower is better) |
| Best result | **Transolver, 0.532%** (published: 0.57%) |

| Model | Test error |
|---|---|
| Transolver | 0.532% |
| FNO | 1.109% |
| GINO | 1.60% |
| WNO | 2.02% |

Each model uses the training recipe of its own paper or reference code. The Transolver figure is the best of 5 seeds; the 5-seed mean is 0.556%, which involves no selection on the test set. See the [comparison page](results/comparative.md) for details.

## Training and evaluation

Every loss and metric is computed on the prediction in the stored units of $u$. The error of one sample is the relative L2 error over the grid points, and the test metric is its mean over the 200 test samples, using the weights after the last epoch:

$$
e(\hat u, u) = \frac{\lVert \hat u - u \rVert_2}{\lVert u \rVert_2}
$$

The training loss is the batch mean of $e$. GINO and Transolver add a gradient term with weight 0.1, as in [[4]](#ref4); FNO and WNO do not. Inputs and outputs are normalised with statistics from the training samples only.

## Reproduce

You need a GPU and the data ([Get the data](../../dataset_guide/darcy_2d.md#get-the-data)). Run from the repository root:

```bash
export DARCY_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/darcy/train.py --run transolver
python scripts/benchmarks/darcy/infer.py --out-dir <OUT_DIR>
```

`--run` also accepts `fno`, `wno`, `gino` or `all`. Training takes about 5 minutes for FNO and about 90 minutes for GINO.

The committed metrics and figures were made by earlier per-model scripts; the current `train.py` and `infer.py` have not been shown to reproduce those files exactly. The [physics-only variant](../darcy_physics/README.md) uses the same data and trains without solutions.

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>
2. <span id="ref2"></span>Tripura and Chakraborty, *Wavelet Neural Operator for Solving Parametric Partial Differential Equations in Computational Mechanics Problems*, Computer Methods in Applied Mechanics and Engineering 404, 115783 (2023). <https://arxiv.org/abs/2205.02191>
3. <span id="ref3"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>
4. <span id="ref4"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
