<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Car-CFD

Predict the steady surface pressure on a car body from its geometry (a 3,586-vertex surface mesh, or its signed distance function on a grid), as a surrogate for a 3-D external-aerodynamics simulation. It is the Car-CFD study of [[1]](#ref1) on the cars of [[2]](#ref2).

## At a glance

| | |
| --- | --- |
| Input | The car shape: its signed distance function on a 64³ grid (GINO) or the position and normal of each surface vertex (Transolver) |
| Output | Pressure at the 3,586 surface vertices |
| Data | 611 ShapeNet cars, 500 for training and 111 for testing |
| Metric | Relative L2 error of the pressure on the test cars (lower is better) |
| Best result | **Transolver, 6.81%** ([[1]](#ref1): 7.12%) |

| Model | Test error |
|---|---|
| Transolver | 6.81% |
| GINO | 8.37% |

Both models have about 75 M parameters. The experimental GIWO is under [Experimental](../experimental/car_cfd_giwo.md). Every figure is for seed 0. The [comparison page](results/comparative.md) has train errors and the published value.

## Training and evaluation

Pressure is standardised with one mean and one standard deviation. The training loss is the relative L2 error of the standardised pressure, per car. The test metric is the same error on the original pressure, averaged over the 111 test cars:

$$
E = \frac{1}{N_\mathrm{test}} \sum_{k=1}^{N_\mathrm{test}} \frac{\lVert \hat p_k - p_k \rVert_2}{\lVert p_k \rVert_2}
$$

The pressure statistics are computed over all 611 cars, including the test cars, as in the neuraloperator reference loader, and every number here was produced that way. The unit of the pressure is not documented.

## Reproduce

You need a GPU. The data are downloaded on first use. Run from the repository root:

```bash
pip install -e ".[benchmark]"
python scripts/benchmarks/car_cfd/train.py --run paper transolver_val giwo_val
python scripts/benchmarks/car_cfd/infer.py --out-root <OUT_DIR>
```

`infer.py` needs the checkpoints of all three runs, including the experimental GIWO's `giwo_val`. Each run takes 20 to 30 minutes on one NVIDIA RTX PRO 6000 Blackwell GPU.

## References

1. <span id="ref1"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>
2. <span id="ref2"></span>Umetani and Bickel, *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*, ACM Transactions on Graphics 37(4), 2018. <https://doi.org/10.1145/3197517.3201325>
