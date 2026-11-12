<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# ShapeNetCar

Predict the steady air flow around a car (velocity everywhere, and pressure on the car surface) from the shape of the car alone. It uses the ShapeNetCar data of [[1]](#ref1), as in [[2]](#ref2).

## At a glance

| | |
| --- | --- |
| Input | The car as a point cloud: position, signed distance to the car surface and surface normal at each point |
| Output | Velocity (3 components) at all points and pressure at the surface points |
| Data | About 889 cars; fold 0 gives 789 for training and 100 for validation ([dataset guide](../../dataset_guide/shapenetcar.md)) |
| Metric | Relative L2 error of velocity and of surface pressure on the validation cars; also the drag coefficient (lower is better, except correlation) |
| Best result | **Transolver**, velocity 0.0223, surface pressure 0.0767 (paper: 0.0207 and 0.0745) |

| Metric | Transolver | GINO | Paper (Transolver) |
|---|---|---|---|
| Velocity error | **0.0223** | 0.0730 | 0.0207 |
| Surface pressure error | **0.0767** | 0.2149 | 0.0745 |
| Drag error | **0.0106** | 0.0779 | 0.0103 |
| Drag correlation | **0.9928** | 0.6685 | 0.9935 |

Both models have about 3.8 M parameters. The experimental GIWO is under [Experimental](../experimental/shapenetcar_giwo.md). The drag coefficient is computed after the fact from the predicted fields and is never trained on.

## Training and evaluation

Inputs and targets are standardised per channel with training-fold statistics. The training loss, per car, is the mean squared error of the velocity at all points plus half the mean squared error of the pressure at the surface points. The reported errors are relative L2 errors on the original (unstandardised) fields, averaged over the validation cars:

$$
\mathrm{rel\_l2\_volume}=\frac{\lVert \hat{\mathbf u}-\mathbf u\rVert_2}{\lVert \mathbf u\rVert_2}, \qquad
\mathrm{rel\_l2\_surface}=\frac{\lVert \hat p-p\rVert_{2}}{\lVert p\rVert_{2}} \ \text{(surface points)}
$$

The drag formula uses constants fixed by the paper's benchmark authors, which are not physical air properties. The paper text and the code also disagree on the flow axis; the code and data use $+z$.

## Reproduce

You need a GPU and the data ([dataset guide](../../dataset_guide/shapenetcar.md)). Run from the repository root:

```bash
pip install -e ".[benchmark]"
python scripts/benchmarks/shapenetcar/train.py --run transolver \
    --data-dir ~/data/mlcfd_data/training_data
python scripts/benchmarks/shapenetcar/infer.py \
    --data-dir ~/data/mlcfd_data/training_data --out-root outputs/shapenetcar
```

Training the Transolver for 200 epochs takes about an hour on one NVIDIA RTX PRO 6000 Blackwell GPU, plus a one-off preprocessing of the VTK files. Add `--epochs 2` for a smoke test only.

## References

1. <span id="ref1"></span>Umetani and Bickel, *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*, ACM Transactions on Graphics 37(4), 2018. <https://doi.org/10.1145/3197517.3201325>
2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
