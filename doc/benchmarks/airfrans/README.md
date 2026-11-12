<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# AirfRANS

Predict the steady RANS flow around a 2-D airfoil (velocity, pressure and turbulent viscosity at every node of the case's own mesh, about 180,000 nodes) from the airfoil geometry and the inflow, and from those fields the lift coefficient. It benchmarks operator learning on irregular, case-specific meshes, scored with the official evaluation of [[1]](#ref1) on the AirfRANS dataset ([[2]](#ref2)).

## At a glance

| | |
| --- | --- |
| Input | Per mesh node: position, distance to the airfoil, wall normal and inflow velocity |
| Output | Velocity, pressure and turbulent viscosity at every node; lift is computed from them |
| Data | The AirfRANS "full" task, scored on 200 test cases ([dataset guide](../../dataset_guide/airfrans.md)) |
| Metric | Volume and Surf (normalised mean squared error away from and on the airfoil wall), lift error and Spearman correlation of lift (Volume, Surf and lift error: lower is better) |
| Best result | **Transolver**, Volume 0.00131 (published Transolver: 0.0037) |

| Model | Volume | Surf | Lift error | Lift correlation |
|---|---|---|---|---|
| Transolver | 0.00131 | 0.00046 | 0.0265 | 0.99959 |
| WNO | 0.00446 | 0.00145 | 0.0297 | 0.99958 |
| GINO | 0.0254 | 0.00158 | 0.0752 | 0.99806 |
| Transolver, published | 0.0037 | 0.0142 | 0.1030 | 0.9978 |

All models have about 3.9 M parameters, matching the paper's Transolver. Drag is also computed but is not in the paper.

## Training and evaluation

Each step trains on one case and a random subset of 32,000 of its nodes. The loss is the mean squared error of the standardised velocity, pressure and turbulent viscosity, with the wall nodes and the other nodes averaged separately and added.

At test time the predictions are mapped back to physical units, velocity and viscosity are set to zero on the wall, and everything is standardised with the official statistics. Volume and Surf are then the mean squared error over the non-wall and wall nodes. Lift is integrated from the predicted pressure and wall shear stress. The paper's caption calls Volume and Surf relative L2 errors, but the official code computes the normalised mean squared error used here.

## Reproduce

You need a GPU and the dataset ([dataset guide](../../dataset_guide/airfrans.md)). Run from the repository root:

```bash
pip install -e ".[benchmark]"
export AIRFRANS_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/airfrans/train.py --run transolver
python scripts/benchmarks/airfrans/infer.py --out-dir <OUT_DIR>
```

`--run` also accepts `wno`, `gino` or `all`. The first call caches the node features (about 4 minutes). Training takes from about 2 hours (WNO) to about 12 hours (GINO).

The Transolver and WNO checkpoints were trained by the research scripts that `train.py` replaces and converted without retraining. A fresh `train.py` run has not been documented. `infer.py` scores GINO with a wider wall mask than the committed scores used, so its GINO volume error is slightly different (see the [AirfRANS notes](results/comparative.md#notes)).

## References

1. <span id="ref1"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
2. <span id="ref2"></span>Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions*, NeurIPS 2022, Datasets and Benchmarks. <https://arxiv.org/abs/2212.07564>
