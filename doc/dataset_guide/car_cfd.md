<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Car-CFD Dataset

*Automotive external aerodynamics*

[All datasets](index.md){ .md-button } [Results](../benchmarks/car_cfd/results/comparative.md){ .md-button .md-button--primary }

Steady air flow around 611 car bodies from ShapeNet's car category, in the processed copy made by neuraloperator. Each car has a surface mesh and its signed distance function on a regular grid. The model is given the car's shape and predicts the pressure on its surface. The simulations are those of [[1]](#ref1), and the protocol is that of the GINO paper [[2]](#ref2). This is not [ShapeNetCar](shapenetcar.md), which reads the raw VTK export and also predicts volume velocity; the two are not comparable.

## At a glance

| | |
| --- | --- |
| Dataset | Car-CFD, "Three-dimensional flow dataset over ShapeNet-Car" (neuraloperator's processed version) |
| Task | GINO and GIWO: the signed distance function on a 64 × 64 × 64 grid → surface pressure. Transolver: the position and unit normal of each surface vertex → surface pressure |
| Samples | 611 cars (500 for training, 111 for testing); each surface has 3,586 vertices |
| Metric | Relative L2 error of the de-normalised pressure on the 111 test cars |
| Source | [Zenodo record 13936501](https://zenodo.org/records/13936501); simulations by [[1]](#ref1); protocol from [[2]](#ref2) |
| Used by | [GINO](../benchmarks/car_cfd/results/comparative.md#gino), [GIWO](../benchmarks/experimental/car_cfd_giwo.md) (experimental), [Transolver](../benchmarks/car_cfd/results/comparative.md#transolver) |

The data is the Zenodo record named above. The licence of the data is stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai).

## Get the data

| | |
| --- | --- |
| Download | Automatic. The loader downloads the [Zenodo record](https://zenodo.org/records/13936501) on first use |
| Files | `train.txt`, `test.txt` and, for each car, the surface mesh, the pressure and the signed distance function |
| Size | About 400 MB |
| Folder | `--data-root`. Without it the loader downloads into a default folder in your home directory; `--no-download` uses files that are already there |

```bash
python scripts/benchmarks/car_cfd/train.py --run all --data-root <DATA_DIR>
```

## How the benchmark reads the data

The loader is `CarCFDDatasetjax` in [`src/scirex/data/datasets/car_cfd.py`](../../src/scirex/data/datasets/car_cfd.py). For each car it returns `vertices`, `vertex_normals`, `press` (the target) and `distance` (the signed distance function).

- Vertices and the grid are scaled into $[0, 1]$ with one bounding box shared by all cars.
- Pressure is normalised with one mean and one standard deviation. **These come from all loaded cars, training and test together.** This is how neuraloperator's reference loader computes them, and every number on the Car-CFD pages was produced this way, so the normalisation is not a training-only fit.
- GINO and GIWO work out the neighbour lists between the latent grid and the surface vertices once per car.

??? note "Technical details"

    `press` has 3,682 stored entries per car, but a car has 3,586 vertices. The loader keeps `press[:, 0:16]` and `press[:, 112:]` and drops the 96 entries at positions 16 to 111, so that the target matches the vertices. Why those 96 entries exist is not recorded anywhere.

## What changes between samples

Only the car shape changes. Every car sees the same inflow, so the task is shape to surface pressure and nothing else; there is no volume velocity.

## Splits

| Split | Cars |
| --- | --- |
| Train | 500 |
| Test | 111 |
| Validation (to choose learning rates only) | 50 |

The train and test sets are the dataset's own `train.txt` and `test.txt`, the split of [[2]](#ref2). The paper has no validation set, so for choosing learning rates this repository trains on the first 450 training cars and scores the last 50 (`train.py --val 50`). The test cars are never scored for that choice.

## Results

See the [Car-CFD results page](../benchmarks/car_cfd/results/comparative.md).

## References

1. <span id="ref1"></span>Umetani and Bickel, *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*, ACM Transactions on Graphics 37(4), 2018. <https://doi.org/10.1145/3197517.3201325>

2. <span id="ref2"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>
