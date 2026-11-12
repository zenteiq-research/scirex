<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# ShapeNetCar Dataset

*Automotive external aerodynamics*

[All datasets](index.md){ .md-button } [Results](../benchmarks/shapenetcar/results/comparative.md){ .md-button .md-button--primary }

Steady air flow around about 889 different car bodies from ShapeNet's car category. The model is given the car as a cloud of points on and near its surface, and predicts the air velocity everywhere and the pressure on the car's surface. The CFD was solved and released by [[1]](#ref1) and used as a benchmark by [[2]](#ref2). [Car-CFD](car_cfd.md) uses the same simulations in a different processed copy, and the two benchmarks are not comparable.

## At a glance

| | |
| --- | --- |
| Dataset | Car-Design-ShapeNetCar (Transolver paper's car-surface/volume CFD benchmark) |
| Task | Per point: position, signed distance to the car surface, surface normal (7 channels) → velocity (3 channels) at every point and pressure on the surface points |
| Samples | About 889 cars in 9 folds; fold 0 gives 789 for training and 100 for validation |
| Metric | Relative L2 error of the volume and surface fields, the drag-coefficient error, and the Spearman drag correlation $\rho_D$, on the held-out fold |
| Source | [[1]](#ref1); used as a benchmark by [[2]](#ref2) |
| Used by | [GINO](../benchmarks/shapenetcar/results/comparative.md#gino), [GIWO](../benchmarks/experimental/shapenetcar_giwo.md) (experimental), [Transolver](../benchmarks/shapenetcar/results/comparative.md#transolver) |

The car shapes come from ShapeNet and the CFD data was released by [[1]](#ref1). The scripts do not redistribute the raw files. The licence of the data is stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai).

## Get the data

| | |
| --- | --- |
| Download | `mlcfd_data.zip` of [[1]](#ref1), linked from the Transolver repository [[2]](#ref2). The commands are below the table |
| Files | `param0` to `param8`, each with one folder per car holding `quadpress_smpl.vtk` (surface pressure) and `hexvelo_smpl.vtk` (volume velocity) |
| Size | About 2 GB zipped, about 6 GB of disk during extraction |
| Folder | `--data-dir`, set to `mlcfd_data/training_data` (the tutorials read `SHAPENETCAR_DATA`). Each sample is cached in the folder given by `--cache-dir` |

```bash
cd <DATA_DIR>
wget http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip
unzip mlcfd_data.zip
cd mlcfd_data/training_data
for f in param*.tar.gz; do tar xzf "$f"; done
```

Then train:

```bash
pip install -e ".[benchmark]"
python scripts/benchmarks/shapenetcar/train.py --run transolver --data-dir <DATA_DIR>/mlcfd_data/training_data
```

## How the benchmark reads the data

- The loader is [`load_shapenet_car()`](../api/data.md#scirex.data.load_shapenet_car) in [`src/scirex/data/datasets/shapenet_car.py`](../../src/scirex/data/datasets/shapenet_car.py).
- The first read of a sample is slow (it parses the VTK files, finds the signed distance to the surface and computes the normals), so each sample is cached to `<cache_dir>/<fold>_<sample_id>.npz`. Sample folders with no VTK output are skipped.
- Channels are normalised with the training fold's mean and standard deviation. The batch size is 1, because the number of points differs from car to car.

## What changes between samples

Only the car shape changes: every sample is a different ShapeNet car, with its own mesh and number of points. The inflow, the far-field boundary and the no-slip car surface are fixed. All shapes are passenger cars, and the "ground truth" is itself a CFD solution, not a wind-tunnel measurement.

## Splits

| Split | Cars |
| --- | --- |
| Train | 789 |
| Validation (the reported test) | 100 |

The nine folds `param0` to `param8` are the dataset's own grouping, and `fold_id` (default 0) picks the held-out fold. There is no separate test set. The normalisation statistics come from the training fold only.

## Results

See the [ShapeNetCar results page](../benchmarks/shapenetcar/results/comparative.md).

## References

1. <span id="ref1"></span>Umetani and Bickel, *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*, ACM Transactions on Graphics 37(4), 2018. <https://doi.org/10.1145/3197517.3201325>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
