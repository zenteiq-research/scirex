<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# AirfRANS Dataset

*Two-dimensional airfoil aerodynamics*

[All datasets](index.md){ .md-button } [Results](../benchmarks/airfrans/results/comparative.md){ .md-button .md-button--primary }

1,000 steady simulations (RANS, k-omega SST, OpenFOAM) of air flowing around NACA 4- and 5-digit airfoils, each on its own unstructured mesh of about 180,000 nodes. The model is given the airfoil shape and the inflow, and predicts the velocity, the pressure and the turbulent viscosity at every node. The lift is then computed from these fields. The data is the benchmark of [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | AirfRANS (Airfoil RANS) |
| Task | Per mesh node: position, inflow, distance to the airfoil, wall normal → $(U_x, U_y, p, \nu_t)$ at every node |
| Samples | 1,000 simulations (489 NACA 4-digit, 511 NACA 5-digit); the `full` task trains on 800 and tests on 200 |
| Metric | The official scores on the 200 test cases: Volume and Surf (normalised MSE away from and on the wall), $C_L$ relative error, and the Spearman correlation $\rho_L$ of the lift |
| Source | [[1]](#ref1); used as a benchmark by [[2]](#ref2) |
| Used by | [WNO](../benchmarks/airfrans/results/comparative.md#wno), [GINO](../benchmarks/airfrans/results/comparative.md#gino), [Transolver](../benchmarks/airfrans/results/comparative.md#transolver) |

The data is distributed by its authors. The licence of the data is stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai). Results you share publicly may need to carry the notice that licence asks for.

## Get the data

| | |
| --- | --- |
| Download | With the [`airfrans` package](https://airfrans.readthedocs.io): `airfrans.dataset.download(root="<DATA_DIR>", unzip=True)` unpacks the data into `<DATA_DIR>/Dataset` |
| Files | `manifest.json`, and one folder per simulation with `<case>_internal.vtu` (volume mesh and fields) and `<case>_aerofoil.vtp` (airfoil surface and normals) |
| Size | About 15 GB unpacked |
| Folder | `AIRFRANS_DATA_DIR` or `--data-dir`, set to the `Dataset` folder. Reading the VTK files needs the `benchmark` extra |

Train:

```bash
pip install -e ".[benchmark]"
export AIRFRANS_DATA_DIR=<DATA_DIR>/Dataset   # or pass --data-dir <DATA_DIR>/Dataset
python scripts/benchmarks/airfrans/train.py --run transolver   # or wno
```

## How the benchmark reads the data

- The loader is [`scripts/benchmarks/airfrans/_data.py`](../../scripts/benchmarks/airfrans/_data.py). It reads every node of the volume mesh and caches the features under `AIRFRANS_CACHE_DIR`.
- Each training step uses a random subset of 32,000 nodes of one case. At test time the whole case is covered with such subsets.
- Wall nodes are the nodes where $U_x = 0$.
- Transolver and WNO add $\log_{10}$ of the wall distance and $\log_{10}$ of the Reynolds number as features (9 in total), and scale the targets by the inflow before normalising.

## What changes between cases

The shape and the operating point change independently in each case:

| | Range |
| --- | --- |
| Inflow speed $U_\infty$ | 31.3 to 93.6 m/s, so Re $\approx$ 2.0 to 6.0 $\times 10^6$ (chord 1) |
| Angle of attack $\alpha$ | $-4.9^\circ$ to $14.9^\circ$ |
| Airfoil | NACA 4-digit (3 parameters) and 5-digit (4 parameters) |

The flow is 2D and steady, with one turbulence model (SST) and no transition.

## Splits

`manifest.json` defines four official tasks:

| Task | Train | Test |
| --- | --- | --- |
| `full` | 800 | 200 |
| `scarce` | 200 | the `full` test set |
| `reynolds` | 504 | 496 |
| `aoa` | 804 | 196 |

The benchmarks use `full` (unseen airfoils and operating points), as [[2]](#ref2) does. Transolver and WNO train on the first 720 training cases, log a validation loss on the last 80, and test on the 200 test cases. The normalisation statistics come from the training cases only.

## Results

See the [AirfRANS results page](../benchmarks/airfrans/results/comparative.md). The scores are defined in [The metric](../benchmarks/airfrans/results/comparative.md#the-metric).

## References

1. <span id="ref1"></span>Bonnet et al., *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions*, NeurIPS 2022, Datasets and Benchmarks. <https://arxiv.org/abs/2212.07564>

2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
