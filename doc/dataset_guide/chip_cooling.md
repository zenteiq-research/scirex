<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Chip Cooling Dataset

*Electronics thermal management*

[All datasets](index.md){ .md-button } [Results](../benchmarks/chip_cooling/results/comparative.md){ .md-button .md-button--primary }

The temperature inside a liquid-cooled copper cold plate that cools two chips, with two inlets and one outlet. Each case is a steady CFD run, reduced to the temperature on the central plane of the plate. The model takes six boundary-condition numbers and predicts that 2D temperature field. The data is the POD-NN benchmark of [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Dataset | Chip Cooling / dual-chip liquid-cooled pin-fin cold plate CFD dataset (POD-NN benchmark) |
| Task | Six numbers (the power of each chip, the flow rate and temperature of each inlet) → the temperature at the 11,110 mesh nodes of the central plane |
| Samples | 2,000 Latin-hypercube samples (1,600 for training, 400 for testing) |
| Metric | Error of the temperature on the test samples (relative L2, and mean absolute error in kelvin) |
| Source | [[1]](#ref1); [data archive](https://doi.org/10.5061/dryad.k0p2ngfp5) |
| Used by | [FNO](../benchmarks/chip_cooling/results/comparative.md#fno), [WNO](../benchmarks/chip_cooling/results/comparative.md#wno), [GINO](../benchmarks/chip_cooling/results/comparative.md#gino), [Transolver](../benchmarks/chip_cooling/results/comparative.md#transolver) |

The data is the Dryad archive named above. The licence of the data is stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai).

## Get the data

| | |
| --- | --- |
| Download | The [Dryad archive](https://doi.org/10.5061/dryad.k0p2ngfp5) of the dataset of [[1]](#ref1) |
| Files | `model_setup.json` and one `sim_*.npz` per simulation (2,000) |
| Size | About 1.3 GB |
| Folder | `CHIPCOOLING_DATA_DIR` or `--data-dir`. The 128 × 256 grid is cached in the folder given by `--cache-dir` |

Put the files in one folder, written `<DATA_DIR>` here, and train:

```bash
export CHIPCOOLING_DATA_DIR=<DATA_DIR>   # or pass --data-dir <DATA_DIR>
python scripts/benchmarks/chip_cooling/train.py --run all
```

## How the benchmark reads the data

- **FNO and WNO** get the point cloud turned into a 128 × 256 grid. The input has 7 channels: the 6 parameters, repeated over the grid, plus a mask of the domain.
- **GINO and Transolver** use the 11,110 mesh points directly.
- Inputs are standardised per parameter, and the target uses one mean and one standard deviation. All statistics come from the training samples only.

## What changes between samples

Only the six parameters change. They are sampled uniformly over a Latin hypercube:

| Parameter | Unit | Min | Max |
| --- | --- | --- | --- |
| Inlet 1 and 2 mass flow rate | g/s | 5 | 10 |
| Inlet 1 and 2 temperature | K | 300 | 340 |
| Chip 1 and 2 power | W | 1280 | 1600 |

The geometry is fixed. Only the steady state is provided, and extrapolation outside these ranges has not been tested.

## Splits

| Split | Samples |
| --- | --- |
| Train | first 1,600 |
| Test | last 400 |

This is the 80:20 protocol of [[1]](#ref1). The test set is the same for every model and every training-set size. There is no separate validation split in the benchmark scripts.

## Results

See the [Chip Cooling results page](../benchmarks/chip_cooling/results/comparative.md).

## References

1. <span id="ref1"></span>Curl and Hu, *Physically interpretable surrogate modeling of thermal fields in electronics cooling using combined proper orthogonal decomposition and neural networks*, AI Thermal Fluids 6, 100040 (2026). <https://doi.org/10.1016/j.aitf.2026.100040>
