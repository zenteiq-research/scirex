<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Chip Cooling

Predict the steady temperature field on the central plane of a liquid-cooled, dual-chip cold plate from six boundary-condition parameters (the power of each chip, and the mass flow rate and temperature of each inlet). It is a cheap stand-in for a 3-D conjugate CFD run in design work, and is compared with the POD-NN surrogate of [[1]](#ref1).

## At a glance

| | |
| --- | --- |
| Input | Six numbers: power of each chip, mass flow rate and temperature of each inlet |
| Output | The temperature $T$ at the 11,110 nodes of the central plane |
| Data | 1,600 training and 400 test simulations ([dataset guide](../../dataset_guide/chip_cooling.md)) |
| Metric | Mean absolute error of the temperature, in kelvin (lower is better) |
| Best result | **FNO, 0.122 K** (POD-NN: 1.534 K) |

| Model | Test MAE in kelvin (K) |
|---|---|
| FNO | 0.122 |
| GINO | 0.425 |
| WNO | 0.640 |
| Transolver | 0.665 |
| POD-NN (reported by [[1]](#ref1), not re-run) | 1.534 |

All models have about 1.8 M parameters. FNO and WNO work on the plane interpolated to a 128 × 256 grid; GINO and Transolver work on the 11,110 nodes directly. $R^2$ and the maximum error are on the [comparison page](results/comparative.md).

## Training and evaluation

The six parameters and the temperature are standardised with statistics from the training samples. The training loss is the relative L2 error of the standardised temperature. Test metrics are computed on the temperature in kelvin, pooled over the 400 test samples:

$$
\mathrm{MAE} = \overline{\left|\hat T - T\right|}
$$

The test set is the same as in [[1]](#ref1): the last 400 simulations. Their POD-NN was trained on the first 100 simulations only, not on 1,600. Because $T$ is an absolute temperature (about 300 to 523 K), relative errors are small by construction, so MAE in kelvin is the more informative number. A difference of 1 K is the same size as 1 °C.

## Reproduce

You need a GPU and the data ([Get the data](../../dataset_guide/chip_cooling.md#get-the-data)). Run from the repository root:

```bash
export CHIPCOOLING_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/chip_cooling/train.py --run fno
python scripts/benchmarks/chip_cooling/infer.py --out-dir <OUT_DIR>
```

`--run` also accepts `wno`, `gino`, `transolver` or `all`. Training takes about 45 minutes for FNO and about 9 hours for GINO, on a shared GPU.

## References

1. <span id="ref1"></span>Curl and Hu, *Physically interpretable surrogate modeling of thermal fields in electronics cooling using combined proper orthogonal decomposition and neural networks*, AI Thermal Fluids 6, 100040 (2026). <https://doi.org/10.1016/j.aitf.2026.100040>
