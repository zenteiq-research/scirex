<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Benchmarks

Accuracy and compute cost of the neural operators on nine PDE problems. Each model is sized to about the same parameter count and keeps the training recipe of its own paper, and every page gives the command that reproduces its numbers.

## How the pages are organised

The matrices put every problem (a row) and every model (a column) on one page:

| Page | What it shows |
| --- | --- |
| [Accuracy](accuracy.md) | The headline test error of every model on every problem |
| [Compute cost](compute_cost.md) | The number of parameters and the time per epoch |
| [Setup](setup.md) | The number of epochs and the batch size |
| [Results summary](comparison.md) | The headline numbers again, with the figures of every problem |

Each problem has a **problem card** (the data, the models and how they are trained and scored), a **comparison page** that puts its models side by side, and a **results page** that holds the setup, predictions, accuracy and compute cost of all its models, one section per model. The table below links all of them. Each cell of a matrix above links to the section of a model on that page.

The [problem classification](classification.md) lists the physics, geometry and published target of each problem. [Reproduce our benchmarks](reproducibility.md) explains how the numbers are regenerated from saved checkpoints, and the [checkpoint cards](cards/index.md) record which checkpoint produced which number.

## All problems

The problem name opens its problem card, and the line under it, which names the metric, opens the comparison of all its models. A ✓ opens the section of that model on the results page of the problem; a dash means the model was not run. The experimental GIWO is benchmarked separately, under [Experimental](experimental/index.md).

| Problem | FNO | WNO | GINO | Transolver |
| --- | --- | --- | --- | --- |
| **[Darcy Flow](darcy/README.md)**<br>[Relative L2 error](darcy/results/comparative.md) | [✓](darcy/results/comparative.md#fno) | [✓](darcy/results/comparative.md#wno) | [✓](darcy/results/comparative.md#gino) | [✓](darcy/results/comparative.md#transolver) |
| **[Darcy Flow, Physics-Only](darcy_physics/README.md)**<br>[Relative L2 error](darcy_physics/results/comparative.md) | [✓](darcy_physics/results/comparative.md#fno) | [✓](darcy_physics/results/comparative.md#wno) | [✓](darcy_physics/results/comparative.md#gino) | [✓](darcy_physics/results/comparative.md#transolver) |
| **[Pipe Flow](pipe/README.md)**<br>[Relative L2 error](pipe/results/comparative.md) | [✓](pipe/results/comparative.md#fno) | [✓](pipe/results/comparative.md#wno) | [✓](pipe/results/comparative.md#gino) | [✓](pipe/results/comparative.md#transolver) |
| **[Navier-Stokes 2D](navier_stokes/README.md)**<br>[Relative L2 error, 10-step rollout](navier_stokes/results/comparative.md) | [✓](navier_stokes/results/comparative.md#fno) | [✓](navier_stokes/results/comparative.md#wno) | [✓](navier_stokes/results/comparative.md#gino) | [✓](navier_stokes/results/comparative.md#transolver) |
| **[Lid-Driven Cavity](cavity/README.md)**<br>[RMSE, one step ahead](cavity/results/comparative.md) | [✓](cavity/results/comparative.md#fno) | [✓](cavity/results/comparative.md#wno) | [✓](cavity/results/comparative.md#gino) | [✓](cavity/results/comparative.md#transolver) |
| **[Chip Cooling](chip_cooling/README.md)**<br>[Mean absolute error in kelvin (K)](chip_cooling/results/comparative.md) | [✓](chip_cooling/results/comparative.md#fno) | [✓](chip_cooling/results/comparative.md#wno) | [✓](chip_cooling/results/comparative.md#gino) | [✓](chip_cooling/results/comparative.md#transolver) |
| **[ShapeNetCar](shapenetcar/README.md)**<br>[Relative L2 error, drag](shapenetcar/results/comparative.md) | – | – | [✓](shapenetcar/results/comparative.md#gino) | [✓](shapenetcar/results/comparative.md#transolver) |
| **[AirfRANS](airfrans/README.md)**<br>[Volume and surface error, lift](airfrans/results/comparative.md) | – | [✓](airfrans/results/comparative.md#wno) | [✓](airfrans/results/comparative.md#gino) | [✓](airfrans/results/comparative.md#transolver) |
| **[Car-CFD](car_cfd/README.md)**<br>[Relative L2 error of pressure](car_cfd/results/comparative.md) | – | – | [✓](car_cfd/results/comparative.md#gino) | [✓](car_cfd/results/comparative.md#transolver) |

Figures are generated offline by the runners in [`scripts/benchmarks/`](../../scripts/benchmarks) (training, checkpoint, inference), so building the documentation never runs training.
