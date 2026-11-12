<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Tutorials

The tutorials are grouped by model. Each of the four models is taught on two
benchmarks, chosen from the problems it was designed for. The other tutorials
run each model on the remaining benchmarks it was run on.

| Model | Tutorials | Why these problems |
|---|---|---|
| [FNO](../theory_guide/operator_learning.md#fno) | [Darcy flow](FNO/Darcy_FNO.md), [Navier-Stokes 2D](FNO/NavierStokes_FNO.md) | The benchmarks of the original FNO paper: fields on a regular grid |
| [WNO](../theory_guide/operator_learning.md#wno) | [Lid-driven cavity](WNO/Cavity_WNO.md), [Chip cooling](WNO/ChipCooling_WNO.md) | Non-periodic fields with local features, where wavelets help |
| [GINO](../theory_guide/operator_learning.md#gino) | [Car-CFD](GINO/CarCFD_GINO.md), [Pipe flow](GINO/Pipe_GINO.md) | Irregular geometry: the car surface of the GINO paper, and a deformed mesh |
| [Transolver](../theory_guide/operator_learning.md#transolver) | [ShapeNetCar](Transolver/ShapeNetCar_Transolver.md), [AirfRANS](Transolver/AirfRANS_Transolver.md) | Large unstructured meshes, two benchmarks of the Transolver paper |

Every tutorial has the same structure: the problem, the data, a step-by-step
walkthrough of a complete script (data, model, loss, optimiser, training,
evaluation, plots), how to run it, and the results. Each page has a script in
[`scripts/tutorials/`](../../scripts/tutorials), named `<problem>_<model>.py`, that runs the same code:

```bash
python scripts/tutorials/darcy_fno.py --data-dir <DATA_DIR> --smoke   # a tiny model, on a CPU
python scripts/tutorials/darcy_fno.py --data-dir <DATA_DIR>           # the benchmark's model, on a GPU
```

## How the tutorials work

Each step of a walkthrough is a cell of the script, and the code on the page is copied from it. Every page for a problem has the same sections. The first tutorial of a problem (FNO, else WNO) holds the text that all models share: the setup, the problem and the metric, the common problems and the shared references. The tutorials for the other models link to it and add only what is specific to their model.

`print_environment` records what is needed to repeat a run: the library versions, the devices JAX sees and the seed. Check the `devices:` line first, because JAX falls back to the CPU without a warning when it cannot find a GPU.

## Darcy flow, physics-only

The four [physics-only Darcy tutorials](DarcyPhysics/DarcyPhysics_FNO.md)
train each model on the Darcy problem **without labelled solutions**, with
the Ritz energy as the loss (see
[Models and Losses](../user_guide/models_and_losses.md#the-ritz-energy)).

## Others

The other tutorials, each model on the remaining benchmarks it was run on, are
in `doc/tutorials/Others/` and listed under each problem in the menu:

| Model | Other tutorials |
|---|---|
| FNO | Pipe flow, lid-driven cavity, chip cooling |
| WNO | Darcy flow, pipe flow, Navier-Stokes 2D, AirfRANS |
| GINO | Darcy flow, Navier-Stokes 2D, lid-driven cavity, chip cooling, AirfRANS, ShapeNetCar |
| Transolver | Darcy flow, pipe flow, Navier-Stokes 2D, lid-driven cavity, chip cooling, Car-CFD |

The tutorials of the experimental GIWO are under [Benchmarks › Experimental](../benchmarks/experimental/index.md).

## Your own problem

[Add a New Example](NewExample/Overview.md) shows how to add a model, a
problem and its tutorial, with a worked example on the 1D Poisson equation.
