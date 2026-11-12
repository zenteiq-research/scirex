<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Datasets

Choose a physical problem. Each dataset guide says what the model learns, where the data comes from, how to
get it, how it is split, and where the results are. A [register of all datasets](../benchmarks/datasets.md)
(source, size and where to download) is generated from these guides.

## Regular grids

<div class="grid cards" markdown>

-   **Darcy Flow 2D**

    ---

    *Subsurface flow*

    Steady 2D Darcy flow. The task maps a two-phase diffusion coefficient to the pressure field. 1,000 training and 200 test samples at 85x85.

    [:octicons-arrow-right-24: Dataset Guide](darcy_2d.md)

-   **Navier-Stokes 2D**

    ---

    *Incompressible fluid flow*

    Vorticity of 2D flow on a periodic square at viscosity 1e-5. Given the first 10 snapshots, predict the next 10.

    [:octicons-arrow-right-24: Dataset Guide](navier_stokes.md)

-   **Lid-Driven Cavity**

    ---

    *Turbulent internal flow*

    Transient k-epsilon cavity flow from OpenFOAM at several lid speeds. Forecast the next snapshot and test on a Reynolds number held out of training.

    [:octicons-arrow-right-24: Dataset Guide](cavity.md)

</div>

## Irregular domains and geometry

<div class="grid cards" markdown>

-   **Pipe Flow**

    ---

    *Internal flow, deformed geometry*

    Steady flow through 2D pipes whose centreline is deformed differently in every sample. The task maps the mesh coordinates to the flow field.

    [:octicons-arrow-right-24: Dataset Guide](pipe.md)

-   **Chip Cooling**

    ---

    *Electronics thermal management*

    Six boundary-condition parameters (power of each chip, inlet flow rate and temperature) to the steady 2D temperature field of a liquid-cooled dual-chip cold plate.

    [:octicons-arrow-right-24: Dataset Guide](chip_cooling.md)

-   **AirfRANS**

    ---

    *Two-dimensional airfoil aerodynamics*

    1,000 steady RANS simulations of NACA airfoils at varying inflow speed and angle of attack, each on its own mesh of about 180,000 nodes.

    [:octicons-arrow-right-24: Dataset Guide](airfrans.md)

-   **ShapeNetCar**

    ---

    *Automotive external aerodynamics*

    About 889 car bodies from ShapeNet as surface and near-volume point clouds. The task predicts the velocity field and the surface pressure.

    [:octicons-arrow-right-24: Dataset Guide](shapenetcar.md)

-   **Car-CFD**

    ---

    *Automotive external aerodynamics*

    611 car bodies with one surface mesh each. The task maps the geometry to the steady surface pressure (the GINO paper's version of the car data).

    [:octicons-arrow-right-24: Dataset Guide](car_cfd.md)

</div>

Darcy flow also has a [physics-only benchmark](../benchmarks/darcy_physics/results/setup.md) that reads the same files and trains on the coefficient fields alone.

## How a dataset guide is laid out

Every guide has the same sections, in this order.

| Section | What it tells you |
| --- | --- |
| At a glance | What the model learns, how many samples there are, the grid, the metric, where the data comes from |
| Get the data | Where to download it, which files to expect, where to put them, and the command that trains on it |
| How the benchmark reads the data | What the loader does with the files, in a few plain points. A collapsed block holds the finer technical details where there are any |
| What changes between samples | What varies from one sample to the next, and so what the data does not cover |
| Splits | How the samples are divided into training and test sets, and how normalisation avoids leaking test data |
| Results | A link to the model comparison for this problem |
