<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Problem types

Scientific machine learning problems differ in what is known, what is predicted, and how the domain is
discretised. Use this page to place your problem, choose a model, and find the matching tutorial. The
tutorials in the sidebar are grouped the same way.

## How problems are classified

| Axis | Options | Why it matters |
| --- | --- | --- |
| **Direction** | Forward problem; inverse problem | A forward problem predicts the solution of a PDE from its inputs. An inverse problem infers unknown inputs from observed solutions. |
| **Supervision** | Data-driven; physics-informed | Data-driven training needs solved examples. Physics-informed training minimises the PDE residual or energy and needs no solutions. |
| **Domain and discretisation** | Regular grid; structured curvilinear mesh; unstructured mesh; point cloud | It decides which architecture applies. FFT and wavelet layers need a regular grid. |
| **Geometry** | Fixed; varying between samples | A model that sees many geometries must take the geometry as input. |
| **Time** | Steady; time-dependent | Time-dependent problems are predicted one step at a time and rolled out. |
| **Input type** | Field to field; geometry to field; parameters to field | It sets the shape of the model input. |
| **Dimension** | 2D; 3D | It sets the cost of the model and the amount of data needed. |

## Forward and inverse problems

**Forward problems.** All nine benchmark problems are forward problems. The model learns the solution
operator: given a coefficient field, a geometry or a set of boundary parameters, it predicts the solution
field in one pass. A trained model replaces repeated PDE solves, for design exploration, uncertainty
studies or real-time use.

**Inverse problems.** Inferring unknown coefficients, sources or geometry from observed solutions is not
covered in release 1.0: there are no inverse-problem losses, datasets or tutorials. A model could be trained on
the swapped map, solution to coefficient, with the data losses, but that use is not tested or documented.

## Regular and irregular domains

| Discretisation | Typical data | Models |
| --- | --- | --- |
| Regular grid | Images of a field on a uniform Cartesian grid | FNO, WNO |
| Structured curvilinear mesh | A grid deformed to fit the domain, with mesh coordinates as input | Transolver on its structured option, or FNO and WNO on the mapped grid |
| Unstructured mesh | Nodes and elements that differ between samples | GINO, Transolver |
| Point cloud | Points with features, no connectivity | GINO, Transolver |

## Choosing a model

| Your problem | Start with | Reason |
| --- | --- | --- |
| Regular grid, smooth fields | [FNO](../theory_guide/operator_learning.md#fno) | Global Fourier convolution, the fastest to train. It assumes periodicity unless you pad the domain |
| Regular grid, sharp local features or non-periodic boundaries | [WNO](../theory_guide/operator_learning.md#wno) | Wavelets are local in space and frequency |
| Varying geometry, 2D or 3D | [GINO](../theory_guide/operator_learning.md#gino) | Graph layers carry the geometry onto a regular latent grid |
| Large unstructured meshes | [Transolver](../theory_guide/operator_learning.md#transolver) | Attention between a small number of learned slices keeps the cost low. Set `geometry="irregular"` for point clouds and meshes, or `"structured_2d"` for images of a field |
| No solution data, a known PDE with an energy | A physics-informed loss with any of the above | See [Physics-informed losses](../loss_guide/overview.md) |

## Tutorials by problem type

Each tutorial trains one model on one benchmark problem. All problems are forward problems.

### Regular grids, data-driven

| Problem | Physics | Time | Tutorials |
| --- | --- | --- | --- |
| Darcy flow, 2D | Porous media | Steady | [FNO](../tutorials/FNO/Darcy_FNO.md) · [WNO](../tutorials/Others/Darcy_WNO.md) · [GINO](../tutorials/Others/Darcy_GINO.md) · [Transolver](../tutorials/Others/Darcy_Transolver.md) |
| Navier-Stokes, 2D | Vorticity transport | Time-dependent | [FNO](../tutorials/FNO/NavierStokes_FNO.md) · [WNO](../tutorials/Others/NavierStokes_WNO.md) · [GINO](../tutorials/Others/NavierStokes_GINO.md) · [Transolver](../tutorials/Others/NavierStokes_Transolver.md) |
| Lid-driven cavity, 2D | Turbulent internal flow | Time-dependent | [FNO](../tutorials/Others/Cavity_FNO.md) · [WNO](../tutorials/WNO/Cavity_WNO.md) · [GINO](../tutorials/Others/Cavity_GINO.md) · [Transolver](../tutorials/Others/Cavity_Transolver.md) |

### Irregular domains and varying geometry, data-driven

| Problem | Physics | Geometry | Tutorials |
| --- | --- | --- | --- |
| Pipe flow, 2D | Internal flow | Deformed pipe, structured curvilinear mesh | [FNO](../tutorials/Others/Pipe_FNO.md) · [WNO](../tutorials/Others/Pipe_WNO.md) · [GINO](../tutorials/GINO/Pipe_GINO.md) · [Transolver](../tutorials/Others/Pipe_Transolver.md) |
| Chip cooling, 3D domain | Electronics cooling | Fixed, unstructured mesh | [FNO](../tutorials/Others/ChipCooling_FNO.md) · [WNO](../tutorials/WNO/ChipCooling_WNO.md) · [GINO](../tutorials/Others/ChipCooling_GINO.md) · [Transolver](../tutorials/Others/ChipCooling_Transolver.md) |
| AirfRANS, 2D | Airfoil aerodynamics | Varies per sample, unstructured mesh | [WNO](../tutorials/Others/AirfRANS_WNO.md) · [GINO](../tutorials/Others/AirfRANS_GINO.md) · [Transolver](../tutorials/Transolver/AirfRANS_Transolver.md) |
| ShapeNetCar, 3D | Automotive aerodynamics | Varies per sample, point cloud | [GINO](../tutorials/Others/ShapeNetCar_GINO.md) · [GIWO](../tutorials/Others/ShapeNetCar_GIWO.md) · [Transolver](../tutorials/Transolver/ShapeNetCar_Transolver.md) |
| Car-CFD, 3D | Automotive aerodynamics | Varies per sample, surface mesh | [GINO](../tutorials/GINO/CarCFD_GINO.md) · [GIWO](../tutorials/Others/CarCFD_GIWO.md) · [Transolver](../tutorials/Others/CarCFD_Transolver.md) |

### Physics-informed, no solution data

| Problem | Physics | Loss | Tutorials |
| --- | --- | --- | --- |
| Darcy flow, 2D, physics-only | Porous media | Finite-element energy (Ritz) functional | [FNO](../tutorials/DarcyPhysics/DarcyPhysics_FNO.md) · [WNO](../tutorials/DarcyPhysics/DarcyPhysics_WNO.md) · [GINO](../tutorials/DarcyPhysics/DarcyPhysics_GINO.md) · [Transolver](../tutorials/DarcyPhysics/DarcyPhysics_Transolver.md) |

The [benchmark classification](../benchmarks/classification.md) lists the equations, boundary conditions,
grids and published targets of the same problems in one table.
