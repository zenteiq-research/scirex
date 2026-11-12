---
hide:
  - toc
---

<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Problem Classification

<style>
/* The 10-column table: small text, with the wrapper's scroll bar as the fallback
   on narrow screens. The page width is set site-wide in stylesheets/extra.css. */
.md-typeset table:not([class]) { font-size: 0.7rem; }
.md-typeset table:not([class]) td:nth-child(6) { min-width: 16rem; }
.md-typeset table:not([class]) th,
.md-typeset table:not([class]) td { padding: 0.5em 0.7em; min-width: 5rem; }
/* Narrow screens: keep the columns at a readable width and scroll sideways. */
@media (max-width: 76.1875em) {
  .md-typeset table:not([class]) { min-width: 80rem; }
}
</style>

One row per benchmark problem and supervision type: its domain, geometry and grid, what the model maps from and to, how it is trained and which published result is the target. The benchmark name links to its problem card; the equations and boundary conditions are in the papers listed there.

| Benchmark | Domain | Dimension | Geometry | Grid | Input to output | Supervision | Benchmark/Target | Metric | Models run |
|---|---|---|---|---|---|---|---|---|---|
| [Darcy flow](darcy/README.md) | Porous media | 2D | Regular (unit square) | Structured | permeability field to pressure field | DD | 0.57% [[1]](#ref1) (Transolver) | Relative L2 of the pressure | Transolver, FNO, WNO, GINO |
| [Darcy flow, physics-only](darcy_physics/README.md) | Porous media | 2D | Regular (unit square) | Structured | permeability field to pressure field | Physics loss | 0.57% [[1]](#ref1) (data-driven Transolver; no physics-only reference exists) | Relative L2 of the pressure | Transolver, FNO, WNO, GINO |
| [Navier-Stokes 2D](navier_stokes/README.md) | Fluid dynamics (vorticity transport) | 2D | Regular (periodic unit square) | Structured | vorticity at t = 1 to 10 to vorticity at t = 11 to 20 | DD | 0.0900 [[1]](#ref1) (Transolver) | Relative L2 over the 10-step autoregressive rollout | Transolver, FNO, WNO, GINO |
| [Lid-driven cavity](cavity/README.md) | Fluid dynamics (internal flow) | 2D | Regular (square cavity) | Structured | 3 past frames of pressure and velocity to next frame | DD | 6.43e-3 m/s [[2]](#ref2) (FNO) | One-step RMSE at the held-out Re 10,000 | FNO, WNO, GINO, Transolver |
| [Pipe flow](pipe/README.md) | Fluid dynamics (internal flow) | 2D | Deformed pipe, centreline varies per sample | Structured (curvilinear) | mesh coordinates to horizontal velocity | DD | 0.33% [[1]](#ref1) (Transolver); Geo-FNO 0.67% [[5]](#ref5) | Relative L2 of the horizontal velocity | Transolver, FNO, WNO, GINO |
| [Chip cooling](chip_cooling/README.md) | Thermal (electronics cooling) | 3D domain, 2D output slice | Fixed (pin-fin cold plate with two chips) | Unstructured (11,110 nodes) | 6 boundary-condition parameters to temperature on the 2D slice | DD | 1.534 K [[3]](#ref3) (POD-NN) | Test MAE in K (also R², max AE) | FNO, WNO, GINO, Transolver |
| [AirfRANS](airfrans/README.md) | Aerodynamics (airfoils) | 2D | Varies per sample (NACA airfoils) | Unstructured | node position, signed distance, wall normal, inflow velocity to velocity, pressure and turbulent viscosity at every node | DD | Volume 0.0037, Surf 0.0142, C_L 0.1030, Spearman 0.9978 [[1]](#ref1) | Volume and Surf normalised MSE, C_L relative error, Spearman rank correlation | Transolver, WNO, GINO |
| [ShapeNetCar](shapenetcar/README.md) | Aerodynamics (automotive) | 3D | Varies per sample (car bodies) | Point cloud | point position, signed distance, surface normal to velocity (all points), pressure (surface points) | DD | Velocity 0.0207, pressure 0.0745 [[1]](#ref1) (Transolver) | Relative L2 of velocity and of surface pressure | Transolver, GINO, GIWO |
| [Car-CFD](car_cfd/README.md) | Aerodynamics (automotive) | 3D | Varies per sample (car bodies) | Unstructured (surface mesh, 3,586 vertices) | car surface geometry to surface pressure | DD | 7.12% [[4]](#ref4) (GINO) | Relative L2 of the surface pressure | GINO, Transolver, GIWO |

## Reading the table

- **Geometry** is the shape of the domain: fixed, or changing from sample to sample.
  **Grid** is how the data is discretised: a structured grid, an unstructured mesh or a
  point cloud. FNO and WNO need a regular grid, so on unstructured data they run on a
  resampled copy: chip cooling is interpolated onto a 128 × 256 grid, and GINO reads each
  Car-CFD car as a signed distance function on a 64³ grid. Transolver reads meshes and
  point clouds directly.
- **Supervision**: DD is trained on solution data, Physics loss on the PDE residual or
  energy only, and H on both. No benchmark has an H run yet.
- **Benchmark/Target** is the published number the benchmark is compared with, taken
  from the problem card. The results of the ScireX runs are on the comparison pages.

## Adding a problem

Add one row with no blank cells. A test ([`tests/docs/test_classification_table.py`](../../tests/docs/test_classification_table.py))
checks that every folder under [`scripts/benchmarks/`](../../scripts/benchmarks) has a row, that each row links to
its problem card, and that no cell is empty.

## References

1. <span id="ref1"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>

2. <span id="ref2"></span>Costa Rocha et al., *Deep Neural Network Modeling for CFD Simulations: Benchmarking the Fourier Neural Operator on the Lid-Driven Cavity Case*, Applied Sciences 13(5), 3165 (2023). <https://doi.org/10.3390/app13053165>

3. <span id="ref3"></span>Curl and Hu, *Physically interpretable surrogate modeling of thermal fields in electronics cooling using combined proper orthogonal decomposition and neural networks*, AI Thermal Fluids 6, 100040 (2026). <https://doi.org/10.1016/j.aitf.2026.100040>

4. <span id="ref4"></span>Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023. <https://arxiv.org/abs/2309.00583>

5. <span id="ref5"></span>Li et al., *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries* (Geo-FNO), Journal of Machine Learning Research 24(388), 2023. <https://arxiv.org/abs/2207.05209>
