<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Results summary

One table and one set of figures for all nine problems. The page of each problem has a model comparison with the details. The headline metric differs from problem to problem,
so compare models **within a row**, not between rows. Lower is better in every row. The models of a problem
are sized to the same parameter count, and each keeps the training recipe of its own paper, so a difference
reflects the architecture and its recipe together. A dash means the model was not run on that problem. The experimental GIWO is benchmarked separately, under [Experimental](experimental/index.md).

## Headline results

| Problem | Metric | FNO | WNO | GINO | Transolver |
| --- | --- | --- | --- | --- | --- |
| [Darcy Flow](darcy/results/comparative.md) | Relative L2, 200 test samples | 1.109% | 2.020% | 1.603% | **0.532%** |
| [Darcy Flow, Physics-Only](darcy_physics/results/comparative.md) | Relative L2, trained on 1000 coefficient fields only | 1.211% | 2.769% | 1.156% | **1.020%** |
| [Navier-Stokes 2D](navier_stokes/results/comparative.md) | Relative L2, 200 test samples, 10-step rollout | 7.05% | 31.98% | **6.66%** | 8.72% |
| [Lid-Driven Cavity](cavity/results/comparative.md) | RMSE as % of the lid speed (1 m/s), one step ahead, Re 10,000 | 0.197% | 0.204% | **0.195%** | 0.204% |
| [Pipe Flow](pipe/results/comparative.md) | Relative L2, best of 3 seeds | 0.934% | 0.529% | 1.024% | **0.386%** |
| [Chip Cooling](chip_cooling/results/comparative.md) | Test relative L2 (not scored on the same points for every model) | **0.041%** | 0.227% | 0.248% | 0.281% |
| [AirfRANS](airfrans/results/comparative.md) | Volume MSE of the normalised fields, in % | – | 0.446% | 2.537% | **0.131%** |
| [ShapeNetCar](shapenetcar/results/comparative.md) | Relative L2 of velocity (volume) | – | – | 7.30% | **2.23%** |
| [Car-CFD](car_cfd/results/comparative.md) | Relative L2 of surface pressure, 111 test cars | – | – | 8.37% | **6.81%** |

Bold marks the lowest value in the row. In the lid-driven cavity row the four models are within 4% of each other,
and the value is the default size of each model. The setup, the other metrics and the caveats of every row
are on the linked comparison page.

## Figures

Every problem has the same three figures: the predictions of the models on one test sample, the test error of each model, and the training time per epoch of each model. Select an image to open it at full size.
The physics-only Darcy predictions show Transolver, the most accurate physics-only model.


=== "Darcy Flow"

    [![Predictions: Darcy flow](../assets/benchmarks/darcy_comparison/contour.png){ loading=lazy }](../assets/benchmarks/darcy_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Darcy flow](../assets/benchmarks/darcy_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/darcy_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Darcy flow](../assets/benchmarks/darcy_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/darcy_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Darcy flow](darcy/results/comparative.md)

=== "Darcy Flow, Physics-Only"

    [![Predictions: Darcy flow, physics-only](../assets/benchmarks/darcy_physics_comparison/contour.png){ loading=lazy }](../assets/benchmarks/darcy_physics_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Darcy flow, physics-only](../assets/benchmarks/darcy_physics_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/darcy_physics_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Darcy flow, physics-only](../assets/benchmarks/darcy_physics_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/darcy_physics_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Darcy flow, physics-only](darcy_physics/results/comparative.md)

=== "Navier-Stokes 2D"

    [![Predictions: Navier-Stokes 2D](../assets/benchmarks/navier_stokes_comparison/contour.png){ loading=lazy }](../assets/benchmarks/navier_stokes_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Navier-Stokes 2D](../assets/benchmarks/navier_stokes_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/navier_stokes_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Navier-Stokes 2D](../assets/benchmarks/navier_stokes_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/navier_stokes_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Navier-Stokes 2D](navier_stokes/results/comparative.md)

=== "Lid-Driven Cavity"

    [![Predictions: Lid-driven cavity](../assets/benchmarks/cavity_comparison/contour.png){ loading=lazy }](../assets/benchmarks/cavity_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Lid-driven cavity](../assets/benchmarks/cavity_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/cavity_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Lid-driven cavity](../assets/benchmarks/cavity_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/cavity_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Lid-driven cavity](cavity/results/comparative.md)

=== "Pipe Flow"

    [![Predictions: Pipe flow](../assets/benchmarks/pipe_comparison/contour.png){ loading=lazy }](../assets/benchmarks/pipe_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Pipe flow](../assets/benchmarks/pipe_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/pipe_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Pipe flow](../assets/benchmarks/pipe_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/pipe_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Pipe flow](pipe/results/comparative.md)

=== "Chip Cooling"

    [![Predictions: Chip cooling](../assets/benchmarks/chip_cooling_comparison/contour.png){ loading=lazy }](../assets/benchmarks/chip_cooling_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Chip cooling](../assets/benchmarks/chip_cooling_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/chip_cooling_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Chip cooling](../assets/benchmarks/chip_cooling_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/chip_cooling_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of Chip cooling](chip_cooling/results/comparative.md)

=== "AirfRANS"

    [![Predictions: AirfRANS](../assets/benchmarks/airfrans_comparison/contour.png){ loading=lazy }](../assets/benchmarks/airfrans_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: AirfRANS](../assets/benchmarks/airfrans_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/airfrans_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: AirfRANS](../assets/benchmarks/airfrans_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/airfrans_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    [Full comparison of AirfRANS](airfrans/results/comparative.md)

=== "ShapeNetCar"

    [![Predictions: ShapeNetCar](../assets/benchmarks/shapenetcar_comparison/contour.png){ loading=lazy }](../assets/benchmarks/shapenetcar_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: ShapeNetCar](../assets/benchmarks/shapenetcar_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/shapenetcar_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: ShapeNetCar](../assets/benchmarks/shapenetcar_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/shapenetcar_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    These figures also show the experimental GIWO ([Experimental](experimental/index.md)).

    [Full comparison of ShapeNetCar](shapenetcar/results/comparative.md)

=== "Car-CFD"

    [![Predictions: Car-CFD](../assets/benchmarks/car_cfd_comparison/contour.png){ loading=lazy }](../assets/benchmarks/car_cfd_comparison/contour.png "Predictions"){ target=_blank }
    [![Accuracy: Car-CFD](../assets/benchmarks/car_cfd_comparison/summary_accuracy.png){ loading=lazy }](../assets/benchmarks/car_cfd_comparison/summary_accuracy.png "Accuracy"){ target=_blank }
    [![Compute cost: Car-CFD](../assets/benchmarks/car_cfd_comparison/summary_cost.png){ loading=lazy }](../assets/benchmarks/car_cfd_comparison/summary_cost.png "Compute cost"){ target=_blank }
    { .zq-figs }

    These figures also show the experimental GIWO ([Experimental](experimental/index.md)).

    [Full comparison of Car-CFD](car_cfd/results/comparative.md)
