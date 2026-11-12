<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Experimental models

Models in [`scirex.experimental`](../../api/experimental.md) are not covered by the 1.x stability promise:
their interface can change in any release, and they are not published architectures. They are benchmarked
here, apart from the main [results](../comparison.md), so that the headline comparisons contain only the
four established operators (FNO, WNO, GINO and Transolver).

## GIWO

**GIWO** (geometry-informed wavelet operator) keeps GINO's graph layers, which map an irregular mesh to a
latent grid and back, and replaces the latent FNO with a [WNO](https://arxiv.org/abs/2205.02191). Its
settings are GINO's, with `wno_*` in place of `fno_*`.

```python
from scirex.experimental.giwo import GIWO
```

Its WNO latent operator has real weights, so any optimiser works; the complex-aware [`adamw_cx`](../../api/training.md#scirex.training.adamw_cx) is not needed
(see [Complex-parameter optimisers](../../user_guide/complex_optimisers.md)). The API is documented under
[`scirex.experimental`](../../api/experimental.md#scirex.experimental.giwo.GIWO).

### Results

GIWO was run on the two 3-D car problems, each at about the parameter count of the other models on that
problem. GINO and Transolver are shown for reference. Lower is better.

| Problem | Metric | GIWO | GINO | Transolver |
| --- | --- | ---: | ---: | ---: |
| [ShapeNetCar](shapenetcar_giwo.md) | Relative L2, volume velocity | 6.96% | 7.30% | **2.23%** |
| [Car-CFD](car_cfd_giwo.md) | Relative L2, surface pressure | 12.01% | 8.37% | **6.81%** |

| Problem | Run | Parameters | Seconds per epoch |
| --- | --- | ---: | ---: |
| ShapeNetCar | `giwo_transolver_cost_matched` | 3,752,638 | ~23.5 |
| Car-CFD | `giwo_val` | 74,197,232 | ~11 |

On ShapeNetCar GIWO is closer to Transolver than GINO is, but well behind it. On Car-CFD it is the least
accurate of the three, and the cheapest per epoch.

### Pages

<div class="grid cards" markdown>

-   :material-car-side: **ShapeNetCar**

    ---

    [Results](shapenetcar_giwo.md) ·
    [Tutorial](../../tutorials/Others/ShapeNetCar_GIWO.md) ·
    [Checkpoint](../cards/experimental.md#shapenetcar)

-   :material-car-sports: **Car-CFD**

    ---

    [Results](car_cfd_giwo.md) ·
    [Tutorial](../../tutorials/Others/CarCFD_GIWO.md) ·
    [Checkpoint](../cards/experimental.md#car-cfd)

</div>
