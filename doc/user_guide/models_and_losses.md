<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Models and Losses

This page lists the models and losses in ScireX 1.0. For each model it gives
the settings that matter most, the input layout and the status. For the
losses it explains the choice between a **data loss**, which needs the
solution of every training sample, and the **Ritz energy**, a physics loss
that needs none.

How each model works is explained in the Concepts pages
([FNO](../theory_guide/operator_learning.md#fno), [WNO](../theory_guide/operator_learning.md#wno),
[GINO](../theory_guide/operator_learning.md#gino), [Transolver](../theory_guide/operator_learning.md#transolver));
every argument is in the [API reference](../api/models.md).

## Models

### Status

**Stable** means the class is in [`scirex.models`](../api/models.md), its interface follows the
[API policy](../dev_guide/api_policy.md), and it is validated on the
benchmarks listed. **Experimental** means it lives in [`scirex.experimental`](../api/experimental.md)
and can change in any release.

| Model | Import | Status | Validated on |
|---|---|---|---|
| FNO | `from scirex.models import FNO` | Stable | Darcy, Darcy physics-only, Pipe, Navier-Stokes, Chip cooling |
| FNOClassic2d | `from scirex.models import FNOClassic2d` | Stable | Lid-driven cavity |
| WNO | `from scirex.models import WNO` | Stable | Darcy, Darcy physics-only, Pipe, Navier-Stokes, Lid-driven cavity, Chip cooling, AirfRANS |
| GINO | `from scirex.models import GINO` | Stable | Darcy, Darcy physics-only, Pipe, Navier-Stokes, Lid-driven cavity, Chip cooling, AirfRANS, ShapeNetCar, Car-CFD |
| Transolver | `from scirex.models import Transolver` | Stable | Darcy, Darcy physics-only, Pipe, Navier-Stokes, Lid-driven cavity, Chip cooling, AirfRANS, ShapeNetCar, Car-CFD |
| GIWO | `from scirex.experimental.giwo import GIWO` | Experimental: not a published architecture | ShapeNetCar, Car-CFD |

Each benchmark's results page gives the error, the parameter count and the
training time of every model on it; see the [results summary](../benchmarks/comparison.md).
To choose a model for your data, see [Choosing a model](../learn/problem_types.md#choosing-a-model).

### Key settings

**FNO** ([`scirex.models.FNO`](../api/models.md#scirex.models.FNO)). Input `(batch, in_channels, *spatial)`,
channels-first.

| Setting | Meaning | Darcy run |
|---|---|---|
| `n_modes` | Fourier modes kept per dimension; its length sets the dimension, `(16,)` is 1D, `(16, 16)` is 2D | `(20, 20)` |
| `hidden_channels` | Width of every Fourier layer; the main size knob | 40 |
| `n_layers` | Number of Fourier layers (default 4) | 4 |
| `positional_embedding` | `"grid"` (default) appends the coordinates as extra input channels | `"grid"` |
| `domain_padding` | Pads the domain for non-periodic data | – |

**FNOClassic2d** ([`scirex.models.FNOClassic2d`](../api/models.md#scirex.models.FNOClassic2d)). The original 2D FNO of [[1]](#ref1):
single linear lifting and projection, one skip per block, and separate
weights for the two frequency bands, so about twice the spectral parameters
of [`FNO`](../api/models.md#scirex.models.FNO) at the same mode count. Input `(batch, in_channels, H, W)`. Settings:
`n_modes1`, `n_modes2`, `hidden_channels`, `n_layers` (default 4),
`projection_hidden_channels` (default 128).

**WNO** ([`scirex.models.WNO`](../api/models.md#scirex.models.WNO)). Input `(batch, *spatial, in_channels)`,
**channels-last**.

| Setting | Meaning | Darcy run |
|---|---|---|
| `hidden_channels` | Width of every wavelet layer (required) | 28 |
| `n_layers` | Number of wavelet layers (required) | 4 |
| `level` | Levels of the wavelet decomposition (required) | 4 |
| `size` | Spatial size of the input, an `int` in 1D or a tuple (required) | `(85, 85)` |
| `wavelet` | Wavelet family (default `"db4"`) | `"db6"` |
| `padding` | Spatial padding before the wavelet layers (default 0) | 1 |

WNO needs the `jaxwt` package; `import scirex.models` works without it, and
the error naming it appears when you first use [`WNO`](../api/models.md#scirex.models.WNO).

**GINO** ([`scirex.models.GINO`](../api/models.md#scirex.models.GINO)). A graph layer maps the input points to a
regular latent grid, an FNO runs on the grid, and a second graph layer maps
back to the output query points. Its call takes the input geometry, the
latent grid and the output queries, not one array.

| Setting | Meaning | Default |
|---|---|---|
| `gno_coord_dim` | Dimension of the coordinates, 2 or 3 | 3 |
| `in_gno_radius`, `out_gno_radius` | Neighbourhood radius of the input and output graph layers; scale them with the point spacing | 0.033 |
| `fno_n_modes` | Fourier modes of the latent FNO | `(16, 16, 16)` |
| `fno_hidden_channels` | Width of the latent FNO | 64 |
| `fno_n_layers` | Layers of the latent FNO | 4 |
| `gno_use_open3d` | Neighbour search with Open3D (faster on 3D point clouds) | `True` |

**Transolver** ([`scirex.models.Transolver`](../api/models.md#scirex.models.Transolver)). Attention over a small number of
learned "slices" of the points. Input `(batch, n_points, channels)`; pass the
coordinates as `x` and the input function as `fx`.

| Setting | Meaning | Default | Darcy run |
|---|---|---|---|
| `space_dim`, `fun_dim`, `out_dim` | Coordinate, input and output channels | 1, 1, 1 | – |
| `n_hidden` | Width | 256 | 128 |
| `n_layers` | Transformer blocks | 5 | 8 |
| `n_head` | Attention heads | 8 | 8 |
| `slice_num` | Number of learned slices | 32 | 64 |
| `geometry` | `"irregular"`, `"structured_2d"` (needs `H`, `W`) or `"structured_3d"` (needs `H`, `W`, `D`) | `"irregular"` | `"structured_2d"` |
| `unified_pos`, `ref` | Replace the coordinates by distances to a `ref`-point reference grid | `False`, 8 | – |

**GIWO** ([`scirex.experimental.giwo.GIWO`](../api/experimental.md#scirex.experimental.giwo.GIWO)). GINO's graph layers around a WNO
latent operator instead of an FNO. Settings as GINO, with `wno_*` in place of
`fno_*`. Experimental: the interface is not settled.

The Darcy column is the run in [`scripts/benchmarks/darcy/_config.py`](../../scripts/benchmarks/darcy/_config.py); each
benchmark's `_config.py` has the settings of its own runs.

## Losses

A loss compares the model's output with something. A **data loss** compares
it with the known solution of each training sample. A **physics loss**
compares it with the PDE itself, and needs only the inputs.

| Loss | Import | Kind | Needs solutions |
|---|---|---|---|
| [`LpLoss(d, p)`](../api/losses.md#scirex.losses.LpLoss) | [`scirex.losses`](../api/losses.md) | Relative Lp error | Yes |
| [`H1Loss(d)`](../api/losses.md#scirex.losses.H1Loss) | [`scirex.losses`](../api/losses.md) | Relative H1 error (value and gradient) | Yes |
| Ritz energy ([`Q4Galerkin`](../api/losses.md#scirex.losses.Q4Galerkin) with `BatchedEnergyFunctionalLossQ4`) | [`scirex.losses`](../api/losses.md), `scirex.losses.darcy_piecewise_coefficient` | Energy of the weak form | No |
| [`DarcyEqnLoss`](../api/losses.md#scirex.losses.DarcyEqnLoss), [`FiniteDiff`](../api/losses.md#scirex.losses.FiniteDiff) | [`scirex.losses`](../api/losses.md) | Residual of the strong form on the grid | No |
| [`PDEResidualLoss`](../api/experimental.md#scirex.experimental.losses.PDEResidualLoss), T3/T6 Galerkin, AD residuals, ... | [`scirex.experimental.losses`](../api/experimental.md) | Experimental physics losses | No |

The [Loss Guide](../loss_guide/overview.md) explains the physics losses in
detail.

### Data losses

[`LpLoss(d=2, p=2)`](../api/losses.md#scirex.losses.LpLoss) is the relative L2 error `||pred - u|| / ||u||` of each
sample, the standard training loss and the metric every benchmark reports.
[`H1Loss(d=2)`](../api/losses.md#scirex.losses.H1Loss) adds the error of the gradient, which often helps on smooth
solutions. Both treat the last `d` axes as space and **sum** over the batch by
default; pass `reduction="mean"` for a loss independent of the batch size.

```python
from scirex.losses import LpLoss

l2 = LpLoss(d=2, p=2, reduction="mean")
loss = l2(pred, u)  # pred, u: (batch, channels, H, W)
```

### The Ritz energy

For an elliptic PDE such as Darcy flow,

$$
-\nabla\cdot(a\nabla u) = f \ \text{in } \Omega, \qquad u = 0 \ \text{on } \partial\Omega,
$$

the solution is the function that minimises the energy

$$
J_a(u) = \frac{1}{2}\int_\Omega a\,|\nabla u|^2 \,dx - \int_\Omega f\,u \,dx .
$$

Training on `J` as the loss teaches the model to output, for every
coefficient field `a`, the function of lowest energy, that is the solution,
**without ever seeing a solution**. `J` is evaluated with bilinear (Q4) finite
elements on the grid, exactly and differentiably:

```python
import jax
import jax.numpy as jnp

from scirex.losses import Q4Galerkin
from scirex.losses.darcy_piecewise_coefficient import (  # helper module, not guaranteed: not stable API
    BatchedEnergyFunctionalLossQ4,
    build_grid_aligned_q4_mesh,
    kappa_fn,
    source_fn,
)

xy, quad, boundary = build_grid_aligned_q4_mesh(16)  # 17 x 17 nodes, 16 x 16 cells
galerkin = Q4Galerkin(jnp.asarray(xy), jnp.asarray(quad))
ritz = BatchedEnergyFunctionalLossQ4(galerkin, kappa_fn, source_fn)

a = jnp.where(jax.random.normal(jax.random.PRNGKey(0), (4, 16, 16)) > 0, 12.0, 3.0)
u = jnp.zeros((4, xy.shape[0]))  # the model's output at the nodes, 0 on the boundary
energy = ritz(u, a)  # mean of J over the batch; differentiable in u
```

!!! note "Not part of the stable API"
    `scirex.losses.darcy_piecewise_coefficient` holds the Darcy helpers of the physics-only benchmark. It is
    not in the [API reference](../api/index.md) and can change in any release; pin the library version if you use it.

`a` holds one value per cell and `u` one value per node. The boundary
condition is not part of `J`: the model's output must already be zero on the
boundary, for example by multiplying it with `sin(pi x) sin(pi y)`. The
complete setup used by the benchmark, with the output scaling and the map
from nodes to the data grid, is in [`scripts/benchmarks/darcy_physics/_ritz.py`](../../scripts/benchmarks/darcy_physics/_ritz.py)
and explained on the [Darcy physics-only](../benchmarks/darcy_physics/README.md)
page.

### Data loss or Ritz energy?

| | Data loss | Ritz energy |
|---|---|---|
| Training data | Inputs **and** their solutions | Inputs only |
| PDE | Any; it need not be known | Must have an energy: symmetric, elliptic, such as Darcy, Poisson, linear elasticity |
| Boundary conditions | Learned from the data | Imposed exactly by the output |
| Loss value | An error, 0 is perfect | An energy, negative, decreasing to the minimum; not an error |
| Steps to converge | Fewer | More |
| Limit of accuracy | Accuracy of the solver that made the data | Accuracy of the finite-element mesh |

On the Darcy benchmark, with the same 1000 coefficient fields and the same
parameter-matched models
([comparison](../benchmarks/darcy_physics/results/comparative.md)):

| Model | Data loss, 1000 solutions | Ritz energy, 1000 fields | Ritz energy, more fields (no solutions) |
|---|---|---|---|
| Transolver | 0.532% | 1.020% | 0.565% (10k fields) |
| GINO | 1.603% | 1.156% | 0.722% (20k fields) |
| FNO | 1.109% | 1.211% | 0.648% (10k fields) |
| WNO | 2.020% | 2.769% | 1.262% (20k fields) |

- With the same 1000 fields, the data loss is better for three of the four
  models; GINO is better with the Ritz energy.
- The Ritz energy needs no solutions, so more coefficient fields cost only
  their sampling. With 10k fields, Transolver reaches 0.565%, comparable to
  its data-driven result with 1000 solutions.
- An exact Q4 solve scored against the released solutions has 0.31% error:
  no Ritz-trained model at this resolution can be more accurate.

Choose the data loss when you have solutions, or when the PDE has no energy.
Choose the Ritz energy when solutions are expensive or unavailable and the PDE
has one.

!!! note "The energy is small; scale it for FNO and WNO"
    At 85 × 85 the Darcy energy is about -0.003. At initialisation 98% (FNO)
    and 95% (WNO) of its gradient components are below AdamW's `eps` of
    `1e-8`, so the optimiser barely moves. The benchmark multiplies the loss
    by the number of interior nodes (6889) for FNO and WNO; GINO trains
    without a scale. Check the gradient size of your own model at
    initialisation before training. The step budgets also differ: FNO
    converges in 100k steps, WNO is not converged at 200k.

## See also

- [Workflow](workflow.md): where the model and the loss enter a run.
- [Loss Guide](../loss_guide/overview.md): the physics losses in detail.
- [Benchmarks](../benchmarks/index.md): every model on every problem.

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>
