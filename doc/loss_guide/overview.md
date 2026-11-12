<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Physics-Informed Losses: the Ritz Energy

!!! abstract "In short"

    A data loss compares the prediction with a known solution. A physics-informed loss trains from the
    equation itself, so it needs no solutions. ScireX trains from the equation with one loss: the
    **Ritz energy**, evaluated with bilinear (Q4) finite elements. It is the loss of the
    [Darcy Flow, Physics-Only](../benchmarks/darcy_physics/README.md) benchmark, where every model is
    trained with it.

## The idea

For an elliptic equation such as Darcy flow,

$$
-\nabla\cdot(a\nabla u) = f \ \text{in } \Omega, \qquad u = 0 \ \text{on } \partial\Omega,
$$

the solution is the function that minimises the energy

$$
J_a(u) = \frac{1}{2}\int_\Omega a\,|\nabla u|^2 \,dx - \int_\Omega f\,u \,dx .
$$

This is the Ritz (variational) form of the same equation, not extra physics. Training a model on $J_a$ as
the loss teaches it to output, for every coefficient field $a$, the function of lowest energy, which is the
solution. The model never sees a solution.

## Why the energy, not a residual

- **It is well scaled.** $J_a(0) = 0$, while at the true solution
  $J_a(u) = -\tfrac{1}{2}\int_\Omega a\,|\nabla u|^2\,dx$, a clear gap. A squared weak-form residual, by
  contrast, takes almost the same tiny value at $u = 0$ and at the true solution, so it gives the optimiser
  little signal.
- **It has one minimum.** For $a$ bounded away from zero, $J_a$ is strictly convex, so there is a single
  minimiser, and $u = 0$ is not a stationary point.
- **It needs only first derivatives.** $J_a$ contains $\nabla u$ but no second derivatives.

## How ScireX evaluates it

The model's output is read as the nodal values of a field on a grid-aligned mesh of bilinear quadrilaterals
(Q4), one element per grid cell. [`Q4Galerkin`](../api/losses.md#scirex.losses.Q4Galerkin) integrates $J_a$ over the elements with the 2 × 2
Gauss-Legendre rule, exactly and differentiably, so the loss can be minimised with any `optax` optimiser.

- $a$ holds one value per **cell**. It is constant inside each element, so material interfaces sit on
  element edges.
- $u$ holds one value per **node**.
- The boundary condition is not part of $J_a$. The model's output must already be zero on the boundary;
  the benchmark multiplies it by $\sin(\pi x)\sin(\pi y)$, so the condition holds exactly.

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
    [`Q4Galerkin`](../api/losses.md#scirex.losses.Q4Galerkin) is stable. `scirex.losses.darcy_piecewise_coefficient` holds the Darcy helpers of the
    physics-only benchmark; it is not in the [API reference](../api/index.md) and can change in any release.
    Pin the library version if you use it.

The value of the loss is an energy, not an error: it is negative and decreases towards the minimum. To judge
accuracy, track the error against reference solutions where you have them.

## Results

On the [Darcy Flow, Physics-Only](../benchmarks/darcy_physics/README.md) benchmark, each model is trained
on the Ritz energy alone and scored against the released solutions:

| Model | Coefficient fields (no solutions) | Test error |
| --- | --- | --- |
| Transolver | 10,000 | 0.565% |
| FNO | 10,000 | 0.648% |
| GINO | 20,000 | 0.722% |
| WNO | 20,000 | 1.262% |

Because no solutions are needed, more coefficient fields cost only their sampling. For comparison, the
data-driven Transolver trained on 1,000 solutions reaches 0.532%. The full comparison is on the
[comparison page](../benchmarks/darcy_physics/results/comparative.md).

## Limits

- **The equation must have an energy:** symmetric and elliptic, such as Darcy, Poisson or linear
  elasticity.
- **The mesh sets the accuracy floor.** An exact Q4 solve on the 85 × 85 benchmark grid, scored against
  the released solutions, has 0.31% error; a perfectly trained model cannot do better.
- **More steps.** Training on the energy takes more steps to converge than training on solutions.

## Where things live

```text
src/scirex/losses/
├── galerkin_q4.py                    # Q4Galerkin: element gradients and the energy integral
└── darcy_piecewise_coefficient.py    # Darcy helpers: mesh, coefficient, source, the batched Ritz loss

scripts/benchmarks/darcy_physics/
├── _ritz.py                          # benchmark set-up: boundary factor, output scale, node-to-data map
└── train_{fno,wno,gino,transolver}.py
```

To choose between a data loss and the Ritz energy, see
[Data loss or Ritz energy?](../user_guide/models_and_losses.md#data-loss-or-ritz-energy)
