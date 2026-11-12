<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Operator Learning
A neural operator learns a map from one function to another, for example from the permeability of a material to the pressure in it. Once trained it predicts a new case in a single forward pass, instead of solving the PDE again.
This page gives the common idea and compares the four models in the library (FNO, WNO, GINO, Transolver). The full derivations are in the original papers, linked in the table.

## The problem

A classical neural network maps vectors to vectors,
$f: \mathbb{R}^n \to \mathbb{R}^m$. Many problems in scientific computing map
*functions* to functions: the coefficient field of a PDE to its solution, an
initial condition to the state at a later time, a geometry to the pressure on
its surface. **Operator learning** approximates such an operator

$$
\mathcal{G}: \mathcal{A} \to \mathcal{U}, \qquad a \mapsto u,
$$

where $\mathcal{A}$ and $\mathcal{U}$ are spaces of functions on a domain
$D \subset \mathbb{R}^d$. A model $\mathcal{G}_\theta \approx \mathcal{G}$ is
trained on pairs $\{(a_i, u_i)\}$ produced by a solver or by experiment.

Why learn an operator rather than a network on a fixed grid:

- **Many queries.** After training, a new input costs one forward pass, often
  orders of magnitude less than a classical solve. This pays off when the same
  PDE is solved many times: design studies, uncertainty quantification,
  control.
- **Discretisation.** The model acts on functions, so the same weights can be
  evaluated on another grid or mesh (how far this holds depends on the model;
  see the table below).
- **Geometry.** Some operators (GINO, Transolver) work directly on irregular
  meshes and point clouds.

## The neural operator layer

A neural operator lifts the input to a hidden width with a pointwise map
$\mathcal{P}$, applies $L$ layers, and projects back with a pointwise map
$\mathcal{Q}$:

$$
\mathcal{G}_\theta = \mathcal{Q} \circ \mathcal{L}_L \circ \cdots \circ \mathcal{L}_1 \circ \mathcal{P}.
$$

Each layer adds a pointwise linear term $W$ and a non-local **kernel integral
operator** $\mathcal{K}$, followed by a nonlinearity $\sigma$:

$$
v_{t+1}(x) = \sigma\Big( W v_t(x) + \big(\mathcal{K} v_t\big)(x) \Big), \qquad
\big(\mathcal{K} v_t\big)(x) = \int_D \kappa_\theta(x, y)\, v_t(y)\, \mathrm{d}y .
$$

The integral lets every point see the whole domain. Computed directly it
costs $O(N^2)$ for $N$ points, so each architecture is a cheaper way to
parameterise and evaluate $\mathcal{K}$ [[1]](#ref1). The four models in the
library follow their papers, cited in the first column, where the derivations are.

## The four models { #models }

| Model | How $\mathcal{K}$ is computed | Input | Cost per layer |
|---|---|---|---|
| <span id="fno"></span>**FNO** [[2]](#ref2) | Convolution, as a product with learned weights on the lowest Fourier modes | Regular grid | $O(N \log N)$ |
| <span id="wno"></span>**WNO** [[3]](#ref3) | Convolution, as a product on the coarsest wavelet coefficients | Regular grid | $O(N)$ |
| <span id="gino"></span>**GINO** [[4]](#ref4) | Graph integral onto a latent grid, an FNO there, graph integral back | Any mesh or point cloud | $O(N k)$ plus a latent FNO |
| <span id="transolver"></span>**Transolver** [[5]](#ref5) | Attention between $G$ learned slices of the points (Physics Attention) | Any mesh or point cloud | $O(N G)$ |

$N$ is the number of points, $k$ the number of neighbours within the graph
radius, $G \ll N$ the number of slices.

## Choosing a model

| Model | Use it for | Limits | Key settings | Reference code | Tutorials |
|---|---|---|---|---|---|
| FNO | Smooth fields on a regular grid; fast, the first model to try | Needs a regular grid; the FFT assumes periodic data | `n_modes`, `hidden_channels`, `n_layers` | [neuraloperator](https://github.com/neuraloperator/neuraloperator) | [Darcy](../tutorials/FNO/Darcy_FNO.md), [Navier-Stokes](../tutorials/FNO/NavierStokes_FNO.md) |
| WNO | Sharp fronts, local features, non-periodic boundaries | Weights fixed by the training resolution; grid padded to a multiple of $2^{\text{level}}$ | `wavelet`, `level`, `hidden_channels`, `n_layers` | [TapasTripura/WNO](https://github.com/TapasTripura/WNO) | [Cavity](../tutorials/WNO/Cavity_WNO.md), [Chip cooling](../tutorials/WNO/ChipCooling_WNO.md) |
| GINO | Geometry that changes from sample to sample | Neighbour search costs memory; GPU graph sums are not bit-reproducible ([why](../user_guide/checkpoints_and_reproducibility.md#determinism-ginos-scatter-add)) | Latent grid, `gno_radius`, `fno_n_modes`, `fno_hidden_channels` | [neuraloperator](https://github.com/neuraloperator/neuraloperator) | [Car-CFD](../tutorials/GINO/CarCFD_GINO.md), [Pipe](../tutorials/GINO/Pipe_GINO.md) |
| Transolver | Large unstructured meshes | Needs more data and tuning than FNO on grids | `slice_num`, `n_hidden`, `n_head`, `n_layers`, `geometry` | [thuml/Transolver](https://github.com/thuml/Transolver) | [ShapeNetCar](../tutorials/Transolver/ShapeNetCar_Transolver.md), [AirfRANS](../tutorials/Transolver/AirfRANS_Transolver.md) |

GIWO, in [`scirex.experimental`](../api/experimental.md), is GINO with the latent FNO replaced by a WNO; see the
[Car-CFD GIWO tutorial](../tutorials/Others/CarCFD_GIWO.md).

## Using the models

The input layout, the constructor settings and the status of each model are
in [Models and Losses](../user_guide/models_and_losses.md); the constructor
arguments are in the [API reference](../api/models.md). The
[tutorials](../tutorials/index.md) train each model end to end.

## References

1. <span id="ref1"></span>N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar.
   *Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs.*
   JMLR 24 (2023). [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)

2. <span id="ref2"></span>Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar.
   *Fourier Neural Operator for Parametric Partial Differential Equations.*
   ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

3. <span id="ref3"></span>T. Tripura, S. Chakraborty.
   *Wavelet Neural Operator for Solving Parametric Partial Differential Equations in Computational Mechanics Problems.*
   CMAME 404 (2023). [arXiv:2205.02191](https://arxiv.org/abs/2205.02191)

4. <span id="ref4"></span>Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, A. Anandkumar.
   *Geometry-Informed Neural Operator for Large-Scale 3D PDEs.*
   NeurIPS 2023. [arXiv:2309.00583](https://arxiv.org/abs/2309.00583)

5. <span id="ref5"></span>H. Wu, H. Luo, H. Wang, J. Wang, M. Long.
   *Transolver: A Fast Transformer Solver for PDEs on General Geometries.*
   ICML 2024. [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
