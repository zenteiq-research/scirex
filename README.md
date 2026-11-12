<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

<div align="center">

<img src="doc/assets/brand/scirex-logo.png" alt="ScireX" width="200">

# ScireX

**Neural operators for scientific machine learning, written in JAX and Flax.**

[![License](https://img.shields.io/badge/License-LGPL_v3.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10_to_3.12-3776AB.svg)](pyproject.toml)
[![JAX](https://img.shields.io/badge/JAX-0.6.x-orange.svg)](https://github.com/jax-ml/jax)
[![Flax](https://img.shields.io/badge/Flax-0.10.x-green.svg)](https://github.com/google/flax)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](doc/index.md)

</div>

| **Documentation** |
|---|
| [Install](doc/install/index.md) &nbsp;\|&nbsp; [API reference](doc/api/index.md) &nbsp;\|&nbsp; [Theory](doc/theory_guide/operator_learning.md) &nbsp;\|&nbsp; [Benchmarks](doc/benchmarks/index.md) &nbsp;\|&nbsp; [Gallery](gallery/README.md) |

ScireX is a [JAX](https://github.com/jax-ml/jax)-native library of neural operators for
scientific machine learning. Neural operators learn mappings between function spaces, not
between fixed-size vectors, so an operator trained on one mesh or grid can be evaluated on
another without retraining. The library is for PDE solving, surrogate modelling and
multi-scale physical systems, and it uses JAX's XLA compilation, automatic
differentiation and accelerator support.

It provides four architectures behind one model interface:
[FNO](doc/theory_guide/operator_learning.md#fno) (Fourier Neural Operator),
[WNO](doc/theory_guide/operator_learning.md#wno) (Wavelet Neural Operator),
[GINO](doc/theory_guide/operator_learning.md#gino) (Geometry-Informed Neural Operator) and
[Transolver](doc/theory_guide/operator_learning.md#transolver) (Transformer solver with
Physics-Attention), plus the experimental GIWO. Every model works with `jit`, `vmap` and `grad`.
Nine benchmark problems compare the models at a matched parameter budget with published
results, and physics-informed losses let you train without labelled solutions. ScireX
builds on [SciREX](https://github.com/zenteiq-sciml/SciREX), the earlier research framework from the AiREX Lab at IISc (a separate project).

## Install

```bash
pip install scirex                # CPU
pip install "scirex[cuda12]"      # NVIDIA GPU, CUDA 12
```

From source:

```bash
git clone https://github.com/zenteiq-research/scirex
cd scirex
python3 -m venv scirex-venv && source scirex-venv/bin/activate
pip install -e .                  # CPU
pip install -e ".[cuda12]"        # NVIDIA GPU, CUDA 12
```

#### *Try your first ScireX program*

```python
import jax
import jax.numpy as jnp
from scirex.models import FNO

model = FNO(n_modes=(16, 16), hidden_channels=64, n_layers=4, in_channels=1, out_channels=1)

x = jnp.ones((4, 1, 64, 64))            # batch of 4, 1 channel, 64 x 64 grid
params = model.init(jax.random.PRNGKey(0), x)
out = model.apply(params, x)

print(out.shape)                        # (4, 1, 64, 64)
```

For more examples see the [tutorials](doc/tutorials) and the [user guide](doc/user_guide).

### Supported platforms

| Platform | Support |
|---|---|
| Ubuntu 22.04 and 24.04 LTS, x86_64, CPU | Supported, tested in CI |
| Ubuntu 22.04 and 24.04 LTS, x86_64, NVIDIA GPU (CUDA 12) | Supported |
| macOS on Apple Silicon (macOS 14 or later), CPU | Supported, tested in CI |
| Native Windows 11, Python 3.10 to 3.12, CPU | Supported; see the Windows tab of [Install with pip](doc/install/index.md#install-with-pip) |
| GPU on native Windows | Not supported (JAX has no CUDA build for native Windows; for an NVIDIA GPU on Windows use Ubuntu under WSL2) |
| GPU on macOS (Apple Silicon) | Not supported (JAX has no GPU backend for it; scirex runs on the CPU) |
| Ubuntu 20.04 and older | Not supported (its standard support ended in 2025) |
| CUDA 13 | Not supported in 1.0 |

See [Install](doc/install/index.md) for the details and the check script, and
[Use Docker](doc/install/docker.md) for the container images.

### Optional extras and GPU settings

`wandb` (Weights & Biases logging) and `open3d` (fast radius search, `MeshDataModule`) are not
installed by default.

```bash
pip install -e ".[wandb]"       # logging only
pip install -e ".[open3d]"      # point-cloud neighbour search
pip install -e ".[benchmark]"   # wandb, open3d, vtk, meshio, pandas
pip install -e ".[dev]"         # development tools
pre-commit install
```

`requirements.txt` holds the exact versions the project is tested with. Use it for a
reproducible environment: `pip install -r requirements.txt`.

For GPU work, these variables help control JAX memory allocation and device selection:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0
```

## Benchmarks

Each problem is trained with up to four models (FNO, WNO, GINO and Transolver; GIWO, which is
experimental, on the car problems) at a matched parameter budget and compared with the numbers in the reference papers. The table
shows the best ScireX model for each problem. Click a problem for the full model
comparison, the setup, the error curves and the commands to reproduce it.

| Problem | Metric | Best ScireX model | Published reference |
|---|---|---|---|
| [Darcy flow 2D](doc/benchmarks/darcy/results/comparative.md), data-driven | Test relative L2 | Transolver, **0.532%** | Transolver 0.57% [[5]](#ref-5) |
| [Darcy flow 2D](doc/benchmarks/darcy_physics/results/comparative.md), physics-only (no solutions) | Test relative L2 | Transolver, **0.565%** | Transolver 0.57%, with 1000 labels [[5]](#ref-5) |
| [Pipe flow](doc/benchmarks/pipe/results/comparative.md) | Test relative L2 | Transolver, **0.386%** | Transolver 0.33% [[5]](#ref-5) |
| [Navier-Stokes 2D](doc/benchmarks/navier_stokes/results/comparative.md) | Rollout relative L2 | GINO, **6.66%** | Transolver 9.00% [[5]](#ref-5) |
| [Lid-driven cavity](doc/benchmarks/cavity/results/comparative.md) | One-step RMSE (m/s) | GINO, **1.95e-3** | FNO 6.43e-3 [[7]](#ref-7) |
| [Chip cooling](doc/benchmarks/chip_cooling/results/comparative.md) | Mean absolute error in kelvin (K) | FNO, **0.122** | POD-NN 1.534 [[8]](#ref-8) |
| [AirfRANS](doc/benchmarks/airfrans/results/comparative.md) | Volume / surface error | Transolver, **0.131%** / **0.046%** | Transolver 0.37% / 1.42% [[5]](#ref-5) |
| [ShapeNetCar](doc/benchmarks/shapenetcar/results/comparative.md) | Volume / surface relative L2 | Transolver, **2.23%** / **7.67%** | Transolver 2.07% / 7.45% [[5]](#ref-5) |
| [Car-CFD](doc/benchmarks/car_cfd/results/comparative.md) | Test relative L2 | Transolver, **6.81%** | GINO 7.12% [[3]](#ref-3) |

Lower is better in every row. Where a problem was trained with several seeds, the table gives the
best seed (Darcy flow: 0.532%, mean 0.556% over five seeds); the problem page lists every seed and
the mean. Where a published number is better than ours (Pipe flow and
ShapeNetCar), the table shows it as it is. The Pipe flow Transolver uses a
[recipe tuned on the test set](doc/benchmarks/pipe/results/comparative.md#tuned-recipe); with the
paper's recipe it reaches 0.424%. The physics-only Darcy model is trained without a single
solution: its loss is the finite-element energy of the PDE.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="animations/darcy_physics/evolution.gif" alt="Darcy flow, Transolver trained without labels, across the test set" width="100%"><br>
      <sub>Darcy flow, physics-only Transolver</sub>
    </td>
    <td align="center" width="50%">
      <img src="animations/navier_stokes/evolution.gif" alt="Navier-Stokes, Transolver over the 10-step rollout" width="100%"><br>
      <sub>Navier-Stokes 2D, 10-step rollout</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="animations/airfrans/evolution.gif" alt="AirfRANS, Transolver across the test set" width="100%"><br>
      <sub>AirfRANS, Transolver</sub>
    </td>
    <td align="center" width="50%">
      <img src="animations/car_cfd/evolution.gif" alt="Car-CFD, Transolver across the test set" width="100%"><br>
      <sub>Car-CFD, Transolver</sub>
    </td>
  </tr>
</table>

Each animation shows the prediction, the ground truth and the absolute error. They are in
the [benchmark gallery](gallery/README.md). The setups are in the
[benchmark docs](doc/benchmarks/index.md).

The trained checkpoints are on Hugging Face, one repository per model and benchmark, in the
[ScireX collection](https://huggingface.co/collections/zenteiq-ai/scirex). The
[checkpoint cards](doc/benchmarks/cards/index.md) list each one with its result.

### Reproduce the results table

[`scripts/reproduce.py`](scripts/reproduce.py) rebuilds the results table from the released checkpoints. It
runs each benchmark's `infer.py` on the released weights (inference only, no training) and
writes the numbers it measures to `results.csv`, `results_table.csv` and `results_table.tex`. The
cell-to-checkpoint map is in [Reproduce our benchmarks](doc/benchmarks/reproducibility.md).

```bash
pip install -e ".[all,cuda12]"                 # everything, with the NVIDIA GPU wheels
scripts/reproduce.sh                           # every cell of the results table, inference only
scripts/reproduce.sh --only "Pipe,Cavity"      # selected rows or models
python scripts/reproduce.py --list             # the cell-to-checkpoint map
python scripts/dev/check_tables.py --strict    # no checkpoints: committed metrics match the doc tables
```

Set `SCIREX_CKPT_ROOT` to the folder that holds the downloaded checkpoints and `SCIREX_DATA_ROOT`
to the folder that holds the datasets (there is no default location). A cell
without a checkpoint is
reported as `MISSING` and the run continues (`--strict-missing` makes it fail).

## Public API

Import from the package, not from the module that defines a name:

```python
from scirex.models import FNO
from scirex.losses import LpLoss, Q4Galerkin
from scirex.training import adamw_cx
```

The stable API is the set of names exported by [`scirex.models`](https://zenteiq-research.github.io/scirex/api/models/), [`scirex.layers`](https://zenteiq-research.github.io/scirex/api/layers/),
[`scirex.losses`](https://zenteiq-research.github.io/scirex/api/losses/), [`scirex.data`](https://zenteiq-research.github.io/scirex/api/data/), [`scirex.training`](https://zenteiq-research.github.io/scirex/api/training/), [`scirex.config`](https://zenteiq-research.github.io/scirex/api/config/) and [`scirex.utils`](https://zenteiq-research.github.io/scirex/api/utils/). It follows
Semantic Versioning from 1.0, and removals go through a deprecation period of at least two minor
releases. Code under [`scirex.experimental`](https://zenteiq-research.github.io/scirex/api/experimental/) (including [`GIWO`](https://zenteiq-research.github.io/scirex/api/experimental/#scirex.experimental.giwo.GIWO)) has no stability guarantee, and
`scirex._internal` is private. The [API reference](doc/api/index.md) lists every name, the
[experimental reference](doc/api/experimental.md) covers [`scirex.experimental`](https://zenteiq-research.github.io/scirex/api/experimental/), and the
[API policy](doc/dev_guide/api_policy.md) has the rules.

## Contribution guidelines

**If you want to contribute, read the [Contribution Guidelines](CONTRIBUTING.md) first. This
project follows its [Code of Conduct](CODE_OF_CONDUCT.md); by taking part you are expected to
uphold it.** Contributions need a signed [Contributor Licence Agreement](CLA.md) (companies use
the [corporate form](CLA-CORPORATE.md)). Governance and decision rules are in
[Governance](GOVERNANCE.md).

**We use GitHub Issues for bugs and feature requests.** Never report a security issue in a
public issue: the [Security policy](SECURITY.md) explains how to report it privately. For anything else
write to [**research@zenteiq.ai**](mailto:research@zenteiq.ai).

Run the tests with:

```bash
make test-unit          # fast, JIT disabled
make test-integration   # JIT compiled, uses XLA cache
make test               # full suite
make lint
```

## Resources

*   [Install guide](doc/install/index.md): Ubuntu, macOS, Windows, [Docker](doc/install/docker.md), [troubleshooting](doc/install/troubleshooting.md)
*   [API reference](doc/api/index.md)
*   [Theory guide](doc/theory_guide/operator_learning.md) and [loss guide](doc/loss_guide)
*   [Tutorials](doc/tutorials) and [user guide](doc/user_guide)
*   [Benchmarks](doc/benchmarks/index.md), [reproducing the paper tables](doc/benchmarks/reproducibility.md) and the [gallery](gallery/README.md)
*   [Dataset guides](doc/dataset_guide)
*   [Developer guide](doc/dev_guide/contributing.md) and [API policy](doc/dev_guide/api_policy.md)
*   [Community](doc/community/index.md)

## Citation

If you use ScireX in research, please cite the software and the relevant method papers
in [References](#references). The citation metadata is in the [Citation](CITATION.cff) file, and
GitHub shows it behind the "Cite this repository" button.

## License

[GNU Lesser General Public License v3.0 only (LGPL-3.0-only)](LICENSE) for the source code,
unless a file states otherwise. Commercial and proprietary applications can use the library under
the LGPLv3 conditions.

ZenteiQ-authored documentation and diagrams are licensed under **CC BY 4.0**; code examples stay
LGPL-3.0-only unless marked otherwise. The licences of the datasets and of the model weights are
stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai). Customer data and proprietary ZenteiQ assets are not
covered by these default licences unless expressly identified. See [LGPL v3 licence](LICENSE), [GPL v3 licence](COPYING), [Notice](NOTICE) and
[Third-party notices](THIRD_PARTY_NOTICES.md). Copyright (C) 2026 ZenteiQ AiTech Innovations
Pvt. Ltd.

## About

ScireX is a ZenteiQ open-source project, developed and maintained by **ZenteiQ AiTech
Innovations Pvt. Ltd.** It builds on SciREX, the earlier research framework (a separate project), developed by ZenteiQ and the AiREX (AI
for Research and Engineering eXcellence) Lab at the Indian Institute of Science (IISc), Bangalore.
We thank the researchers, engineers, students and collaborators whose work shaped it.

## References

1. <a id="ref-1"></a>Z. Li, N. Kovachki, K. Azizzadenesheli, et al.,
   *Fourier Neural Operator for Parametric Partial Differential Equations*,
   ICLR 2021.
   https://arxiv.org/abs/2010.08895

2. <a id="ref-2"></a>N. Kovachki, Z. Li, B. Liu, et al.,
   *Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs*,
   Journal of Machine Learning Research (JMLR), 2023.
   https://arxiv.org/abs/2108.08481

3. <a id="ref-3"></a>Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli, and A. Anandkumar,
   *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*,
   NeurIPS 2023.
   https://arxiv.org/abs/2309.00583

4. <a id="ref-4"></a>T. Tripura and S. Chakraborty,
 *Wavelet Neural Operator for Solving Parametric Partial Differential Equations*, 2022.
 https://arxiv.org/abs/2205.02191

5. <a id="ref-5"></a>H. Wu et al.,
  *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024.
https://arxiv.org/abs/2402.02366

6. <a id="ref-6"></a>Z. Li, D. Z. Huang, B. Liu, and A. Anandkumar,
   *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries* (Geo-FNO),
   JMLR 2023.
   https://arxiv.org/abs/2207.05209

7. <a id="ref-7"></a>Costa Rocha et al.,
   *Deep Neural Network Modeling for CFD Simulations: Benchmarking the Fourier Neural Operator on the Lid-Driven Cavity Case*,
   Applied Sciences 13(5), 3165, 2023.
   https://doi.org/10.3390/app13053165

8. <a id="ref-8"></a>D. Curl and H. Hu,
   *Physically interpretable surrogate modeling of thermal fields in electronics cooling using combined proper orthogonal decomposition and neural networks*,
   AI Thermal Fluids 6, 100040, 2026.
   https://doi.org/10.1016/j.aitf.2026.100040

9. <a id="ref-9"></a>F. Bonnet et al.,
    *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions*,
    NeurIPS 2022 Datasets and Benchmarks.
    https://arxiv.org/abs/2212.07564

10. <a id="ref-10"></a>N. Umetani and B. Bickel,
    *Learning Three-Dimensional Flow for Interactive Aerodynamic Design*,
    ACM Transactions on Graphics 37(4), 2018.
    https://dl.acm.org/doi/10.1145/3197517.3201325
