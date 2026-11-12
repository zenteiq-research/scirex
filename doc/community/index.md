<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Community

ScireX is **a ZenteiQ open-source project**: a JAX library of neural
operators for scientific machine learning, with benchmarks that anyone can
re-run and check. It builds on the research of
[SciREX](https://github.com/zenteiq-sciml/SciREX), the earlier research framework from the AiREX Lab at IISc (a separate project). Everyone who uses it,
reports a problem, shares a result or contributes code is part of its
community.

## What the project stands for

<div class="grid cards" markdown>

-   **01 &nbsp; Models you can trust**

    ---

    FNO, WNO, GINO and Transolver with a stable public API, each validated on
    published benchmarks.

    [:octicons-arrow-right-24: Models and Losses](../user_guide/models_and_losses.md)

-   **02 &nbsp; Comparable results**

    ---

    Every model trained on the same splits at a matched parameter budget, and
    compared with the published numbers.

    [:octicons-arrow-right-24: Benchmarks](../benchmarks/index.md)

-   **03 &nbsp; Evidence behind every number**

    ---

    Archived checkpoints, checksums and provenance, so a reported number can be
    re-computed.

    [:octicons-arrow-right-24: Reproduce our benchmarks](../benchmarks/reproducibility.md)

</div>

## Take part

<div class="grid cards" markdown>

-   :material-forum-outline: **Ask a question**

    ---

    How to use a model, which one fits your problem, why a run behaves as it
    does. Search first: your question may already be answered.

    [:octicons-arrow-right-24: GitHub Discussions]({{ repo_url }}/discussions)

-   :material-bug-outline: **Report a bug**

    ---

    Something fails or gives a wrong result. The template asks for the
    version, the platform and a short example that shows the problem.

    [:octicons-arrow-right-24: Open an Issue]({{ repo_url }}/issues/new/choose)

-   :material-chart-line: **Share your results**

    ---

    You trained a model on your own problem, or re-ran a benchmark on other
    hardware. Post the setup and the numbers; they help others choose.

    [:octicons-arrow-right-24: Show and Tell]({{ repo_url }}/discussions)

-   :material-puzzle-outline: **Add a model or an example**

    ---

    A new architecture, a new problem or a tutorial, with the tests it needs.

    [:octicons-arrow-right-24: Add a New Example](../tutorials/NewExample/Overview.md)

-   :material-database-plus-outline: **Propose a benchmark dataset**

    ---

    A public dataset that the benchmarks do not cover yet. Say what it
    contains, its licence and which published results use it.

    [:octicons-arrow-right-24: Request a Feature]({{ repo_url }}/issues/new/choose)

-   :material-source-pull: **Improve the code or the docs**

    ---

    Fix a bug, speed something up or make a page clearer. Small pull requests
    are reviewed fastest.

    [:octicons-arrow-right-24: Contributing](../dev_guide/contributing.md)

</div>

Everyone taking part follows the
[Code of Conduct](../project/code_of_conduct.md).

## People and governance

The project is developed and maintained by the **ZenteiQ SciML Research Team** at
ZenteiQ AiTech Innovations Pvt. Ltd. The team is presented by what it does,
not by the names of its members, as the
[governance policy](../project/governance.md)
sets out.

| Function | What it covers |
|---|---|
| Maintainers | The direction of the project, the public API and its stability policy |
| Review | Every pull request, routed by [`CODEOWNERS`](../../.github/CODEOWNERS) |
| Benchmarks | The benchmark protocols, the archived checkpoints and the published numbers |
| Release | Versions, packages, container images and the [changelog](../project/changelog.md) |
| Security | Private vulnerability reports ([security policy](../project/security.md)) |

External contributions are welcome. They are credited in the pull request
and the changelog, and may need a Contributor Licence Agreement before they
can be merged; the [Contributing guide](../dev_guide/contributing.md)
has the terms.

## Built on the work of others

ScireX implements architectures and runs on datasets published by
other research groups. Please cite their work as well as this library.

**Architectures**

| Model | Original work |
|---|---|
| FNO | Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |
| WNO | Tripura and Chakraborty, *Wavelet Neural Operator for Solving Parametric Partial Differential Equations in Computational Mechanics Problems*, CMAME 2023, [arXiv:2205.02191](https://arxiv.org/abs/2205.02191) |
| GINO | Li et al., *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*, NeurIPS 2023, [arXiv:2309.00583](https://arxiv.org/abs/2309.00583) |
| Transolver | Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024, [arXiv:2402.02366](https://arxiv.org/abs/2402.02366) |

**Datasets**

| Benchmark | Data released by |
|---|---|
| Darcy flow, Navier-Stokes | Li et al., with FNO (ICLR 2021) |
| Pipe flow | Li et al., with Geo-FNO (JMLR 2023, [arXiv:2207.05209](https://arxiv.org/abs/2207.05209)) |
| Lid-driven cavity | Costa Rocha et al., Applied Sciences 2023, [doi:10.3390/app13053165](https://doi.org/10.3390/app13053165) |
| Chip cooling | Curl and Hu, AI Thermal Fluids 2026, [doi:10.1016/j.aitf.2026.100040](https://doi.org/10.1016/j.aitf.2026.100040) |
| AirfRANS | Bonnet et al., NeurIPS 2022 Datasets and Benchmarks, [arXiv:2212.07564](https://arxiv.org/abs/2212.07564) |
| ShapeNetCar, Car-CFD | Umetani and Bickel, ACM Trans. Graph. 2018; Car-CFD as processed for GINO ([Zenodo 13936501](https://zenodo.org/records/13936501)) |

Versions and sizes are in [Datasets](../benchmarks/datasets.md).

## Cite ScireX

If ScireX helps your work, cite the software
([Citation](../project/citation.md);
GitHub's *Cite this repository* button gives the same entry):

```bibtex
@software{scirex,
  author  = {{ZenteiQ AiTech Innovations Pvt. Ltd.}},
  title   = {ScireX},
  version = {1.0.0},
  date    = {2026-11-12},
  url     = {{{ repo_url }}},
  license = {LGPL-3.0-only}
}
```

## Get in touch

| For | Where |
|---|---|
| Questions and ideas | [GitHub Discussions]({{ repo_url }}/discussions) |
| Bugs and feature requests | [GitHub issues]({{ repo_url }}/issues) |
| Security vulnerabilities | Privately, as described in the [security policy](../project/security.md); never in a public issue |
| Anything else: collaborations, datasets, press | [**research@zenteiq.ai**](mailto:research@zenteiq.ai) |
