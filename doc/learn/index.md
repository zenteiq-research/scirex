<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Introduction to ScireX

ScireX is an open-source library for scientific machine learning. It provides neural
operators, models that learn the solution map of a partial differential equation (PDE), together with
the losses, optimisers, data loaders and benchmarks needed to train and evaluate them. It is written in
JAX and Flax, so every model compiles with XLA and runs on CPU, NVIDIA GPU or TPU.

[Install](../install/index.md){ .md-button .md-button--primary }
[Quickstart](../user_guide/quickstart.md){ .md-button }

## What the library provides

<div class="grid cards" markdown>

-   :material-function-variant: **Neural operators**

    ---

    Four architectures for different data layouts: FNO and WNO for regular grids, GINO for irregular
    meshes and changing geometry, and Transolver for large unstructured meshes.

    [:octicons-arrow-right-24: Models](../api/models.md)

-   :material-sigma: **Data and physics losses**

    ---

    Train on solved examples, or on the governing equation alone with the Ritz energy loss.

    [:octicons-arrow-right-24: Losses](../loss_guide/overview.md)

-   :material-chart-line: **Reproducible benchmarks**

    ---

    Nine benchmark problems, with the models compared at a matched parameter count, and a script
    that reproduces the published tables from saved checkpoints.

    [:octicons-arrow-right-24: Benchmarks](../benchmarks/index.md)

-   :material-speedometer: **Built on JAX**

    ---

    Compile with `jit`, vectorise with `vmap`, differentiate with `grad`, and run unchanged on CPU,
    GPU or TPU. A complex-aware AdamW handles the complex spectral weights correctly.

    [:octicons-arrow-right-24: Training Scripts](../user_guide/training.md)

</div>

## Get started

=== "1. Install"

    ```bash
    python3 -m venv scirex-venv && source scirex-venv/bin/activate
    pip install scirex                # CPU; use "scirex[cuda12]" for an NVIDIA GPU
    python -m scirex.info
    ```

=== "2. Build a model"

    ```python
    import jax
    import jax.numpy as jnp
    from scirex.models import FNO

    model = FNO(n_modes=(8, 8), in_channels=1, out_channels=1, hidden_channels=16, n_layers=2)
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 1, 32, 32))  # (batch, channels, height, width)
    params = model.init(jax.random.PRNGKey(1), x)
    print(model.apply(params, x).shape)                            # (8, 1, 32, 32)
    ```

=== "3. Train it"

    ```python
    import optax
    from scirex.losses import LpLoss
    from scirex.training import adamw_cx

    y = jnp.sin(x)                                      # a stand-in target
    loss_fn = LpLoss(d=2, p=2, reduction="mean")
    optimizer = adamw_cx(learning_rate=1e-3)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(lambda p: loss_fn.rel(model.apply(p, x), y))(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    for i in range(3):
        params, state, loss = step(params, state)
        print(i, float(loss))
    ```

The [Quickstart](../user_guide/quickstart.md) trains an FNO end to end on generated data, and the
[tutorials](../tutorials/FNO/Darcy_FNO.md) train each model on a benchmark problem.

To place your problem and choose a model, read [Problem types](problem_types.md).

## How the pieces fit together

| Stage | What you use | Where to read |
| --- | --- | --- |
| Data | Dataset loaders, containers and normalisers in [`scirex.data`](../api/data.md) | [Datasets](../dataset_guide/darcy_2d.md) |
| Model | [`FNO`](../api/models.md#scirex.models.FNO), [`WNO`](../api/models.md#scirex.models.WNO), [`GINO`](../api/models.md#scirex.models.GINO), [`Transolver`](../api/models.md#scirex.models.Transolver), built directly or with [`get_model_jax`](../api/models.md#scirex.models.get_model_jax) | [Concepts](../theory_guide/operator_learning.md) |
| Loss | Data losses ([`LpLoss`](../api/losses.md#scirex.losses.LpLoss), [`H1Loss`](../api/losses.md#scirex.losses.H1Loss)) or the physics-informed Ritz energy loss | [Physics-informed losses](../loss_guide/overview.md) |
| Training | `optax` optimisers and the complex-aware [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) | [Training](../user_guide/training.md) |
| Evaluation | Relative L2 error, checkpoints, reproduction scripts | [Reproducing the tables](../benchmarks/reproducibility.md) |

## Choose your starting point

<div class="grid cards" markdown>

-   :material-school: **New to neural operators**

    ---

    Read the short background on operator learning, then train an FNO on Darcy flow.

    [:octicons-arrow-right-24: Operator Learning](../theory_guide/operator_learning.md)

-   :material-flask: **Researchers**

    ---

    Compare the four models on nine problems and reproduce the numbers on your own hardware.

    [:octicons-arrow-right-24: Benchmarks](../benchmarks/index.md)

-   :material-server: **Installing on a GPU or in Docker**

    ---

    Install the CUDA 12 or TPU wheels, use the container images, and check your system.

    [:octicons-arrow-right-24: Install](../install/index.md)

-   :material-source-pull: **Contributors**

    ---

    Add a model or a problem, with the tests and the pull-request checklist.

    [:octicons-arrow-right-24: Add a New Example](../tutorials/NewExample/Overview.md)

</div>

## Results

The [results summary](../benchmarks/comparison.md) puts all models side by side on every problem.

## Requirements

| | |
| --- | --- |
| Python | 3.10, 3.11 or 3.12 |
| Accelerators | CPU on Linux, macOS and Windows; NVIDIA GPU (CUDA 12) on Linux; TPU through the `tpu` extra |
| Core dependencies | JAX 0.6, Flax, Optax, NumPy, SciPy |
| Licence | Code LGPL-3.0-only, documentation CC BY 4.0 |

Run [`scripts/dev/check_platform.sh`](../../scripts/dev/check_platform.sh) to test your system. See
[Install](../install/index.md) for the options.

## Guides

| Guide | What it covers |
| --- | --- |
| [Training scripts](../user_guide/training.md) | How training is organised and how to run the benchmark scripts |
| [Hyperparameter tuning](../user_guide/hpo.md) | Tuning with Optuna and MLflow (experimental) |
| [Datasets](../dataset_guide/darcy_2d.md) | Source, schema, splits and limits of each dataset |
| [Physics-informed losses](../loss_guide/overview.md) | The Ritz energy loss: train from the equation alone |

## Community and support

- Report a problem or request a feature on the
  [issue tracker]({{ repo_url }}/issues).
- Read [Contributing](../dev_guide/contributing.md) before you open a pull request.
- Contact the team at [**research@zenteiq.ai**](mailto:research@zenteiq.ai). For security reports, see the [Security policy](../project/security.md).
- Cite the software with the [Citation](../project/citation.md) entry.
