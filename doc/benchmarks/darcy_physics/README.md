<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Darcy Flow, Physics-Only

Predict the pressure of steady 2-D Darcy flow from a permeability field, as in the [data-driven Darcy benchmark](../darcy/README.md), but train without any labelled solution. Each network minimises the finite-element energy of the Darcy equation, and is scored against the solutions released by [[1]](#ref1). The benchmark shows how far a label-free physics loss gets, and how much unlabelled permeability fields add.

## At a glance

| | |
| --- | --- |
| Input | The permeability field $a$ on an 85 × 85 grid |
| Output | The pressure field $u$ on the same grid |
| Training data | Permeability fields only; the training solutions are dropped after loading |
| Metric | Relative L2 error of the pressure on the 200 test samples (lower is better) |
| Best result | **Transolver with 10,000 unlabelled fields, 0.565%** |

| Model | Unlabelled fields | Test error |
|---|---|---|
| Transolver | 10,000 | 0.565% |
| FNO | 10,000 | 0.648% |
| GINO | 20,000 | 0.722% |
| WNO | 20,000 | 1.262% |
| Transolver | 1,000 (released fields only) | 1.020% |

For context, the data-driven Transolver with 1,000 labelled samples reaches 0.57% ([[2]](#ref2)). A perfectly trained physics-only model would still have 0.31% error, because the finite-element mesh is coarser than the data.

## Training and evaluation

The loss is the energy of the equation $-\nabla\cdot(a\nabla u) = 1$, evaluated with bilinear (Q4) finite elements on the 85 × 85 grid ([`Q4Galerkin`](../../api/losses.md#scirex.losses.Q4Galerkin); see [the Ritz energy](../../user_guide/models_and_losses.md#the-ritz-energy)). The minimiser of this energy is the solution, so no labels are needed. The network output is multiplied by $\sin(\pi x)\sin(\pi y)$, so the zero boundary condition holds exactly.

Besides the 1,000 released fields, training can add 1,000 to 19,000 unlabelled fields sampled from the same two-phase law. No solutions are generated for them.

The test metric is the mean relative L2 error against the released solutions:

$$
E = \frac{1}{N_{\mathrm{test}}}\sum_{k=1}^{N_{\mathrm{test}}} \frac{\lVert \hat U_k - U_k \rVert_2}{\lVert U_k \rVert_2}, \qquad N_{\mathrm{test}} = 200
$$

The released solutions sit half a data cell inside the finite-element nodes, so the prediction is interpolated onto the solution points before the error is taken.

## Reproduce

You need the data ([Get the data](../../dataset_guide/darcy_2d.md#get-the-data)). Run from the repository root:

```bash
export DARCY_DATA_DIR=<DATA_DIR>

# Headline run: Transolver, 1,000 released + 9,000 sampled fields
python scripts/benchmarks/darcy_physics/train_transolver.py \
    --extra-coefficients 9000 --epochs 50 --out-dir <OUT_DIR>/transolver_fields_10000

# Same model on the 1,000 released fields only
python scripts/benchmarks/darcy_physics/train_transolver.py --out-dir <OUT_DIR>/transolver_fields_1000
```

`train_fno.py`, `train_gino.py` and `train_wno.py` take the same options. The commands for each model are in the [Reproduce section of the results page](results/comparative.md#reproduce). Each run writes `metrics.json` and figures to `--out-dir`; its last log line prints the final test error. Runs take from about 14 minutes (FNO) to about 3 hours (WNO). `check_floor.py` reproduces the 0.31% formulation floor.

### Evaluate a released checkpoint

The extra-fields runs of the four models are on Hugging Face, in the [ScireX collection](https://huggingface.co/collections/zenteiq-ai/scirex). `infer.py` evaluates one of them, or any run folder with the same `config.json` and weights, on a CPU or a GPU:

```bash
hf download zenteiq-ai/scirex-fno-darcy-physics-extra-fields \
    --local-dir ckpt/scirex-fno-darcy-physics-extra-fields

# Check the download against its sample.safetensors (no dataset needed)
python scripts/benchmarks/darcy_physics/infer.py \
    --checkpoint ckpt/scirex-fno-darcy-physics-extra-fields --sample

# The benchmark metric on the 200 test fields
export DARCY_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/darcy_physics/infer.py \
    --checkpoint ckpt/scirex-fno-darcy-physics-extra-fields
```

The model, its architecture, the input normalisation and the output scale are read from the checkpoint's `config.json`, and the network is rebuilt with the functions of its `train_<model>.py`. The test-set run prints the mean relative L2 error at the released sample points next to the figure recorded in `config.json`.

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021. <https://arxiv.org/abs/2010.08895>
2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
