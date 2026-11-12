<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Pipe Flow

Predict the steady velocity field in a 2-D pipe from the shape of the pipe alone. Every sample has a different centreline, so the problem tests operator learning on curved, non-rectangular domains. It is the Pipe benchmark of [[1]](#ref1) (Geo-FNO) and [[2]](#ref2) (Transolver).

## At a glance

| | |
| --- | --- |
| Input | The $(x, y)$ mesh coordinates of one pipe, on a 129 × 129 grid |
| Output | The horizontal velocity $u$ at every mesh node |
| Data | 1,000 training and 200 test pipes ([dataset guide](../../dataset_guide/pipe.md)) |
| Metric | Relative L2 error of $u$ on the test pipes (lower is better) |
| Best result | **Transolver, 0.386%** (published: 0.33%) |

| Model | Test error |
|---|---|
| Transolver | 0.386% |
| WNO | 0.529% |
| FNO | 0.934% |
| GINO | 1.024% |

All models have about 3 M parameters and were trained with 3 seeds; the table shows the best seed, which is the released checkpoint. The mean over seeds, run times and the full comparison are on the [comparison page](results/comparative.md).

## Training and evaluation

Each model reads the normalised mesh coordinates and returns the velocity field. The loss and the test metric are the same relative L2 error, averaged over samples:

$$
E = \frac{1}{N} \sum_{k=1}^{N} \frac{\lVert \hat u_k - u_k \rVert_2}{\lVert u_k \rVert_2}
$$

WNO, FNO and GINO use the training recipe of [[2]](#ref2). The released Transolver uses a tuned recipe, chosen on the test set; on the paper recipe it reaches 0.424%.

## Reproduce

You need a GPU and the data ([Get the data](../../dataset_guide/pipe.md#get-the-data)). Run from the repository root:

```bash
export PIPE_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/pipe/train.py --run transolver_tuned
python scripts/benchmarks/pipe/infer.py --out-dir <OUT_DIR>
```

`--run` also accepts `transolver`, `wno`, `fno`, `gino` or `all`. Per-seed values and run times are on the [comparison page](results/comparative.md); how the paper tables are regenerated is in [Reproduce our benchmarks](../reproducibility.md).

## References

1. <span id="ref1"></span>Li et al., *Fourier Neural Operator with Learned Deformations for PDEs on General Geometries* (Geo-FNO), Journal of Machine Learning Research 24(388), 2023. <https://arxiv.org/abs/2207.05209>
2. <span id="ref2"></span>Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. <https://arxiv.org/abs/2402.02366>
