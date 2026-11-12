<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Third-Party Notices

ScireX includes or depends on third-party software and other materials.
Each component remains subject to its own licence. This file records those
components and preserves their required notices.

## Components

Do not remove or alter upstream copyright or licence notices. Vendored or bundled
components are listed here with their original notices preserved verbatim.

## Transolver
Version: `main` branch, read September–October 2026
Source: https://github.com/thuml/Transolver
Licence: MIT
Use in ScireX: code derived from it (ported to JAX). `scripts/benchmarks/pipe/_transolver.py` ports `PDE-Solving-StandardBenchmark/exp_pipe.py`; `scripts/benchmarks/navier_stokes/_data.py` and `scripts/benchmarks/navier_stokes/train.py` port the data split and the training and evaluation loops of `PDE-Solving-StandardBenchmark/exp_ns.py`; `scripts/benchmarks/airfrans/_data.py`, `scripts/benchmarks/airfrans/_metrics.py` and `scripts/benchmarks/airfrans/_gino.py` re-implement the node features and the evaluation (`Infer_test`, `Airfoil_test`, `Compute_coefficients`, `Results_test`) of `Airfoil-Design-AirfRANS`. `src/scirex/models/transolver.py`, `src/scirex/layers/blocks/transolver_block.py` and `src/scirex/layers/blocks/physics_attention.py` port the model's own layers (`Model`, `Transolver_block`, `Physics_Attention_*`); `src/scirex/data/datasets/shapenet_car_drag.py` ports `Car-Design-ShapeNetCar/utils/drag_coefficient.py`. Each of these files carries a "Portions derived from Transolver" line.
Copyright / notice:

```
MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## neuraloperator
Version: `main` branch, read September–October 2026
Source: https://github.com/neuraloperator/neuraloperator
Licence: MIT
Use in ScireX: code derived from it (ported to JAX/Flax). The FNO model (`src/scirex/models/fno.py`), its spectral convolution (`src/scirex/layers/blocks/spectral_convolution.py`), GINO (`src/scirex/models/gino.py`), the Darcy/Navier–Stokes dataset loaders, the unit-Gaussian normalizer, data processors, positional embeddings, normalization/padding/resample layers, the AdamW/patching/tensor-GaLore/Trainer training utilities, `src/scirex/utils.py`'s ported helpers (and those split out into `src/scirex/_internal` and `src/scirex/experimental/analysis.py`) and the `src/scirex/experimental/distributed` model-parallel utilities are ports of the corresponding `neuralop` modules; their unit tests mirror `neuralop`'s own test suite one-for-one where an upstream test exists. Each of these files carries a "Portions derived from neuraloperator" line.
Copyright / notice:

```
MIT License

Copyright (c) 2023 NeuralOperator developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## x-transformers
Version: `main` branch, read October 2026
Source: https://github.com/lucidrains/x-transformers
Licence: MIT
Use in ScireX: `src/scirex/layers/common/embeddings.py`'s rotary-embedding helpers (`rotate_half` and related functions) are modified from `x_transformers/x_transformers.py`, per that file's own "modified from" comment.
Copyright / notice:

```
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Third-party datasets (referenced, not redistributed)

ScireX documents and trains on external datasets. They are **not redistributed** in this
repository: each must be obtained from its upstream source, as the per-dataset guides in
`doc/dataset_guide/` describe. The licences of the datasets and of the model weights are
stated on the Hugging Face repositories ([zenteiq-ai](https://huggingface.co/zenteiq-ai)), and a dataset's licence is
independent of the ScireX code licence (LGPL-3.0-only).
