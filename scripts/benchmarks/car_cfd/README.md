<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Car-CFD benchmark scripts

The problem card (equations, geometry, BCs, parameters, data, model, quickstart) lives in [doc/benchmarks/car_cfd/README.md](../../../doc/benchmarks/car_cfd/README.md).

- `train.py`: step 1, train and save checkpoints
- `infer.py`: step 2, evaluate saved checkpoints
- `animate.py`: animation of the best model's surface pressure across the test cars
- `_config.py`, `_data.py`, `_models.py`, `_checkpoint.py`, `_reference.py`, `_giwo.py`, `_fast_dwt.py`, `_vtk.py`: shared helpers
