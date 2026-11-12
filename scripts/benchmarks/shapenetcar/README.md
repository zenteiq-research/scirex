<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Car-Design-ShapeNetCar benchmark scripts

The problem card (equations, geometry, BCs, parameters, data, model, quickstart) lives in
[doc/benchmarks/shapenetcar/README.md](../../../doc/benchmarks/shapenetcar/README.md).

- `train.py`: train and save checkpoints
- `infer.py`: evaluate saved checkpoints
- `animate.py`: re-render the training-progress GIF
- Helpers: `_checkpoint.py`, `_config.py`, `_data.py`, `_gino_geometry.py`, `_models.py`, `_physics.py`, `_reference.py`
