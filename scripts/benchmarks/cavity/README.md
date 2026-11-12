<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Lid-Driven Cavity benchmark scripts

The problem card (equations, geometry, boundary conditions, parameters, data,
model and quickstart) lives in
[doc/benchmarks/cavity/README.md](../../../doc/benchmarks/cavity/README.md).

Scripts in this folder:

- `train.py`: step 1 of 2, trains the runs and saves one checkpoint per run.
- `infer.py`: step 2 of 2, evaluates the saved checkpoints on the held-out Re = 10,000 case.
- `animate.py`: re-renders the training-progress GIF from the cached frames.
- `_checkpoint.py`, `_config.py`, `_data.py`, `_models.py`, `_physics.py`, `_reference.py`: shared helpers.
