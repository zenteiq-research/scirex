<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Training Neural Operators

!!! note "How to train in 1.0"
    scirex 1.0 ships the building blocks (models, layers, losses, data loaders,
    the [`adamw_cx`](../api/training.md#scirex.training.adamw_cx) optimiser), **not a packaged training entry point**. Training
    goes through the benchmark scripts in the repository
    ([`scripts/benchmarks/`](../../scripts/benchmarks)),
    which are part of a source checkout and are **not** installed by `pip install
    scirex`. To train on your own problem, copy the script of the closest
    benchmark and change the data and model; each one is a short, self-contained
    jitted loop.

The curated benchmarks train with their own jitted training loops, using the
complex-aware optimiser [`scirex.training.adamw_cx`](../api/training.md#scirex.training.adamw_cx). The package also has an
**experimental** generic [`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) ([`scirex.experimental.training`](../api/experimental.md)), described
[below](#the-experimental-trainer). Experimental code has no stability guarantee.

## The benchmark scripts

Every curated benchmark lives in its own package under
[`scripts/benchmarks/`](../../scripts/benchmarks),
with the same layout:

```text
scripts/benchmarks/<problem>/
    _config.py      the data split and every published run (model, size, recipe)
    _data.py        loading and normalising the dataset
    _models.py      building each architecture and wiring it to the data
    _checkpoint.py  the checkpoint format shared by train.py and infer.py
    train.py        train one or more runs and save their checkpoints
    infer.py        evaluate the checkpoints and write the doc figures and metrics
    animate.py      render the problem's GIF under animations/<problem>/
```

| Package | Problem | Models |
|---------|---------|--------|
| `darcy/` | Darcy flow (2D elliptic) | FNO, WNO, GINO, Transolver |
| `darcy_physics/` | Darcy flow, physics-only | FNO, WNO, GINO, Transolver |
| `pipe/` | Pipe flow | FNO, WNO, GINO, Transolver |
| `navier_stokes/` | Navier-Stokes 2D | FNO, WNO, GINO, Transolver |
| `cavity/` | Lid-driven cavity | FNO, WNO, GINO, Transolver |
| `chip_cooling/` | Chip cooling | FNO, WNO, GINO, Transolver |
| `airfrans/` | AirfRANS airfoils | WNO, GINO, Transolver |
| `shapenetcar/` | ShapeNetCar | GINO, Transolver |
| `car_cfd/` | Car-CFD | GINO, Transolver, GIWO |

For example:

```bash
export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth{1,2}.mat
python scripts/benchmarks/darcy/train.py --run fno      # or --run all
python scripts/benchmarks/darcy/infer.py
```

## The experimental Trainer

[`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) automates a training loop: parameter management, the optimiser step,
scheduling, evaluation and checkpointing. Only the Car-CFD benchmark uses it. It drives
a model *wrapper*, an object that holds the parameters and optimiser and provides
`train_step` and `__call__`; the Car-CFD wrapper is in
[`scripts/benchmarks/car_cfd/_models.py`](../../scripts/benchmarks/car_cfd/_models.py). The library models ([`FNO`](../api/models.md#scirex.models.FNO), [`GINO`](../api/models.md#scirex.models.GINO), ...) are
plain Flax modules, so they cannot be passed to [`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) directly. The flow is:

```python
from scirex.experimental.training import Trainer
from scirex.losses import H1Loss, LpLoss

train_loss = H1Loss(d=2)
eval_losses = {"h1": H1Loss(d=2), "l2": LpLoss(d=2, p=2)}

trainer = Trainer(
    model=model,                  # a wrapper with train_step, see above
    n_epochs=300,
    data_processor=data_processor,
    eval_interval=1,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

## Losses

- **[`LpLoss(d, p)`](../api/losses.md#scirex.losses.LpLoss)** – relative $L^p$ data loss (the standard regression loss).
- **[`H1Loss(d)`](../api/losses.md#scirex.losses.H1Loss)** – Sobolev $H^1$ loss, which penalizes errors in the gradient as
  well as the value; often improves FNO accuracy on smooth PDE solutions.

The [`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) expects losses to **sum** across the batch dimension; see the API
reference for the reduction conventions.

## Capturing per-epoch error curves

`Trainer.train` returns the **final** epoch's metrics. To record the full
train-vs-test history (as the benchmark gallery does), drive the loop manually
with `train_one_epoch` + `evaluate_all`:

```python
for epoch in range(n_epochs):
    train_err, *_ = trainer.train_one_epoch(epoch, train_loader, train_loss)
    metrics = trainer.evaluate_all(
        epoch=epoch, eval_losses=eval_losses,
        test_loaders=test_loaders, eval_modes={},
    )
    test_err = metrics["16_l2"]   # "<resolution>_<loss>"
```

The benchmark packages above record the same history with their own training
loops; see e.g. [`scripts/benchmarks/darcy/train.py`](../../scripts/benchmarks/darcy/train.py).

## Distributed and high-resolution training

The [`Trainer`](../api/experimental.md#scirex.experimental.training.Trainer) supports multi-GPU data parallelism (sharding via JAX `pmap`) and
multi-grid patching ([`scirex.experimental.patching.MGPatchingDataProcessor`](../api/experimental.md#scirex.experimental.patching.MGPatchingDataProcessor), experimental) for high-resolution inputs. See
the [API reference](../api/training.md) for the relevant options.
