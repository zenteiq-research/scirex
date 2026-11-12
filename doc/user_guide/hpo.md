<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Hyperparameter Tuning

!!! warning "Experimental"
    This page describes [`scirex.experimental`](../api/experimental.md), which is not part of the stable API of ScireX 1.0:
    the interface can change in a minor release. Pin the exact version when you use it.

[`scirex.experimental.tuning`](../api/experimental.md) tunes the hyperparameters of **any** model and training loop with
[Optuna](https://optuna.org) and records every trial in [MLflow](https://mlflow.org).
It is framework-agnostic: you hand it a function that builds a model and a function
that trains it, and it handles sampling, early stopping of weak trials, and tracking.

```bash
pip install -e ".[hpo]"
```

## What you provide

| Piece | Signature | Role |
|---|---|---|
| `build_model` | `(params) -> model` | Build a fresh model for one trial. |
| `train_fn` | `(model, params, ctx) -> float \| None` | Your normal training loop. Call `ctx.report(value, step)` once per epoch. |
| `search_space` | dict spec or `(trial) -> dict` | The parameters to search. |

```python
from scirex.experimental.tuning import tune

search_space = {
    "lr":     {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
    "hidden": {"type": "int", "low": 16, "high": 256, "log": True},
    "act":    {"type": "categorical", "choices": ["gelu", "relu"]},
    "epochs": 30,                      # a plain value is a constant
}

def build_model(params):
    return make_my_model(hidden=params["hidden"], act=params["act"])

def train_fn(model, params, ctx):
    for epoch in range(1, params["epochs"] + 1):
        val_loss = train_one_epoch_and_validate(model, lr=params["lr"])
        ctx.report(val_loss, step=epoch)   # may raise TrialPruned: let it propagate

study = tune(
    build_model, train_fn, search_space,
    n_trials=50, max_resource=30, metric_name="val_loss",
    experiment_name="my-experiment",
    tracking_uri="sqlite:///mlflow.db",
)
print(study.best_params, study.best_value)
```

For conditional spaces, pass a callable `(trial) -> dict` that uses
`trial.suggest_*` directly.

## What gets tracked

* **One parent MLflow run per study** (`study-<name>`) holding the settings, the best
  parameters and score, a `trials.csv`, and Optuna's optimisation-history,
  intermediate-values, parameter-importance and parallel-coordinate plots.
* **One child run per trial** (`trial-000`, ...) with its parameters, the metric per step,
  the final value, and an `optuna.state` tag (`COMPLETE`, `PRUNED` or `FAIL`). Pruned
  trials show as *Killed* in the MLflow UI.

## Early stopping (multi-fidelity)

Each `ctx.report` is forwarded to Optuna's pruner, [Hyperband](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)
by default. Set `max_resource` to the most epochs a trial may run, and `min_resource` to
the first rung (usually the reporting interval). A non-finite value (NaN or inf loss)
prunes the trial immediately. Pass `pruner=` or `sampler=` to use other Optuna
implementations.

## Resuming and running workers in parallel

Give `tune` a `storage` and a `study_name` and the study becomes resumable: calling it
again adds trials to the same study. Several **processes** can share one
`storage`/`study_name` to search in parallel; each gets its own parent run, tagged with
its host name. Threads are not supported, because MLflow's tracking URI and active-run
stack are process-global.

## Running without a search

`tune(..., enabled=False, default_params={...})` runs `train_fn` once on fixed values,
logs a single MLflow run, and returns a one-trial study. This lets one script serve both
as a plain training run and as a search.

## Tracking server (optional)

`sqlite:///...` works for a single machine. For a shared MLflow UI and a database that
several workers can use, start the bundled stack:

```bash
python docker/hpo/make_env.py   # once: writes the gitignored docker/hpo/.env
docker compose -f docker/hpo/docker-compose.yml up -d --build
```

Then point the tuner at it (note the `+psycopg` driver in the Optuna URL):

```python
tune(
    ...,
    tracking_uri="http://localhost:5000",
    storage="postgresql+psycopg://mlflow:mlflow@localhost:5433/optuna",
    study_name="my-study",
)
```

The MLflow UI is at `http://localhost:5000` (on the machine that runs the stack). Both ports are bound to `localhost` and
MLflow has no authentication, so do not expose the stack beyond the machine without
putting authentication in front of it. Set `HPO_POSTGRES_PASSWORD` before `up` to change
the database password, and use the same value in `storage`.
