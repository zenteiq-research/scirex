<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Add a Problem

A problem is a dataset, a fixed train/test protocol and a metric, with one or
more models trained on it. In ScireX each problem is a **benchmark folder**
of scripts plus a **problem card**, and each model on it gets a **tutorial**.
[`scripts/benchmarks/darcy/`](../../../scripts/benchmarks/darcy) is the reference to copy; the
[worked example](Poisson1D_FNO.md) has the same pieces in one file.

Below, `<problem>` is the snake_case name (`navier_stokes`) and `<Problem>`
the CamelCase name (`NavierStokes`).

## Layout

```text
scripts/benchmarks/<problem>/
    __init__.py        what the benchmark is and its entry points (docstring)
    _config.py         PROTOCOL, RUNS: every setting, in one place
    _data.py           the data loader, normalisers, the loss and the metric
    _models.py         builds each model from its RunConfig
    _checkpoint.py     save and load a run (weights, config, normalisation)
    train.py           trains the runs and saves one checkpoint per run
    infer.py           evaluates the saved checkpoints, writes figures and tables
scripts/tests/test_<problem>_benchmark.py
doc/benchmarks/<problem>/README.md            the problem card
doc/benchmarks/<problem>/results/<model>.md   one results page per model
```

Files starting with `_` are private to the benchmark; `train.py` and
`infer.py` are the only entry points.

## 1. Data loader

Put the loader in `scripts/benchmarks/<problem>/_data.py`. It must:

- **Read from a folder given on the command line**, with a default from an
  environment variable (`DARCY_DATA_DIR`, `PIPE_DATA_DIR`, ...); never a
  hard-coded personal path.
- **Fail clearly**: a missing file raises `FileNotFoundError` naming the file
  expected and where to get it; asking for more samples than the file has
  raises `ValueError`.
- **Take the protocol from the config**: which files, how many training and
  test samples, which subsampling.
- **Return plain arrays in a documented layout**, channels-first, float32.
- **Fit normalisation on the training samples only**, and store the
  statistics in the checkpoint so `infer.py` uses the same ones.

Test it on small synthetic files written by the test, in the real file
format, so the test needs no download (see `fake_darcy` in
[`tests/docs/darcy_tutorial_helpers.py`](../../../tests/docs/darcy_tutorial_helpers.py)).

If the loader is useful outside the benchmark, it can become a public loader
in `src/scirex/data/datasets/<problem>.py` (like [`load_darcy_pt`](../../api/data.md#scirex.data.load_darcy_pt) or
[`load_airfrans`](../../api/data.md#scirex.data.load_airfrans)). That is a public name, so follow the
[API policy](../../dev_guide/api_policy.md#adding-a-public-name).

## 2. Config

All settings live in `_config.py`, as in [`scripts/benchmarks/darcy/_config.py`](../../../scripts/benchmarks/darcy/_config.py):

```python
PROTOCOL: dict = dict(
    train_file="...", test_file="...",
    n_train=1000, n_test=200,
    subsample_rate=5,
    eval_every=10,
)


@dataclass(frozen=True)
class RunConfig:
    """One training run: the model, its size, its recipe and its seed."""

    name: str
    model: str
    arch: dict = field(default_factory=dict)
    opt: dict = field(default_factory=dict)
    seed: int = 0


RUNS: dict[str, RunConfig] = {r.name: r for r in (RunConfig("fno", "fno", ...), ...)}
```

Rules:

- **One source of truth.** A setting is written once, here, and read
  everywhere else; the tutorial script repeats the values and a test checks
  they agree (`test_full_size_and_recipe_match_the_benchmark` in
  [`tests/docs/test_tutorial_darcy_fno.py`](../../../tests/docs/test_tutorial_darcy_fno.py)).
- **Say where each recipe comes from** in the module docstring: the paper,
  table or reference script.
- **Unknown keys fail.** `RunConfig` is a dataclass, so an unknown field is a
  `TypeError`; the keys of `arch` go to the model constructor, which refuses
  unknown ones; `train.py --run` refuses a run name not in `RUNS`. If you add
  free-form dictionaries, check their keys the way
  [`make_config`](Poisson1D_FNO.md#1-the-config) does.

## 3. `train.py`

`train.py` trains one or more runs from `RUNS` and saves a checkpoint per run.
Its command line, from [`scripts/benchmarks/darcy/train.py`](../../../scripts/benchmarks/darcy/train.py):

| Option | Meaning |
|---|---|
| `--run NAME [NAME ...]` or `--run all` | the runs to train; an unknown name is an error |
| `--data-dir` | the data folder (default from the environment variable) |
| `--ckpt-root` | where checkpoints go (`scripts/benchmarks/checkpoints/<problem>/`) |
| `--epochs` | override the epoch budget, for quick checks only |
| `--overwrite` | retrain a run that already has a complete checkpoint |
| `--allow-cpu` | permit a CPU run; without it a missing GPU is an error |

It must set and print the seed, print the parameter count and the versions,
and save, next to the weights, the run's config, the normalisation statistics
and the training history.

## 4. `infer.py`

`infer.py` loads every complete checkpoint, evaluates it on the test split
and writes the figures and numbers for the results pages
(`doc/assets/benchmarks/<problem>_<model>/`). It must recompute the
normalisation from the training split and refuse a checkpoint whose stored
statistics differ (`check_normalisation` in the Darcy `_checkpoint.py`), so a
checkpoint is never evaluated on a different split by accident.

## 5. Problem card

Write `doc/benchmarks/<problem>/README.md` with the same sections as the existing problem cards, for example the
[Darcy flow card](../../benchmarks/darcy/README.md): At a glance, Training and evaluation, Reproduce and References, plus Method,
Notes on the data and Notes on the results when they apply. State how the data are normalised, the loss, the
metric with its unit and the commands that reproduce the headline number. Point to the paper for the equations,
boundary conditions and geometry, and to the dataset guide for the files and the splits. Then:

- add it to `mkdocs.yml` under *Benchmarks*, with the results pages;
- keep `Status: draft` until a reviewer has checked every claim against the code or the paper; the reviewer changes
  the line to `Status: checked by <name> on <YYYY-MM-DD>`.

[`tests/docs/test_problem_cards.py`](../../../tests/docs/test_problem_cards.py) checks the headings, the `Status:` line,
that `doc/benchmarks/<problem>/` matches `scripts/benchmarks/<problem>/`, and
the navigation entry.

## The tutorial

Each model on the problem gets a tutorial: a page
`doc/tutorials/Others/<Problem>_<Model>.md` and a script
`scripts/tutorials/<problem>_<model>.py` (lower case, no underscore inside the
problem name: `navierstokes_fno.py`). The page goes in
`doc/tutorials/Others/`: the folders `FNO/`, `WNO/`, `GINO/` and `Transolver/`
hold only each model's two main tutorials (see the [overview](../index.md)).
The worked example
[Poisson1D_FNO](Poisson1D_FNO.md) and its script
[`scripts/tutorials/poisson1d_fno.py`](../../../scripts/tutorials/poisson1d_fno.py) show the format.

The script:

- uses only the public API ([`scirex.models`](../../api/models.md), [`scirex.training`](../../api/training.md),
  [`scirex.utils`](../../api/utils.md), ...), never `scripts.*` or a private module;
- is split into cells with `# %%` (at least eight) and ends with
  `if __name__ == "__main__":`;
- has a `--smoke` mode that runs on a CPU in seconds;
- prints its environment and seed.

The page:

- has these headings, in this order: *Summary*, *Setup*, *The problem*,
  *Step-by-step walkthrough*, *Run it*, *Results*, *Next steps*, *References*
  (a *Troubleshooting* section is optional, after *Results*);
- copies its code from the script: each block marked
  `<!-- tutorial-snippet -->` must appear in the script word for word (at
  least ten), every function and class of the script except `main` is named on
  the page, and the page ends its walkthrough by naming the complete script;
- shows at least one figure, stored in
  `doc/assets/tutorials/<Problem>_<Model>/`;
- states where each number comes from, and leaves out what was not measured;
- is listed in `mkdocs.yml` under *Tutorials*, in *Others* under its model.

[`tests/docs/test_tutorial_pages.py`](../../../tests/docs/test_tutorial_pages.py) checks all of this for every page named
`<Problem>_<Model>.md`; the script's own behaviour is tested in
`tests/docs/test_tutorial_<problem>_<model>.py`.
