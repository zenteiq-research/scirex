<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Add a New Example (Model + Problem)

An *example* in ScireX is a model trained on a problem, with a tutorial page
that someone else can follow from start to finish. This guide says where each
piece goes, what it must look like and which tests it must pass. It is
written so that a contributor who has never seen the code base can add an
example from it alone.

## The pieces

| Piece | Where it lives | Page |
|---|---|---|
| A model (only if it is new) | `src/scirex/experimental/<model>.py`, registered with the model factory | [Add a model](Add_a_Model.md) |
| A problem: data loader, config, `train.py`, `infer.py`, problem card | `scripts/benchmarks/<problem>/` and `doc/benchmarks/<problem>/README.md` | [Add a problem](Add_a_Problem.md) |
| A tutorial: one page and one script | `doc/tutorials/Others/<Problem>_<Model>.md` and `scripts/tutorials/<problem>_<model>.py` | [Add a problem](Add_a_Problem.md#the-tutorial) |
| Tests | `tests/` for the library, [`tests/docs/`](../../../tests/docs) for the tutorial, [`scripts/tests/`](../../../scripts/tests) for the benchmark | [Required tests](Required_Tests.md) |
| A pull request that ticks every box | GitHub, against `develop` | [PR checklist](PR_Checklist.md), [Contributing](../../dev_guide/contributing.md) |

An example does not always need all of them. A tutorial for an existing model
on an existing benchmark needs only the tutorial and its tests. A new model on
an existing benchmark adds the model. A new problem adds the benchmark folder
and its problem card.

## The order to work in

1. **Read the [worked example](Poisson1D_FNO.md).** It is the smallest
   complete example: 1D Poisson with FNO, one script that generates its own
   data, trains in about a minute on a CPU and has the four required tests.
2. **Add the model**, if it is new, and its four required tests. Nothing else
   can be tested until the model builds.
3. **Add the problem**: the data loader first, with a test on synthetic data,
   then the config, `train.py`, `infer.py` and the problem card.
4. **Write the tutorial** from a script that runs end to end with `--smoke` on
   a CPU.
5. **Measure** what the tutorial page reports; leave out anything you did not
   measure.
6. **Open the pull request** with the [PR checklist](PR_Checklist.md).

## Conventions that apply everywhere

- Every `.py` file starts with the SPDX header; every page starts with the
  licence comment of the other tutorial pages.
- Names: `<Problem>` is CamelCase in page and folder names (`NavierStokes`),
  `<problem>` is snake_case in script and benchmark names (`navier_stokes`,
  but `navierstokes_fno.py` for tutorial scripts), `<Model>` is the model's
  class name ([`FNO`](../../api/models.md#scirex.models.FNO), [`Transolver`](../../api/models.md#scirex.models.Transolver)). See [Naming rules](../../dev_guide/naming.md).
- Arrays: [`FNO`](../../api/models.md#scirex.models.FNO) is channels-first, `(batch, channels, *spatial)`; [`WNO`](../../api/models.md#scirex.models.WNO) is channels-last, [`Transolver`](../../api/models.md#scirex.models.Transolver)
  takes point lists and [`GINO`](../../api/models.md#scirex.models.GINO) takes coordinates and query points. State the layout of a new model.
- Numbers on a page are measured on the stated hardware; a number that was not
  measured is left out. Never copy a number from a paper as if it were ours.
- Commit messages follow `<type>(<scope>): <description>`, for example
  `docs(tutorials): add the Poisson1D FNO example`.
