<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Problem Card Template

A problem card describes how one benchmark problem is run and scored, so that a reader can reproduce the headline number and judge the results without opening the code. It does not repeat what the papers and the dataset guide already say: the equations, boundary conditions and geometry are in the source paper, and the files, the splits and how to get them are in the dataset guide (`doc/dataset_guide/<name>.md`). Each retained benchmark has one card at `doc/benchmarks/<name>/README.md`, where `<name>` is the snake_case benchmark name (the same name as `scripts/benchmarks/<name>/`), and the result pages sit next to it in `doc/benchmarks/<name>/results/` (`<model>.md` per model, `comparative.md` for all models side by side, which also holds the setup of the models). The author fills it in. A reviewer then checks every claim against the code or the source paper, and records the sign-off in the `Status:` line.

## Template

Copy the block below into `doc/benchmarks/<name>/README.md` and replace every prompt and every `TODO(owner)` marker. Keep the headings that are not marked optional, in this order.

````markdown
# <Problem name>

Status: draft / checked by <name> on <date>


## At a glance

*Fill in:* the models and their parameter budget, the metric, and the headline results with their source. TODO(owner)

## Method

*Optional.* Only when the method is the benchmark (for example a physics loss): what is minimised and how it is discretised. TODO(owner)

## Training and evaluation

*Fill in:* how the data are normalised, the training loss, and the metric with its unit (or "dimensionless" for a relative error), as implemented. Give the formulas. Say which weights are reported and whether anything was selected on the test set. TODO(owner)

## Notes on the data

*Optional.* Findings about the files or the splits that change how a result reads. Do not repeat the dataset guide. TODO(owner)

## Reproduce

*Fill in:* commands that run as written from the repository root, with `<DATA_DIR>` and `<OUT_DIR>` in place of local paths, and roughly how long training takes. The expected values and tolerances live on the comparison page.

```bash
export <NAME>_DATA_DIR=<DATA_DIR>
python scripts/benchmarks/<name>/train.py --run <model>
python scripts/benchmarks/<name>/infer.py --out-dir <OUT_DIR>
```

## Notes on the results

*Optional.* Caveats about the published numbers that are not on the comparison page. TODO(owner)

## References

*Fill in:* the source paper with the table or section the numbers come from, and the published targets. The dataset and its licence are in the dataset guide. TODO(owner)

**Checklist before review**

- [ ] The training and evaluation section gives the normalisation, the loss and the metric with its unit.
- [ ] The reproduce commands run and reach the metric on the comparison page.
- [ ] Paths in the commands are placeholders.
- [ ] Names link to the API reference, and the dataset guide is linked for the data.
- [ ] Equations and parameters are cross-checked against the code or the paper.
- [ ] No `TODO(owner)` markers remain.
````

## Rules

- Do not rename or reorder the headings. A test ([`tests/docs/test_problem_cards.py`](../../tests/docs/test_problem_cards.py)) checks every card against the list in this template, and checks that the template itself has not drifted. The headings marked optional may be left out.
- Point to the paper for the governing equations, the boundary conditions and the geometry, and to the dataset guide for the files and the splits. Do not copy them.
- Use SI units unless the problem is dimensionless. If it is, say so explicitly in the metric.
- Link to results tables that already live on the benchmark's Results page; do not copy them.
- Keep `Status: draft` while any `TODO(owner)` marker remains. A card whose status says `checked` must contain no such marker.
- The checker is not the owner. They compare the card with the code or the source paper, fix or flag every mismatch, and then change the `Status:` line.
