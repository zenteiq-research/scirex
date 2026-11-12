<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# PR Checklist for a New Example

Copy this list into the description of the pull request and tick each box.
Strike through a box that does not apply and say why (for example, "no new
model").

```markdown
### New example: <Problem> with <Model>

**Model** (skip if the model already exists)
- [ ] `src/scirex/experimental/<model>.py`, a `BaseModel` subclass with a unique `name=`
- [ ] Added to `_BUILTIN_ARCHS` in `src/scirex/models/base.py` and to `tests/models/test_init.py`
- [ ] `<Model>Config(ModelConfig)` in `src/scirex/config/models.py`: `model_arch` + one field per constructor argument
- [ ] Numpy-style docstring: summary, Parameters (type, meaning, default), Notes on layout, References

**Problem** (skip if the benchmark already exists)
- [ ] `scripts/benchmarks/<problem>/` with `_config.py`, `_data.py`, `train.py`, `infer.py`
- [ ] Every setting is in `_config.py`, with the source of each recipe
- [ ] Data folder from `--data-dir` or an environment variable; no personal paths
- [ ] Problem card `doc/benchmarks/<problem>/README.md` from the template, `Status: draft`, in `mkdocs.yml`

**Tutorial**
- [ ] `doc/tutorials/Others/<Problem>_<Model>.md` and `scripts/tutorials/<problem>_<model>.py`
- [ ] The script uses only public API and runs with `--smoke` on a CPU
- [ ] Every number on the page is measured (hardware, versions, runtime stated); a number that was not measured is left out
- [ ] Figures in `doc/assets/tutorials/<Problem>_<Model>/`; page in `mkdocs.yml`

**Tests**
- [ ] Output shape
- [ ] Finite gradients
- [ ] Tiny overfit, with the measured values in a comment
- [ ] Unknown config key refused (problem config and model constructor)
- [ ] No test downloads data; synthetic files in the real format instead
- [ ] `pytest tests/` and `pytest scripts/tests/` pass locally; the new tests take under two minutes on a CPU

**Housekeeping**
- [ ] SPDX header on every new `.py` file, licence comment on every new page
- [ ] `pre-commit run --all-files` is clean; `mkdocs build --strict` passes
- [ ] `CHANGELOG.md` entry under *Added*
- [ ] Commit messages follow `<type>(<scope>): <description>`
- [ ] Someone else has followed the tutorial page from a clean checkout and it worked
```

## Before you ask for review

```bash
pre-commit run --all-files
pytest tests/ -m "not gpu"
pytest scripts/tests/ -m "not gpu"
mkdocs build --strict
```

The last box matters most: a second person follows the page from a clean
checkout, without help, and every gap they hit is fixed in the guide or the
page before merging.
