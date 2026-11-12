<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.

GENERATED from CONTRIBUTING.md by scripts/dev/check_contributing.py --write-docs.
Edit CONTRIBUTING.md, not this file.
-->

# Contributing to ScireX

Thanks for contributing to `ScireX`. This page summarizes the development
workflow; see also [JAX Patterns](jax_patterns.md).

Please read the [Code of Conduct](../project/code_of_conduct.md) before taking part, and report
security problems as described in the [Security policy](../project/security.md), never in a public issue.
For anything else that doesn't fit a GitHub issue, reach the maintainers at
[**research@zenteiq.ai**](mailto:research@zenteiq.ai).

We welcome contributions from researchers, engineers, students, and practitioners from both academia and industry. Contributions may include bug fixes, new Neural Operator architectures, performance improvements, documentation updates, examples, benchmarks, and testing enhancements.

> If you like the project but don't have time to contribute, there are other ways to support it:
> - Star the project on GitHub
> - Share it in academic and professional networks
> - Cite it in your research papers
> - Mention it at conferences and workshops
> - Use it in your research or industrial projects

## Licensing of Contributions

By submitting a contribution to ScireX, you represent that you have the
right to submit it and agree that the contribution will be licensed under the
licence that applies to the part of the repository to which it is contributed.

- Contributions to ScireX source code, tests and code examples:
  **LGPL-3.0-only**.
- Contributions to ZenteiQ-authored documentation and diagrams: **CC BY 4.0**.
- Contributions of datasets or model artifacts: accepted only with an explicit,
  approved licence and provenance statement.

For ZenteiQ employees and contractors whose applicable agreements assign the
relevant IP rights to ZenteiQ, public file-level attribution is to ZenteiQ
rather than to individual developers. ZenteiQ retains internal provenance and
contributor records (see the [Governance](../project/governance.md)).

Do not submit code, data, text, images or model artifacts copied from a third
party unless their licence permits the proposed inclusion and all required
attribution and notices are provided. Do not remove upstream copyright or
licence notices; list third-party components in
[Third-party notices](../project/third_party_notices.md).

External contributors must sign the ScireX
[Contributor Licence Agreement](../project/cla.md) before a pull request can be merged. A bot asks for it on your first pull
request: comment `I have read the CLA Document and I hereby sign the CLA` once, and it covers all
your later contributions. If a company owns what you write, the [corporate agreement](../project/cla_corporate.md) is needed
instead; write to [**research@zenteiq.ai**](mailto:research@zenteiq.ai). ZenteiQ staff covered by their employment or
contractor agreements, and automated accounts such as dependency bots, do not sign. The CLA is a
licence grant (individual or corporate), not a copyright assignment: you keep ownership of your
contribution and grant ZenteiQ the rights needed to distribute it under the licences above.
External or third-party attribution required by a licence or agreement must not
be removed.

## Ways to Contribute

### Reporting Bugs

#### Before Submitting a Bug Report

- Verify you're using the latest version
- Check if it's a known issue in our [bug tracker]({{ repo_url }}/issues)
- Search academic literature if it's a known scientific computing issue
- Collect information:
  - Full error traceback
  - OS and hardware details
  - Minimal reproducible example
  - Scientific context if applicable

#### Submitting a Bug Report

> **Security Issues**: Never report security vulnerabilities through public GitHub issues. See the [Security policy](../project/security.md) for how to report privately.

Submit bugs through GitHub issues:
1. Describe expected vs actual behavior
2. Provide reproduction steps
3. Include relevant scientific context
4. Attach minimal test data if needed

### Suggesting Enhancements

For enhancement suggestions:
1. Verify the functionality doesn't exist
2. Explain the scientific/technical motivation
3. Provide mathematical foundations if applicable
4. Describe implementation approach
5. Reference relevant research papers


### Code Contributions

#### Development setup

Clone, create a virtual environment, install the package with the development
extras, and install the git hooks. Python 3.10, 3.11 or 3.12 is required.

1. Fork the repository and add the main repository as `upstream`
   (skip this if you have write access):
```bash
git clone https://github.com/<YOUR_USERNAME>/scirex.git
cd scirex
git remote add upstream {{ repo_url }}.git
```
2. Create the environment, install, and enable the hooks. These four commands are run
   verbatim by the `contributing` CI job:

<!-- contributing:setup -->
```bash
python3 -m venv scirex-venv
source scirex-venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

3. Check the installation. This is also run by the `contributing` CI job:

<!-- contributing:check -->
```bash
python -c "import scirex; print(scirex.__version__)"
pytest tests/test_version.py tests/test_public_api.py tests/test_dependency_pins.py -m "not gpu" -q
python scripts/check_copyright.py $(git ls-files '*.py')
```

4. Create a branch from `develop` (see [Pull Requests](#pull-requests)):
```bash
git checkout -b <feat>/<descriptive-name>
```

[uv](https://docs.astral.sh/uv/) can replace `pip` in step 2 (`uv pip install -e ".[dev]"`).
Optional extras (`hpo`, `mesh`, `wandb`, `open3d`, `benchmark`, `docs`) are listed in
`pyproject.toml`; the exact tested versions are in `requirements.txt`.

### Environment Variables
For GPU-based workflows, the following environment variables can help control JAX memory allocation and device selection:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0
```

### Development Workflow

1. Open an issue first (bug report or feature request template) and agree the approach
2. Write tests first
3. Implement your changes
4. Add documentation
5. Add a `CHANGELOG.md` entry
6. Run the tests and the style checks (below)
7. Open a pull request against `develop`; a maintainer reviews it

Adding a model, a problem or a benchmark? Follow the
[new-example guide](../tutorials/NewExample/Overview.md) and fill in its
[PR checklist](../tutorials/NewExample/PR_Checklist.md).

### Documentation

Documentation contributions should:
- Follow scientific writing standards
- Include mathematical notation where appropriate
- Provide practical examples
- Reference relevant literature
- Follow our documentation style guide

Write names in prose the way the TensorFlow documentation does:

- A public API symbol (a module, class, function or `Class.method`) is in code font and linked to its entry in the API reference, at every mention.
- A full repository path such as [`scripts/reproduce.py`](../../scripts/reproduce.py) is in code font and linked to the file.
- An argument, a command-line flag, an environment variable, a value or a bare file name such as `infer.py` is in code font and not linked.

`python scripts/dev/check_inline_links.py --strict` lists the names that are not linked, and `--fix` writes the links.

### Research Contributions

For research-related contributions:
- Provide theoretical foundations
- Include mathematical proofs if applicable
- Add benchmark results
- Reference related work
- Follow scientific reproducibility guidelines

## Development Guidelines

### Code Style

- Follow PEP 8
- Use meaningful scientific variable names
- Document mathematical equations
- Include algorithm complexity analysis
- Add references to papers/methods

All code is formatted and linted with `ruff` (version pinned in
`.pre-commit-config.yaml`). Pre-commit hooks enforce this automatically.

```bash
pre-commit run --all-files   # what CI runs: ruff format, SPDX check, file hygiene
ruff check src/scirex/       # lint
ruff format src/scirex/      # format
```

Library code under `src/` uses `logging.getLogger(__name__)` and `warnings.warn(...)`,
not `print`; commented-out code and `TODO` without an owner are rejected by ruff
(`ERA`, `T20`, `TD`, `FIX`). Names follow [Naming Rules](naming.md).

## Copyright Header

Every `.py` file, including empty `__init__.py` files, must begin with the SPDX header:

```python
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
```

The pre-commit `copyright-check` hook and CI will reject files without this header.
The same two lines (with `#` comments) go at the top of YAML, TOML, `Makefile`,
`Dockerfile`, `.ini` and shell files. Markdown pages start with an HTML comment holding
`SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.` and
`SPDX-License-Identifier: CC-BY-4.0`; a page that mixes prose and code uses the
"License notice" wording instead (copy it from any page under `doc/`). Do not add
comments to JSON, lockfiles or vendored third-party files, and do not put individual
names in headers. These rules keep the licence notice of every file in one form.
Check the whole tree at once with
`python scripts/dev/check_contributing.py --strict` (it lists every tracked `.py`
file without the header). Documentation pages carry the licence notice described in
the [licensing section](#licensing-of-contributions).

## Testing

- Write unit tests for new features
- Include numerical accuracy tests
- Add performance benchmarks
- Test edge cases
- Verify mathematical correctness

### Running Tests

Package tests are in `tests/` (mirroring [`src/scirex/`](../../src/scirex/)); tests of the benchmark scripts
are in [`scripts/tests/`](../../scripts/tests/). They run in **separate pytest processes**, because several
scripts switch off `jax_enable_x64` at import time.

The complete suites, the benchmark scripts, tuning, meshing and the docs need the optional
extras. One command installs all of them (CPU):

```bash
pip install -e ".[all]"            # add cuda12 for an NVIDIA GPU: pip install -e ".[all,cuda12]"
```

Without `hpo` the tuning tests cannot be collected; without `benchmark` (open3d, wandb, vtk)
the Car-CFD benchmark tests are skipped. `python scripts/dev/check_extras.py --strict` checks that
`all` covers every import in the repository.

```bash
make test-unit          # fast, JIT disabled (JAX_DISABLE_JIT=1), no GPU required
make test-integration   # JIT compiled
make test               # full suite: tests/ with coverage, then scripts/tests/
pytest tests/ -m "not gpu"            # what CI runs for the package
pytest scripts/tests/ -m "not gpu"    # what CI runs for the scripts
```

Markers (declared in `pytest.ini`):

| Marker | Meaning | Use |
| --- | --- | --- |
| `unit` | fast, JIT disabled, no GPU | `-m unit` |
| `integration` | JIT compiled, uses the XLA cache | `-m integration` |
| `gpu` | needs an NVIDIA GPU; deselected in CI | `-m "not gpu"` |
| `needs_x64` | needs float64; the fixture in [`tests/conftest.py`](../../tests/conftest.py) turns it on | add it to float64 tests |

The suite is large; to run part of it, give a path, for example
`pytest tests/models -m "not gpu" -q`.

### Numerical tolerances

Do not write a fixed `atol` or `rtol` in a test that must pass on every device. Give the value
the test needs on the CPU in float32 to [`tests/tolerances.py`](../../tests/tolerances.py), which returns what to use on the
device and precision the test runs on:

```python
from tests import tolerances

assert jnp.allclose(got, want, atol=tolerances.atol(1e-5))
np.testing.assert_allclose(got, want, **tolerances.both(rtol=1e-5, atol=1e-6))
```

- **float32:** the value is multiplied by 1 on the CPU, 10 on a GPU and 1000 on a TPU.
- **float64** (a `needs_x64` test): the tolerance is tightened to `rtol` 1e-10 and `atol` 1e-12,
  and not scaled. A test that compares against an approximation, such as a finite difference
  or a quadrature, keeps its own looser fixed value.
- The GPU and TPU factors are **provisional**: nobody has measured them, and CI has no GPU or
  TPU. When you run the suite on one, lower or raise the factor in [`tests/tolerances.py`](../../tests/tolerances.py).

## Porting Rules (JAX/Flax)

- Use `flax.linen` (`nn.Module` with `setup()` + attribute assignment)
- No in-place mutations – JAX arrays are immutable
- All dtype-sensitive initializers must pass `dtype=jnp.float32` explicitly
- No `jax.block_until_ready()` in production code – debugging artifact only
- PyTorch layout is `(B, C, H, W)`; Flax Conv expects `(B, H, W, C)` – transpose at module boundaries


## Commit Style

- Use clear, descriptive commit messages
- Follow the format: `<type>(<scope>): <description>`
- Types: feat, fix, docs, test, chore
- Include issue numbers if applicable

Example:

```text
feat: add Tucker-factorized spectral convolution
fix: remove block_until_ready from FNO forward pass
docs: add theory guide for FNO derivation
test: add integration test for GINO on sphere mesh
```

## Pull Requests

1. **Issue first.** Open or pick an issue and agree the approach before you write code.
2. Branch from `develop`, name the branch `feat/`, `fix/`, `docs/` or `test/`.
3. **Tests**: new behaviour has tests; the whole suite passes.
4. **Docs**: user-visible changes are documented under `doc/`; `mkdocs build --strict` passes.
5. **CHANGELOG**: add an entry to `CHANGELOG.md` (Keep a Changelog format).
6. Fill in the pull request template; every box must be ticked or struck out with a reason.
7. All CI checks must pass before review; at least one maintainer approves
   (review routing is in [`.github/CODEOWNERS`](../../.github/CODEOWNERS)).
8. No direct commits to `develop` or `main` (enforced by the pre-commit hook).

## Branch Strategy

```text
feature/* --PR--> develop --(promotion PR)--> main --(auto)--> staging
                                                |
                                           v1.2.3 tag --(approval)--> production
```

- **`develop`** is the integration branch – open your PR here, not against `main`.
- **`main`** only receives promotion PRs from `develop`. Every commit on `main`
  automatically publishes a pre-release build to staging and redeploys the
  docs site, so nothing should land there that hasn't already passed CI on
  `develop`.
- **Production** is a separate step: pushing a `v*.*.*` tag on `main`
  publishes that exact commit, gated by manual approval.
- Promotion travels both ways by PR, merged with a merge commit: `develop → main`
  when `develop` is ready to ship, and `main → develop` immediately afterward to keep
  `develop` from drifting out of sync.

## Where to find more

- [Code of Conduct](../project/code_of_conduct.md) and [Security policy](../project/security.md)
- [New-example guide](../tutorials/NewExample/Overview.md) and its
  [PR checklist](../tutorials/NewExample/PR_Checklist.md)
- Issue templates: [`.github/ISSUE_TEMPLATE`](../../.github/ISSUE_TEMPLATE) (bug report, feature request)
- [Pull request template](../../.github/PULL_REQUEST_TEMPLATE.md)
- [Governance](../project/governance.md), [API policy](api_policy.md)
