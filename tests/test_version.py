# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The version is written once, in ``pyproject.toml``, and everything else reads it.

``scirex.__version__`` comes from the package metadata (``importlib.metadata``), which
the build copies from ``pyproject.toml``. These tests pin that arrangement: no second
copy of the version in the source, ``CITATION.cff`` in step, the deploy workflows
reading the same file, and a sensible answer both for an installed package and for a
source checkout run through ``PYTHONPATH=src``.
"""

import re
from importlib import metadata
from pathlib import Path

import pytest
from packaging.version import Version

from scirex._internal import version as version_module

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"


def _pyproject_version() -> str:
    found = version_module.version_from_pyproject(PYPROJECT)
    assert found is not None, "pyproject.toml has no [project] version"
    return found


def test_pyproject_declares_a_static_canonical_version() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert not re.search(r"^dynamic\s*=.*version", text, re.M), "version must be static"
    assert "[tool.hatch.version]" not in text, "hatch must not read a version from code"
    version = _pyproject_version()
    # devpi rejects metadata whose version differs from the wheel name, so the written
    # form must already be the normalised one
    assert str(Version(version)) == version, f"{version!r} is not canonical PEP 440"


def test_no_second_copy_of_the_version_in_the_package() -> None:
    offenders = [
        str(p.relative_to(REPO))
        for p in sorted((REPO / "src" / "scirex").rglob("*.py"))
        if re.search(r"^__version__\s*(:[^=]+)?=", p.read_text(encoding="utf-8"), re.M)
    ]
    assert (
        not offenders
    ), f"__version__ is assigned in {offenders}; read it from metadata"


def test_dunder_version_is_the_package_version() -> None:
    import scirex

    assert scirex.__version__ == version_module.package_version()
    assert "__version__" in dir(scirex)
    assert Version(scirex.__version__)  # a PEP 440 string, not the placeholder


def test_installed_metadata_matches_pyproject() -> None:
    """Whenever the package is installed, its metadata must be pyproject's version.

    A mismatch means the install is stale: run ``pip install -e .`` again after changing
    the version.
    """
    try:
        installed = metadata.version(version_module.DISTRIBUTION)
    except metadata.PackageNotFoundError:
        pytest.skip("ScireX is not installed; the pyproject fallback is used")
    assert installed == _pyproject_version(), (
        f"installed metadata says {installed} but pyproject.toml says "
        f"{_pyproject_version()}: reinstall with `pip install -e .`"
    )


def test_not_installed_falls_back_to_pyproject(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_module.metadata, "version", _missing)
    assert version_module.package_version() == _pyproject_version()


def test_not_installed_and_no_pyproject_is_reported_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_module.metadata, "version", _missing)
    monkeypatch.setattr(version_module, "version_from_pyproject", lambda path: None)
    assert version_module.package_version() == version_module.UNKNOWN


def test_pyproject_of_another_project_is_not_trusted(tmp_path: Path) -> None:
    other = tmp_path / "pyproject.toml"
    other.write_text('[project]\nname = "something-else"\nversion = "9.9.9"\n')
    assert version_module.version_from_pyproject(other) is None
    assert version_module.version_from_pyproject(tmp_path / "missing.toml") is None


def test_citation_file_names_the_same_release() -> None:
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    cited = re.search(r'^version:\s*"?([^"\n]+)"?', cff, re.M)
    assert cited is not None, "CITATION.cff has no version"
    # CFF may write a release candidate as 1.0.0-rc.1; PEP 440 normalises it to 1.0.0rc1
    assert Version(cited.group(1)) == Version(_pyproject_version())


@pytest.mark.parametrize("workflow", ["deploy-production.yml", "deploy-staging.yml"])
def test_deploy_workflows_read_the_version_from_pyproject(workflow: str) -> None:
    path = REPO / ".github" / "workflows" / workflow
    if not path.is_file():
        pytest.skip("the deploy workflows are maintainers' files, not in this copy")
    text = path.read_text(encoding="utf-8")
    assert "pyproject.toml" in text
    assert "src/scirex/__init__.py" not in text, "a workflow still edits the package"
    assert "__version__" not in text.replace(
        "# ", ""
    ), "a workflow still reads __version__"
