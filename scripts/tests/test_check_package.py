# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the release tooling: package checker, committer list, publish workflow."""

import importlib.util
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = load("check_package", "scripts/dev/check_package.py")
HAS_LC = (ROOT / "scripts/dev/list_committers.py").is_file()
lc = load("list_committers", "scripts/dev/list_committers.py") if HAS_LC else None
# The committer and image tools are maintainers' tools; the public copy does not ship them.
needs_lc = pytest.mark.skipif(
    not HAS_LC, reason="maintainers' tool is not in this copy"
)
needs_images = pytest.mark.skipif(
    not (ROOT / "scripts/release/build_and_scan_images.sh").is_file(),
    reason="maintainers' tool is not in this copy",
)


def make_wheel(path: Path, extra: dict[str, bytes] | None = None) -> Path:
    files = {
        "scirex/__init__.py": b"",
        "scirex/py.typed": b"",
        "scirex-1.dist-info/licenses/LICENSE": b"x",
        **(extra or {}),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def make_sdist(path: Path, extra: dict[str, bytes] | None = None) -> Path:
    files = {
        "src/scirex/__init__.py": b"",
        "src/scirex/py.typed": b"",
        "LICENSE": b"x",
        "COPYING": b"x",
        "NOTICE": b"x",
        "README.md": b"x",
        "pyproject.toml": b"x",
        **(extra or {}),
    }
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(f"pkg-1/{name}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return path


def problems_for(archive: Path) -> list[str]:
    return cp.check_contents(archive, cp.members(archive))


def test_clean_archives_pass(tmp_path) -> None:
    assert problems_for(make_wheel(tmp_path / "a.whl")) == []
    assert problems_for(make_sdist(tmp_path / "a.tar.gz")) == []


@pytest.mark.parametrize(
    "name",
    [
        "scirex/model.pt",
        "scirex/ckpt/best.ckpt",
        "scirex/data.npz",
        "scirex/nb.ipynb",
        "scirex/.env",
        "scirex/.env.local",
        "scirex/__pycache__/x.py",
        "data/train.csv",
    ],
)
def test_forbidden_files_are_reported(tmp_path, name) -> None:
    wheel = make_wheel(tmp_path / "a.whl", {name: b"x"})
    assert any(name in p for p in problems_for(wheel))


def test_scirex_data_source_package_is_allowed(tmp_path) -> None:
    wheel = make_wheel(tmp_path / "a.whl", {"scirex/data/darcy.py": b"x = 1\n"})
    assert problems_for(wheel) == []


def test_personal_path_in_a_source_file_is_reported(tmp_path) -> None:
    wheel = make_wheel(tmp_path / "a.whl", {"scirex/a.py": b"p = '/home/alice/x'\n"})
    assert any("personal path" in p for p in problems_for(wheel))


def test_missing_legal_and_typed_files_are_reported(tmp_path) -> None:
    sdist = make_sdist(tmp_path / "a.tar.gz")
    with tarfile.open(sdist) as t:  # rebuild without NOTICE
        pass
    bad = tmp_path / "b.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        info = tarfile.TarInfo("pkg-1/src/scirex/__init__.py")
        t.addfile(info, io.BytesIO(b""))
    found = problems_for(bad)
    assert any("COPYING is missing" in p for p in found)
    assert any("py.typed is missing" in p for p in found)


def test_wheel_without_licence_files_is_reported(tmp_path) -> None:
    wheel = tmp_path / "a.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("scirex/__init__.py", "")
        z.writestr("scirex/py.typed", "")
    assert any("licence files" in p for p in problems_for(wheel))


def test_oversized_file_is_reported(tmp_path) -> None:
    wheel = make_wheel(tmp_path / "a.whl", {"scirex/big.py": b"#" * 1_100_000})
    assert any("over the file limit" in p for p in problems_for(wheel))


GOOD_META = """Metadata-Version: 2.4
Name: scirex
Version: {version}
Summary: s
Description-Content-Type: text/markdown
License-Expression: LGPL-3.0-only
License-File: LICENSE
License-File: COPYING
Author-email: A <a@example.org>
Maintainer-email: B <b@example.org>
Project-URL: Homepage, https://x
Project-URL: Documentation, https://x
Project-URL: Repository, https://x
Project-URL: Issues, https://x
Project-URL: Changelog, https://x
Classifier: License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Classifier: Programming Language :: Python :: 3.10
Requires-Python: >=3.10
Requires-Dist: jax
{extras}
{body}"""


def meta(version=None, body="x" * 600, extras=None, drop=None):
    import email

    version = version or cp.declared_version(ROOT)
    if extras is None:
        extras = "\n".join(f"Provides-Extra: {e}" for e in cp.declared_extras(ROOT))
    text = GOOD_META.format(version=version, extras=extras, body="\n" + body)
    if drop:
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.startswith(drop)
        )  # keeps the body
    return email.message_from_string(text)


def test_good_metadata_passes() -> None:
    assert cp.check_metadata(meta(), ROOT) == []


def test_metadata_problems_are_reported() -> None:
    assert any("Version" in p for p in cp.check_metadata(meta(version="0.0.1"), ROOT))
    assert any(
        "relative links" in p
        for p in cp.check_metadata(meta(body="[a](doc/x.md)" + "x" * 600), ROOT)
    )
    assert any(
        "License-Expression" in p
        for p in cp.check_metadata(meta(drop="License-Expression"), ROOT)
    )
    assert any(
        "License-File COPYING" in p
        for p in cp.check_metadata(meta(drop="License-File: COPYING"), ROOT)
    )
    assert any(
        "Changelog" in p
        for p in cp.check_metadata(meta(drop="Project-URL: Changelog"), ROOT)
    )
    assert any("extra" in p for p in cp.check_metadata(meta(extras=""), ROOT))


def test_declared_values_come_from_pyproject() -> None:
    assert cp.declared_version(ROOT).startswith("1.")
    assert {"dev", "docs"} <= set(cp.declared_extras(ROOT))


def test_checker_never_uploads() -> None:
    text = (ROOT / "scripts" / "dev" / "check_package.py").read_text()
    assert "upload" not in text.replace("nothing was uploaded", "").replace(
        "Never uploads", ""
    ).replace("never uploads", "")


# --- committers ---------------------------------------------------------------------


@needs_lc
def test_identities_are_merged_by_email() -> None:
    rows = [(3, "Ann", "a@x.org"), (2, "ann b", "a@x.org"), (1, "Bo", "b@x.org")]
    assert lc.merge(rows) == [("Ann", "a@x.org", 5), ("Bo", "b@x.org", 1)]


@needs_lc
def test_tagging_and_strict_exit(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = ["-c", "user.name=T", "-c", "user.email=t@example.org"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "add", "f"], cwd=repo, check=True)
    subprocess.run(["git", *env, "commit", "-qm", "c"], cwd=repo, check=True)
    out = tmp_path / "out" / "c.csv"
    assert lc.main(["--root", str(repo), "--out", str(out), "--strict"]) == 1
    assert "UNTAGGED" in out.read_text()
    mapping = tmp_path / "map.csv"
    mapping.write_text("t@example.org,contractor  # comment\n")
    assert (
        lc.main(
            ["--root", str(repo), "--out", str(out), "--map", str(mapping), "--strict"]
        )
        == 0
    )
    assert "contractor" in out.read_text()


@needs_lc
def test_bad_category_is_rejected(tmp_path) -> None:
    mapping = tmp_path / "m.csv"
    mapping.write_text("a@x.org,boss\n")
    with pytest.raises(SystemExit):
        lc.load_map(mapping)


@needs_lc
def test_output_inside_the_repository_is_refused(tmp_path) -> None:
    assert lc.main(["--root", str(ROOT), "--out", str(ROOT / "committers.csv")]) == 2
    assert not (ROOT / "committers.csv").exists()


# --- publish workflow and image script -----------------------------------------------

WORKFLOW = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text()


@pytest.mark.parametrize(
    "needle",
    [
        "workflow_dispatch:",
        "github.repository == 'zenteiq-research/scirex'",
        "startsWith(github.ref, 'refs/tags/v')",
        "pypa/gh-action-pypi-publish@",
        "attestations: true",
        "id-token: write",
        "python -m build",
        "check_package.py",
        "twine",
    ],
)
def test_publish_workflow_has(needle) -> None:
    assert needle in WORKFLOW


def test_pushing_a_tag_publishes_nothing() -> None:
    # Publishing is started by hand on release day; a pushed tag must not upload.
    top, _, _ = WORKFLOW.partition("\njobs:\n")
    assert "push:" not in top and "tags:" not in top
    assert "environment:" not in WORKFLOW  # the trusted publisher names no environment


def test_publish_workflow_is_pinned_and_has_no_token() -> None:
    import re

    for ref in re.findall(r"uses: (\S+)", WORKFLOW):
        assert re.search(r"@[0-9a-f]{40}$", ref), f"{ref} is not pinned to a commit"
    for secret in ("PYPI_API_TOKEN", "password:", "TWINE_PASSWORD"):
        assert secret not in WORKFLOW
    assert "test.pypi.org" not in WORKFLOW


def test_only_the_publish_job_can_write_the_oidc_token() -> None:
    top, _, jobs = WORKFLOW.partition("\njobs:\n")
    assert "id-token" not in top
    assert jobs.count("id-token: write") == 1


@needs_images
def test_image_script_builds_scans_and_never_pushes() -> None:
    script = ROOT / "scripts" / "release" / "build_and_scan_images.sh"
    text = script.read_text()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert "docker push" not in text.replace("# ", "")
    assert "archive --format=tar" in text
    assert "trivy" in text
    assert "digests.md" in text
