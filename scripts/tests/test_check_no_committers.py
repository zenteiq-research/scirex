# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Release identity: no committer in files, in dists or in the public snapshot."""

import datetime
import importlib.util
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


nc = load("check_no_committers", "scripts/dev/check_no_committers.py")
rm = load("check_release_metadata", "scripts/dev/check_release_metadata.py")
sd = load("set_release_date", "scripts/release/set_release_date.py")


def make_repo(
    tmp_path: Path, files: dict[str, str], author="Jane Roe", email="jroe@corp.test"
):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", f"user.name={author}", "-c", f"user.email={email}",
         "commit", "-qm", "c"],
        cwd=tmp_path, check=True,
    )  # fmt: skip
    return tmp_path


def test_the_repository_names_no_committer() -> None:
    assert nc.main(["--root", str(ROOT), "--strict"]) == 0


def test_name_email_and_local_part_are_found(tmp_path) -> None:
    repo = make_repo(
        tmp_path, {"a.md": "by Jane Roe\nmail jroe@corp.test\nuser jroe\n"}
    )
    assert nc.main(["--root", str(repo), "--strict"]) == 1


def test_clean_tree_passes_and_allowed_mailbox_is_fine(tmp_path) -> None:
    repo = make_repo(tmp_path, {"a.md": "contact research@zenteiq.ai\n"})
    assert nc.main(["--root", str(repo), "--strict"]) == 0


def test_allow_name_exempts_a_word(tmp_path) -> None:
    repo = make_repo(
        tmp_path, {"a.md": "the roe deer\n"}, author="Roe Doe", email="rd@c.test"
    )
    assert nc.main(["--root", str(repo), "--strict"]) == 0


def test_distributions_are_scanned(tmp_path) -> None:
    repo = make_repo(tmp_path, {"a.md": "clean\n"})
    dist = tmp_path.parent / (tmp_path.name + "-dist")
    dist.mkdir()
    with zipfile.ZipFile(dist / "x-1-py3-none-any.whl", "w") as z:
        z.writestr("pkg/a.py", "# written by Jane Roe\n")
    assert nc.main(["--root", str(repo), "--dist", str(dist), "--strict"]) == 1
    (dist / "x-1-py3-none-any.whl").unlink()
    with tarfile.open(dist / "x-1.tar.gz", "w:gz") as t:
        path = dist / "f.txt"
        path.write_text("jroe@corp.test")
        t.add(path, arcname="x-1/f.txt")
    assert nc.main(["--root", str(repo), "--dist", str(dist), "--strict"]) == 1


needs_snapshot = pytest.mark.skipif(
    not (ROOT / "scripts" / "release" / "make_public_snapshot.sh").is_file(),
    reason="maintainers' tool is not in this copy",
)


@needs_snapshot
def test_public_snapshot_has_only_the_project_identity(tmp_path) -> None:
    out = tmp_path / "public"
    done = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "release" / "make_public_snapshot.sh"),
            "HEAD",
            str(out),
            "1.0.0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    log = subprocess.run(
        ["git", "log", "--all", "--format=%an <%ae> | %cn <%ce>"],
        cwd=out,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert log == (
        "ZenteiQ Open Source <research@zenteiq.ai> | "
        "ZenteiQ Open Source <research@zenteiq.ai>"
    )
    assert nc.check_history(out) == []
    tags = subprocess.run(
        ["git", "tag"], cwd=out, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tags == ["v1.0.0"]


def test_history_with_a_developer_identity_is_reported(tmp_path) -> None:
    repo = make_repo(tmp_path, {"a.md": "x\n"})
    assert any("Jane Roe" in p for p in nc.check_history(repo))


@needs_snapshot
def test_snapshot_script_refuses_a_non_empty_folder(tmp_path) -> None:
    (tmp_path / "keep").write_text("x")
    done = subprocess.run(
        ["bash", str(ROOT / "scripts" / "release" / "make_public_snapshot.sh"),
         "HEAD", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 2
    assert (tmp_path / "keep").exists()


# --- release date -------------------------------------------------------------------

CFF = 'cff-version: 1.2.0\nversion: "0.0.1"\ndate-released: "2020-01-01"\n'


def release_repo(tmp_path: Path, version="1.2.3", cff=CFF) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CITATION.cff").write_text(cff)
    return make_repo(tmp_path, {})


def test_set_release_date_writes_version_and_date(tmp_path) -> None:
    release_repo(tmp_path)
    assert sd.main(["--root", str(tmp_path), "--date", "2026-12-01"]) == 0
    text = (tmp_path / "CITATION.cff").read_text()
    assert 'version: "1.2.3"' in text
    assert 'date-released: "2026-12-01"' in text


def test_set_release_date_rejects_a_bad_date(tmp_path) -> None:
    release_repo(tmp_path)
    assert sd.main(["--root", str(tmp_path), "--date", "01/12/2026"]) == 2


def test_stale_date_and_version_are_reported(tmp_path) -> None:
    release_repo(tmp_path)
    found = rm.check(tmp_path)
    assert any("differs from pyproject.toml" in p for p in found)
    assert any("not the date of HEAD" in p for p in found)


@pytest.mark.parametrize("offset", [0, 1])
def test_date_of_the_commit_passes(tmp_path, offset) -> None:
    release_repo(tmp_path)
    today = datetime.date.today() + datetime.timedelta(days=offset)
    sd.main(["--root", str(tmp_path), "--date", today.isoformat()])
    assert rm.check(tmp_path) == []


def test_missing_date_is_reported(tmp_path) -> None:
    release_repo(tmp_path, cff='version: "1.2.3"\n')
    assert any("missing or not YYYY-MM-DD" in p for p in rm.check(tmp_path))
