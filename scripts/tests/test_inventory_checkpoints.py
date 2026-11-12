# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Checkpoint inventory: built on a fake archive, plus the committed inventory (no NAS)."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "inventory_checkpoints", SCRIPTS / "inventory_checkpoints.py"
)
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)
from reproduce_paper_map import CELLS  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _ckpt(root: Path, rel: str, weights: bytes = b"w", config: bool = True) -> Path:
    d = root / rel
    d.mkdir(parents=True)
    (d / "params.msgpack").write_bytes(weights)
    if config:
        (d / "config.json").write_text(
            json.dumps(
                {"n_params": 7, "provenance": {"git_commit": "abc", "git_dirty": False}}
            )
        )
    return d


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "ar"
    _ckpt(root, "pipe/fno", b"one")
    _ckpt(root, "Checkpoints_Benchmarks/pipe/fno", b"one")  # a copy
    _ckpt(root, "lid_driven_cavity_wno/wno_3to1", b"two")
    _ckpt(root, "car_cfd_gino", b"three", config=False)
    (root / "tools").mkdir()
    (root / "tools" / "x.msgpack").write_bytes(b"ignored")
    return root


def test_finds_checkpoints_and_skips_tools(archive):
    found = {p.as_posix() for p in inv.find_checkpoints(archive)}
    assert found == {
        "pipe/fno",
        "Checkpoints_Benchmarks/pipe/fno",
        "lid_driven_cavity_wno/wno_3to1",
        "car_cfd_gino",
    }


def test_rows_classify_hash_and_flag_duplicates(archive):
    rows = {r["path"]: r for r in inv.build(archive)[0]}
    assert rows["lid_driven_cavity_wno/wno_3to1"]["problem"] == "cavity"
    assert rows["lid_driven_cavity_wno/wno_3to1"]["model"] == "wno"
    assert rows["car_cfd_gino"]["problem"] == "car_cfd"
    assert rows["car_cfd_gino"]["loader_format"].startswith("legacy")
    assert rows["pipe/fno"]["loader_format"] == "standard"
    assert rows["pipe/fno"]["weights_sha256"] == _sha(b"one")
    assert rows["pipe/fno"]["git_commit"] == "abc"
    assert rows["pipe/fno"]["same_weights_as"] == "Checkpoints_Benchmarks/pipe/fno"
    assert rows["car_cfd_gino"]["same_weights_as"] == ""


def test_manifest_and_archive_checks_catch_changes(archive):
    d = archive / "pipe" / "fno"
    (d / "MANIFEST.json").write_text(
        json.dumps({"sha256": {"params.msgpack": _sha(b"one")}})
    )
    (archive / "Checkpoints_Benchmarks" / "CHECKSUMS.sha256").write_text(
        f"{_sha(b'old')}  pipe/fno/params.msgpack\n"
    )
    rows = {r["path"]: r for r in inv.build(archive)[0]}
    assert rows["pipe/fno"]["manifest_check"] == "ok"
    assert rows["pipe/fno"]["archive_checksums_check"].startswith("mismatch")


def test_write_then_verify_detects_a_changed_file(archive, tmp_path):
    out = tmp_path / "out"
    rows, hashes = inv.build(archive)
    inv.write(rows, hashes, out)
    assert inv.diff(inv.read_hashes(out / inv.SHA_NAME), hashes) == {
        "added": [],
        "removed": [],
        "changed": [],
    }
    (archive / "pipe" / "fno" / "params.msgpack").chmod(0o644)
    (archive / "pipe" / "fno" / "params.msgpack").write_bytes(b"retrained")
    _, new = inv.build(archive)
    d = inv.diff(inv.read_hashes(out / inv.SHA_NAME), new)
    assert d["changed"] == ["pipe/fno/params.msgpack"]


def test_missing_paper_checkpoint_is_reported(archive):
    rows, _ = inv.build(archive)
    assert "darcy_s85/data_driven/fno/seed0" in inv.missing_from_inventory(rows, CELLS)


def test_committed_inventory_covers_every_paper_checkpoint():
    """The inventory shipped in doc/benchmarks lists each checkpoint the registry cites."""
    path = SCRIPTS.parent / "doc" / "benchmarks" / inv.CSV_NAME
    listed = {r["path"] for r in csv.DictReader(open(path))}
    cited = {
        c.ckpt.strip("/") for c in CELLS if c.ckpt and not inv.is_unreleased(c.ckpt)
    }
    assert cited <= listed, sorted(cited - listed)
    # the published files name no unreleased checkpoint, no archive-internal path or column
    assert not any(
        inv.is_unreleased(p) or p.startswith(inv.ARCHIVE_PREFIX) for p in listed
    )
    assert not set(inv.INTERNAL_COLUMNS) & set(next(csv.DictReader(open(path))))
    sums = (
        (SCRIPTS.parent / "doc" / "benchmarks" / inv.SHA_NAME).read_text().splitlines()
    )
    assert not any(
        inv.is_unreleased(line.split("  ", 1)[1]) for line in sums if "  " in line
    )


def test_the_public_inventory_has_no_unreleased_internal_or_duplicate_entries(archive):
    rows, hashes = inv.public(*inv.build(archive))
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths))  # the archive's own copy is not listed twice
    assert not any(p.startswith(inv.ARCHIVE_PREFIX) for p in paths)
    assert not any(inv.is_unreleased(p) for p in paths)
    assert not any(inv.is_unreleased(p) for p in hashes)
    assert not any(p.startswith(inv.ARCHIVE_PREFIX) for p in hashes)
    assert all(not (set(inv.INTERNAL_COLUMNS) & set(r)) for r in rows)
    assert "pipe/fno" in paths and "car_cfd_gino" in paths
    assert inv.classify(Path("lid_driven_cavity_wno/wno_3to1")) == ("cavity", "", "wno")
    assert inv.is_unreleased("lid_driven_cavity_fno/fno_3to1")
    assert inv.is_unreleased("Checkpoints_Benchmarks/lid_driven_cavity_gino/x")
    assert inv.is_unreleased("drivaernet/data_driven/transolver/x")
    assert not inv.is_unreleased("pipe/fno")
