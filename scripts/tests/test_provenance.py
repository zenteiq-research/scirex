# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""metrics.json provenance: commit, checkpoint SHA-256 and environment (no GPU needed)."""

import hashlib
import json
import logging
from pathlib import Path

from scripts.benchmarks import _provenance as prov


def _ckpt(root: Path, name: str = "fno", weights: bytes = b"weights") -> Path:
    d = root / "toyproblem" / name
    d.mkdir(parents=True)
    (d / "params.msgpack").write_bytes(weights)
    (d / "config.json").write_text("{}")
    return d


def _checksums(root: Path, weights: bytes) -> None:
    folder = root / "Checkpoints_Benchmarks"
    folder.mkdir()
    lines = [
        f"{hashlib.sha256(weights).hexdigest()}  toyproblem/fno/params.msgpack",
        f"{hashlib.sha256(b'{}').hexdigest()}  toyproblem/fno/config.json",
    ]
    (folder / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def test_attach_records_hashes_commit_and_environment(tmp_path):
    ck = _ckpt(tmp_path)
    m = prov.attach({"loss": 1.0}, [ck])
    p = m["inference_provenance"]
    assert m["loss"] == 1.0  # existing keys untouched
    assert p["schema_version"] == prov.SCHEMA_VERSION
    assert (
        p["checkpoints"][0]["sha256"]["params.msgpack"]
        == hashlib.sha256(b"weights").hexdigest()
    )
    assert (
        p["checkpoints"][0]["checksum_verified"] is None
    )  # no archive checksum listed
    assert {"sha", "branch", "dirty"} <= set(p["commit"])
    assert {"python", "packages", "devices", "hostname"} <= set(p["environment"])
    json.dumps(m)  # serialisable


def test_checksum_verified_against_sibling_archive_file(tmp_path):
    ck = _ckpt(tmp_path)
    _checksums(tmp_path, b"weights")
    assert prov.checkpoint_info(ck)["checksum_verified"] is True


def test_mismatch_is_recorded_and_warned_not_fatal(tmp_path, caplog):
    ck = _ckpt(tmp_path, weights=b"retrained")
    _checksums(tmp_path, b"weights")
    with caplog.at_level(logging.WARNING):
        info = prov.checkpoint_info(ck)
    assert info["checksum_verified"] is False
    assert "CHECKSUM MISMATCH" in caplog.text


def test_longest_listed_path_wins():
    entries = {"fno/params.msgpack": "a" * 64, "x/fno/params.msgpack": "b" * 64}
    assert prov._expected(entries, Path("/r/x/fno/params.msgpack")) == "b" * 64
    assert prov._expected(entries, Path("/r/y/fno/params.msgpack")) == "a" * 64
    assert prov._expected(entries, Path("/r/y/wno/params.msgpack")) is None


def test_repo_checksum_list_is_well_formed():
    lines = prov.REPO_CHECKSUMS.read_text().splitlines()
    assert lines and all(len(ln.split("  ")[0]) == 64 for ln in lines)
    assert prov.REPO_CHECKSUMS.name == "checkpoint_inventory.sha256"
