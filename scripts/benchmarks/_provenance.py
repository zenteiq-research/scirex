# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Provenance block for every ``metrics.json`` an ``infer.py`` writes.

``attach(metrics, [checkpoint_dir, ...])`` adds a top-level ``"inference_provenance"``
key (airfrans metrics already use ``"provenance"`` for the training run):

* ``commit``: git SHA, branch and dirty flag of this repository (``null`` outside git);
* ``checkpoints``: per checkpoint directory, the SHA-256 of its weights (``*.msgpack``) and
  ``config.json``, and ``checksum_verified``: ``true`` / ``false`` against the
  archive's ``CHECKSUMS.sha256``, ``null`` when the archive lists no checksum for it;
* ``environment``: Python, JAX, jaxlib, Flax, NumPy and scirex versions, devices, host;
* ``command``, ``timestamp_utc`` and ``schema_version``.

A checksum mismatch is logged as a warning and recorded; the run still completes.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import socket
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA_VERSION = 1
CHECKSUM_FILE = "CHECKSUMS.sha256"
_REPO = Path(__file__).resolve().parents[2]
#: Every checkpoint file's hash, from ``scripts/inventory_checkpoints.py``.
REPO_CHECKSUMS = _REPO / "doc" / "benchmarks" / "checkpoint_inventory.sha256"
_log = logging.getLogger(__name__)
_hash_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a file; repeated calls for an unchanged file are free."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict:
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "branch": None, "dirty": None}
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": {
            p: _version(p) for p in ("jax", "jaxlib", "flax", "numpy", "scirex")
        },
    }
    try:
        import jax

        env["devices"] = [
            {"platform": d.platform, "kind": d.device_kind} for d in jax.devices()
        ]
    except Exception as exc:  # - a report must never break a run
        env["devices"] = f"unavailable: {exc}"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode == 0:
            env["nvidia_driver"] = smi.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return env


def _read_checksums(f: Path) -> dict[str, str]:
    entries = {}
    for line in f.read_text().splitlines():
        digest, _, rel = line.strip().partition("  ")
        if rel:
            entries[rel.strip()] = digest
    return entries


def _archive_checksums(run_dir: Path) -> dict[str, str]:
    """Known checksums (path -> sha256) for ``run_dir``.

    The archive's own ``CHECKSUMS.sha256`` wins (searched in ``run_dir``'s ancestors
    and their immediate subfolders, since it sits in ``Checkpoints_Benchmarks/``);
    ``doc/benchmarks/checkpoint_inventory.sha256`` fills in what it does not list.
    """
    entries: dict[str, str] = {}
    if REPO_CHECKSUMS.is_file():
        entries.update(_read_checksums(REPO_CHECKSUMS))
    for parent in [run_dir, *run_dir.parents][:4]:
        for f in [parent / CHECKSUM_FILE, *sorted(parent.glob(f"*/{CHECKSUM_FILE}"))]:
            if f.is_file():
                return {**entries, **_read_checksums(f)}
    return entries


def _expected(entries: dict[str, str], path: Path) -> str | None:
    """The recorded checksum for ``path``: the longest listed path it ends with."""
    posix = path.as_posix()
    hits = [rel for rel in entries if posix.endswith("/" + rel)]
    return entries[max(hits, key=len)] if hits else None


def checkpoint_info(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    entries = _archive_checksums(run_dir)
    names = sorted({p.name for p in run_dir.glob("*.msgpack")} | {"config.json"})
    files, verified = {}, []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            continue
        actual = file_sha256(path)
        files[name] = actual
        expected = _expected(entries, path)
        if expected is None:
            continue
        verified.append(expected == actual)
        if expected != actual:
            _log.warning(
                "CHECKSUM MISMATCH for %s: archive lists %s, file is %s",
                path,
                expected,
                actual,
            )
    return {
        "path": str(run_dir),
        "sha256": files,
        "checksum_verified": all(verified) if verified else None,
    }


def provenance(checkpoints: Iterable[Path] = ()) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "commit": git_info(),
        "checkpoints": [checkpoint_info(Path(c)) for c in checkpoints],
        "environment": environment(),
    }


def attach(metrics: dict, checkpoints: Iterable[Path] = ()) -> dict:
    """Add the provenance block to ``metrics`` (in place) and return it."""
    metrics["inference_provenance"] = provenance(checkpoints)
    return metrics
