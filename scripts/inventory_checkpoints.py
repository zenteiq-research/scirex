# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Inventory of every archived checkpoint, with SHA-256 of every file.

    python scripts/inventory_checkpoints.py            # write the inventory
    python scripts/inventory_checkpoints.py --verify   # compare the archive with it

A checkpoint is a directory of the archive that directly holds a ``*.msgpack`` weights
file. The archive is only read. Two files are written to ``--out-dir``
(default ``doc/benchmarks/``):

``checkpoint_inventory.csv``
    one row per checkpoint: problem, regime, model, run, the weights file and its
    SHA-256, whether the standard loader can read it, training commit and creation
    time (from ``config.json``) or run commit (from ``MANIFEST.json``), PR, whether
    the archive's own checksums agree, other checkpoints with identical weights, and
    the paper-table cells that use it.
``checkpoint_inventory.sha256``
    ``sha256sum`` format, one line per file of every checkpoint, paths relative to
    the archive root. ``scripts/benchmarks/_provenance.py`` reads it to set
    ``checksum_verified`` in every ``metrics.json``.

The hashes record the archive as it is now; they do not prove it is what was trained.
``--verify`` exits non-zero if a file was added, removed or changed, or if a
checkpoint cited by ``reproduce_paper_map.py`` is missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

DEFAULT_ROOT = Path(os.environ.get("SCIREX_CKPT_ROOT", "<SCIREX_CKPT_ROOT>"))
DEFAULT_OUT = REPO / "doc" / "benchmarks"
CSV_NAME = "checkpoint_inventory.csv"
SHA_NAME = "checkpoint_inventory.sha256"
SKIP_TOP = {"tools"}  # verification helpers, not checkpoints
# Checkpoints that are not released (the data is not public) or are not part of 1.0 stay out
# of the public inventory.
UNRELEASED_PREFIXES = ("lid_driven_cavity_", "drivaernet")
ARCHIVE_PREFIX = "Checkpoints_Benchmarks/"  # the archive's own copy of the same layout
# columns of the archive's working notes, not of the public inventory
INTERNAL_COLUMNS = ("pr", "same_weights_as")


def is_unreleased(path: str | Path) -> bool:
    """True for a checkpoint path (archive layout, with or without the
    ``Checkpoints_Benchmarks/`` prefix) of a benchmark that is not released."""
    parts = Path(path).parts
    if parts and parts[0] == "Checkpoints_Benchmarks":
        parts = parts[1:]
    return bool(parts) and parts[0].startswith(UNRELEASED_PREFIXES)


COLUMNS = [
    "path",
    "problem",
    "regime",
    "model",
    "run",
    "weights_file",
    "weights_sha256",
    "weights_bytes",
    "n_files",
    "loader_format",
    "n_params",
    "git_commit",
    "git_dirty",
    "created_utc",
    "note",
    "manifest_check",
    "archive_checksums_check",
    "paper_cells",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_checkpoints(root: Path) -> list[Path]:
    """Directories (relative to ``root``) that directly hold ``*.msgpack`` weights."""
    found = set()
    for p in root.rglob("*.msgpack"):
        rel = p.parent.relative_to(root)
        if rel.parts and rel.parts[0] not in SKIP_TOP:
            found.add(rel)
    return sorted(found)


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def classify(rel: Path) -> tuple[str, str, str]:
    """(problem, regime, model) from the archive layout; ``""`` where it has none."""
    parts = list(rel.parts)
    if parts[0] == "Checkpoints_Benchmarks":  # a copy of the same layout
        parts = parts[1:]
    top = parts[0]
    for prefix in ("lid_driven_cavity_", "car_cfd_", "shapenetcar_"):
        if top.startswith(prefix):
            problem = "cavity" if prefix.startswith("lid") else prefix.rstrip("_")
            return problem, "", top[len(prefix) :]
    if top in ("darcy_s85", "drivaernet"):  # <problem>/<regime>/<model>/<run>
        return top, parts[1], parts[2]
    if top == "airfrans" and len(parts) > 1 and parts[1] == "data_driven":
        return top, parts[1], parts[2]
    return top, "", parts[1] if len(parts) > 1 else ""


def _manifest_check(root: Path, rel: Path, files: dict[str, str]) -> str:
    """Compare against the SHA-256 recorded in the folder's own MANIFEST.json."""
    man = _json(root / rel / "MANIFEST.json")
    # Two layouts: {"files": {name: {"sha256": ...}}} and {"sha256": {name: ...}}.
    listed = {n: i.get("sha256") for n, i in man.get("files", {}).items()}
    listed = listed or dict(man.get("sha256", {}))
    if not listed:
        return ""
    bad = [n for n, digest in listed.items() if n in files and digest != files[n]]
    return "mismatch: " + ", ".join(bad) if bad else "ok"


def _archive_check(entries: dict[str, str], rel: Path, files: dict[str, str]) -> str:
    """Compare against ``Checkpoints_Benchmarks/CHECKSUMS.sha256`` (paths in it are
    relative to ``Checkpoints_Benchmarks``)."""
    if not entries:
        return ""
    prefix = rel.parts[1:] if rel.parts[0] == "Checkpoints_Benchmarks" else rel.parts
    checked, bad = 0, []
    for name, digest in files.items():
        want = entries.get("/".join([*prefix, name]))
        if want is None:
            continue
        checked += 1
        if want != digest:
            bad.append(name)
    if bad:
        return "mismatch: " + ", ".join(bad)
    return "ok" if checked else ""


def paper_cells(registry) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = {}
    for c in registry:
        if c.ckpt:
            cells.setdefault(c.ckpt.strip("/"), []).append(f"{c.row}/{c.model}")
    return cells


def build(root: Path, registry=()) -> tuple[list[dict], dict[str, str]]:
    """The inventory rows and the per-file hashes (relative path -> sha256)."""
    cells = paper_cells(registry)
    ckpts = find_checkpoints(root)
    archive = {}
    sums = root / "Checkpoints_Benchmarks" / "CHECKSUMS.sha256"
    if sums.is_file():
        for line in sums.read_text().splitlines():
            digest, _, name = line.strip().partition("  ")
            if name:
                archive[name.strip()] = digest

    all_hashes: dict[str, str] = {}
    rows: list[dict] = []
    for rel in ckpts:
        d = root / rel
        files = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file()}
        for name, digest in files.items():
            all_hashes[f"{rel.as_posix()}/{name}"] = digest
        weights = sorted(n for n in files if n.endswith(".msgpack"))
        main = "params.msgpack" if "params.msgpack" in files else weights[0]
        cfg, man = _json(d / "config.json"), _json(d / "MANIFEST.json")
        prov = cfg.get("provenance", {})
        problem, regime, model = classify(rel)
        rows.append(
            {
                "path": rel.as_posix(),
                "problem": problem,
                "regime": man.get("regime", regime),
                "model": man.get("model", model),
                "run": rel.name,
                "weights_file": main,
                "weights_sha256": files[main],
                "weights_bytes": (d / main).stat().st_size,
                "n_files": len(files),
                "loader_format": "standard"
                if "config.json" in files and "params.msgpack" in files
                else "legacy (no config.json or params.msgpack)",
                "n_params": cfg.get("n_params", ""),
                "git_commit": prov.get("git_commit") or man.get("run_git_commit") or "",
                "git_dirty": prov.get("git_dirty", ""),
                "created_utc": prov.get("created_utc", man.get("copied", "")),
                "pr": man.get("pr", ""),
                "note": man["run"] if man.get("run") not in (None, rel.name) else "",
                "manifest_check": _manifest_check(root, rel, files),
                "archive_checksums_check": _archive_check(archive, rel, files),
                "same_weights_as": "",
                "paper_cells": "; ".join(cells.get(rel.as_posix(), [])),
            }
        )
    by_hash: dict[str, list[str]] = {}
    for r in rows:
        by_hash.setdefault(r["weights_sha256"], []).append(r["path"])
    for r in rows:
        r["same_weights_as"] = "; ".join(
            p for p in by_hash[r["weights_sha256"]] if p != r["path"]
        )
    return rows, all_hashes


def public(
    rows: list[dict], hashes: dict[str, str]
) -> tuple[list[dict], dict[str, str]]:
    """The inventory as published: without unreleased checkpoints, without the archive's own
    ``Checkpoints_Benchmarks/`` copies (each path appears once, without the prefix) and
    without the columns of the archive's working notes."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in sorted(rows, key=lambda r: r["path"].startswith(ARCHIVE_PREFIX)):
        path = r["path"].removeprefix(ARCHIVE_PREFIX)
        if is_unreleased(path) or path in seen:
            continue
        seen.add(path)
        out.append(
            {k: v for k, v in r.items() if k not in INTERNAL_COLUMNS} | {"path": path}
        )
    out.sort(key=lambda r: r["path"])
    files = {
        p.removeprefix(ARCHIVE_PREFIX): h
        for p, h in sorted(hashes.items())
        if not is_unreleased(p)
    }
    return out, files


def missing_from_inventory(rows: list[dict], registry) -> list[str]:
    """Checkpoints cited by the paper-table registry that the inventory does not list."""
    have = {r["path"] for r in rows}
    cited = {
        c.ckpt.strip("/") for c in registry if c.ckpt and not is_unreleased(c.ckpt)
    }
    return sorted(cited - have)


def write(rows: list[dict], hashes: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / CSV_NAME, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = [f"{h}  {p}" for p, h in sorted(hashes.items())]
    (out_dir / SHA_NAME).write_text("\n".join(lines) + "\n")


def read_hashes(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if name:
            out[name] = digest
    return out


def diff(old: dict[str, str], new: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(p for p in set(old) & set(new) if old[p] != new[p]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="checkpoint archive"
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="compare the archive with the written inventory instead of writing it",
    )
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        ap.error(f"archive {args.root} not found; set SCIREX_CKPT_ROOT or pass --root")

    from reproduce_paper_map import CELLS

    rows, hashes = public(*build(args.root, CELLS))
    missing = missing_from_inventory(rows, CELLS)
    for p in missing:
        print(
            f"MISSING: the paper's results table cites {p}, which is not in the archive"
        )

    if args.verify:
        old = args.out_dir / SHA_NAME
        if not old.is_file():
            print(f"no inventory at {old}; run without --verify first")
            return 2
        d = diff(read_hashes(old), hashes)
        for kind, paths in d.items():
            for p in paths:
                print(f"{kind.upper()}: {p}")
        ok = not any(d.values()) and not missing
        print(
            f"{len(rows)} checkpoints, {len(hashes)} files: "
            + ("inventory matches the archive" if ok else "inventory DIFFERS")
        )
        return 0 if ok else 1

    write(rows, hashes, args.out_dir)
    print(
        f"wrote {args.out_dir / CSV_NAME} ({len(rows)} checkpoints) and "
        f"{args.out_dir / SHA_NAME} ({len(hashes)} files)"
    )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
