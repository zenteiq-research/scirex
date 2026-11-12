# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Register of the datasets behind the paper's results: source, size, where to get them.

    python scripts/list_datasets.py            # measure the files and write the register
    python scripts/list_datasets.py --check    # fail if the written register is stale

Writes ``doc/benchmarks/datasets.md`` and ``datasets.csv``. The source and name come
from the dataset guide of each benchmark (``doc/dataset_guide/*.md``). The size is
measured here: the bytes of the files a benchmark reads, with archive copies
(``.zip``, ``.tar.gz``) beside an extracted folder left out, in GiB (1024**3 bytes).

The host column links to the "Get the data" section of the guide, which names where the
files come from. The licences of the datasets are on the Hugging Face repositories, not in
this register.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

NAS = Path(os.environ.get("SCIREX_DATA_ROOT", "<SCIREX_DATA_ROOT>"))
NAS_ALL = Path(os.environ.get("SCIREX_DATA_ALL", "<SCIREX_DATA_ALL>"))
GUIDES = REPO / "doc" / "dataset_guide"
DEFAULT_OUT = REPO / "doc" / "benchmarks"
CSV_NAME, MD_NAME = "datasets.csv", "datasets.md"

UNKNOWN = "UNKNOWN"
ARCHIVES = (".zip", ".tar.gz", ".tgz")
COLUMNS = [
    "dataset",
    "benchmarks",
    "source",
    "size",
    "host",
    "download",
    "guide",
]


@dataclass(frozen=True)
class Dataset:
    guide: str  # file in doc/dataset_guide
    benchmarks: str
    location: tuple[Path, ...]  # what the benchmark reads, under the NAS roots
    alternate: str = ""
    host: str = ""  # where the files come from, as the guide's Download row says


DATASETS = [
    Dataset(
        "darcy_2d.md",
        "Darcy (data-driven and label-free)",
        (
            NAS_ALL / "piececonst_r421_N1024_smooth1.mat",
            NAS_ALL / "piececonst_r421_N1024_smooth2.mat",
        ),
        host="Zenodo",
    ),
    Dataset("pipe.md", "Pipe", (NAS_ALL / "pipe",), host="Geo-FNO data release"),
    Dataset(
        "navier_stokes.md",
        "Navier-Stokes",
        (NAS_ALL / "ns_V1e-5" / "NavierStokes_V1e-5_N1200_T20.mat",),
        host="FNO data release",
    ),
    Dataset(
        "cavity.md",
        "Cavity",
        (NAS / "lid_driven_cavity" / "cavity_paper",),
        str(NAS_ALL / "cavity"),
        host="Not public",
    ),
    Dataset(
        "chip_cooling.md", "Chip cooling", (NAS_ALL / "Chip_Cooling",), host="Dryad"
    ),
    Dataset(
        "airfrans.md",
        "AirfRANS",
        (NAS / "AirfRANS" / "Dataset",),
        str(NAS_ALL / "AirfRANS" / "Dataset"),
        host="airfrans package",
    ),
    Dataset(
        "shapenetcar.md",
        "ShapeNetCar",
        (NAS / "shapenetcar" / "training_data",),
        host="Umetani website",
    ),
    Dataset(
        "car_cfd.md",
        "Car-CFD",
        (NAS / "car_cfd" / "processed-car-pressure-data",),
        host="Zenodo",
    ),
]


def measure(path: Path) -> int | None:
    """Bytes of the regular files under ``path`` (a file counts as itself), leaving out
    archive copies; ``None`` if it does not exist."""
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size
    total, stack = 0, [path]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file() and not e.name.endswith(ARCHIVES):
                    total += e.stat().st_size
    return total


def human(n: int | None) -> str:
    if n is None:
        return UNKNOWN
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def _field(text: str, label: str) -> str:
    """The value of a ``- **<label>:** ...`` line of a dataset guide."""
    m = re.search(rf"^- \*\*{re.escape(label)}:\*\*\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else UNKNOWN


def _row(text: str, label: str) -> str:
    """The value of a ``| <label> | ... |`` row of a guide's dataset summary table."""
    m = re.search(
        rf"^\|\s*{re.escape(label)}\s*\|\s*(.+?)\s*\|\s*$", text, re.MULTILINE
    )
    return m.group(1).strip() if m else UNKNOWN


def guide_facts(guides: Path, name: str) -> tuple[str, str]:
    """(dataset name, source) from a dataset guide: the "Dataset name" and "Version /
    release date" bullets, or the "Dataset" and "Source" rows of its summary table."""
    path = guides / name
    if not path.is_file():
        return UNKNOWN, UNKNOWN
    text = path.read_text()
    name = _field(text, "Dataset name")
    source = _field(text, "Version / release date")
    if name == UNKNOWN:  # a guide with the summary table instead of the bullet list
        name, source = _row(text, "Dataset"), _row(text, "Source")
        source = _expand_citations(source, text)
    return name, source


def _expand_citations(source: str, text: str) -> str:
    """Replace the citation numbers [1], [2] in a Source row by the entries of the
    guide's numbered References list, so the register stands on its own."""
    refs = {
        n: re.sub(r'<span id="[^"]*"></span>', "", entry)
        for n, entry in re.findall(
            r"^(\d+)\. (.+)$", text.split("\n## References", 1)[-1], re.M
        )
    }
    numbers = re.findall(r"\[(\d+)\]", source)
    if not numbers or any(n not in refs for n in numbers):
        return source
    return " ".join(refs[n] for n in dict.fromkeys(numbers))


def build(guides: Path = GUIDES, datasets=DATASETS) -> list[dict]:
    rows = []
    for d in datasets:
        name, source = guide_facts(guides, d.guide)
        sizes = [measure(p) for p in d.location]
        size = None if any(s is None for s in sizes) else sum(sizes)
        rows.append(
            {
                "dataset": name,
                "benchmarks": d.benchmarks,
                "source": source,
                "size": human(size),
                "host": d.host,
                "download": f"doc/dataset_guide/{d.guide}#get-the-data",
                "guide": f"doc/dataset_guide/{d.guide}",
            }
        )
    return rows


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.",
        "License notice: Documentation text and diagrams on this page are licensed",
        "under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only",
        "unless the example states otherwise.",
        "-->",
        "",
        "# Datasets behind the paper's results",
        "",
        "<!-- generated by scripts/list_datasets.py; do not edit by hand -->",
        "",
        "One row per dataset the benchmarks use. The source is quoted from the "
        "dataset guide of the benchmark. Size is the bytes of the files the "
        "benchmark reads, in GiB, with `.zip` and `.tar.gz` copies left out.",
        "",
        "The last column names where the files come from and opens the "
        '"Get the data" section of the guide. The licences of the datasets are '
        "stated on the [Hugging Face repositories](https://huggingface.co/zenteiq-ai).",
        "",
        "| Dataset | Benchmarks | Size | Files from |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['benchmarks']} | {r['size']} | "
            f"[{r['host']}](../dataset_guide/{Path(r['download']).name}) |"
        )
    for r in rows:
        lines += [
            "",
            f"## {r['dataset']}",
            "",
            f"- **Source:** {r['source']}",
        ]
        lines.append(f"- **Dataset guide:** `{r['guide']}`")
    return "\n".join(lines) + "\n"


def keep_known_sizes(rows: list[dict[str, str]], committed: Path) -> None:
    """Where the data is not on this machine, keep the size the committed register has,
    so regenerating the register without the NAS does not erase it."""
    if not committed.is_file():
        return
    with open(committed, newline="") as f:
        known = {r["dataset"]: r["size"] for r in csv.DictReader(f)}
    for r in rows:
        if r["size"] == UNKNOWN and known.get(r["dataset"], UNKNOWN) != UNKNOWN:
            r["size"] = known[r["dataset"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--check", action="store_true", help="fail if the register is stale"
    )
    args = ap.parse_args(argv)

    rows = build()
    keep_known_sizes(rows, args.out_dir / CSV_NAME)
    out = {CSV_NAME: to_csv(rows), MD_NAME: to_markdown(rows)}
    if args.check:
        stale = [
            n
            for n, text in out.items()
            if not (args.out_dir / n).is_file()
            or (args.out_dir / n).read_text() != text
        ]
        for n in stale:
            print(f"STALE: {args.out_dir / n}")
        print(f"{len(rows)} datasets: " + ("up to date" if not stale else "STALE"))
        return 1 if stale else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, text in out.items():
        (args.out_dir / n).write_text(text)
    missing = [r["dataset"] for r in rows if r["size"] == UNKNOWN]
    for m in missing:
        print(f"NOT FOUND on the NAS: {m}")
    print(f"wrote {len(rows)} datasets to {args.out_dir}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
