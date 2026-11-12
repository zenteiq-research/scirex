# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reproduce the results table from archived checkpoints.

    python scripts/reproduce.py --mode infer --all
    python scripts/reproduce.py --mode infer --only "Pipe,Cavity, 1 step"

Each cell of the results table is evaluated with the benchmark's own ``infer.py`` on the
archived weights. Writes ``results.csv`` (one row per cell), ``results_table.csv`` and
``results_table.tex`` (the table, one column per model) to ``--out-dir``. The numbers are the
ones measured from the weights; nothing is compared with another table. A cell without a
checkpoint, a runner, or a readable result is reported as MISSING and the run continues.

Every metric of the table is an error, so lower is better. The exit status is non-zero only
with ``--strict-missing`` when a cell is MISSING.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reproduce_paper_map import (
    CELLS,
    MODELS,
    PROBLEMS,
    Cell,
    hf_path,
    problem_of,
    rows,
)

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "scripts" / "benchmarks"
# No default location: set SCIREX_CKPT_ROOT and SCIREX_DATA_ROOT or pass --ckpt-root / --data-root.
DEFAULT_CKPT = None
DEFAULT_DATA = None

# Folder of each dataset under the data root (see doc/benchmarks/reproducibility.md).
DATA_SUBDIRS = {
    "pipe": "pipe",
    "navier_stokes": "navier_stokes",
    "airfrans": "airfrans/Dataset",
    "chip_cooling": "chip_cooling",
    "car_cfd": "car_cfd/processed-car-pressure-data",
    "shapenetcar": "shapenetcar/training_data",
    "cavity": "cavity",
}

OK, MISSING = "OK", "MISSING"
STATUSES = (OK, MISSING)


def _summary(n: dict[str, int]) -> str:
    return ", ".join(f"{n[s]} {s}" for s in STATUSES)


class Unavailable(Exception):
    """A cell cannot be evaluated; the message is shown in the report."""


def dig(d: dict, key: str):
    for part in key.split("."):
        d = d[part]
    return d


# ------------------------------------------------------------------- locations
def _need(value, what: str, env: str, flag: str) -> Path:
    if not value:
        raise Unavailable(f"{what} is not set: export {env} or pass {flag}")
    return Path(value)


def ckpt_root_of(a: argparse.Namespace) -> Path:
    return _need(
        a.ckpt_root, "the checkpoint folder", "SCIREX_CKPT_ROOT", "--ckpt-root"
    )


def data_root_of(a: argparse.Namespace) -> Path:
    return _need(a.data_root, "the data folder", "SCIREX_DATA_ROOT", "--data-root")


def ckpt_dir(a: argparse.Namespace, c: Cell) -> Path:
    """Folder of the cell's checkpoint: Hugging Face layout (default) or the original archive layout."""
    layout = getattr(a, "layout", "hf")
    return ckpt_root_of(a) / (hf_path(c) if layout == "hf" else c.ckpt)


# ------------------------------------------------------------------- runners
def _call(script: Path, args: list[str]) -> str:
    cmd = [sys.executable, str(script), *args]
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        tail = [ln for ln in r.stderr.splitlines() if ln.strip()][-3:]
        raise Unavailable(
            f"{script.parent.name}/infer.py exit {r.returncode}: {' | '.join(tail)[:400]}"
        )
    return r.stdout


def _read(path: Path) -> dict:
    if not path.exists():
        raise Unavailable(f"{path.name} not written")
    return json.loads(path.read_text())


def _run_infer(
    group: str, cells: list[Cell], a: argparse.Namespace
) -> dict[Cell, float]:
    """One ``infer.py --checkpoint`` call per distinct run of one benchmark."""
    cells = [c for c in cells if c.ckpt and ckpt_dir(a, c).exists()]
    if not cells:
        raise Unavailable("no checkpoint found")
    data_root = data_root_of(a)
    datadirs = {g: str(data_root / sub) for g, sub in DATA_SUBDIRS.items()}
    for g, given in (
        ("pipe", a.pipe_data),
        ("navier_stokes", a.ns_data),
        ("airfrans", a.airfrans_data),
        ("chip_cooling", a.chip_data),
    ):
        if given:
            datadirs[g] = given
    res: dict[Cell, float] = {}
    errors: list[str] = []
    for ckpt, run in sorted({(c.ckpt, c.run) for c in cells}):
        mine = [c for c in cells if (c.ckpt, c.run) == (ckpt, run)]
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            flags = [
                "--checkpoint",
                str(ckpt_dir(a, mine[0])),
                "--run",
                run,
                "--out",
                str(out),
            ]
            if a.device == "cpu":
                flags.append("--allow-cpu")
            if group == "cavity":  # reads the prepared cache, not the raw data
                flags += [
                    "--data-dir",
                    datadirs["cavity"],
                    "--cache-dir",
                    str(Path(datadirs["cavity"]) / "cache"),
                ]
            else:
                flags += ["--data-dir", datadirs[group]]
            if group == "car_cfd":
                flags += ["--no-download", "--no-vtk"]
            try:
                _call(BENCH / group / "infer.py", flags)
            except Unavailable as e:
                print(f"  run {run}: {e}", flush=True)
                errors.append(str(e))
                continue
            for c in mine:
                m = None
                try:
                    path = out / "none"
                    if group == "car_cfd":
                        path = out / run / "metrics.json"
                    else:
                        path = next(
                            iter(out.glob(f"*{c.model.lower()}*/metrics.json")), path
                        )
                    m = _read(path)
                    if group == "car_cfd":
                        val = m["metrics"]
                    elif group == "chip_cooling":
                        val = m["test"]
                    elif "runs" in m and c.run in m["runs"]:
                        val = m["runs"][c.run]
                    else:
                        val = m
                    res[c] = float(dig(val, c.key)) * c.scale
                except (Unavailable, KeyError, TypeError) as e:
                    keys = sorted(m) if isinstance(m, dict) else "?"
                    print(
                        f"  {c.row} / {c.model}: result {c.key!r} not readable "
                        f"({type(e).__name__}: {e}); keys: {keys}",
                        flush=True,
                    )
    if not res:
        raise Unavailable("; ".join(sorted(set(errors))) or "no result readable")
    return res


def run_infer_group(
    group: str, cells: list[Cell], a: argparse.Namespace
) -> dict[Cell, float]:
    """Evaluate a benchmark's cells (each run is its own infer.py call)."""
    return _run_infer(group, cells, a)


def run_unsupported(why: str):
    def _r(group, cells, a):
        raise Unavailable(why)

    return _r


def run_darcy_archived(
    group: str, cells: list[Cell], a: argparse.Namespace
) -> dict[Cell, float]:
    """Read the published metrics.json stored next to each Darcy checkpoint.

    Opt-in (``--allow-archived-metrics``). These are NOT re-evaluated here: Darcy
    checkpoints carry no config.json and need the pinned training commits, which
    are not in this repository. ``SciREX_checkpoints/VERIFICATION.md`` records that
    28/28 were re-evaluated within 1e-5 of these numbers.
    """
    if not a.allow_archived_metrics:
        raise Unavailable(
            "These checkpoints cannot be re-evaluated by infer.py (Darcy: no config.json, pinned commits not in the repo; chip Transolver: different normalisation). "
            "Pass --allow-archived-metrics to read the published metrics.json instead (not re-evaluated)."
        )
    res = {}
    for c in cells:
        f = ckpt_dir(a, c) / "metrics.json"
        if f.exists():
            res[c] = float(json.loads(f.read_text())[c.key]) * c.scale
    if not res:
        raise Unavailable("no Darcy metrics.json found")
    return res


def run_airfrans_gino(
    group: str, cells: list[Cell], a: argparse.Namespace
) -> dict[Cell, float]:
    """Score the published AirfRANS GINO with the arguments of its original evaluation.

    ``airfrans/infer.py`` evaluates GINO with ``GINO_ARGS``, which includes
    ``--loader-surf-mask``. The published run did not use that flag, so ``_gino.py`` is
    called directly with the published settings.
    """
    ck = ckpt_dir(a, cells[0])
    if not (ck / "params_final.msgpack").exists():
        raise Unavailable(f"no params_final.msgpack under {ck}")
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "out"
        flags = [
            "--data-dir",
            a.airfrans_data or str(data_root_of(a) / DATA_SUBDIRS["airfrans"]),
            "--cache-dir",
            str(Path(t) / "cache"),
            "--output-dir",
            str(out),
            "--task",
            "full",
            "--epochs",
            "500",
            "--latent-sdf",
            "--decoder-point-features",
            "--latent-res",
            "128",
            "64",
            "--out-gno-mlp",
            "512",
            "362",
            "--eval-only",
            str(ck / "params_final.msgpack"),
            "--history",
            str(ck / "loss_curve.csv"),
        ]
        _call(BENCH / "airfrans" / "_gino.py", flags)
        m = _read(out / "metrics.json")
    try:
        return {c: float(m[c.key]) * c.scale for c in cells}
    except KeyError as e:
        raise Unavailable(f"key {e} not in GINO metrics.json; keys: {sorted(m)}")


RUNNERS = {
    "pipe": run_infer_group,
    "navier_stokes": run_infer_group,
    "airfrans": run_infer_group,
    "airfrans_gino": run_airfrans_gino,
    "cavity": run_infer_group,
    "shapenetcar": run_infer_group,
    "car_cfd": run_infer_group,
    "chip_cooling": run_infer_group,
    "darcy": run_darcy_archived,
    "archived": run_darcy_archived,
}


# -------------------------------------------------------------------- output
def fmt(v, sig=4):
    return "n/a" if v is None else f"{v:.{sig}g}"


def write_outputs(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["row"], r["model"]): r for r in results}
    metric = {r["row"]: (r["metric"], r["unit"]) for r in results}
    present = [r for r in rows() if r in metric]

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "model", "unit", "ours", "status", "note"])
        for r in results:
            w.writerow(
                [
                    r["row"],
                    r["model"],
                    r["unit"],
                    fmt(r["ours"], 6),
                    r["status"],
                    r["note"],
                ]
            )

    with open(out_dir / "results_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Problem", "Metric", *MODELS])
        for row in present:
            w.writerow(
                [
                    row,
                    "{} ({})".format(*metric[row]),
                    *[
                        fmt(by[(row, m)]["ours"]) if (row, m) in by else "-"
                        for m in MODELS
                    ],
                ]
            )

    lines = [
        r"\begin{tabular}{ll" + "r" * len(MODELS) + "}",
        r"\hline",
        "Problem & Metric & " + " & ".join(MODELS) + r" \\",
        r"\hline",
    ]
    for row in present:
        vals = {
            m: by[(row, m)]["ours"]
            for m in MODELS
            if (row, m) in by and by[(row, m)]["ours"] is not None
        }
        best = min(vals.values()) if vals else None
        cells = []
        for m in MODELS:
            v = vals.get(m)
            s = "--" if (row, m) not in by else ("n/a" if v is None else fmt(v))
            cells.append(rf"\textbf{{{s}}}" if v is not None and v == best else s)
        mt = "{} ({})".format(*metric[row]).replace("%", r"\%").replace("_", r"\_")
        lines.append(f"{row.replace('_', ' ')} & {mt} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out_dir / "results_table.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- main
def select(
    only: str | None,
    skip: str | None = None,
    model: str | None = None,
    problem: str | None = None,
) -> list[Cell]:
    """Cells matching ``only``/``skip`` (comma lists, OR within each) and, combined with AND,
    the exact ``model`` (``fno``) and Hugging Face ``problem`` folder (``pipe``)."""

    def hit(c: Cell, spec: str) -> bool:
        return any(
            k in c.row.lower() or k == c.model.lower()
            for k in (x.strip().lower() for x in spec.split(",") if x.strip())
        )

    return [
        c
        for c in CELLS
        if (not only or hit(c, only))
        and not (skip and hit(c, skip))
        and (not model or c.model.lower() == model.lower())
        and (not problem or problem_of(c) == problem)
    ]


def evaluate(cells: list[Cell], a: argparse.Namespace) -> list[dict]:
    groups: dict[str, list[Cell]] = {}
    for c in cells:
        groups.setdefault(c.group, []).append(c)
    values: dict[Cell, float] = {}
    why: dict[Cell, str] = {}
    for g, gc in groups.items():
        print(f"[{g}] {len(gc)} cell(s)", flush=True)
        try:
            values.update(RUNNERS[g](g, gc, a))
        except Unavailable as e:
            for c in gc:
                why[c] = str(e)
            print(f"  MISSING: {e}", flush=True)
        for c in gc:
            why.setdefault(c, "no result from infer.py (see log above)")
    out = []
    for c in cells:
        r = dict(
            row=c.row,
            model=c.model,
            metric=c.metric,
            unit=c.unit,
            ours=values.get(c),
            note="",
        )
        if c in values:
            r["status"] = OK
            if c.group in ("darcy", "archived"):
                r["note"] = "published metrics.json, not re-evaluated"
        else:
            r["status"], r["note"] = MISSING, why.get(c, "not evaluated")
        out.append(r)
    return out


def merge(dirs: list[str]) -> list[dict]:
    """Combine the results.csv files of several runs (later ones win per cell)."""
    got: dict[tuple[str, str], dict] = {}
    for d in dirs:
        with open(Path(d) / "results.csv", newline="") as f:
            for r in csv.DictReader(f):
                num = lambda k: None if r[k] in ("", "n/a") else float(r[k])  # noqa: E731
                got[(r["row"], r["model"])] = dict(
                    row=r["row"],
                    model=r["model"],
                    unit=r["unit"],
                    ours=num("ours"),
                    status=r["status"],
                    note=r["note"],
                )
    cells = {(c.row, c.model): c for c in CELLS}
    out = []
    for key, r in got.items():
        r["metric"] = cells[key].metric
        out.append(r)
    order = {k: i for i, k in enumerate(cells)}
    return sorted(out, key=lambda r: order[(r["row"], r["model"])])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--table", type=int, help=argparse.SUPPRESS)  # accepted, ignored
    ap.add_argument("--mode", choices=["infer", "train"], default="infer")
    ap.add_argument(
        "--all",
        action="store_true",
        help="evaluate every cell (default if --only is not given)",
    )
    ap.add_argument("--only", help="comma list of row substrings or model names")
    ap.add_argument(
        "--skip", help="comma list of row substrings or model names to leave out"
    )
    ap.add_argument(
        "--model",
        choices=[m.lower() for m in MODELS],
        help="only this model (combines with --problem and --only by AND)",
    )
    ap.add_argument(
        "--problem",
        choices=PROBLEMS,
        help="only this problem, named as its Hugging Face folder (combines with --model by AND)",
    )
    ap.add_argument(
        "--layout",
        choices=["hf", "archive"],
        default="hf",
        help="checkpoint folder layout under --ckpt-root: the Hugging Face repository "
        "(<model>/<problem>, default) or the original archive",
    )
    ap.add_argument(
        "--ckpt-root", default=os.environ.get("SCIREX_CKPT_ROOT", DEFAULT_CKPT)
    )
    ap.add_argument(
        "--data-root", default=os.environ.get("SCIREX_DATA_ROOT", DEFAULT_DATA)
    )
    ap.add_argument("--pipe-data"), ap.add_argument("--ns-data")
    ap.add_argument("--airfrans-data"), ap.add_argument("--chip-data")
    ap.add_argument("--out-dir", default="reproduce_out")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument(
        "--allow-archived-metrics",
        action="store_true",
        help="Darcy: read the published metrics.json instead of re-evaluating (labelled in results.csv)",
    )
    ap.add_argument(
        "--strict-missing",
        action="store_true",
        help="exit non-zero if any cell is MISSING",
    )
    ap.add_argument(
        "--merge",
        nargs="+",
        metavar="DIR",
        help="combine results.csv from earlier runs into --out-dir and exit",
    )
    ap.add_argument("--list", action="store_true", help="print the cell map and exit")
    a = ap.parse_args(argv)

    if a.merge:
        results = merge(a.merge)
        write_outputs(results, Path(a.out_dir))
        n = {k: sum(r["status"] == k for r in results) for k in STATUSES}
        print(f"merged {len(results)} cells: {_summary(n)} -> {a.out_dir}")
        return 1 if a.strict_missing and n[MISSING] else 0
    cells = select(a.only, a.skip, a.model, a.problem)
    if a.list:
        for c in cells:
            where = (hf_path(c) if a.layout == "hf" else c.ckpt) or "-"
            print(f"{c.row:34s} {c.model:11s} {where:60s} {c.run or '-'}")
        return 0
    if a.mode == "train":
        sys.exit(
            "train mode is not implemented yet. Each cell's training command is "
            "scripts/benchmarks/<problem>/train.py; see doc/benchmarks/reproducibility.md."
        )
    if not cells:
        sys.exit("no cell matches --only/--skip/--model/--problem; see --list")
    results = evaluate(cells, a)
    write_outputs(results, Path(a.out_dir))
    n = {s: sum(r["status"] == s for r in results) for s in STATUSES}
    print(f"\n{_summary(n)} -> {a.out_dir}")
    return 1 if a.strict_missing and n[MISSING] else 0


if __name__ == "__main__":
    raise SystemExit(main())
