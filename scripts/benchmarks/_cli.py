# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Flags shared by every benchmark ``infer.py``.

``--checkpoint DIR``  one run directory (holds ``params.msgpack`` + ``config.json``)
``--data-dir DIR``    the benchmark's dataset root
``--out DIR``         where figures and ``metrics.json`` are written

``--run NAME`` names the run when the directory name is not a key of
``_config.RUNS``. The older spellings (``--out-dir``, ``--out-root``, ``--data-root``)
still work as aliases, and ``--ckpt-root`` / ``--cache-dir`` are unchanged. Without
``--checkpoint`` a script evaluates every run it finds under ``--ckpt-root``, as it
always did.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def add_standard_args(
    ap: argparse.ArgumentParser,
    *,
    out_flag: str,
    out_default: Path,
    data_flag: str | None = None,
    data_default: Path | None = None,
    run_flag: bool = True,
) -> None:
    """Add ``--checkpoint``, ``--run`` and ``--out`` (plus ``--data-dir`` if missing).

    ``out_flag`` is the script's legacy output flag (``--out-dir`` or ``--out-root``);
    ``--out`` is an alias that fills the same ``args.<legacy>`` attribute. A script
    whose data flag is not ``--data-dir`` (car_cfd: ``--data-root``) passes it as
    ``data_flag`` and gets ``--data-dir`` as an alias for it. ``run_flag=False`` for a
    script that already defines its own ``--run``.
    """
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="evaluate only this run directory (default: every run under --ckpt-root)",
    )
    if run_flag:
        ap.add_argument(
            "--run",
            default=None,
            metavar="NAME",
            help="name from _config.RUNS of the --checkpoint run (default: its directory name)",
        )
    ap.add_argument(
        "--out",
        out_flag,
        dest=out_flag.lstrip("-").replace("-", "_"),
        type=Path,
        default=out_default,
        help=f"output directory (legacy spelling: {out_flag})",
    )
    if data_flag:
        ap.add_argument(
            data_flag,
            "--data-dir",
            dest=data_flag.lstrip("-").replace("-", "_"),
            type=Path,
            default=data_default,
            help=f"dataset root (legacy spelling: {data_flag})",
        )


def select_run(
    ap: argparse.ArgumentParser, args: argparse.Namespace, runs: Iterable[str]
) -> str | None:
    """Validate ``--checkpoint`` and return the run name to evaluate.

    Returns ``None`` when ``--checkpoint`` is absent (evaluate every run).
    """
    if args.checkpoint is None:
        return None
    names = list(runs)
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.is_dir():
        ap.error(f"--checkpoint {args.checkpoint} is not a directory")
    args.run = args.run or args.checkpoint.name
    if args.run not in names:
        ap.error(
            f"cannot tell which run {args.checkpoint.name!r} is; "
            f"pass --run with one of {names}"
        )
    return args.run


def checkpoint_dir(args: argparse.Namespace, name: str) -> Path:
    """Directory holding run ``name``: ``--checkpoint`` if it is that run, else under ``--ckpt-root``."""
    if args.checkpoint is not None and args.run == name:
        return args.checkpoint
    return args.ckpt_root / name
