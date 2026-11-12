# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Lid-driven cavity training-progress GIF: prediction, ground truth and
absolute error on one fixed test frame, one frame per captured epoch.

``train.py --animate`` writes ``animations/cavity/evolution.gif`` and caches
its frames next to it (``frames.npz``); this re-renders the GIF from that
cache, for a cosmetic change that needs no retraining. The published GIF is
of the FNO run::

    python scripts/benchmarks/cavity/train.py --run fno_3to1 --animate
    python scripts/benchmarks/cavity/animate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks._animate import rebuild_from_cache  # noqa: E402

DEFAULT_DIR = _REPO_ROOT / "animations" / "cavity"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", type=Path, default=DEFAULT_DIR / "frames.npz")
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR / "evolution.gif")
    args = ap.parse_args()
    rebuild_from_cache(args.frames, args.out, title="Lid-Driven Cavity")


if __name__ == "__main__":
    main()
