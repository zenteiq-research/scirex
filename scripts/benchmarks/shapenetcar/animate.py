# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Car-Design-ShapeNetCar training-progress GIF: predicted, true and
absolute-error surface pressure on one validation car, one frame per
captured epoch.

``train.py --animate`` writes ``animations/shapenetcar/evolution.gif`` and
caches its frames next to it (``frames.npz``); this re-renders the GIF from
that cache, for a cosmetic change that needs no retraining. The published GIF
is of the Transolver run::

    python scripts/benchmarks/shapenetcar/train.py --run transolver --animate
    python scripts/benchmarks/shapenetcar/animate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks._animate import rebuild_point_cloud_from_cache  # noqa: E402

DEFAULT_DIR = _REPO_ROOT / "animations" / "shapenetcar"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--frames", type=Path, default=DEFAULT_DIR / "frames.npz")
    ap.add_argument("--out", type=Path, default=DEFAULT_DIR / "evolution.gif")
    args = ap.parse_args()
    rebuild_point_cloud_from_cache(
        args.frames, args.out, title="Car-Design-ShapeNetCar"
    )


if __name__ == "__main__":
    main()
