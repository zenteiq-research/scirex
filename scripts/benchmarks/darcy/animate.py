# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Darcy training-progress GIF: prediction, ground truth and absolute error
on one fixed test case, one frame per captured epoch.

Renders the ``frames.npz`` that ``train.py --animate`` saves (the test case
with the median final error, at ``_config.ANIM_EPOCHS``). The published GIF
is of the best model, Transolver::

    python scripts/benchmarks/darcy/train.py --run transolver --animate
    python scripts/benchmarks/darcy/animate.py

Written to ``animations/darcy_flow/evolution.gif`` (a training diagnostic,
not part of the published doc assets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks._animate import rebuild_from_cache  # noqa: E402
from scripts.benchmarks.darcy._config import (  # noqa: E402
    DEFAULT_ANIM_OUT,
    DEFAULT_CKPT_ROOT,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frames",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "transolver" / "frames.npz",
        help="frames.npz saved by train.py --animate",
    )
    ap.add_argument("--model-label", default="Transolver")
    ap.add_argument("--out", type=Path, default=DEFAULT_ANIM_OUT)
    args = ap.parse_args()
    rebuild_from_cache(
        args.frames,
        args.out,
        title_template=f"Darcy Flow, Model: {args.model_label}, Epoch: {{epoch}}",
        field_fn=lambda f: f[..., 0],
        value_label="pressure u",
        error_label="|u error|",
    )


if __name__ == "__main__":
    main()
