#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Reproduce the paper's results table from the archived checkpoints.
#   scripts/reproduce.sh [extra reproduce.py args]   e.g. --only "Pipe,Cavity"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SCIREX_CKPT_ROOT:?set SCIREX_CKPT_ROOT to the folder with the downloaded checkpoints}"
: "${SCIREX_DATA_ROOT:?set SCIREX_DATA_ROOT to the folder with the datasets}"
export SCIREX_CKPT_ROOT SCIREX_DATA_ROOT
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
exec python "$HERE/reproduce.py" --mode infer --all --out-dir "${OUT_DIR:-reproduce_out}" "$@"
