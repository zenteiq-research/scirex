<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# scripts

## Reproducing the paper tables

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[cuda]"
scripts/reproduce.sh                        # every cell of the paper table, inference only
scripts/reproduce.sh --only "Pipe,Cavity"   # selected rows or models
python scripts/reproduce.py --list          # show the cell-to-checkpoint map
```

Outputs go to `reproduce_out/`: `results.csv`, `results_table.csv`, `results_table.tex`.
Set `SCIREX_CKPT_ROOT` to the folder with the downloaded checkpoints and `SCIREX_DATA_ROOT` to the
folder with the datasets (there is no default location). Full documentation:
[doc/benchmarks/reproducibility.md](../doc/benchmarks/reproducibility.md).
