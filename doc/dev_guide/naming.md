<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# Naming Rules

| What | Rule | Examples |
|---|---|---|
| Modules and packages | `snake_case` | `scirex.layers.blocks.fno_block`, `scirex.data.datasets.darcy` |
| Functions | `snake_case` | [`get_model_jax`](../api/models.md#scirex.models.get_model_jax), `cgelu` |
| Classes | `CapWords` | `TransolverBlock`, [`FNOSmall2d`](../api/config.md#scirex.config.FNOSmall2d), [`TransolverPipe`](../api/config.md#scirex.config.TransolverPipe) |
| Model names | As written in the paper | [`FNO`](../api/models.md#scirex.models.FNO), [`GINO`](../api/models.md#scirex.models.GINO), [`WNO`](../api/models.md#scirex.models.WNO), [`GIWO`](../api/experimental.md#scirex.experimental.giwo.GIWO), [`Transolver`](../api/models.md#scirex.models.Transolver) |
| Config keys | `snake_case` | `scheduler_t_max`, `height`, `width`, `depth` |
| Benchmark folders | `snake_case` | [`scripts/benchmarks/darcy`](../../scripts/benchmarks/darcy), [`scripts/benchmarks/chip_cooling`](../../scripts/benchmarks/chip_cooling) |

## Notes

- **Models.** A model class keeps the name used in its publication, including
  acronyms ([`FNO`](../api/models.md#scirex.models.FNO), not `Fno`). The model registry looks names up case-insensitively,
  so `fno` and [`FNO`](../api/models.md#scirex.models.FNO) select the same model.
- **Config presets** are classes, so they follow `CapWords` and start with the model
  name followed by the variant: [`FNOSmall2d`](../api/config.md#scirex.config.FNOSmall2d), [`GINODarcy2d`](../api/config.md#scirex.config.GINODarcy2d), [`TransolverPipeOfficial`](../api/config.md#scirex.config.TransolverPipeOfficial).
  No underscores.
- **Config keys** are the attribute names of config classes and the keys of YAML or
  command-line settings. This applies to every key, including sizes such as `height`
  and `width`.
- **Checkpoints.** Model constructor argument names are stored inside checkpoint
  metadata. Renaming a model argument breaks loading older checkpoints, so a config
  key and the model argument it feeds may differ; the forwarding code maps one to the
  other.

## Checking the rules

```bash
python scripts/dev/check_naming.py --strict
```

The script checks module, function, class and config-key names in [`src/scirex`](../../src/scirex) and the
folders under [`scripts/benchmarks`](../../scripts/benchmarks), and exits with 1 when a name breaks a rule above.
