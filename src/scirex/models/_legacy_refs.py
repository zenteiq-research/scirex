# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Module paths stored in checkpoints, translated after a module moves.

Checkpoint metadata stores each class or function reference (an activation passed as
an argument, for instance) as a module path and a qualified name. When a module moves,
the path in an existing checkpoint no longer imports. `remap_legacy_module`
translates it to the current location with the move table in
`scirex._internal.aliases`, so the checkpoint keeps loading. The same table keeps
the old import paths working.
"""

from __future__ import annotations

from scirex._internal.aliases import MOVED

# Old module path -> current location (longest package prefix wins).
MOVED_MODULES: dict[str, str] = MOVED


def _remap_prefix(table: dict[str, str], module_name: str) -> str:
    """Apply `table` to `module_name`: exact match, else longest package prefix."""
    if module_name in table:
        return table[module_name]
    parts = module_name.split(".")
    for n in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:n])
        if head in table:
            return ".".join([table[head], *parts[n:]])
    return module_name


def remap_legacy_module(module_name: str) -> str:
    """Return the current module path for a module path stored in a checkpoint.

    Exact matches win; otherwise the longest moved package prefix is applied. Other
    names are returned unchanged.
    """
    return _remap_prefix(MOVED_MODULES, module_name)
