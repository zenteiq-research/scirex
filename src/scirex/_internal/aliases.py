# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Old import paths of modules that moved, kept working.

When a module moves, its old dotted path stays importable: importing it gives the *same
module object* as the new path (so `isinstance` checks, monkeypatching and private
names behave identically) and emits a `DeprecationWarning` that names the new path and
the release that removes the old one. Submodules follow their package, so
`scirex.distributed.comm` resolves to `scirex.experimental.distributed.comm`.

The table `MOVED` is the single list of moves. It also feeds
`scirex.models._legacy_refs`, which translates the same paths when a checkpoint is
loaded. Add a line here in the same change that moves a module.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType

from scirex._internal.deprecation import deprecation_message, warn_deprecated

#: release in which the old paths were deprecated, and the earliest that may drop them
SINCE = "1.0.0"
REMOVAL = "1.2.0"

#: old dotted module path -> where the module lives now
MOVED: dict[str, str] = {
    # not yet covered by the stability promise
    "scirex.distributed": "scirex.experimental.distributed",
    "scirex.geometry": "scirex.experimental.geometry",
    "scirex.models.giwo": "scirex.experimental.giwo",
    "scirex.training.patching": "scirex.experimental.patching",
    "scirex.tuning": "scirex.experimental.tuning",
    "scirex.losses.ad_differentiation": "scirex.experimental.losses.ad_differentiation",
    "scirex.losses.ad_pde_terms": "scirex.experimental.losses.ad_pde_terms",
    "scirex.losses.exact_ad_differentiation": "scirex.experimental.losses.exact_ad_differentiation",
    "scirex.losses.patch_quadrature_ad": "scirex.experimental.losses.patch_quadrature_ad",
    "scirex.losses.quadrature": "scirex.experimental.losses.quadrature",
    "scirex.losses.domain_membership": "scirex.experimental.losses.domain_membership",
    "scirex.losses.galerkin_t6": "scirex.experimental.losses.galerkin_t6",
    "scirex.losses.darcy_hybrid_loss": "scirex.experimental.losses.darcy_hybrid_loss",
    "scirex.config.distributed": "scirex.experimental.distributed.config",
    # private helpers
    "scirex.data.web_utils": "scirex._internal.web_utils",
    "scirex.layers.common.einsum_utils": "scirex._internal.einsum_utils",
    "scirex.layers.common.segment_csr": "scirex._internal.segment_csr",
    "scirex.training.trainer": "scirex.experimental.training.trainer",
    "scirex.training.training_state": "scirex.experimental.training.training_state",
    "scirex.training.torch_setup": "scirex._internal.torch_setup",
    "scirex.training.wno_train_utils": "scirex._internal.wno_train_utils",
}


def moved_message(old: str, new: str) -> str:
    """The text of the `DeprecationWarning` for the import of `old`."""
    return deprecation_message(old, since=SINCE, removal=REMOVAL, alternative=new)


class _AliasLoader(importlib.abc.Loader):
    """Hands out the module that lives at the new path instead of loading a copy."""

    def __init__(self, old: str, new: str) -> None:
        self.old, self.new = old, new
        self._real_spec: ModuleSpec | None = None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.new)
        self._real_spec = module.__spec__
        # stacklevel 3: this frame, warn_deprecated's, then the importing line (the
        # import machinery's own frames are skipped)
        warn_deprecated(
            self.old,
            since=SINCE,
            removal=REMOVAL,
            alternative=self.new,
            stacklevel=3,
        )
        return module

    def exec_module(self, module: ModuleType) -> None:
        # the import system stamped the alias spec on the shared module; put the real
        # one back so `module.__spec__.name` still agrees with `module.__name__`
        if self._real_spec is not None:
            module.__spec__ = self._real_spec


class AliasFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder for the keys of a move table (and their submodules)."""

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = MOVED if table is None else table

    def target(self, fullname: str) -> str | None:
        """The new dotted path for `fullname`, or `None` if it was not moved."""
        for old, new in self.table.items():
            if fullname == old:
                return new
            if fullname.startswith(old + "."):
                return new + fullname[len(old) :]
        return None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        new = self.target(fullname)
        if new is None:
            return None
        try:
            real = importlib.util.find_spec(new)
        except (ImportError, ValueError):
            return None
        if real is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, new),
            is_package=real.submodule_search_locations is not None,
        )


def install(table: dict[str, str] | None = None) -> AliasFinder:
    """Put an `AliasFinder` first on `sys.meta_path` (once) and return it.

    It must come first: the ordinary path finder would otherwise find the submodules of
    a moved package inside its new directory and load them a second time under the old
    name.
    """
    wanted = MOVED if table is None else table
    for finder in sys.meta_path:
        if isinstance(finder, AliasFinder) and finder.table is wanted:
            return finder
    finder = AliasFinder(wanted)
    sys.meta_path.insert(0, finder)
    return finder
