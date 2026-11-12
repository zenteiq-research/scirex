# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The package-level exports of ``scirex.training`` and ``scirex.training.optim``."""

import importlib

import pytest

import scirex.training
from tests.public_api_helpers import (
    assert_all_is_well_formed,
    assert_exports_resolve,
    assert_lazy_table_matches,
    assert_run_ok,
)

_PACKAGES = ["scirex.training", "scirex.training.optim"]


def test_all_is_the_decided_surface() -> None:
    assert_all_is_well_formed(scirex.training)
    assert set(scirex.training.__all__) == {"adamw_cx"}
    optim = importlib.import_module("scirex.training.optim")
    assert_all_is_well_formed(optim)
    assert set(optim.__all__) == {"adamw_cx", "scale_by_adam_cx"}


@pytest.mark.parametrize("package", _PACKAGES)
def test_every_export_resolves_and_is_documented(package: str) -> None:
    assert_exports_resolve(importlib.import_module(package))


@pytest.mark.parametrize("package", _PACKAGES)
def test_lazy_table_matches_all(package: str) -> None:
    assert_lazy_table_matches(importlib.import_module(package))


def test_adamw_cx_is_the_same_object_in_both_packages() -> None:
    from scirex.training.optim import adamw_cx

    assert scirex.training.adamw_cx is adamw_cx


def test_deprecated_names_are_not_part_of_all() -> None:
    deprecated = set(scirex.training._DEPRECATED_REEXPORTS)
    assert deprecated
    assert deprecated.isdisjoint(scirex.training.__all__)
    assert deprecated.isdisjoint(dir(scirex.training))


def test_deprecation_shim_works_alongside_the_lazy_exports() -> None:
    with pytest.warns(DeprecationWarning, match="resolve_dtype"):
        deprecated = scirex.training.resolve_dtype
    assert callable(deprecated)

    assert callable(scirex.training.adamw_cx)


@pytest.mark.parametrize(
    "name",
    ["SimpleDataLoader", "Trainer", "load_training_state", "save_training_state"],
)
def test_names_that_moved_to_experimental_still_resolve_with_a_warning(
    name: str,
) -> None:
    import scirex.experimental.training as experimental

    assert name not in scirex.training.__all__
    with pytest.warns(DeprecationWarning, match="scirex.experimental.training"):
        obj = getattr(scirex.training, name)
    assert obj is getattr(experimental, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    for package in _PACKAGES:
        with pytest.raises(AttributeError, match="does_not_exist"):
            importlib.import_module(package).does_not_exist


@pytest.mark.parametrize("package", _PACKAGES)
def test_fresh_interpreter_imports_the_package_and_star_import_works(
    package: str,
) -> None:
    assert_run_ok(
        f"""
        import {package} as pkg
        namespace = {{}}
        exec("from {package} import *", namespace)
        missing = set(pkg.__all__) - set(namespace)
        assert not missing, missing
        """
    )


def test_import_does_not_load_matplotlib_wandb_or_open3d() -> None:
    assert_run_ok(
        """
        import sys
        import scirex.training
        import scirex.training.optim
        heavy = ["matplotlib", "wandb", "open3d", "jax", "optax", "flax", "tensorly"]
        loaded = [m for m in heavy if m in sys.modules]
        assert not loaded, loaded
        """
    )


def test_deprecated_name_in_fresh_interpreter_warns_and_resolves() -> None:
    assert_run_ok(
        """
        import warnings
        import scirex.training
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            scirex.training.resolve_dtype
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        """
    )
