# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The package-level exports of ``scirex.data`` and its sub-packages."""

import importlib
import sys
from pathlib import Path

import pytest

import scirex.data
from tests.public_api_helpers import (
    assert_all_is_well_formed,
    assert_exports_resolve,
    assert_lazy_table_matches,
    assert_run_ok,
)

_PACKAGES = [
    "scirex.data",
    "scirex.data.containers",
    "scirex.data.transforms",
    "scirex.data.datasets",
]

_EXPECTED = {
    "scirex.data": {
        "DefaultDataProcessor",
        "DictDataset",
        "TensorDataset",
        "UnitGaussianNormalizer",
        "load_airfrans",
        "load_cavity_kepsilon",
        "load_darcy_flow_small",
        "load_darcy_pt",
        "load_navier_stokes_pt",
        "load_shapenet_car",
    },
    "scirex.data.containers": {"DictDataset", "TensorDataset"},
    "scirex.data.transforms": {"DefaultDataProcessor", "UnitGaussianNormalizer"},
    "scirex.data.datasets": {
        "load_airfrans",
        "load_cavity_kepsilon",
        "load_darcy_flow_small",
        "load_darcy_pt",
        "load_navier_stokes_pt",
        "load_shapenet_car",
    },
}


@pytest.mark.parametrize("package", _PACKAGES)
def test_all_is_the_decided_surface(package: str) -> None:
    module = importlib.import_module(package)
    assert_all_is_well_formed(module)
    assert set(module.__all__) == _EXPECTED[package]


@pytest.mark.parametrize("package", _PACKAGES)
def test_every_export_resolves_and_is_documented(package: str) -> None:
    assert_exports_resolve(importlib.import_module(package))


@pytest.mark.parametrize("package", _PACKAGES)
def test_lazy_table_matches_all(package: str) -> None:
    assert_lazy_table_matches(importlib.import_module(package))


def test_subpackages_export_the_same_objects_as_the_facade() -> None:
    for package in _PACKAGES[1:]:
        module = importlib.import_module(package)
        for name in module.__all__:
            assert getattr(module, name) is getattr(scirex.data, name)


def test_unknown_attribute_raises_attribute_error() -> None:
    for package in _PACKAGES:
        with pytest.raises(AttributeError, match="does_not_exist"):
            importlib.import_module(package).does_not_exist


def test_deep_module_paths_keep_working() -> None:
    from scirex.data.containers.tensor_dataset import TensorDataset
    from scirex.data.datasets.darcy import load_darcy_flow_small
    from scirex.data.transforms.normalizers import UnitGaussianNormalizer

    assert scirex.data.TensorDataset is TensorDataset
    assert scirex.data.load_darcy_flow_small is load_darcy_flow_small
    assert scirex.data.UnitGaussianNormalizer is UnitGaussianNormalizer


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


def test_import_does_not_load_heavy_or_optional_dependencies() -> None:
    assert_run_ok(
        """
        import sys
        import scirex.data
        import scirex.data.containers, scirex.data.transforms, scirex.data.datasets
        heavy = ["jax", "flax", "sklearn", "scipy", "requests", "open3d", "vtk",
                 "matplotlib", "wandb"]
        loaded = [m for m in heavy if m in sys.modules]
        assert not loaded, loaded
        """
    )


def test_accessing_a_name_imports_only_its_module() -> None:
    assert_run_ok(
        """
        import sys
        from scirex.data import load_darcy_flow_small
        assert "scirex.data.datasets.darcy" in sys.modules
        assert "scirex.data.datasets.airfrans" not in sys.modules
        assert "scirex.data.datasets.shapenet_car" not in sys.modules
        assert "sklearn" not in sys.modules
        """
    )


def test_vtk_loader_without_vtk_names_the_benchmark_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A raw sample whose VTK files exist but cannot be parsed without vtk.
    for fold in range(9):
        sample = tmp_path / f"param{fold}" / "sample0"
        sample.mkdir(parents=True)
        (sample / "quadpress_smpl.vtk").write_bytes(b"")
        (sample / "hexvelo_smpl.vtk").write_bytes(b"")
    monkeypatch.setitem(sys.modules, "vtk", None)  # makes `import vtk` fail

    with pytest.raises(ImportError, match=r"scirex\[benchmark\]") as info:
        scirex.data.load_shapenet_car(tmp_path)
    assert "'vtk'" in str(info.value)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_vtk_loaders_are_wrapped_so_the_hint_covers_the_whole_call() -> None:
    assert hasattr(scirex.data.load_airfrans, "__wrapped__")
    assert hasattr(scirex.data.load_shapenet_car, "__wrapped__")
