# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The package-level exports of ``scirex.config`` and preset selection by name."""

import pytest

import scirex.config
from scirex.config import ModelConfig, OptimizationConfig
from tests.public_api_helpers import (
    assert_all_is_well_formed,
    assert_exports_resolve,
    assert_run_ok,
)

_MODEL_PRESETS = [
    "FNOHuge2d",
    "FNOLarge2d",
    "FNOMedium2d",
    "FNOMedium3d",
    "FNOSmall2d",
    "GINODarcy2d",
    "GINOPoisson2d",
    "GINOSmall3d",
    "TransolverAirfRANS",
    "TransolverDarcy2dAligned",
    "TransolverKappaOperator2d",
    "TransolverLaplace2d",
    "TransolverPipe",
    "TransolverPipeOfficial",
    "TransolverShapeNetCar",
]


def test_all_is_the_decided_surface() -> None:
    assert_all_is_well_formed(scirex.config)
    assert len(scirex.config.__all__) == 22
    assert set(_MODEL_PRESETS) < set(scirex.config.__all__)
    assert {
        "FNOConfig",
        "GINOConfig",
        "ModelConfig",
        "OptimizationConfig",
        "PatchingConfig",
        "SimpleFNOConfig",
        "TransolverConfig",
    } < set(scirex.config.__all__)


def test_every_export_resolves_and_is_documented() -> None:
    assert_exports_resolve(scirex.config)


def test_experiment_configs_are_not_exported() -> None:
    for name in ("Default", "CarCFDDatasetConfig", "CarCFDOptConfig"):
        assert name not in scirex.config.__all__
        assert not hasattr(scirex.config, name)


@pytest.mark.parametrize("preset", _MODEL_PRESETS)
def test_preset_is_selectable_by_name(preset: str) -> None:
    selected = ModelConfig(_config_name=preset)
    assert type(selected) is getattr(scirex.config, preset)
    # zencfg names are case-insensitive.
    assert type(ModelConfig(_config_name=preset.lower())) is type(selected)


def test_unknown_preset_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown subclass"):
        ModelConfig(_config_name="NoSuchPreset")


def test_experiment_config_classes_are_registered_by_importing_the_package() -> None:
    assert "transolverpipeoptconfig" in OptimizationConfig._registry


def test_the_two_default_classes_do_not_collide() -> None:
    from scirex.config import gino_carcfd_config, transolver_pipe_config

    gino, pipe = gino_carcfd_config.Default, transolver_pipe_config.Default
    assert gino is not pipe
    # Each is a root ConfigBase subclass with its own registry, so neither
    # overwrites the other under the name "default".
    assert gino._registry is not pipe._registry
    assert gino().to_dict()["model"]["model_arch"] == "gino"
    assert pipe().to_dict()["model"]["model_arch"] == "transolver"


def test_presets_are_selectable_in_a_fresh_interpreter_after_import() -> None:
    assert_run_ok(
        f"""
        import scirex.config
        from scirex.config import ModelConfig
        # Resolved through the base class only: this works only if the defining
        # module registered its classes during `import scirex.config`.
        for name in {_MODEL_PRESETS!r}:
            assert type(ModelConfig(_config_name=name)).__name__ == name, name
        """
    )


def test_import_is_eager_not_lazy() -> None:
    assert_run_ok(
        """
        import sys
        import scirex.config
        assert "scirex.config.models" in sys.modules
        assert "scirex.config.opt" in sys.modules
        assert "scirex.experimental.distributed.config" in sys.modules
        assert "__getattr__" not in vars(scirex.config)
        for heavy in ("jax", "flax", "matplotlib", "wandb"):
            assert heavy not in sys.modules, heavy
        """
    )
