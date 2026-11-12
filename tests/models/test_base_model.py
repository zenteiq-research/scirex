# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for BaseModel and the model registry (JAX/Flax).

neuraloperator has no dedicated ``test_base_model.py``; the upstream contract
for BaseModel is exercised indirectly through a ``DummyModel(BaseModel,
name="Dummy")`` mock in ``neuralop/tests/test_utils.py`` and the checkpoint
round-trip in ``neuralop/training/tests/test_trainer.py``. This file reproduces
that ``DummyModel`` idiom for the Flax port and covers the behaviours BaseModel
is responsible for:

* subclass registration / naming (``__init_subclass__`` + ``available_models``),
* init-kwarg capture for save/load (``__new__`` -> ``_init_kwargs``),
* params (de)serialization (``get_state_dict`` / ``load_state_dict``),
* on-disk checkpointing (``save_checkpoint`` / ``load_checkpoint`` /
  ``from_checkpoint``) including the version-mismatch warning, and
* config-driven instantiation (``get_model_jax``).

Following the Phase 1 ``layers/tests`` structure, tests that touch device
placement (forward pass and params (de)serialization) are parametrized over the
available CPU/GPU backends; the pure registry/metadata tests are device
independent.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from scirex._internal.serialization import load_msgpack, save_msgpack
from scirex.models.base import (
    BaseModel,
    _decode_init_kwargs,
    _encode_init_kwargs,
    available_models,
    get_model_jax,
)


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# Module-level mock models (registered with BaseModel at import time, mirroring
# the upstream DummyModel(BaseModel, name="Dummy") pattern).
# ---------------------------------------------------------------------------


class DummyModel(BaseModel, name="DummyBaseTest"):
    """Minimal linear model mocking the model API (parallels upstream DummyModel)."""

    features: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.features)(x)


class UnnamedModel(BaseModel):
    """A BaseModel subclass declared without an explicit ``name=``."""

    features: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.features)(x)


class CfgModel(BaseModel, name="CfgBaseTest"):
    """Model whose signature accepts ``in_channels`` for get_model_jax tests."""

    in_channels: int = 1
    out_channels: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.out_channels)(x)


class NoChannelsModel(BaseModel, name="NoChannelsBaseTest"):
    """Model whose signature has neither in_channels nor out_channels, mocking
    Transolver's differently-named space_dim/fun_dim/out_dim parameterization
    for get_model_jax tests."""

    hidden: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.hidden)(x)


class _Config(dict):
    """Bunch-like config: a dict (for ``config['patching']``) with a ``.model``."""

    def __init__(self, model: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model


# ---------------------------------------------------------------------------
# Subclass registration / naming (__init_subclass__, available_models)
# ---------------------------------------------------------------------------


def test_explicit_name_registers_lowercased() -> None:
    """A subclass declared with name= is registered under the lowercased name."""
    assert "dummybasetest" in available_models()
    assert BaseModel._models["dummybasetest"] is DummyModel
    # the class remembers its (cased) registered name
    assert DummyModel._name == "DummyBaseTest"


def test_unnamed_subclass_registers_under_class_name() -> None:
    """Without an explicit name=, the class name (lowercased) is used."""
    assert "unnamedmodel" in available_models()
    assert BaseModel._models["unnamedmodel"] is UnnamedModel
    assert UnnamedModel._name == "UnnamedModel"


def test_available_models_includes_builtin_operators() -> None:
    """The shared registry exposes the built-in operators alongside test models."""
    models = available_models()
    assert {"fno", "gino", "transolver"}.issubset(set(models))


# ---------------------------------------------------------------------------
# __new__ init-kwarg capture
# ---------------------------------------------------------------------------


def test_init_kwargs_records_passed_and_default_args() -> None:
    """_init_kwargs captures explicit args plus defaults for unspecified ones."""
    model = DummyModel(features=7)
    assert model._init_kwargs["features"] == 7
    # version, registered name and positional args are always recorded
    assert model._init_kwargs["_version"] == DummyModel._version
    assert model._init_kwargs["_name"] == "DummyBaseTest"
    assert model._init_kwargs["args"] == ()


def test_init_kwargs_fills_missing_default() -> None:
    """An unspecified keyword falls back to its signature default in _init_kwargs."""
    model = DummyModel()
    assert model._init_kwargs["features"] == 1


# ---------------------------------------------------------------------------
# State-dict (de)serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_state_dict_round_trip(backend: str) -> None:
    """get_state_dict -> load_state_dict restores identical params + metadata."""
    with jax.default_device(jax.devices(backend)[0]):
        model = DummyModel(features=3)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)

        state_dict = model.get_state_dict(params)
        assert set(state_dict.keys()) == {"params_bytes", "_metadata"}
        assert state_dict["_metadata"]["_version"] == model._version

        restored = model.load_state_dict(params, state_dict)
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda a, b: bool((a == b).all()), params, restored)
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_load_state_dict_warns_on_version_mismatch(backend: str) -> None:
    """Loading a state dict saved under a different _version warns the user."""
    with jax.default_device(jax.devices(backend)[0]):
        model = DummyModel(features=2)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)

        state_dict = model.get_state_dict(params)
        state_dict["_metadata"] = dict(state_dict["_metadata"])
        state_dict["_metadata"]["_version"] = "9.9.9"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.load_state_dict(params, state_dict)
        assert any("version" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# On-disk checkpointing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_and_load_checkpoint_round_trip(backend: str, tmp_path: Path) -> None:
    """save_checkpoint writes params+metadata; load_checkpoint restores params."""
    with jax.default_device(jax.devices(backend)[0]):
        model = DummyModel(features=3)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)

        model.save_checkpoint(tmp_path, "ckpt", params)
        assert (tmp_path / "ckpt_params.bytes").exists()
        assert (tmp_path / "ckpt_metadata.msgpack").exists()

        restored = model.load_checkpoint(tmp_path, "ckpt", params)
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda a, b: bool((a == b).all()), params, restored)
        )


def test_save_checkpoint_creates_missing_folder(tmp_path: Path) -> None:
    """save_checkpoint creates the target folder if it does not yet exist."""
    model = DummyModel(features=1)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((2, 4), dtype=jnp.float32))
    nested = tmp_path / "does" / "not" / "exist"
    model.save_checkpoint(nested, "ckpt", params)
    assert (nested / "ckpt_params.bytes").exists()


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_checkpoint_rebuilds_instance_and_params(
    backend: str, tmp_path: Path
) -> None:
    """from_checkpoint reconstructs the model from saved kwargs and its params."""
    with jax.default_device(jax.devices(backend)[0]):
        model = DummyModel(features=5)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)
        model.save_checkpoint(tmp_path, "ckpt", params)

        rebuilt, rebuilt_params = DummyModel.from_checkpoint(tmp_path, "ckpt", params)
        assert isinstance(rebuilt, DummyModel)
        assert rebuilt.features == 5
        assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda a, b: bool((a == b).all()), params, rebuilt_params
            )
        )


# ---------------------------------------------------------------------------
# get_model_jax (config-driven instantiation)
# ---------------------------------------------------------------------------


def test_get_model_jax_instantiates_from_config() -> None:
    """get_model_jax builds the model named by config and maps data_channels."""
    config = _Config(
        model={"model_arch": "CfgBaseTest", "data_channels": 5, "out_channels": 2}
    )
    model = get_model_jax(config)
    assert isinstance(model, CfgModel)
    # data_channels -> in_channels
    assert model.in_channels == 5
    assert model.out_channels == 2


def test_get_model_jax_multiplies_channels_by_patching_levels() -> None:
    """With patching levels=L, in_channels = data_channels * (L + 1)."""
    config = _Config(
        model={"model_arch": "CfgBaseTest", "data_channels": 3, "out_channels": 1},
        patching={"levels": 2},
    )
    model = get_model_jax(config)
    assert model.in_channels == 9  # 3 * (2 + 1)


def test_get_model_jax_unknown_arch_raises() -> None:
    """An architecture name not in the registry raises ValueError."""
    config = _Config(model={"model_arch": "does_not_exist", "data_channels": 1})
    with pytest.raises(ValueError, match="expected one of"):
        get_model_jax(config)


def test_get_model_jax_missing_data_channels_raises_keyerror() -> None:
    """A model that declares in_channels requires data_channels in config --
    a missing key must fail loudly (KeyError), not silently build the model
    with in_channels=None."""
    config = _Config(model={"model_arch": "CfgBaseTest", "out_channels": 1})
    with pytest.raises(KeyError):
        get_model_jax(config)


def test_get_model_jax_skips_in_channels_for_model_without_the_field() -> None:
    """A model with no in_channels parameter (e.g. Transolver's
    space_dim/fun_dim parameterization) builds fine even with no
    data_channels in config, and data_channels/out_channels are simply
    dropped rather than injected."""
    config = _Config(
        model={"model_arch": "NoChannelsBaseTest", "hidden": 4, "out_channels": 2}
    )
    model = get_model_jax(config)
    assert isinstance(model, NoChannelsModel)
    assert model.hidden == 4
    assert not hasattr(model, "in_channels")


def test_get_model_jax_strips_underscore_prefixed_keys() -> None:
    """zencfg internal bookkeeping keys (e.g. "_config_name", re-added by
    make_config_from_cli's CLI-override revalidation) must not reach the
    model constructor."""
    config = _Config(
        model={
            "model_arch": "CfgBaseTest",
            "data_channels": 2,
            "out_channels": 1,
            "_config_name": "configbase",
        }
    )
    model = get_model_jax(config)
    assert isinstance(model, CfgModel)


def test_get_model_jax_unknown_key_raises() -> None:
    """A key the architecture does not accept (typo, or meant for a
    different architecture) must fail loudly, not reach a bare TypeError."""
    config = _Config(
        model={
            "model_arch": "CfgBaseTest",
            "data_channels": 2,
            "out_channels": 1,
            "dropout_rate": 0.1,  # Transolver-only, not accepted by CfgModel
        }
    )
    with pytest.raises(ValueError, match="dropout_rate"):
        get_model_jax(config)


def test_get_model_jax_permissive_warns_and_drops_unknown_key() -> None:
    """permissive=True turns the same error into a warning and still builds
    the model, with the unknown key dropped."""
    config = _Config(
        model={
            "model_arch": "CfgBaseTest",
            "data_channels": 2,
            "out_channels": 1,
            "dropout_rate": 0.1,
        }
    )
    with pytest.warns(UserWarning, match="dropout_rate"):
        model = get_model_jax(config, permissive=True)
    assert isinstance(model, CfgModel)


# ---------------------------------------------------------------------------
# verbose=True paths in __new__ (source lines 70-71, 79)
# ---------------------------------------------------------------------------


def test_verbose_unknown_kwarg_warns() -> None:
    """verbose=True warns when an unknown kwarg is passed.

    BaseModel.__new__ warns before Flax's __init__ rejects the kwarg.
    """
    with pytest.warns(UserWarning, match="_unknown_kwarg_xyz"):
        with pytest.raises(TypeError):
            # verbose/_unknown_kwarg_xyz are deliberately invalid Flax fields,
            # exercised to confirm __new__ warns before Flax's __init__ rejects.
            DummyModel(  # type: ignore[call-arg]
                features=1, verbose=True, _unknown_kwarg_xyz="oops"
            )


def test_verbose_missing_default_logs(caplog: pytest.LogCaptureFixture) -> None:
    """verbose=True logs when a kwarg with a default is not explicitly passed.

    BaseModel.__new__ logs before Flax's __init__ rejects the kwarg.
    """
    caplog.set_level(logging.INFO)
    with pytest.raises(TypeError):
        DummyModel(verbose=True)  # type: ignore[call-arg]
    assert "features" in caplog.text


# ---------------------------------------------------------------------------
# load_state_dict with metadata missing _version (source line 133)
# ---------------------------------------------------------------------------


def test_load_state_dict_warns_when_no_stored_version() -> None:
    """load_state_dict warns when _metadata has no _version key (source line 133)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = DummyModel(features=2)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)

        state_dict = model.get_state_dict(params)
        meta_without_version = {
            k: v for k, v in state_dict["_metadata"].items() if k != "_version"
        }
        state_dict = dict(state_dict, _metadata=meta_without_version)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.load_state_dict(params, state_dict)
        messages = [str(w.message) for w in caught]
        assert any("no stored version" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# save_checkpoint creates missing folder (source line 156) - explicit device ctx
# ---------------------------------------------------------------------------


def test_save_checkpoint_creates_missing_folder_with_device(
    tmp_path: Path,
) -> None:
    """save_checkpoint creates nested folders that do not yet exist
    (source line 156)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = DummyModel(features=1)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)
        nested = tmp_path / "a" / "b" / "c"
        model.save_checkpoint(nested, "ckpt", params)
        assert (nested / "ckpt_params.bytes").exists()


# ---------------------------------------------------------------------------
# from_checkpoint: version mismatch warning (source line 210)
# ---------------------------------------------------------------------------


def test_from_checkpoint_version_mismatch_warns(tmp_path: Path) -> None:
    """from_checkpoint warns when the saved version differs from
    cls._version (source line 210)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = DummyModel(features=3)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)
        model.save_checkpoint(tmp_path, "ckpt", params)

        meta_path = tmp_path / "ckpt_metadata.msgpack"
        meta = load_msgpack(meta_path)
        meta["_version"] = "0.0.0-mismatched"
        save_msgpack(meta, meta_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rebuilt, _ = DummyModel.from_checkpoint(tmp_path, "ckpt", params)
        messages = [str(w.message) for w in caught]
        assert any("version" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# from_checkpoint: no "args" key in saved metadata (source line 220)
# ---------------------------------------------------------------------------


def test_from_checkpoint_no_args_key(tmp_path: Path) -> None:
    """from_checkpoint falls back to init_args=[] when 'args' is absent
    (source line 220)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = DummyModel(features=4)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)
        model.save_checkpoint(tmp_path, "ckpt", params)

        meta_path = tmp_path / "ckpt_metadata.msgpack"
        meta = load_msgpack(meta_path)
        meta.pop("args", None)
        save_msgpack(meta, meta_path)

        rebuilt, rebuilt_params = DummyModel.from_checkpoint(tmp_path, "ckpt", params)
        assert isinstance(rebuilt, DummyModel)
        assert rebuilt.features == 4


# ---------------------------------------------------------------------------
# Checkpoint metadata references: allowlisted modules only
# ---------------------------------------------------------------------------


def _ref(module: object, qualname: object) -> dict[str, Any]:
    return {"__ref_module__": module, "__ref_qualname__": qualname}


def test_reference_round_trip_for_real_hyperparameters() -> None:
    """Callables used as model hyperparameters survive encode -> decode."""
    from scirex.layers.blocks.wno_block import mish
    from scirex.models.fno import FNO

    kwargs = {"non_linearity": nn.gelu, "activation": mish, "cls": FNO, "n": 3}
    decoded = _decode_init_kwargs(_encode_init_kwargs(kwargs))
    assert decoded["non_linearity"] is nn.gelu
    assert decoded["activation"] is mish
    assert decoded["cls"] is FNO
    assert decoded["n"] == 3


@pytest.mark.parametrize(
    ("module", "qualname"),
    [
        ("os", "system"),
        ("subprocess", "Popen"),
        ("builtins", "eval"),
        ("scirex_evil", "x"),  # shares a string prefix with "scirex"
        ("jaxx.core", "x"),
        (None, "x"),
        (["os"], "system"),
        ("scirex", 3),
    ],
)
def test_decode_rejects_references_outside_allowed_packages(
    module: object, qualname: object
) -> None:
    with pytest.raises(ValueError, match="Refusing to load reference"):
        _decode_init_kwargs(_ref(module, qualname))


def test_moved_module_table_is_well_formed() -> None:
    """Every ``MOVED_MODULES`` key is a vanished ``scirex.*`` path, every target exists."""
    import scirex
    from scirex.models._legacy_refs import MOVED_MODULES

    src = Path(scirex.__file__).parent.parent

    def exists(dotted: str) -> bool:
        path = src / dotted.replace(".", "/")
        return (path / "__init__.py").is_file() or path.with_suffix(".py").is_file()

    assert MOVED_MODULES
    for old, new in MOVED_MODULES.items():
        assert old.startswith("scirex."), old
        assert new.startswith("scirex."), new
        assert old != new
        assert not exists(old), f"{old} still exists; it is not a moved module"
        assert exists(new), new
        assert new not in MOVED_MODULES, f"{old} -> {new} chains through another row"


def test_decode_maps_moved_scirex_references() -> None:
    """References written by the pre-1.0 ``scirex.*`` layout (commit e9d3043) resolve
    to the modules' final locations, with longest-prefix matching for submodules."""
    import scirex.experimental.distributed.comm as comm
    import scirex.experimental.geometry.mesh_io as mesh_io
    from scirex._internal.segment_csr import segment_csr
    from scirex.models._legacy_refs import remap_legacy_module

    assert _decode_init_kwargs(_ref("scirex.distributed.comm", "init")) is comm.init
    assert (
        _decode_init_kwargs(_ref("scirex.geometry.mesh_io", "save_mesh"))
        is mesh_io.save_mesh
    )
    assert (
        _decode_init_kwargs(_ref("scirex.layers.common.segment_csr", "segment_csr"))
        is segment_csr
    )
    assert (
        remap_legacy_module("scirex.distributed") == "scirex.experimental.distributed"
    )
    # A name that merely shares a string prefix is not remapped.
    assert remap_legacy_module("scirex.distributed_x") == "scirex.distributed_x"
    assert (
        remap_legacy_module("scirex.config.distributed")
        == "scirex.experimental.distributed.config"
    )


@pytest.mark.parametrize("module", ["operators_evil", "operatorsx.models", "operators"])
def test_decode_legacy_prefix_is_not_a_blanket_allow(module: str) -> None:
    with pytest.raises((ValueError, AttributeError, ImportError)):
        _decode_init_kwargs(_ref(module, "system"))


def test_decode_rejects_qualname_walking_out_of_allowed_module() -> None:
    """An allowed module that merely imported something cannot be used to reach it."""
    # base does ``import importlib`` -> importlib.import_module
    with pytest.raises(ValueError, match="resolves outside"):
        _decode_init_kwargs(_ref("scirex.models.base", "importlib.import_module"))
    with pytest.raises(ValueError, match="resolves outside"):
        _decode_init_kwargs(_ref("scirex.models.base", "__builtins__"))


def test_from_checkpoint_rejects_tampered_reference(tmp_path: Path) -> None:
    """A checkpoint whose metadata references ``os.system`` is refused on load."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = DummyModel(features=4)
        x = jnp.ones((2, 4), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), x)
        model.save_checkpoint(tmp_path, "ckpt", params)

        meta_path = tmp_path / "ckpt_metadata.msgpack"
        meta = load_msgpack(meta_path)
        meta["features"] = _ref("os", "system")
        save_msgpack(meta, meta_path)

        with pytest.raises(ValueError, match="Refusing to load reference"):
            DummyModel.from_checkpoint(tmp_path, "ckpt", params)


def test_load_checkpoint_returns_jax_arrays(tmp_path: Path) -> None:
    """Restored parameters are JAX arrays, not the NumPy arrays flax decodes to."""
    model = DummyModel(features=3)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((2, 4), dtype=jnp.float32))
    model.save_checkpoint(tmp_path, "ckpt", params)
    restored = model.load_checkpoint(tmp_path, "ckpt", params)
    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(restored))
