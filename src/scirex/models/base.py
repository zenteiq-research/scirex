# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Base model class, model registry and the `get_model_jax` factory."""

import importlib
import inspect
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp

from scirex._internal.lazy import import_feature
from scirex._internal.serialization import load_msgpack, save_msgpack
from scirex.models._legacy_refs import remap_legacy_module

logger = logging.getLogger(__name__)

# JAX/Flax port: minimal changes from base_model.py

_REF_MODULE_KEY = "__ref_module__"
_REF_QUALNAME_KEY = "__ref_qualname__"

# Top-level packages a checkpoint may reference a class/function from. A
# checkpoint's metadata is data, not code: without this allowlist a tampered
# file could make load resolve any importable attribute (e.g. `os.system`).
_ALLOWED_REF_PACKAGES = ("scirex", "jax", "flax")

# Config keys are snake_case, but some model constructors keep older parameter
# names that are stored in checkpoint metadata (`_init_kwargs`) -- renaming
# those would break loading existing checkpoints. `get_model_jax` translates
# the config key to the constructor name for models that declare the latter.
_CONFIG_KEY_TO_MODEL_PARAM = {"height": "H", "width": "W", "depth": "D"}


def _is_allowed_ref_module(module_name: object) -> bool:
    """True if `module_name` is, or is inside, an allowed package."""
    return isinstance(module_name, str) and any(
        module_name == pkg or module_name.startswith(pkg + ".")
        for pkg in _ALLOWED_REF_PACKAGES
    )


def _encode_init_kwargs(obj: Any) -> Any:
    """Makes `_init_kwargs` msgpack-safe.

    Model constructors can take a class or function as a hyperparameter
    (e.g. `conv_module`, `non_linearity`) -- pickle serialized these by
    module+qualname reference, which msgpack can't represent directly. Do
    the same thing explicitly and safely: store the reference, and restore
    it with an import + attribute lookup (never arbitrary code execution)
    in `_decode_init_kwargs`.
    """
    if isinstance(obj, dict):
        return {k: _encode_init_kwargs(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_encode_init_kwargs(v) for v in obj]
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return {_REF_MODULE_KEY: obj.__module__, _REF_QUALNAME_KEY: obj.__qualname__}
    return obj


def _decode_init_kwargs(obj: Any) -> Any:
    """Inverse of `_encode_init_kwargs`.

    Raises
    ------
    ValueError
        If a stored reference points outside `_ALLOWED_REF_PACKAGES`.
    """
    if isinstance(obj, dict):
        if set(obj) == {_REF_MODULE_KEY, _REF_QUALNAME_KEY}:
            module_name = obj[_REF_MODULE_KEY]
            qualname = obj[_REF_QUALNAME_KEY]
            if isinstance(module_name, str):
                # a module that moved after the checkpoint was written
                module_name = remap_legacy_module(module_name)
            if not _is_allowed_ref_module(module_name) or not isinstance(qualname, str):
                raise ValueError(
                    f"Refusing to load reference {module_name!r}:{qualname!r} from "
                    f"checkpoint metadata; only {_ALLOWED_REF_PACKAGES} are allowed."
                )
            target: Any = importlib.import_module(module_name)
            for part in qualname.split("."):
                target = getattr(target, part)
            # The module check above only constrains where the lookup starts; a
            # qualname can still walk to something a module merely imported
            # (`os.system`, `__globals__`...). Require the object itself to
            # be defined inside an allowed package too.
            if not _is_allowed_ref_module(getattr(target, "__module__", None)):
                raise ValueError(
                    f"Refusing to load reference {module_name!r}:{qualname!r} from "
                    f"checkpoint metadata; it resolves outside "
                    f"{_ALLOWED_REF_PACKAGES}."
                )
            return target
        return {k: _decode_init_kwargs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_init_kwargs(v) for v in obj]
    return obj


class BaseModel(nn.Module):
    """Base class for all models

    This class has two main functionalities:
    * It monitors the creation of subclass, that are automatically registered
      for users to use by name using the library's config system
    * When a new instance of this class is created, the init call is intercepted
      so we can store the parameters used to create the instance.
      This makes it possible to save trained models along with their init parameters,
      and therefore load saved models easily.

    Notes
    -----
    Model can be versioned using the _version class attribute.
    This can be used for sanity check when loading models from checkpoints to verify the
    model hasn't been updated since.
    """

    _models: ClassVar[dict[str, type["BaseModel"]]] = {}
    _version = "0.1.0"
    _name: ClassVar[str] = ""

    def __init_subclass__(cls, name: str | None = None, **kwargs: Any) -> None:
        """When a subclass is created, register it in _models
        We look for an existing name attribute.
        If not give, then we use the class' name.
        """
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseModel._models[name.lower()] = cls
            cls._name = name
        else:
            BaseModel._models[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __new__(cls, *args: Any, **kwargs: Any) -> "BaseModel":
        """Verify arguments and save init kwargs for loading/saving

        We inspect the class' signature and check for unused parameters, or
        parameters not passed.

        We store all the args and kwargs given so we can duplicate the
        instance transparently.
        """
        sig = inspect.signature(cls)
        model_name = cls.__name__

        verbose = kwargs.get("verbose", False)
        for key in kwargs:
            if key not in sig.parameters:
                if verbose:
                    warnings.warn(
                        f"Given argument key={key} "
                        f"that is not in {model_name}'s signature.",
                        stacklevel=2,
                    )

        # flax.linen.Module's own built-in fields, not model hyperparameters:
        # "parent" defaults to a flax-internal sentinel object that is never
        # meaningful to checkpoint (and isn't safely serializable at all).
        _module_builtin_fields = {"parent", "name"}

        for key, value in sig.parameters.items():
            if (
                (value.default is not inspect._empty)
                and (key not in kwargs)
                and (key not in _module_builtin_fields)
            ):
                if verbose:
                    logger.info(
                        f"Keyword argument {key} not specified for model {model_name}, "
                        f"using default={value.default}."
                    )
                kwargs[key] = value.default

        if hasattr(cls, "_version"):
            kwargs["_version"] = cls._version
        kwargs["args"] = args
        kwargs["_name"] = cls._name
        instance = super().__new__(cls)
        instance._init_kwargs = kwargs

        return instance

    def get_state_dict(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Returns a state dict containing model params and version metadata.

        Parameters
        ----------
        params : dict
            Flax parameter dict obtained from model.init(...)

        Returns
        -------
        dict
            state dict with params bytes and metadata
        """
        state_dict = {
            "params_bytes": flax.serialization.to_bytes(params),
            "_metadata": self._init_kwargs,
        }
        return state_dict

    def load_state_dict(
        self, params_target: Mapping[str, Any], state_dict: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Loads params from a state dict, with version checking.

        Parameters
        ----------
        params_target : dict
            A reference param structure (e.g. from model.init) used to
            restore shape info
        state_dict : dict
            state dictionary generated by `get_state_dict`

        Returns
        -------
        dict
            restored Flax parameter dict
        """
        metadata = state_dict.get("_metadata", None)

        if metadata is not None:
            saved_version = metadata.get("_version", None)
            if saved_version is None:
                warnings.warn(
                    f"Saved instance of {self.__class__} has no stored "
                    "version attribute.",
                    stacklevel=2,
                )
            if saved_version != self._version:
                warnings.warn(
                    f"Attempting to load a {self.__class__} of version "
                    f"{saved_version}, but current version of "
                    f"{self.__class__} is {self._version}",
                    stacklevel=2,
                )

        return flax.serialization.from_bytes(params_target, state_dict["params_bytes"])

    def save_checkpoint(
        self, save_folder: str | Path, save_name: str, params: Mapping[str, Any]
    ) -> None:
        """Saves the model params and init kwargs under the given folder and name.

        Parameters
        ----------
        save_folder : str or Path
        save_name : str
        params : dict
            Flax parameter dict obtained from model.init(...)
        """
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        params_filepath = save_folder / f"{save_name}_params.bytes"
        params_filepath.write_bytes(flax.serialization.to_bytes(params))

        metadata_filepath = save_folder / f"{save_name}_metadata.msgpack"
        save_msgpack(_encode_init_kwargs(self._init_kwargs), metadata_filepath)

    def load_checkpoint(
        self, save_folder: str | Path, save_name: str, params_target: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Loads and returns the model params from a checkpoint.

        Parameters
        ----------
        save_folder : str or Path
        save_name : str
        params_target : dict
            A reference param structure used by flax.serialization.from_bytes

        Returns
        -------
        dict
            restored Flax parameter dict, as JAX arrays ready for `apply`
        """
        save_folder = Path(save_folder)
        params_filepath = save_folder / f"{save_name}_params.bytes"
        restored = flax.serialization.from_bytes(
            params_target, params_filepath.read_bytes()
        )
        # from_bytes returns NumPy arrays; layers such as SpectralConv need JAX arrays.
        return jax.tree_util.tree_map(jnp.asarray, restored)

    @classmethod
    def from_checkpoint(
        cls, save_folder: str | Path, save_name: str, params_target: Mapping[str, Any]
    ) -> tuple["BaseModel", dict[str, Any]]:
        """Instantiates the model from a checkpoint and returns (instance, params).

        Parameters
        ----------
        save_folder : str or Path
        save_name : str
        params_target : dict
            A reference param structure used by flax.serialization.from_bytes

        Returns
        -------
        tuple[BaseModel, dict]
            (model instance, restored params dict)
        """
        save_folder = Path(save_folder)

        metadata_filepath = save_folder / f"{save_name}_metadata.msgpack"
        init_kwargs = _decode_init_kwargs(load_msgpack(metadata_filepath))

        version = init_kwargs.pop("_version", None)
        if hasattr(cls, "_version") and version != cls._version:
            warnings.warn(
                f"Checkpoint saved for version {version} of model {cls._name} "
                f"but current code is version {cls._version}",
                stacklevel=2,
            )

        init_kwargs.pop("_name", None)

        if "args" in init_kwargs:
            init_args = init_kwargs.pop("args")
        else:
            init_args = []

        instance = cls(*init_args, **init_kwargs)
        params = instance.load_checkpoint(save_folder, save_name, params_target)
        return instance, params


# Architectures that `get_model_jax` imports on demand: registry name -> module.
# A model registers itself in `BaseModel._models` when its module is imported,
# but `scirex.models` is lazy, so the factory cannot rely on the caller having
# imported the class. GIWO is experimental and lives outside `scirex.models`.
_BUILTIN_ARCHS: dict[str, str] = {
    "fno": "scirex.models.fno",
    "fnoclassic2d": "scirex.models.fno_classic",
    "giwo": "scirex.experimental.giwo",
    "gino": "scirex.models.gino",
    "transolver": "scirex.models.transolver",
    "wno": "scirex.models.wno",
}


def available_models() -> list[str]:
    """List the names accepted by `get_model_jax`, sorted.

    Contains every built-in architecture (without importing it) plus any model
    registered by subclassing `BaseModel`.
    """
    return sorted({*BaseModel._models, *_BUILTIN_ARCHS})


def get_model_jax(config: Any, permissive: bool = False) -> nn.Module:
    """Returns an instantiated model for the given config

    * Reads the model to be used from ``config.model["model_arch"]``
    * Passes the other keys of ``config.model`` to its constructor

    Parameters
    ----------
    config : Bunch or dict-like
        configuration, must have ``config.model``, a mapping with the key
        ``model_arch`` (a string, the name of the architecture) and the keyword
        arguments of the model
    permissive : bool, by default False
        A `config.model` key the chosen architecture does not accept (a typo,
        or a key meant for a different architecture) raises `ValueError`.
        With `permissive=True` it instead warns and drops the key.

    Returns
    -------
    model : nn.Module
        the instantiated module
    """
    arch = config.model["model_arch"].lower()
    model_config = config.model.copy()

    model_config.pop("model_arch", None)
    # zencfg internal bookkeeping (e.g. "_config_name") can leak into
    # config.model when it was rebuilt via a CLI override (make_config_from_cli
    # re-validates every field, re-adding "_config_name"); strip any such
    # underscore-prefixed keys since they're never real model constructor kwargs.
    for key in [k for k in model_config if k.startswith("_")]:
        model_config.pop(key)

    # Architectures register themselves only when their module is imported; load
    # a built-in one on demand so the caller need not have imported its class.
    if arch not in BaseModel._models and arch in _BUILTIN_ARCHS:
        import_feature(_BUILTIN_ARCHS[arch], f"The {arch!r} architecture")

    try:
        model_cls = BaseModel._models[arch]
    except KeyError:
        raise ValueError(
            f"Got config.arch={arch}, expected one of {available_models()}."
        ) from None

    # data_channels/in_channels and out_channels are renamed/injected only
    # for architectures whose model class actually declares those fields
    # (e.g. FNO, GINO). Architectures with a different input/output
    # parameterization (e.g. Transolver's space_dim/fun_dim/out_dim) keep
    # their own field names untouched, and the interface-parity-only
    # data_channels/out_channels config fields are simply dropped before
    # instantiation. Reuses the same inspect.signature(cls) idiom as
    # BaseModel.__new__ above, so there's a single source of truth for a
    # model's declared parameters.
    model_params = inspect.signature(model_cls).parameters

    if "in_channels" in model_params:
        # Required: if the model needs in_channels, data_channels must be
        # present in config - fail loudly here rather than building a model
        # with in_channels=None and failing later with a confusing arithmetic
        # TypeError.
        data_channels = model_config.pop("data_channels")
        try:
            patching_levels = config["patching"]["levels"]
        except KeyError:
            patching_levels = 0
        if patching_levels:
            data_channels *= patching_levels + 1
        model_config["in_channels"] = data_channels
    else:
        model_config.pop("data_channels", None)

    if "out_channels" not in model_params:
        model_config.pop("out_channels", None)

    for config_key, param in _CONFIG_KEY_TO_MODEL_PARAM.items():
        if config_key in model_config and config_key not in model_params:
            if param not in model_params:
                continue
            value = model_config.pop(config_key)
            if param in model_config:
                # The old key set directly on the config, e.g. `config.model.H`.
                warnings.warn(
                    f"config.model.{param} is deprecated; use "
                    f"config.model.{config_key}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if value is None:
                    continue
            model_config[param] = value

    accepts_any_kwarg = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in model_params.values()
    )
    if not accepts_any_kwarg:
        unknown = sorted(k for k in model_config if k not in model_params)
        if unknown:
            message = (
                f"config.model has key(s) {unknown} not accepted by "
                f"{arch!r} (accepts {sorted(model_params)})."
            )
            if permissive:
                warnings.warn(message, stacklevel=2)
                for key in unknown:
                    model_config.pop(key)
            else:
                raise ValueError(message)

    return model_cls(**model_config)
