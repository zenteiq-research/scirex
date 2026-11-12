# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for the top-level ``scirex`` public API: exports, laziness, version."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import scirex

_SRC_DIR = Path(scirex.__file__).resolve().parent.parent
_INIT_FILE = Path(scirex.__file__).resolve()


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter that has imported nothing yet."""
    env = {**os.environ, "PYTHONPATH": str(_SRC_DIR), "JAX_PLATFORMS": "cpu"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


def test_all_is_the_documented_surface() -> None:
    assert scirex.__all__ == ["FNO", "GINO", "WNO", "Transolver", "get_model_jax"]


@pytest.mark.parametrize("name", scirex.__all__)
def test_every_exported_name_resolves(name: str) -> None:
    assert getattr(scirex, name) is not None


def test_exports_are_the_objects_of_scirex_models() -> None:
    import scirex.models

    for name in scirex.__all__:
        assert getattr(scirex, name) is getattr(scirex.models, name)


def test_lazy_table_matches_all() -> None:
    assert set(scirex._LAZY_EXPORTS) == set(scirex.__all__)


def test_dir_lists_exports_and_subpackages() -> None:
    listing = dir(scirex)
    assert set(scirex.__all__) <= set(listing)
    assert {"models", "layers", "losses", "utils", "__version__"} <= set(listing)


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="does_not_exist"):
        scirex.does_not_exist
    assert not hasattr(scirex, "__wrapped__")


def test_subpackages_are_reachable_as_attributes() -> None:
    assert scirex.models.__name__ == "scirex.models"
    assert scirex.losses.__name__ == "scirex.losses"


def test_import_scirex_does_not_import_jax_or_optional_dependencies() -> None:
    result = _run_fresh(
        """
        import sys
        import scirex

        heavy = {"jax", "flax", "jaxwt", "open3d", "torch", "wandb", "tensorly"}
        loaded = sorted(heavy & set(sys.modules))
        assert not loaded, loaded
        assert not any(m.startswith("scirex.models") for m in sys.modules)
        assert scirex.__version__

        scirex.FNO  # first use of a lazy name imports what it needs
        assert "jax" in sys.modules
        assert "scirex.models.fno" in sys.modules
        assert "scirex.models.wno" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_version_is_a_string_read_lazily() -> None:
    """``__version__`` is looked up on first use (see ``tests/test_version.py``)."""
    assert isinstance(scirex.__version__, str) and scirex.__version__
