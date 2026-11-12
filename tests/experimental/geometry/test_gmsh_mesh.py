# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""These tests deliberately do not require gmsh to be installed (it is not
expected to be present on CI/training machines -- see the module's own
docstring): they only exercise the domain-dispatch table and the
clear-error-when-gmsh-is-missing path, both of which are independent of
gmsh actually being importable. A separate, opt-in test exercising a real
gmsh call belongs on a developer's own machine, not here."""

import pytest

from scirex.experimental.geometry.gmsh_mesh import generate_domain_mesh


def test_generate_domain_mesh_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unknown domain"):
        generate_domain_mesh("hexagon")


def test_generate_domain_mesh_raises_clear_error_without_gmsh() -> None:
    try:
        import gmsh  # noqa: F401

        pytest.skip("gmsh is installed in this environment; nothing to check here")
    except (ImportError, OSError):
        pass

    with pytest.raises(ImportError, match="gmsh is not installed"):
        generate_domain_mesh("square")
