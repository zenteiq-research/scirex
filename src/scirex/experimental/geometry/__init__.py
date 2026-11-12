# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Mesh I/O and generation for the Galerkin (weak-form) physics losses in
`src/scirex/losses/galerkin.py`. Kept separate from that module on
purpose: `galerkin.py` is pure JAX/numpy numerics with no mesh-file or
`gmsh` dependency, so it runs unmodified on a server that has neither
installed -- only `mesh_io.load_mesh` needs a mesh-format library
(`meshio`, optional), and only `gmsh_mesh.py` needs `gmsh` itself,
which is expected to run on a developer's machine, not the training server
(see that module's docstring).
"""
