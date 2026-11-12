# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""CLI: generate a T3 mesh with gmsh and save it in this repo's own ``.npz``
convention, ready to sync to a training server that has neither ``gmsh``
nor ``meshio`` installed.

Run this on your own machine (wherever ``gmsh`` is installed), then
``scp``/``rsync`` the resulting ``.npz`` to the server and pass it to a
Galerkin benchmark script's ``--mesh-path``.

Usage
-----
    python scripts/mesh/generate_gmsh_mesh.py --domain circular --resolution 0.05
        # -> meshes/circular_r0.05.npz
    python scripts/mesh/generate_gmsh_mesh.py --domain lshape --resolution 0.05
        # -> meshes/lshape_r0.05.npz
    python scripts/mesh/generate_gmsh_mesh.py --domain square --resolution 0.05 \
        --order 2
        # -> meshes/square_r0.05_t6.npz -- the _t6 suffix is automatic for
        # --order 2, so this never collides with the --order 1 mesh at the
        # same resolution. Pass --out to choose the path explicitly instead.

Note on imports: this script deliberately loads ``gmsh_mesh.py`` by file
path (``importlib``) rather than via ``import scirex.experimental.geometry.gmsh_mesh``.
``src/scirex/__init__.py`` eagerly imports the JAX/Flax model stack, so a
normal package import would require installing ``jax``/``flax`` on this
mesh-generation machine even though ``gmsh_mesh.py`` itself needs neither --
exactly the dependency split (heavy ML stack only on the server, ``gmsh``
only here) this whole workflow exists to keep.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(rel_path: str, name: str):
    file_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"couldn't find {rel_path} (looked at {file_path})")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gmsh_mesh = _load_module("src/scirex/experimental/geometry/gmsh_mesh.py", "gmsh_mesh")
mesh_io = _load_module("src/scirex/experimental/geometry/mesh_io.py", "mesh_io")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["square", "circular", "lshape"], required=True)
    ap.add_argument(
        "--resolution", type=float, default=0.05, help="target element size"
    )
    ap.add_argument(
        "--order",
        type=int,
        choices=[1, 2],
        default=1,
        help="1 for T3 (linear), 2 for T6 (quadratic)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="output .npz path (default: meshes/<domain>_r<resolution>"
        "[_t6].npz -- the _t6 suffix is added automatically for --order 2, "
        "so a T3 and a T6 mesh at the same resolution never collide)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite --out (or the default path) if it already exists; "
        "without this, an existing file is left untouched and the script "
        "exits with an error",
    )
    args = ap.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = "_t6" if args.order == 2 else ""
        out_path = (
            _REPO_ROOT / "meshes" / f"{args.domain}_r{args.resolution}{suffix}.npz"
        )

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists -- pass --force to overwrite it "
            "(mesh .npz files here are committed artifacts, not scratch "
            "output)."
        )

    xy, tri, boundary_mask = gmsh_mesh.generate_domain_mesh(
        args.domain, args.resolution, order=args.order
    )
    mesh_io.save_mesh(out_path, xy, tri, boundary_mask)
    print(
        f"[{args.domain}] {xy.shape[0]} nodes, {tri.shape[0]} triangles, "
        f"{int(boundary_mask.sum())} boundary nodes -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
