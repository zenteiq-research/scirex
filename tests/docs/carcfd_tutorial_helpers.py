# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared helpers for the Car-CFD tutorial tests.

``fake_car_cfd`` writes a small dataset in the layout of the Zenodo archive
(``train.txt``, ``test.txt`` and ``data/data/<id>/{tri_mesh.ply,press.npy}``)
with closed torus meshes of exactly 3,586 vertices (22 x 163) and 3,682
pressure entries per car, as in the real files, so that both the tutorials
and the benchmark's own loader can read it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from tests.public_api_helpers import REPO

SCRIPTS = REPO / "scripts" / "tutorials"


def load_script(name: str):
    """Import ``scripts/tutorials/<name>.py`` as a module.

    The module is registered in ``sys.modules`` before it runs: its Flax
    dataclasses resolve their string annotations through it.
    """
    module_name = f"{name}_tutorial"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def fake_car_cfd(root: Path, n_train: int = 4, n_test: int = 2) -> Path:
    """Write the fake dataset under ``root/processed-car-pressure-data``.

    Returns that folder, the tutorials' ``--data-dir``; ``root`` is the
    benchmark loader's ``data_root``.
    """
    import open3d as o3d

    item = Path(root) / "processed-car-pressure-data"
    rng = np.random.default_rng(0)
    ids = [f"{k:03d}" for k in range(1, n_train + n_test + 1)]
    for k, car_id in enumerate(ids):
        mesh = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=1.0 + 0.05 * k,
            tube_radius=0.3 + 0.02 * k,
            radial_resolution=163,
            tubular_resolution=22,
        )
        v = np.asarray(mesh.vertices) * np.array([0.9, 0.6, 2.8]) + np.array([0, 1, 0])
        mesh.vertices = o3d.utility.Vector3dVector(v)
        folder = item / "data" / "data" / car_id
        folder.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(folder / "tri_mesh.ply"), mesh)
        kept = 100.0 * np.sin(v[:, 2]) - 30.0 * v[:, 1] + rng.normal(0, 5, len(v))
        press = np.concatenate([kept[:16], rng.normal(0, 50, 96), kept[16:]])
        np.save(folder / "press.npy", press)
    (item / "train.txt").write_text(",".join(ids[:n_train]))
    (item / "test.txt").write_text(",".join(ids[n_train:]))
    return item


def max_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))
