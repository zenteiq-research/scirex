# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared pieces of the tests of the four Darcy tutorial scripts."""

import importlib.util
from pathlib import Path

import numpy as np

from tests.public_api_helpers import REPO

TRAIN_FILE = "piececonst_r421_N1024_smooth1.mat"
TEST_FILE = "piececonst_r421_N1024_smooth2.mat"


def load_script(model: str):
    """Import ``scripts/tutorials/darcy_<model>.py`` as a module."""
    path = REPO / "scripts" / "tutorials" / f"darcy_{model}.py"
    spec = importlib.util.spec_from_file_location(f"darcy_{model}_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_darcy(folder: Path, n: int = 12, native: int = 106) -> Path:
    """Write both ``.mat`` files with the real layout: ``coeff`` in {3, 12}
    (uint8) and ``sol`` (float64), ``(n, native, native)``. With the
    benchmark's stride of 5, ``native=106`` gives a 22x22 grid."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    s = np.linspace(0.0, 1.0, native)
    bump = np.outer(np.sin(np.pi * s), np.sin(np.pi * s))
    for name in (TRAIN_FILE, TEST_FILE):
        coeff = np.where(rng.normal(size=(n, native, native)) > 0, 12, 3)
        sol = 0.01 * bump[None] * (1.0 + 0.1 * rng.normal(size=(n, 1, 1)))
        savemat(folder / name, dict(coeff=coeff.astype(np.uint8), sol=sol))
    return folder


def benchmark_split(n_side: int = 85, n_train: int = 4, n_test: int = 2) -> dict:
    """A random split in the layout of ``scripts/benchmarks/darcy/_data.load_split``."""
    rng = np.random.default_rng(1)

    def field(n: int, kind: str) -> np.ndarray:
        if kind == "a":
            return np.where(rng.normal(size=(n, n_side, n_side)) > 0, 12, 3).astype(
                np.float32
            )
        return (0.01 * rng.random((n, n_side, n_side))).astype(np.float32)

    return dict(
        train_a=field(n_train, "a"),
        train_u=field(n_train, "u"),
        test_a=field(n_test, "a"),
        test_u=field(n_test, "u"),
    )
