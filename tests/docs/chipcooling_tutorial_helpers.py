# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Shared pieces of the tests of the four chip cooling tutorial scripts."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from tests.public_api_helpers import REPO

FIELD_KEY = "yz-mid|temperature"
COORD_KEY = "yz-mid|coordinates"


def load_script(model: str):
    """Import ``scripts/tutorials/chipcooling_<model>.py`` as a module."""
    path = REPO / "scripts" / "tutorials" / f"chipcooling_{model}.py"
    spec = importlib.util.spec_from_file_location(f"chipcooling_{model}_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_chip_cooling(folder: Path, n: int = 12, n_nodes: int = 1500) -> Path:
    """Write ``model_setup.json`` and ``sim_0001.npz``... in the real layout.

    The nodes fill a 20 x 60 mm plane with a rectangular hole, so that the
    grid's domain mask is not trivial. The temperature depends smoothly on
    the position and on the six parameters.
    """
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    yz = rng.random((4 * n_nodes, 2)) * np.array([0.02, 0.06])
    hole = (
        (yz[:, 0] > 0.008) & (yz[:, 0] < 0.014) & (yz[:, 1] > 0.02) & (yz[:, 1] < 0.04)
    )
    yz = yz[~hole][:n_nodes]
    coords = np.stack([np.zeros(len(yz)), yz[:, 0], yz[:, 1]], axis=-1).astype(
        np.float32
    )
    ranges = dict(
        chip1={"thermal.heat_flux": (8e5, 1e6)},
        chip2={"thermal.heat_flux": (8e5, 1e6)},
        inlet1={
            "momentum.mass_flow_rate": (0.005, 0.01),
            "thermal.total_temperature": (300.0, 340.0),
        },
        inlet2={
            "momentum.mass_flow_rate": (0.005, 0.01),
            "thermal.total_temperature": (300.0, 340.0),
        },
    )
    doe = {
        bc: {k: list(rng.uniform(lo, hi, n)) for k, (lo, hi) in v.items()}
        for bc, v in ranges.items()
    }
    doe["chip1"]["thermal.material"] = "copper"  # a scalar entry the loader skips
    (folder / "model_setup.json").write_text(json.dumps(dict(doe_configuration=doe)))
    y, z = coords[:, 1] / 0.02, coords[:, 2] / 0.06
    for i in range(n):
        q1 = doe["chip1"]["thermal.heat_flux"][i] / 1e6
        q2 = doe["chip2"]["thermal.heat_flux"][i] / 1e6
        t1 = doe["inlet1"]["thermal.total_temperature"][i]
        t2 = doe["inlet2"]["thermal.total_temperature"][i]
        temp = (
            0.5 * (t1 + t2)
            + 80 * q1 * np.exp(-((z - 0.25) ** 2 + (y - 0.5) ** 2) / 0.02)
            + 80 * q2 * np.exp(-((z - 0.75) ** 2 + (y - 0.5) ** 2) / 0.02)
            + (t2 - t1) * (z - 0.5)
        )
        np.savez(
            folder / f"sim_{i + 1:04d}.npz",
            **{FIELD_KEY: temp.astype(np.float32), COORD_KEY: coords},
        )
    return folder


def benchmark_raw(folder: Path) -> dict:
    """The benchmark's own read of a folder (``_data.load_raw``)."""
    from scripts.benchmarks.chip_cooling._data import load_raw

    return load_raw(folder)
