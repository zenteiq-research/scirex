# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the AirfRANS WNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's grid
operator, geometry grid and case reader are compared with the benchmark's own
code on small synthetic cases written in the real file layout, and its
``main`` runs end to end on them. The data, schedule and metric code is the
same as in the Transolver tutorial and is checked there too. The test that
reads the real dataset runs only when ``AIRFRANS_DATA_DIR`` points at it.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from tests import tolerances

pytest.importorskip("vtk")

import jax
import jax.numpy as jnp

from tests.docs.airfrans_tutorial_helpers import write_dataset
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "airfrans_wno.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("airfrans_wno_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Flax modules look their class up here
    spec.loader.exec_module(module)
    jax.config.update("jax_enable_x64", False)
    return module


@pytest.fixture(scope="module")
def tutorial():
    return _load_script()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("airfrans") / "Dataset")


def _first_case(root: Path) -> str:
    return json.loads((root / "manifest.json").read_text())["full_train"][0]


def test_case_and_geometry_match_the_benchmark(tutorial, dataset) -> None:
    from scripts.benchmarks.airfrans import _data as D
    from scripts.benchmarks.airfrans._config import RUNS
    from scripts.benchmarks.airfrans._models import get_spec

    cid = _first_case(dataset)
    mine, ref = tutorial.read_case(dataset, cid), D.build_case(dataset, cid)
    for key in ("x7", "y", "pos", "surf", "segments"):
        np.testing.assert_array_equal(mine[key], ref[key])
    for size, arch in ((tutorial.SIZES["full"], RUNS["wno"].arch),
                       (tutorial.SIZES["smoke"], tutorial.SIZES["smoke"])):  # fmt: skip
        np.testing.assert_array_equal(
            tutorial.case_geometry(mine["segments"], size),
            get_spec("wno").extra(arch, ref),
        )


def test_full_size_init_and_output_match_the_benchmark(tutorial, dataset) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.airfrans._config import RUNS
    from scripts.benchmarks.airfrans._models import get_spec

    assert tutorial.SIZES["full"] == RUNS["wno"].arch
    case = tutorial.read_case(dataset, _first_case(dataset))
    d = tutorial.prepare_case(case, tutorial.fit_normalisation([case]))
    x, pos = jnp.asarray(d["x"]), jnp.asarray(d["pos"])
    geo = tutorial.case_geometry(case["segments"], tutorial.SIZES["full"])
    model = tutorial.build_model(tutorial.SIZES["full"])
    params = tutorial.init_params(model, 0, x, pos, geo)
    assert count_model_params(params) == 3_964_728
    spec = get_spec("wno")
    bench_model = spec.build(RUNS["wno"].arch)
    bench_params = spec.init(bench_model, jax.random.PRNGKey(0), x, pos, geo)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda u, v: bool(jnp.array_equal(u, v)), params, bench_params
        )
    )
    np.testing.assert_array_equal(
        tutorial.apply_model(model, params, x, pos, geo),
        spec.apply(bench_model, bench_params, x, pos, geo),
    )


def test_bilinear_reproduces_a_linear_field(tutorial) -> None:
    H, W = 9, 17
    pts = tutorial.grid_points(H, W).reshape(H, W, 2)
    field = 2.0 * pts[..., :1] - 3.0 * pts[..., 1:]  # linear in x and y
    q = jnp.array([[0.3, 0.1], [-1.9, 1.4], [3.7, -1.2]])
    np.testing.assert_allclose(
        tutorial.bilinear(field, q)[:, 0],
        2.0 * q[:, 0] - 3.0 * q[:, 1],
        atol=tolerances.atol(1e-5),
    )


def test_smoke_main_on_synthetic_cases(
    tutorial, dataset, tmp_path, monkeypatch
) -> None:
    monkeypatch.setitem(tutorial.PROTOCOL, "n_val", 2)
    monkeypatch.setitem(tutorial.PROTOCOL, "subsample", 500)
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(dataset),
            "--smoke",
            "--n-train", "3",
            "--n-val", "2",
            "--n-test", "3",
            "--epochs", "2",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    for key in ("volume", "surf", "relerr_cl", "spearman_cl"):
        assert np.isfinite(metrics[key]), key
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("AIRFRANS_DATA_DIR"),
    reason="set AIRFRANS_DATA_DIR to the real AirfRANS Dataset folder",
)
def test_smoke_main_on_real_data(tutorial, tmp_path) -> None:
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "2", "--n-val", "1",
         "--n-test", "2", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["AIRFRANS_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["volume"])
