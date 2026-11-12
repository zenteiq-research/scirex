# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the AirfRANS Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's pieces
are compared with the benchmark's own code (``scripts/benchmarks/airfrans``)
on small synthetic cases written in the real file layout, and its ``main``
runs end to end on them, so nothing is downloaded. The test that reads the
real dataset runs only when ``AIRFRANS_DATA_DIR`` points at it.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vtk")

import jax
import jax.numpy as jnp

from tests.docs.airfrans_tutorial_helpers import write_dataset
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "airfrans_transolver.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "airfrans_transolver_tutorial", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    jax.config.update("jax_enable_x64", False)
    return module


@pytest.fixture(scope="module")
def tutorial():
    return _load_script()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("airfrans") / "Dataset")


def _ids(root: Path) -> list[str]:
    return json.loads((root / "manifest.json").read_text())["full_train"]


def _same_tree(a, b) -> bool:
    return jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda u, v: bool(jnp.array_equal(u, v)), a, b)
    )


def test_cases_and_normalisation_match_the_benchmark(tutorial, dataset) -> None:
    from scripts.benchmarks.airfrans import _data as D

    ids = _ids(dataset)
    mine = [tutorial.read_case(dataset, c) for c in ids]
    ref = [D.build_case(dataset, c) for c in ids]
    for a, b in zip(mine, ref, strict=True):
        for key in ("x7", "y", "pos", "surf"):
            np.testing.assert_array_equal(a[key], b[key])
    norm, stats = tutorial.fit_normalisation(mine), D.model_stats(ref)
    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        np.testing.assert_array_equal(norm[key], getattr(stats, key))
    a, b = tutorial.prepare_case(mine[0], norm), D.prepare_case(ref[0], stats)
    for key in ("x", "y", "scale"):
        np.testing.assert_array_equal(a[key], b[key])
    pred = np.random.default_rng(0).normal(size=a["y"].shape).astype(np.float32)
    np.testing.assert_array_equal(
        tutorial.to_physical(pred, a["scale"], norm),
        D.to_physical(pred, b["scale"], stats),
    )
    for u, v in zip(
        tutorial.official_output_stats(mine), D.official_output_stats(ref), strict=True
    ):
        np.testing.assert_array_equal(u, v)


def test_protocol_matches_the_benchmark(tutorial) -> None:
    from scripts.benchmarks.airfrans._config import PROTOCOL as BENCH

    p = tutorial.PROTOCOL
    for key in ("n_val", "subsample", "epochs", "lr", "pct_start", "div_factor",
                "final_div_factor", "base_momentum", "max_momentum", "surf_weight",
                "val_every", "val_repeats"):  # fmt: skip
        assert p[key] == BENCH[key], key
    add = BENCH["additions"]
    assert p["ema_decay"] == add["ema_decay"]
    assert p["grad_clip_global_norm"] == add["grad_clip_global_norm"]
    assert p["matmul_precision"] == add["matmul_precision"]


def test_full_size_init_and_output_match_the_benchmark(tutorial, dataset) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.airfrans._config import RUNS
    from scripts.benchmarks.airfrans._models import get_spec

    assert tutorial.SIZES["full"] == RUNS["transolver"].arch
    case = tutorial.read_case(dataset, _ids(dataset)[0])
    d = tutorial.prepare_case(case, tutorial.fit_normalisation([case]))
    x, pos = jnp.asarray(d["x"]), jnp.asarray(d["pos"])
    model = tutorial.build_model(tutorial.SIZES["full"])
    params = tutorial.init_params(model, 0, x, pos)
    assert count_model_params(params) == 3_894_596
    spec = get_spec("transolver")
    bench_model = spec.build(RUNS["transolver"].arch)
    bench_params = spec.init(bench_model, jax.random.PRNGKey(0), x, pos, None)
    assert _same_tree(params, bench_params)
    np.testing.assert_array_equal(
        tutorial.apply_model(model, params, x, pos),
        spec.apply(bench_model, bench_params, x, pos, None),
    )


def test_schedule_and_loss_match_the_benchmark(tutorial) -> None:
    from scripts.benchmarks.airfrans._config import PROTOCOL as BENCH
    from scripts.benchmarks.airfrans.train import onecycle, region_losses

    lr_a, b1_a = tutorial.onecycle(1000, tutorial.PROTOCOL)
    lr_b, b1_b = onecycle(1000, BENCH)
    for k in (0, 1, 150, 299, 300, 700, 999):
        assert float(lr_a(k)) == float(lr_b(k))
        assert float(b1_a(k)) == float(b1_b(k))
    rng = np.random.default_rng(1)
    out, y = (jnp.asarray(rng.normal(size=(50, 4)), jnp.float32) for _ in range(2))
    surf = jnp.asarray(rng.random(50) < 0.2)
    for a, b in zip(
        tutorial.region_losses(out, y, surf), region_losses(out, y, surf), strict=True
    ):
        assert float(a) == float(b)
    opt = tutorial.make_optimizer()
    state = opt.init({"w": jnp.ones(3, jnp.complex64)})
    assert all(not jnp.iscomplexobj(v) for v in state.hyperparams.values())


def test_scores_and_forces_match_the_benchmark(tutorial, dataset) -> None:
    from scripts.benchmarks.airfrans._metrics import (
        CaseGeometry,
        cover_predict,
        field_scores,
        impose_wall,
    )

    for cid in _ids(dataset)[:3]:
        case = tutorial.read_case(dataset, cid)
        geo = CaseGeometry(dataset, cid)
        np.testing.assert_allclose(
            tutorial.force_coefficients(dataset, cid), geo.coefficients(), rtol=1e-12
        )
        pred = impose_wall(case["y"] * 1.05, case["surf"])
        np.testing.assert_allclose(
            tutorial.force_coefficients(dataset, cid, pred),
            geo.coefficients(pred),
            rtol=1e-12,
        )
        mean, std = case["y"].mean(0), case["y"].std(0)
        mine = tutorial.field_scores(pred, case["y"], case["surf"], mean, std)
        ref = field_scores(pred, case["y"], case["surf"], mean, std)
        assert mine["volume"] == pytest.approx(ref["volume"], rel=1e-12)
        assert mine["surf"] == pytest.approx(ref["surf"], rel=1e-12)

    def predict(idx):
        return np.repeat(idx[:, None].astype(float), 4, axis=1)

    a = tutorial.cover_predict(predict, 100, 30, np.random.default_rng(0))
    b, _ = cover_predict(predict, 100, 30, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


def test_smoke_main_on_synthetic_cases(
    tutorial, dataset, tmp_path, monkeypatch
) -> None:
    # The synthetic manifest has 6 training cases, so hold out 2, not 80.
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
