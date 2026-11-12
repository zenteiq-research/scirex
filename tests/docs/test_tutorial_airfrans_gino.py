# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the AirfRANS GINO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script is
compared with the benchmark's GINO trainer (``scripts/benchmarks/airfrans/
_gino.py`` with ``_config.GINO_ARGS``) on small synthetic cases written in the
real file layout, and its ``main`` runs end to end on them. The test that
reads the real dataset runs only when ``AIRFRANS_DATA_DIR`` points at it.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vtk")
pytest.importorskip("sklearn")

import jax
import jax.numpy as jnp

from tests.docs.airfrans_tutorial_helpers import write_dataset
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "airfrans_gino.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("airfrans_gino_tutorial", SCRIPT)
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


def _bench_args(size: dict, *extra: str):
    from scripts.benchmarks.airfrans import _gino

    return _gino._parser().parse_args(
        [
            "--data-dir", "unused",
            "--latent-sdf",
            "--decoder-point-features",
            "--latent-res", *map(str, size["latent_res"]),
            "--in-radius", str(size["in_radius"]),
            "--out-radius", str(size["out_radius"]),
            "--in-k", str(size["in_k"]),
            "--out-k", str(size["out_k"]),
            "--fno-modes", *map(str, size["fno_modes"]),
            "--fno-hidden", str(size["fno_hidden"]),
            "--fno-layers", str(size["fno_layers"]),
            "--fno-lifting-ratio", str(size["fno_lifting_ratio"]),
            "--out-gno-mlp", *map(str, size["out_gno_mlp"]),
            *extra,
        ]
    )  # fmt: skip


def test_full_size_and_protocol_are_the_published_run(tutorial) -> None:
    from scirex.utils import count_model_params
    from scripts.benchmarks.airfrans import _gino
    from scripts.benchmarks.airfrans._config import GINO_ARGS

    published = _gino._parser().parse_args([*GINO_ARGS, "--data-dir", "unused"])
    mine = _bench_args(tutorial.SIZES["full"], "--epochs", "500", "--loader-surf-mask")

    def norm(ns):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in vars(ns).items()}

    assert norm(mine) == norm(published)
    p = tutorial.PROTOCOL
    assert p["epochs"] == published.epochs
    assert p["n_points"] == published.n_points
    assert p["learning_rate"] == published.learning_rate
    assert p["final_div_factor"] == published.final_div_factor
    assert p["surf_weight"] == published.weight
    size = tutorial.SIZES["full"]
    lq = jnp.asarray(tutorial.latent_grid(size["latent_res"]))
    model = tutorial.build_model(size)
    params = jax.eval_shape(lambda: tutorial.init_params(model, lq, size, 0, 4096))
    assert count_model_params(params) == 3_894_069


def test_init_forward_loss_and_forces_match_the_benchmark(tutorial, dataset) -> None:
    from scirex.data import load_airfrans
    from scripts.benchmarks.airfrans import _gino

    size = tutorial.SIZES["smoke"]
    args = _bench_args(size)
    lq_np = tutorial.latent_grid(size["latent_res"])
    lq = jnp.asarray(lq_np)
    model, bench_model = tutorial.build_model(size), _gino.build_model(args)
    params = tutorial.init_params(model, lq, size, 0, 2000)
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (2000, 2), minval=-1.0, maxval=1.0)
    x = jnp.zeros((2000, 7))
    bench_params = bench_model.init(
        key,
        input_geom=pos[None],
        latent_queries=lq[None],
        output_queries=pos[None],
        x=x[None],
        latent_features=jnp.zeros((*lq.shape[:-1], 1))[None],
        neighbors_in=_gino.radius_neighbors(
            pos, lq.reshape(-1, 2), args.in_radius, args.in_k
        ),
        neighbors_out=_gino.radius_neighbors(
            lq.reshape(-1, 2), pos, args.out_radius, args.out_k
        ),
        output_query_features=x[None],
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda u, v: bool(jnp.array_equal(u, v)), params, bench_params
        )
    )

    data = load_airfrans(dataset, task="full")
    s = data["train"][0]
    lat = tutorial.latent_sdf(s, lq_np)
    np.testing.assert_array_equal(lat, _gino.latent_sdf(s, lq_np))
    xs, ps = jnp.asarray(s["x"]), jnp.asarray(s["pos"])
    out = tutorial.make_forward(model, lq, size)(params, xs, ps, jnp.asarray(lat))
    ref = _gino.make_forward(bench_model, lq, args)(bench_params, xs, ps, lat)
    np.testing.assert_array_equal(out, ref)
    y, surf = jnp.asarray(s["y"]), jnp.asarray(s["surf"])
    assert float(tutorial.region_losses(out, y, surf, 1.0)[0]) == float(
        _gino.region_losses(ref, y, surf, 1.0)[0]
    )

    np.testing.assert_array_equal(
        tutorial.wall_mask(s, dataset), _gino.wall_mask(s, dataset)
    )
    cid = s["case_id"]
    internal = tutorial.read_vtk(dataset / cid / f"{cid}_internal.vtu")
    foil = tutorial.read_vtk(dataset / cid / f"{cid}_aerofoil.vtp")
    mean_out, std_out = data["coef_norm"][2:]
    phys = s["y"] * (std_out + 1e-8) + mean_out
    u_inf, angle = float(cid.split("_")[2]), float(cid.split("_")[3])
    np.testing.assert_allclose(
        tutorial.lift_drag(
            internal, foil, phys[:, :2], phys[:, 2], s["surf"], u_inf, angle
        ),
        _gino.lift_drag(
            internal, foil, phys[:, :2], phys[:, 2], s["surf"], u_inf, angle
        ),
        rtol=1e-12,
    )


def test_radius_neighbors_finds_exactly_the_nodes_in_range(tutorial) -> None:
    rng = np.random.default_rng(0)
    src = jnp.asarray(rng.uniform(-1, 1, (300, 2)), jnp.float32)
    qry = jnp.asarray(rng.uniform(-1, 1, (40, 2)), jnp.float32)
    nb = tutorial.radius_neighbors(src, qry, 0.3, 300, chunk=16)
    seg, idx = np.asarray(nb["segment_ids"]), np.asarray(nb["neighbors_index"])
    for q in range(40):
        found = set(idx[seg == q].tolist())
        d = np.linalg.norm(np.asarray(src) - np.asarray(qry[q]), axis=1)
        assert found == set(np.flatnonzero(d <= 0.3).tolist())


def test_optimizer_conjugates_complex_gradients(tutorial) -> None:
    tx, _ = tutorial.make_optimizer(tutorial.PROTOCOL, 10)
    params = dict(w=jnp.array([1.0 + 1.0j], jnp.complex64), b=jnp.ones(1))
    loss = lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2)  # noqa: E731
    updates, _ = tx.update(jax.grad(loss)(params), tx.init(params), params)
    step = complex(updates["w"][0])
    assert step.real < 0 and step.imag < 0  # towards 0 along -(1 + 1j)
    assert updates["b"].dtype == params["b"].dtype


def test_smoke_main_on_synthetic_cases(tutorial, dataset, tmp_path) -> None:
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(dataset),
            "--smoke",
            "--n-train", "3",
            "--n-test", "3",
            "--epochs", "2",
            "--n-points", "600",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    for key in ("volume", "surf", "relerr_cl", "spearman_cl"):
        assert np.isfinite(metrics[key]), key
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0
    small = json.loads((out / "manifest_small.json").read_text())
    assert len(small["full_train"]) == 3


@pytest.mark.skipif(
    not os.environ.get("AIRFRANS_DATA_DIR"),
    reason="set AIRFRANS_DATA_DIR to the real AirfRANS Dataset folder",
)
def test_smoke_main_on_real_data(tutorial, tmp_path) -> None:
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--n-train", "2", "--n-test", "2",
         "--n-points", "4000", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["AIRFRANS_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["volume"])
