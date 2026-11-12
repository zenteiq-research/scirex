# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Navier-Stokes FNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions are compared with the benchmark's (``scripts/benchmarks/
navier_stokes``), and its ``main`` runs on a small synthetic ``.mat`` file,
so nothing is downloaded. The test that reads the real dataset runs only when
``NS_DATA_DIR`` points at the folder with ``NavierStokes_V1e-5_N1200_T20.mat``.
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.io

from scirex.utils import count_model_params
from scripts.benchmarks.navier_stokes import _data as bench_data
from scripts.benchmarks.navier_stokes._config import PROTOCOL as BENCH_PROTOCOL
from scripts.benchmarks.navier_stokes._config import RUNS
from scripts.benchmarks.navier_stokes._models import get_spec
from scripts.tests.test_navier_stokes_benchmark import TINY_ARCH
from tests import tolerances
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "navierstokes_fno.py"
DATA_FILE = "NavierStokes_V1e-5_N1200_T20.mat"
H = W = 16


def _load_script():
    spec = importlib.util.spec_from_file_location("ns_fno_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_mat(folder: Path, n: int, grid: int) -> Path:
    """``u`` with the real layout ``(n, grid, grid, 20)``; sample i is i + 1."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(n, grid, grid, 20)).astype(np.float32)
    u += np.arange(n, dtype=np.float32)[:, None, None, None] + 1.0
    scipy.io.savemat(folder / DATA_FILE, {"u": u})
    return folder


def _tiny_arch() -> dict:
    arch = dict(TINY_ARCH["fno"])
    arch.pop("gelu", None)  # the tutorial always uses the exact GELU
    return arch


def test_load_takes_the_benchmark_split(tmp_path: Path) -> None:
    tutorial = _load_script()
    folder = _fake_mat(tmp_path, n=9, grid=H)
    data = tutorial.load_ns(folder / DATA_FILE, ntrain=4, ntest=3, t_in=10, t_out=10)
    bench = bench_data.load_split(
        folder / DATA_FILE, dict(BENCH_PROTOCOL, ntrain=4, ntest=3)
    )
    for key in ("a_train", "u_train", "a_test", "u_test"):
        np.testing.assert_array_equal(
            data[key].reshape(bench[key].shape), bench[key], err_msg=key
        )
    with pytest.raises(ValueError, match="need"):
        tutorial.load_ns(folder / DATA_FILE, ntrain=8, ntest=3, t_in=10, t_out=10)


def test_full_size_and_protocol_match_the_benchmark() -> None:
    tutorial = _load_script()
    arch = dict(RUNS["fno"].arch)
    arch.pop("gelu", None)
    assert tutorial.SIZES["full"] == arch
    _, params = tutorial.build_model(
        tutorial.SIZES["full"], 10, np.zeros((64, 64, 10), np.float32), seed=0
    )
    assert count_model_params(params) == 11_183_467
    p = tutorial.PROTOCOL
    for key in ("ntrain", "ntest", "t_in", "t_out", "epochs", "batch_size",
                "weight_decay", "adam_eps", "pct_start", "div_factor",
                "final_div_factor"):  # fmt: skip
        assert p[key] == BENCH_PROTOCOL[key], key
    assert p["learning_rate"] == BENCH_PROTOCOL["lr"]
    assert BENCH_PROTOCOL["max_grad_norm"] is None


def test_step_matches_the_benchmark_model() -> None:
    """Same weights, same history: the tutorial's channels-last step gives the
    benchmark spec's next frame, so the layouts agree."""
    tutorial = _load_script()
    rng = np.random.default_rng(1)
    fx = rng.normal(size=(2, H, W, 10)).astype(np.float32)
    model, params = tutorial.build_model(_tiny_arch(), 10, fx[0], seed=0)
    ours = tutorial.make_step(model)(params, jnp.asarray(fx))
    spec = get_spec("fno")
    bench_model = spec.build(TINY_ARCH["fno"], (H, W), 10)
    theirs = spec.apply(
        bench_model, params, jnp.asarray(fx.reshape(2, H * W, 10)), (H, W)
    )
    np.testing.assert_allclose(
        np.asarray(ours).reshape(2, H * W, 1),
        np.asarray(theirs),
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )


def test_teacher_forcing_and_rollout_match_the_benchmark() -> None:
    tutorial = _load_script()
    rng = np.random.default_rng(2)
    fx = jnp.asarray(rng.normal(size=(2, H, W, 10)).astype(np.float32))
    yy = jnp.asarray(rng.normal(size=(2, H, W, 4)).astype(np.float32))
    model, params = tutorial.build_model(_tiny_arch(), 10, np.asarray(fx[0]), seed=0)
    step = tutorial.make_step(model)

    def flat_step(h):
        return step(params, h.reshape(2, H, W, -1)).reshape(2, H * W, 1)

    loss = tutorial.teacher_forced(step, params, fx, yy)
    bench_loss, _ = bench_data.teacher_forced(
        flat_step, fx.reshape(2, H * W, 10), yy.reshape(2, H * W, 4)
    )
    np.testing.assert_allclose(
        float(loss), float(bench_loss), rtol=tolerances.rtol(1e-5)
    )
    ours = tutorial.rollout(step, params, fx, 4)
    theirs = bench_data.rollout(flat_step, fx.reshape(2, H * W, 10), 4)
    np.testing.assert_allclose(
        np.asarray(ours).reshape(2, H * W, 4),
        np.asarray(theirs),
        rtol=tolerances.rtol(1e-5),
        atol=tolerances.atol(1e-5),
    )


def test_optimizer_keeps_the_hyperparameters_real() -> None:
    tutorial = _load_script()
    _, params = tutorial.build_model(
        _tiny_arch(), 10, np.zeros((H, W, 10), np.float32), seed=0
    )
    tx, _ = tutorial.make_optimizer(tutorial.PROTOCOL, 10, "adamw_cx")
    state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, state = tx.update(grads, state, params)
    for p, u in zip(
        jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updates)
    ):
        assert u.dtype == p.dtype
    hyper = [
        s.hyperparams
        for s in jax.tree_util.tree_leaves(
            state, is_leaf=lambda s: hasattr(s, "hyperparams")
        )
        if hasattr(s, "hyperparams")
    ]
    assert hyper and all(not jnp.iscomplexobj(v) for h in hyper for v in h.values())


@pytest.mark.parametrize("optimizer", ["adamw_cx", "adamw"])
def test_optimizer_conjugates_complex_gradients(optimizer: str) -> None:
    tutorial = _load_script()
    tx, _ = tutorial.make_optimizer(tutorial.PROTOCOL, 10, optimizer)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))

    def loss(p):
        return jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2)

    updates, _ = tx.update(jax.grad(loss)(params), tx.init(params), params)
    assert updates["w"].dtype == jnp.complex64
    assert updates["b"].dtype == params["b"].dtype
    step = complex(updates["w"][0])
    # adamw_cx moves w towards 0 along -(1 + 1j); plain optax moves the
    # imaginary part the wrong way
    assert step.real < 0
    assert (step.imag < 0) == (optimizer == "adamw_cx")


def test_benchmark_run_uses_adamw_cx() -> None:
    assert RUNS["fno"].optimizer == "adamw_cx"
    assert not RUNS["fno"].cycle_momentum


def test_smoke_main_on_a_synthetic_file(tmp_path: Path) -> None:
    tutorial = _load_script()
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(_fake_mat(tmp_path / "data", n=4, grid=64)),
            "--smoke",
            "--n-train", "2",
            "--n-test", "2",
            "--epochs", "1",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
    assert np.isfinite(metrics["copy_last_frame"])
    for name in ("prediction.png", "rollout.png", "loss.png"):
        assert (out / name).stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("NS_DATA_DIR"),
    reason="set NS_DATA_DIR to the folder with NavierStokes_V1e-5_N1200_T20.mat",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = _load_script()
    metrics = tutorial.main(
        ["--smoke", "--n-train", "2", "--n-test", "2", "--epochs", "1",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["NS_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
