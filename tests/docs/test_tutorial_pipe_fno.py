# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Pipe FNO tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on small synthetic ``Pipe_*.npy`` files, so
nothing is downloaded. The test that reads the real dataset runs only when
``PIPE_DATA_DIR`` points at a folder with the three files.
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import tolerances
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "pipe_fno.py"
N = 32  # mesh size of the synthetic files; large enough for the smoke FNO's modes


def _load_script():
    spec = importlib.util.spec_from_file_location("pipe_fno_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_pipe(folder: Path, n: int = 1200) -> Path:
    """Write ``Pipe_X/Y/Q.npy`` with the real layout on an ``N x N`` mesh."""
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    s = np.linspace(0.0, 1.0, N, dtype=np.float32)
    bend = rng.normal(scale=0.3, size=(n, 1, 1)).astype(np.float32)
    x = np.broadcast_to(10.0 * s[None, :, None], (n, N, N)).astype(np.float32)
    y = (s[None, None, :] - 0.5 + bend * np.sin(np.pi * s)[None, :, None]).astype(
        np.float32
    )
    q = np.zeros((n, 3, N, N), dtype=np.float32)
    q[:, 0] = 0.25 * (1.0 - (2.0 * s[None, None, :] - 1.0) ** 2) * (1.0 + bend)
    np.save(folder / "Pipe_X.npy", x)
    np.save(folder / "Pipe_Y.npy", y)
    np.save(folder / "Pipe_Q.npy", q)
    return folder


def test_load_pipe_takes_the_benchmark_split(tmp_path: Path) -> None:
    tutorial = _load_script()
    data = tutorial.load_pipe(_fake_pipe(tmp_path), ntrain=6, ntest=3)
    q = np.load(tmp_path / "Pipe_Q.npy")
    assert data["x_train"].shape == (6, N, N, 2)
    assert data["y_test"].shape == (3, N, N)
    np.testing.assert_array_equal(data["y_train"], q[:6, 0])
    np.testing.assert_array_equal(data["y_test"], q[1197:1200, 0])
    with pytest.raises(ValueError, match="need"):
        tutorial.load_pipe(tmp_path, ntrain=1000, ntest=201)


def test_unit_gaussian_round_trip() -> None:
    tutorial = _load_script()
    v = np.random.default_rng(1).normal(3.0, 2.0, size=(5, 4, 4, 2)).astype(np.float32)
    norm = tutorial.UnitGaussian(v, axes=(0, 1, 2))
    enc = norm.encode(v)
    np.testing.assert_allclose(
        enc.mean(axis=(0, 1, 2)), 0.0, atol=tolerances.atol(1e-5)
    )
    np.testing.assert_allclose(enc.std(axis=(0, 1, 2)), 1.0, atol=1e-4)
    np.testing.assert_allclose(
        norm.decode(enc), v, rtol=tolerances.rtol(1e-5), atol=tolerances.atol(1e-5)
    )


def test_rel_l2_matches_definition() -> None:
    tutorial = _load_script()
    true = jnp.array([[3.0, 4.0]])
    assert float(tutorial.rel_l2_per_sample(true, true)[0]) == 0.0
    assert float(
        tutorial.rel_l2_per_sample(jnp.zeros((1, 2)), true)[0]
    ) == pytest.approx(1.0)


def test_full_size_matches_the_benchmark() -> None:
    from scirex.models import FNO
    from scirex.utils import count_model_params
    from scripts.benchmarks.pipe._config import PROTOCOL as BENCH_PROTOCOL
    from scripts.benchmarks.pipe._config import RUNS

    tutorial = _load_script()
    assert tutorial.SIZES["full"] == RUNS["fno"].arch
    model = FNO(
        in_channels=2,
        out_channels=1,
        positional_embedding=None,
        **tutorial.SIZES["full"],
    )
    params = jax.eval_shape(
        lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2, 129, 129)))
    )
    assert count_model_params(params) == 2_999_629
    p = tutorial.PROTOCOL
    for key in ("ntrain", "ntest", "n_total", "epochs", "batch_size", "weight_decay",
                "adam_eps", "pct_start", "div_factor", "final_div_factor",
                "max_grad_norm"):  # fmt: skip
        assert p[key] == BENCH_PROTOCOL[key], key
    assert p["learning_rate"] == BENCH_PROTOCOL["lr"]
    assert BENCH_PROTOCOL["optimizer"] == "adamw_cx"


@pytest.mark.parametrize("optimizer", ["adamw_cx", "adamw"])
def test_optimizer_keeps_dtypes_and_conjugates_complex_gradients(
    optimizer: str,
) -> None:
    tutorial = _load_script()
    tx, _ = tutorial.make_optimizer(tutorial.PROTOCOL, 10, optimizer)
    params = dict(w=jnp.array([1.0 + 1.0j], dtype=jnp.complex64), b=jnp.ones(1))
    loss = lambda p: jnp.sum(jnp.abs(p["w"]) ** 2) + jnp.sum(p["b"] ** 2)  # noqa: E731
    grads = jax.grad(loss)(params)
    updates, _ = tx.update(grads, tx.init(params), params)
    assert updates["w"].dtype == jnp.complex64
    assert updates["b"].dtype == params["b"].dtype
    step = complex(updates["w"][0])
    if optimizer == "adamw_cx":
        # descent on |w|^2 moves w towards 0 along -(1 + 1j)
        assert step.real < 0 and step.imag < 0
    else:
        # plain optax: the imaginary part moves the wrong way
        assert step.real < 0 and step.imag > 0


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = _load_script()
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(_fake_pipe(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "2",
            "--epochs", "2",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
    assert metrics["worst"] >= metrics["median"]
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("PIPE_DATA_DIR"),
    reason="set PIPE_DATA_DIR to the folder with the real Pipe_*.npy files",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = _load_script()
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["PIPE_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
