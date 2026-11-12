# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/pipe/.

Everything runs on a few synthetic 129x129 samples on the CPU, so no dataset
or GPU is needed: the data split and normalisation, the model layouts, the
size of the published runs, and the train -> checkpoint -> infer round trip.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.pipe import infer  # noqa: E402
from scripts.benchmarks.pipe._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
)
from scripts.benchmarks.pipe._config import (  # noqa: E402
    PROTOCOL,
    RUNS,
    TRANSOLVER_PARAMS,
)
from scripts.benchmarks.pipe._data import load_split  # noqa: E402
from scripts.benchmarks.pipe._models import (  # noqa: E402
    SPECS,
    get_spec,
    template_inputs,
)
from scripts.benchmarks.pipe.train import make_optimizer, train_one  # noqa: E402

N = 129
SMALL = dict(PROTOCOL, ntrain=4, ntest=2, n_total=6, batch_size=2)
TINY_ARCH = {
    "transolver": dict(
        n_hidden=8,
        n_layers=1,
        n_head=2,
        mlp_ratio=1,
        slice_num=4,
        ref=8,
        dropout_rate=0.0,
    ),
    "wno": dict(
        hidden_channels=4,
        n_layers=1,
        level=1,
        wavelet="db4",
        projection_hidden_channels=8,
    ),
    "fno": dict(
        n_modes=(4, 4),
        hidden_channels=4,
        n_layers=1,
        projection_channel_ratio=2,
        domain_padding=8 / 129,
        fno_skip="linear",
        use_channel_mlp=False,
    ),
    "gino": dict(
        in_gno_radius=0.05,
        out_gno_radius=0.05,
        fno_n_modes=(4, 4),
        fno_hidden_channels=4,
        fno_n_layers=1,
        fno_lifting_channel_ratio=1,
        in_gno_channel_mlp_hidden_layers=(8,),
        out_gno_channel_mlp_hidden_layers=(8,),
    ),
}


def test_fno_run_has_its_published_size() -> None:
    run = RUNS["fno"]
    spec = get_spec("fno")
    prepared = spec.prepare(run.arch, template_inputs((N, N)), (N, N))
    params = spec.build(run.arch, (N, N)).init(
        jax.random.PRNGKey(0), **spec.take(prepared, np.arange(1))
    )
    assert int(count_model_params(params)) == 2_999_629


def test_fno_run_keeps_pipe_py_modes() -> None:
    """Geo-FNO pipe.py keeps 12 modes each way: a 24 x 12 weight block."""
    run = RUNS["fno"]
    spec = get_spec("fno")
    prepared = spec.prepare(run.arch, template_inputs((N, N)), (N, N))
    params = spec.build(run.arch, (N, N)).init(
        jax.random.PRNGKey(0), **spec.take(prepared, np.arange(1))
    )
    weight = params["params"]["fno_blocks"]["convs_0"]["weight"]
    assert jnp.iscomplexobj(weight)
    assert weight.shape[-2:] == (24, 12)


def test_transolver_run_has_the_paper_size() -> None:
    run = RUNS["transolver"]
    spec = get_spec("transolver")
    prepared = spec.prepare(run.arch, template_inputs((N, N)), (N, N))
    params = spec.build(run.arch, (N, N)).init(
        jax.random.PRNGKey(0), **spec.take(prepared, np.arange(1))
    )
    assert int(count_model_params(params)) == TRANSOLVER_PARAMS


def test_transolver_has_the_paper_ports_parameters() -> None:
    """The benchmark's Transolver has the parameter tree of _transolver.py's
    model (the port of exp_pipe.py) at the same size; only the numerics
    differ (exact GELU, LayerNorm eps 1e-5, PyTorch's Conv2d init)."""
    from scirex import get_model_jax
    from scripts.benchmarks.pipe import _transolver as T
    from scripts.benchmarks.pipe._config import TRANSOLVER_ARGS

    config = T.build_config(T._parser().parse_args(TRANSOLVER_ARGS))
    config.model.height = config.model.width = N
    x = jnp.zeros((1, N * N, 2))
    port = jax.eval_shape(
        lambda: get_model_jax(config).init(jax.random.PRNGKey(0), x=x, fx=None)
    )
    ours = jax.eval_shape(
        lambda: (
            get_spec("transolver")
            .build(RUNS["transolver"].arch, (N, N))
            .init(jax.random.PRNGKey(0), x=x, fx=None)
        )
    )
    assert jax.tree_util.tree_map(lambda a: a.shape, ours) == jax.tree_util.tree_map(
        lambda a: a.shape, port
    )


def test_fno_layout_puts_channels_first() -> None:
    x = template_inputs((N, N))
    prepared = get_spec("fno").prepare(TINY_ARCH["fno"], x, (N, N))
    assert prepared.shape == (1, 2, N, N)
    np.testing.assert_array_equal(prepared[0, :, 3, 5], x[0, 3 * N + 5])


def test_gino_run_has_its_published_size() -> None:
    """Also checks that GINO's template graph fits in max_edges."""
    run = RUNS["gino"]
    spec = get_spec("gino")
    prepared = spec.prepare(run.arch, spec.template((N, N)), (N, N))
    params = spec.build(run.arch, (N, N)).init(
        jax.random.PRNGKey(0), **spec.take(prepared, np.arange(1))
    )
    assert int(count_model_params(params)) == 3_001_956


def test_gino_rejects_different_gno_radii() -> None:
    """Both GNO layers share one graph, built with a single radius."""
    arch = dict(TINY_ARCH["gino"], out_gno_radius=0.08)
    with pytest.raises(ValueError, match="out_gno_radius"):
        get_spec("gino").prepare(arch, get_spec("gino").template((N, N)), (N, N))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Six synthetic samples in the Pipe_X/Pipe_Y/Pipe_Q layout."""
    rng = np.random.default_rng(0)
    n = SMALL["n_total"]
    x = np.linspace(0.0, 10.0, N)[None, :, None] + 0.1 * rng.standard_normal((n, N, N))
    y = np.linspace(-1.0, 1.0, N)[None, None, :] + 0.1 * rng.standard_normal((n, N, N))
    q = rng.random((n, 3, N, N))
    np.save(tmp_path / "Pipe_X.npy", x.astype(np.float32))
    np.save(tmp_path / "Pipe_Y.npy", y.astype(np.float32))
    np.save(tmp_path / "Pipe_Q.npy", q.astype(np.float32))
    return tmp_path


def test_split_takes_first_train_and_last_test_samples(data_dir: Path) -> None:
    data = load_split(data_dir, SMALL)
    q = np.load(data_dir / "Pipe_Q.npy")[:, 0].reshape(SMALL["n_total"], -1)

    assert data["grid"] == (N, N)
    assert data["x_train"].shape == (4, N * N, 2)
    assert data["y_test"].shape == (2, N * N, 1)
    np.testing.assert_array_equal(data["y_train"][..., 0], q[:4])
    np.testing.assert_array_equal(data["y_test"][..., 0], q[-2:])
    # Inputs are normalised with training statistics; targets stay physical.
    np.testing.assert_allclose(data["x_train"].mean(axis=(0, 1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(data["x_train"].std(axis=(0, 1)), 1.0, atol=1e-4)


def test_split_refuses_a_truncated_target_file(data_dir: Path) -> None:
    """The Transolver loader would fall back to Pipe_Q_interp_x.npy."""
    q = data_dir / "Pipe_Q.npy"
    q.write_bytes(q.read_bytes()[:-1000])
    with pytest.raises(ValueError, match="truncated"):
        load_split(data_dir, SMALL)


@pytest.mark.parametrize("model", sorted(SPECS))
def test_spec_maps_samples_to_flat_predictions(model: str) -> None:
    """prepare -> take -> apply gives one (H * W, 1) prediction per sample."""
    spec = get_spec(model)
    arch = TINY_ARCH[model]
    x = np.concatenate([spec.template((N, N))] * 3)
    prepared = spec.prepare(arch, x, (N, N))
    nn_model = spec.build(arch, (N, N))
    idx = np.arange(spec.batch_size or 2)
    params = nn_model.init(jax.random.PRNGKey(0), **spec.take(prepared, idx))
    out = spec.apply(nn_model, params, spec.take(prepared, idx))
    assert out.shape == (len(idx), N * N, 1)


@pytest.mark.parametrize("name", sorted(RUNS))
def test_published_runs_match_transolver_size(name: str) -> None:
    """Every run stays within 2.5% of Transolver's parameter count."""
    run = RUNS[name]
    spec = get_spec(run.model)
    prepared = spec.prepare(run.arch, spec.template((N, N)), (N, N))
    params = spec.build(run.arch, (N, N)).init(
        jax.random.PRNGKey(0), **spec.take(prepared, np.arange(1))
    )
    assert abs(int(count_model_params(params)) / TRANSOLVER_PARAMS - 1) < 0.025


def test_wno_run_has_its_published_size() -> None:
    run = RUNS["wno"]
    model = get_spec("wno").build(run.arch, (N, N))
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, N, N, 2)))
    assert int(count_model_params(params)) == 3_102_157


def test_schedule_peaks_at_pct_start() -> None:
    _, schedule = make_optimizer(PROTOCOL, total_steps=1000)
    assert float(schedule(300)) == pytest.approx(PROTOCOL["lr"], rel=1e-6)
    assert float(schedule(0)) == pytest.approx(
        PROTOCOL["lr"] / PROTOCOL["div_factor"], rel=1e-6
    )


def test_optimizer_descends_on_complex_weights() -> None:
    """Plain optax.adamw moves a complex weight's imaginary part uphill,
    because JAX's gradient is the conjugate of PyTorch's."""
    target = jnp.asarray([1.0 + 2.0j, -0.5 - 1.0j], jnp.complex64)

    def loss(w):
        return jnp.sum(jnp.abs(w - target) ** 2)

    protocol = dict(PROTOCOL, lr=0.05, weight_decay=0.0, max_grad_norm=None)
    tx, _ = make_optimizer(protocol, total_steps=200)
    w = jnp.zeros_like(target)
    state = tx.init(w)
    start = float(loss(w))
    for _ in range(200):
        updates, state = tx.update(jax.grad(loss)(w), state, w)
        w = optax.apply_updates(w, updates)
    assert float(loss(w)) < 0.05 * start


def test_optimizer_matches_adamw_on_real_weights() -> None:
    """On real weights the optimiser is AdamW (eps 1e-8, decoupled decay)."""
    w0 = jnp.asarray([0.3, -1.2, 2.0], jnp.float32)
    g = jnp.asarray([0.5, -0.1, 0.02], jnp.float32)
    protocol = dict(PROTOCOL, max_grad_norm=None)
    tx, schedule = make_optimizer(protocol, total_steps=100)
    updates, _ = tx.update(g, tx.init(w0), w0)
    lr, wd = float(schedule(0)), protocol["weight_decay"]
    # torch.optim.AdamW, first step: m/(1-b1) = g, sqrt(v/(1-b2)) = |g|.
    expected = -lr * (g / (jnp.abs(g) + protocol["adam_eps"]) + wd * w0)
    np.testing.assert_allclose(updates, expected, rtol=1e-4)


def test_per_sample_rel_l2() -> None:
    truth = np.ones((2, 4, 1), np.float32)
    pred = truth.copy()
    pred[1] *= 1.5
    np.testing.assert_allclose(infer.per_sample_rel_l2(pred, truth), [0.0, 0.5])


@pytest.mark.parametrize("model", sorted(SPECS))
def test_train_checkpoint_infer_round_trip(
    model: str, data_dir: Path, tmp_path: Path
) -> None:
    """The test error logged during training is what infer.py measures on the
    reloaded checkpoint."""
    data = load_split(data_dir, SMALL)
    run = replace(RUNS[model], arch=TINY_ARCH[model], epochs=2)
    run_dir = tmp_path / "ckpt" / run.name / "seed0"

    train_one(run, data, run_dir, SMALL, seed=0)

    assert is_complete(run_dir)
    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert len(ck["history"]) == 2
    np.testing.assert_array_equal(ck["x_normalizer"].mean, data["x_normalizer"].mean)
    np.testing.assert_array_equal(ck["y_normalizer"].std, data["y_normalizer"].std)

    pred = infer.predict(ck, data["xy_test"], SMALL["batch_size"])
    assert pred.dtype == np.float32
    rel = infer.per_sample_rel_l2(pred, data["y_test"])
    # Every model computes in float32, so infer.py reproduces the logged metric.
    assert rel.mean() == pytest.approx(ck["history"][-1]["test_rel_l2"], rel=1e-4)


def test_wno_computes_in_float32() -> None:
    model = get_spec("wno").build(TINY_ARCH["wno"], (N, N))
    x = jnp.zeros((1, N, N, 2))
    params = model.init(jax.random.PRNGKey(0), x)
    assert model.apply(params, x).dtype == jnp.float32


def test_one_sample_at_a_time_matches_the_full_batch(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model fed one sample at a time sums the gradients over the batch, so
    it takes the same optimiser steps as one fed the whole batch."""
    from scripts.benchmarks.pipe import train

    data = load_split(data_dir, SMALL)
    run = replace(RUNS["fno"], arch=TINY_ARCH["fno"], epochs=2)
    train_one(run, data, tmp_path / "batched", SMALL, seed=0)
    one_at_a_time = replace(get_spec("fno"), batch_size=1)
    monkeypatch.setattr(train, "get_spec", lambda name: one_at_a_time)
    train_one(run, data, tmp_path / "single", SMALL, seed=0)

    batched = load_checkpoint(tmp_path / "batched")
    single = load_checkpoint(tmp_path / "single")
    for a, b in zip(
        jax.tree_util.tree_leaves(batched["params"]),
        jax.tree_util.tree_leaves(single["params"]),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)
    assert single["config"]["training"]["samples_per_step"] == SMALL["batch_size"]
