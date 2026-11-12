# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/navier_stokes/.

Everything runs on a few synthetic samples on a small grid on the CPU, so no
dataset or GPU is needed: the data split, the model layouts, the size of the
published runs, teacher forcing and rollout, the optimiser schedule, and the
train -> checkpoint -> infer round trip.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks.navier_stokes import infer, train  # noqa: E402
from scripts.benchmarks.navier_stokes._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
)
from scripts.benchmarks.navier_stokes._config import (  # noqa: E402
    DATA_FILE,
    OPTIMIZERS,
    PROTOCOL,
    RUNS,
    TRANSOLVER_PARAMS,
    RunConfig,
)
from scripts.benchmarks.navier_stokes._data import (  # noqa: E402
    copy_last_frame,
    load_split,
    rel_l2_sum,
    rollout,
    teacher_forced,
)
from scripts.benchmarks.navier_stokes._models import SPECS, get_spec  # noqa: E402
from scripts.benchmarks.navier_stokes.train import (  # noqa: E402
    make_optimizer,
    momentum_schedule,
    train_one,
)

H = W = 16
T = 20
SMALL = dict(PROTOCOL, ntrain=4, ntest=2, batch_size=2)
TINY_ARCH = {
    "transolver": dict(
        n_layers=1,
        n_hidden=16,
        n_head=2,
        mlp_ratio=1,
        slice_num=4,
        ref=2,
        unified_pos=True,
        gelu="exact",
        layer_norm_epsilon=1e-5,
        conv_init="torch",
    ),
    "fno": dict(hidden_channels=4, n_modes=(4, 4), n_layers=1),
    "wno": dict(
        hidden_channels=4,
        n_layers=1,
        level=1,
        wavelet="db4",
        projection_hidden_channels=8,
    ),
    "gino": dict(
        in_gno_radius=0.15,
        out_gno_radius=0.15,
        fno_n_modes=(4, 4),
        fno_hidden_channels=4,
        fno_n_layers=1,
    ),
}


def _n_params(model, inputs) -> int:
    """Trainable parameters, counting each complex weight as two."""
    shapes = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), **inputs))
    return sum(
        int(np.prod(x.shape))
        * (2 if jnp.issubdtype(x.dtype, jnp.complexfloating) else 1)
        for x in jax.tree_util.tree_leaves(shapes)
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A synthetic data file: 7 samples of 20 frames on a 16x16 grid, where
    sample i is filled with i + 1 (plus frame index / 100)."""
    n = 7
    u = np.zeros((n, H, W, T), dtype=np.float32)
    for i in range(n):
        u[i] = i + np.arange(T, dtype=np.float32) / 100 + 1.0
    scipy.io.savemat(tmp_path / DATA_FILE, {"u": u})
    return tmp_path


def test_split_takes_first_train_and_last_test_samples(data_dir: Path) -> None:
    d = load_split(data_dir / DATA_FILE, SMALL)
    assert d["grid"] == (H, W)
    assert d["a_train"].shape == (4, H * W, 10)
    assert d["u_train"].shape == (4, H * W, 10)
    # train = samples 0..3, test = the last two (5, 6)
    assert d["a_train"][:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert d["a_test"][:, 0, 0].tolist() == [6.0, 7.0]
    # input frames 0..9, targets 10..19
    np.testing.assert_allclose(d["a_test"][0, 0, -1], 6.09, rtol=1e-6)
    np.testing.assert_allclose(d["u_test"][0, 0, 0], 6.10, rtol=1e-6)


def test_split_rejects_more_samples_than_the_file_has(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="fewer than"):
        load_split(data_dir / DATA_FILE, dict(SMALL, ntrain=6, ntest=2))


def test_split_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NS_DATA_DIR"):
        load_split(tmp_path / DATA_FILE, SMALL)


@pytest.mark.parametrize("name", list(RUNS))
def test_published_runs_match_transolver_size(name: str) -> None:
    """Every run is within 1% of Transolver's parameter count."""
    run = RUNS[name]
    spec = get_spec(run.model)
    model = spec.build(run.arch, (64, 64), PROTOCOL["t_in"])
    fx = jnp.zeros((1, 64 * 64, PROTOCOL["t_in"]))
    n = _n_params(model, spec._inputs(model, fx, (64, 64)))
    if name == "transolver":
        assert n == TRANSOLVER_PARAMS
    else:
        assert abs(n / TRANSOLVER_PARAMS - 1) < 0.01


def test_transolver_run_uses_the_pytorch_reference_numerics() -> None:
    run = RUNS["transolver"]
    assert run.arch["gelu"] == "exact"
    assert run.arch["layer_norm_epsilon"] == 1e-5
    assert run.arch["conv_init"] == "torch"
    assert run.cycle_momentum
    assert not any(RUNS[name].cycle_momentum for name in ("fno", "wno", "gino"))


@pytest.mark.parametrize("model_name", list(SPECS))
def test_spec_maps_a_history_to_the_next_frame(model_name: str) -> None:
    spec = SPECS[model_name]
    model = spec.build(TINY_ARCH[model_name], (H, W), 10)
    fx = jax.random.normal(jax.random.PRNGKey(0), (2, H * W, 10))
    params = spec.init(model, jax.random.PRNGKey(0), fx, (H, W))
    out = spec.apply(model, params, fx, (H, W))
    assert out.shape == (2, H * W, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_fno_reads_the_history_channels_first() -> None:
    """Frame k of every point must arrive as FNO input channel k."""
    spec = get_spec("fno")
    model = spec.build(TINY_ARCH["fno"], (H, W), 10)
    fx = jnp.broadcast_to(jnp.arange(10.0), (1, H * W, 10))
    x = spec._inputs(model, fx, (H, W))["x"]
    assert x.shape == (1, 10, H, W)
    np.testing.assert_array_equal(np.asarray(x[0, :, 3, 5]), np.arange(10.0))


def test_gino_rejects_different_gno_radii() -> None:
    arch = dict(TINY_ARCH["gino"], out_gno_radius=0.3)
    with pytest.raises(ValueError, match="in_gno_radius == out_gno_radius"):
        get_spec("gino").build(arch, (H, W), 10)


def test_teacher_forcing_feeds_the_ground_truth() -> None:
    """With a persistence 'model', step t predicts frame t - 1 of the truth."""
    fx = jnp.arange(3.0).reshape(1, 1, 3)
    yy = jnp.array([[[10.0, 20.0, 30.0]]])
    _, pred = teacher_forced(lambda h: h[..., -1:], fx, yy)
    np.testing.assert_array_equal(np.asarray(pred), [[[2.0, 10.0, 20.0]]])


def test_rollout_feeds_back_its_own_predictions() -> None:
    fx = jnp.arange(3.0).reshape(1, 1, 3)
    pred = rollout(lambda h: h[..., -1:], fx, 4)
    np.testing.assert_array_equal(np.asarray(pred), [[[2.0, 2.0, 2.0, 2.0]]])
    np.testing.assert_array_equal(np.asarray(pred), copy_last_frame(np.asarray(fx), 4))


def test_rel_l2_sum_is_summed_over_the_batch() -> None:
    true = jnp.ones((2, 4, 1))
    pred = jnp.stack([jnp.ones((4, 1)) * 2.0, jnp.ones((4, 1))])
    assert float(rel_l2_sum(pred, true)) == pytest.approx(1.0)


def test_momentum_schedule_cycles_like_onecycle() -> None:
    total = 1000
    b1 = momentum_schedule(PROTOCOL, total)
    warm = int(PROTOCOL["pct_start"] * total)
    assert float(b1(0)) == pytest.approx(0.95)
    assert float(b1(warm)) == pytest.approx(0.85)
    assert float(b1(total)) == pytest.approx(0.95)


@pytest.mark.parametrize("name", list(RUNS))
def test_runs_with_complex_weights_use_adamw_cx(name: str) -> None:
    run = RUNS[name]
    spec = get_spec(run.model)
    model = spec.build(TINY_ARCH[run.model], (H, W), 10)
    inputs = spec._inputs(model, jnp.zeros((1, H * W, 10)), (H, W))
    shapes = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), **inputs))
    has_complex = any(
        jnp.issubdtype(x.dtype, jnp.complexfloating)
        for x in jax.tree_util.tree_leaves(shapes)
    )
    assert run.optimizer == ("adamw_cx" if has_complex else "adamw")


@pytest.mark.parametrize("name", list(RUNS))
@pytest.mark.parametrize("cycle_momentum", [False, True])
def test_updates_keep_each_weights_dtype(name: str, cycle_momentum: bool) -> None:
    """Real weights get real updates even next to complex spectral weights
    (the injected hyperparameters stay float32)."""
    run = RUNS[name]
    spec = get_spec(run.model)
    model = spec.build(TINY_ARCH[run.model], (H, W), 10)
    inputs = spec._inputs(model, jnp.zeros((1, H * W, 10)), (H, W))
    params = model.init(jax.random.PRNGKey(0), **inputs)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    tx, _ = make_optimizer(PROTOCOL, 10, cycle_momentum, run.optimizer)
    state = tx.init(params)
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


def test_optimizer_names_match_the_builders() -> None:
    assert set(OPTIMIZERS) == set(train.OPTIMIZERS)
    with pytest.raises(ValueError, match="not one of"):
        RunConfig("fno", "fno", optimizer="adamw-cx")


def test_train_refuses_a_checkpoint_from_another_optimizer(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete checkpoint trained with plain adamw is not silently kept
    for a run that now uses adamw_cx."""
    data = load_split(data_dir / DATA_FILE, SMALL)
    run = replace(RUNS["fno"], arch=TINY_ARCH["fno"], epochs=1, optimizer="adamw")
    train_one(run, data, tmp_path / "ckpt" / "fno", SMALL)
    monkeypatch.setattr(train, "require_gpu", lambda allow_cpu: None)
    monkeypatch.setattr(train, "load_split", lambda path, protocol: data)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--run", "fno", "--ckpt-root", str(tmp_path / "ckpt")],
    )
    with pytest.raises(SystemExit, match="--overwrite"):
        train.main()


@pytest.mark.parametrize("optimizer, converges", [("adamw_cx", True), ("adamw", False)])
def test_only_adamw_cx_descends_on_a_complex_weight(
    optimizer: str, converges: bool
) -> None:
    """|z - c|^2 from z = 0: jax.grad gives the conjugate of the descent
    direction, so plain AdamW moves Im(z) away from Im(c)."""
    c = jnp.asarray(1e-3 + 2e-3j, jnp.complex64)
    loss = lambda z: jnp.abs(z - c) ** 2  # noqa: E731
    tx, _ = make_optimizer(dict(PROTOCOL, weight_decay=0.0), 400, optimizer=optimizer)
    z = jnp.asarray(0j, jnp.complex64)
    state = tx.init(z)
    for _ in range(400):
        updates, state = tx.update(jax.grad(loss)(z), state, z)
        z = z + updates
    assert (float(loss(z)) < 0.1 * float(loss(jnp.zeros_like(z)))) is converges
    assert (float(z.imag) > 0) is converges


def test_schedule_peaks_at_pct_start() -> None:
    _, schedule = make_optimizer(PROTOCOL, 1000)
    lrs = np.array([float(schedule(s)) for s in range(1001)])
    assert int(lrs.argmax()) == pytest.approx(300, abs=1)
    assert lrs.max() == pytest.approx(PROTOCOL["lr"])


@pytest.mark.parametrize("name", ["transolver", "fno"])
def test_train_checkpoint_infer_round_trip(
    name: str, data_dir: Path, tmp_path: Path
) -> None:
    """One epoch of training, saved, reloaded and re-evaluated, gives the
    test error the training logged."""
    data = load_split(data_dir / DATA_FILE, SMALL)
    run = replace(RUNS[name], arch=TINY_ARCH[name], epochs=1)
    run_dir = tmp_path / "ckpt" / name
    training = train_one(run, data, run_dir, SMALL)
    assert is_complete(run_dir)

    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert ck["grid"] == (H, W)
    pred = infer.predict(ck, data["a_test"], SMALL["t_out"], batch_size=2)
    rel = infer.per_sample_rel_l2(pred, data["u_test"])
    assert float(rel.mean()) == pytest.approx(training["final_test_rel_l2"], rel=1e-4)
    assert len(ck["history"]) == 1
