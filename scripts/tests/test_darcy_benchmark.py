# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/darcy/.

Everything runs on a few synthetic ``.mat`` samples on the CPU, so no dataset
or GPU is needed: loading and striding, the normalisers and losses, the size
of the published runs, and the train -> checkpoint -> infer round trip of
every model.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.io import savemat  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.darcy import infer  # noqa: E402
from scripts.benchmarks.darcy._checkpoint import is_complete  # noqa: E402
from scripts.benchmarks.darcy._config import PROTOCOL, RUNS  # noqa: E402
from scripts.benchmarks.darcy._data import (  # noqa: E402
    GlobalNormalizer,
    PixelNormalizer,
    batched_rel_l2_loss,
    central_diff,
    gradient_regularization_loss,
    load_split,
    rel_l2,
    zero_border,
)
from scripts.benchmarks.darcy._models import build_grid_geometry, prepare  # noqa: E402
from scripts.benchmarks.darcy.train import train_one  # noqa: E402

TINY_ARCH = {
    "fno": dict(modes=4, hidden_channels=4, n_layers=1),
    "wno": dict(hidden_channels=4, level=1, wavelet="db4", n_layers=1, padding=1),
    "gino": dict(hidden_channels=4, modes=4, n_layers=1),
    "transolver": dict(
        n_hidden=8, n_layers=1, n_head=2, slice_num=4, ref=4, mlp_ratio=1, kernel=3
    ),
}
# The published runs' sizes (doc/benchmarks/darcy/results/comparative.md).
PUBLISHED_PARAMS = dict(
    fno=2_836_281, wno=2_829_601, gino=2_801_340, transolver=2_826_945
)


def _write_mat(path: Path, n: int, side: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    fields = {
        "coeff": rng.choice([3.0, 12.0], size=(n, side, side)),
        "sol": rng.standard_normal((n, side, side)) * 0.01 + 0.02,
        # Li et al.'s files carry extra variables the loader must ignore.
        "Kcoeff": rng.standard_normal((n, side, side)),
    }
    savemat(path, fields)
    return fields


def _protocol(**overrides) -> dict:
    return dict(PROTOCOL, train_file="train.mat", test_file="test.mat", **overrides)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("darcy")
    _write_mat(d / "train.mat", n=6, side=21, seed=0)
    _write_mat(d / "test.mat", n=4, side=21, seed=1)
    return d


@pytest.fixture(scope="module")
def small_protocol() -> dict:
    return _protocol(n_train=4, n_test=2, subsample_rate=1, eval_every=1)


def test_load_strides_and_keeps_the_first_samples(tmp_path: Path) -> None:
    train = _write_mat(tmp_path / "train.mat", n=6, side=21, seed=0)
    test = _write_mat(tmp_path / "test.mat", n=5, side=21, seed=1)
    data = load_split(tmp_path, _protocol(n_train=4, n_test=3, subsample_rate=5))
    assert data["train_a"].shape == (4, 5, 5) and data["test_u"].shape == (3, 5, 5)
    assert data["train_a"].dtype == np.float32
    np.testing.assert_allclose(data["train_a"], train["coeff"][:4, ::5, ::5])
    np.testing.assert_allclose(data["test_u"], test["sol"][:3, ::5, ::5], rtol=1e-6)


def test_load_rejects_more_samples_than_the_file_has(tmp_path: Path) -> None:
    _write_mat(tmp_path / "train.mat", n=3, side=6, seed=0)
    _write_mat(tmp_path / "test.mat", n=3, side=6, seed=1)
    with pytest.raises(ValueError, match="only has 3"):
        load_split(tmp_path, _protocol(n_train=4, n_test=1, subsample_rate=1))


def test_load_missing_file_names_the_expected_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="piececonst_r421_N1024"):
        load_split(tmp_path, _protocol())


def test_pixel_normalizer_is_per_pixel_and_round_trips() -> None:
    x = np.random.default_rng(0).standard_normal((8, 5, 5)).astype(np.float32)
    norm = PixelNormalizer(x)
    assert norm.mean.shape == (1, 5, 5)
    np.testing.assert_allclose(norm.decode(norm.encode(x)), x, atol=1e-5)


def test_global_normalizer_is_one_scalar_per_channel() -> None:
    x = np.random.default_rng(0).standard_normal((8, 5, 5, 1)).astype(np.float32)
    norm = GlobalNormalizer(x)
    assert norm.mean.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(norm.mean.item(), x.mean(), rtol=1e-5)
    np.testing.assert_allclose(norm.decode(norm.encode(x)), x, atol=1e-5)


def test_batched_loss_is_the_mean_per_sample_rel_l2() -> None:
    rng = np.random.default_rng(0)
    pred, true = rng.standard_normal((3, 10)), rng.standard_normal((3, 10))
    expected = np.mean([rel_l2(pred[i], true[i]) for i in range(3)])
    assert float(batched_rel_l2_loss(jnp.asarray(pred), jnp.asarray(true))) == (
        pytest.approx(expected, rel=1e-5)
    )


def test_zero_border_only_touches_the_outer_ring() -> None:
    grid = zero_border(jnp.ones((1, 5, 5, 1)))
    assert float(grid[0, 1:-1, 1:-1, 0].min()) == 1.0
    assert float(grid[0, 0].max()) == float(grid[0, :, -1].max()) == 0.0


def test_central_diff_matches_the_gradient_of_a_linear_field() -> None:
    n, dx = 7, 0.1
    yy, xx = np.meshgrid(np.arange(n) * dx, np.arange(n) * dx, indexing="ij")
    field = jnp.asarray((2.0 * xx + 3.0 * yy)[None, ..., None])
    gx, gy = central_diff(field, dx)
    np.testing.assert_allclose(gx[0, 1:-1, 1:-1, 0], 2.0, rtol=1e-5)
    np.testing.assert_allclose(gy[0, 1:-1, 1:-1, 0], 3.0, rtol=1e-5)


def test_gradient_loss_is_zero_for_identical_bordered_fields() -> None:
    field = zero_border(jnp.asarray(np.random.default_rng(0).random((2, 6, 6, 1))))
    assert float(gradient_regularization_loss(field, field, 0.2)) == pytest.approx(0)


def test_grid_geometry_flattens_like_the_data() -> None:
    xy = build_grid_geometry(4).reshape(4, 4, 2)
    # Point (i, j) of the C-order flatten has x = i / 3 and y = j / 3.
    np.testing.assert_allclose(xy[2, 1], [2 / 3, 1 / 3], rtol=1e-6)


@pytest.mark.parametrize("name", sorted(RUNS))
def test_published_runs_have_their_published_size(name: str) -> None:
    rng = np.random.default_rng(0)
    data = {k: rng.random((2, 85, 85)).astype(np.float32) for k in
            ("train_a", "train_u", "test_a", "test_u")}  # fmt: skip
    run = RUNS[name]
    prep = prepare(run.model, run.arch, data)
    params = prep.init(jax.random.PRNGKey(0))
    assert int(count_model_params(params)) == PUBLISHED_PARAMS[name]


@pytest.mark.parametrize("model", sorted(RUNS))
def test_train_checkpoint_infer_round_trip(
    model: str, data_dir: Path, small_protocol: dict, tmp_path: Path
) -> None:
    run = replace(
        RUNS[model],
        arch=TINY_ARCH[model],
        opt={**RUNS[model].opt, "epochs": 3, "batch_size": 2},
    )
    data = load_split(data_dir, small_protocol)
    run_dir = tmp_path / "ckpt" / model
    training = train_one(run, data, run_dir, small_protocol, animate=True)
    assert is_complete(run_dir)
    assert training["steps_run"] == 6

    res = infer.evaluate(run_dir, data)
    assert res["run"] == run
    assert res["pred"].shape == (2, 21 * 21)
    # The last-epoch weights reproduce the error recorded during training.
    assert res["errs"].mean() == pytest.approx(training["final_test_rel_l2"], rel=1e-4)

    out = tmp_path / "assets"
    infer.write_assets(res, data, infer.load_history(run_dir), out)
    assert sorted(p.name for p in out.iterdir()) == [
        "compute_cost.png",
        "contour.png",
        "error_curve.png",
        "metrics.json",
    ]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["test_rel_l2_mean"] == pytest.approx(res["errs"].mean())

    frames = np.load(run_dir / "frames.npz")
    assert frames["pred_frames"].shape == (3, 21, 21, 1)
    assert list(frames["epochs"]) == [1, 2, 3]


def test_checkpoint_refuses_a_different_split(
    data_dir: Path, small_protocol: dict, tmp_path: Path
) -> None:
    run = replace(
        RUNS["fno"], arch=TINY_ARCH["fno"], opt={**RUNS["fno"].opt, "epochs": 1}
    )
    run_dir = tmp_path / "ckpt"
    train_one(run, load_split(data_dir, small_protocol), run_dir, small_protocol)
    other = load_split(data_dir, dict(small_protocol, n_train=3))
    with pytest.raises(ValueError, match="normalisation"):
        infer.evaluate(run_dir, other)
