# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/airfrans/.

Most tests use small synthetic cases, so no dataset or GPU is needed: the
features and target scaling, both normalisations, the official scores, the
cover inference, the OneCycleLR schedule, the published model size and init,
and the checkpoint round trip. The last test checks the force integration on
one real case and is skipped when the AirfRANS dataset is not available.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.models import Transolver  # noqa: E402
from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.airfrans._checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    template_params,
)
from scripts.benchmarks.airfrans._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PAPER_REFERENCE,
    PROTOCOL,
    RUNS,
    RunConfig,
)
from scripts.benchmarks.airfrans._data import (  # noqa: E402
    N_FEATURES,
    NU,
    features,
    model_stats,
    official_output_stats,
    prepare_case,
    target_scale,
    to_physical,
)
from scripts.benchmarks.airfrans._metrics import (  # noqa: E402
    cover_predict,
    field_scores,
    impose_wall,
)
from scripts.benchmarks.airfrans._models import get_spec  # noqa: E402
from scripts.benchmarks.airfrans.train import onecycle, region_losses  # noqa: E402

jax.config.update("jax_enable_x64", False)


def synthetic_case(n=500, u_inf=50.0, aoa_deg=4.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.deg2rad(aoa_deg)
    pos = rng.uniform(-1.0, 2.0, (n, 2)).astype(np.float32)
    surf = np.zeros(n, bool)
    surf[:40] = True
    sdf = np.where(surf, 0.0, rng.uniform(1e-5, 1.0, n))
    x7 = np.column_stack(
        [
            pos,
            np.full(n, u_inf * np.cos(a)),
            np.full(n, u_inf * np.sin(a)),
            sdf,
            np.zeros((n, 2)),
        ]
    ).astype(np.float32)
    y = rng.normal(size=(n, 4)).astype(np.float32) * np.array(
        [u_inf, u_inf, u_inf**2, 1e-4], np.float32
    )
    y[surf, 0] = y[surf, 1] = y[surf, 3] = 0.0
    return dict(x7=x7, y=y, pos=pos, surf=surf)


def test_features_append_log_wall_distance_and_log_re():
    c = synthetic_case(u_inf=40.0)
    x9 = features(c["x7"])
    assert x9.shape == (500, N_FEATURES)
    np.testing.assert_array_equal(x9[:, :7], c["x7"])
    np.testing.assert_allclose(
        x9[:, 7], np.log10(np.abs(c["x7"][:, 4]) + 1e-6), rtol=1e-6
    )
    np.testing.assert_allclose(x9[:, 8], np.log10(40.0 / NU), rtol=1e-6)


def test_target_scaling_and_normalisation_round_trip():
    cases = [synthetic_case(u_inf=u, seed=i) for i, u in enumerate((31.0, 55.0, 80.0))]
    stats = model_stats(cases)
    for c in cases:
        d = prepare_case(c, stats)
        np.testing.assert_allclose(
            to_physical(d["y"], d["scale"], stats), c["y"], rtol=1e-4, atol=1e-4
        )
    np.testing.assert_array_equal(target_scale(2.0), [2.0, 2.0, 4.0, 2.0])


def test_official_output_stats_match_plain_float64_statistics():
    cases = [synthetic_case(seed=i) for i in range(3)]
    y = np.concatenate([c["y"] for c in cases]).astype(np.float64)
    mean, std = official_output_stats(cases)
    np.testing.assert_allclose(mean, y.mean(0), rtol=1e-5)
    np.testing.assert_allclose(std, y.std(0), rtol=1e-5)


def test_wall_values_are_imposed_and_scores_are_normalised_mse():
    c = synthetic_case()
    pred = c["y"] + 1.0
    pred_w = impose_wall(pred, c["surf"])
    assert np.all(pred_w[c["surf"]][:, [0, 1, 3]] == 0.0)
    np.testing.assert_array_equal(pred_w[c["surf"], 2], pred[c["surf"], 2])
    std = np.array([2.0, 2.0, 4.0, 1.0], np.float32)
    s = field_scores(pred_w, c["y"], c["surf"], np.zeros(4, np.float32), std)
    # off the wall every channel is off by 1 (in physical units)
    np.testing.assert_allclose(
        s["volume_per_channel"], (1.0 / (std + 1e-8)) ** 2, rtol=1e-5
    )
    # on the wall only the pressure is wrong
    # (the predictions are float32, so the +1 offset is exact only to ~1e-5)
    np.testing.assert_allclose(
        s["surf_per_channel"], [0.0, 0.0, (1 / 4.0) ** 2, 0.0], rtol=1e-5, atol=1e-12
    )
    assert s["surf"] == pytest.approx((1 / 4.0) ** 2 / 4, rel=1e-5)


def test_cover_predict_hits_every_node_and_averages():
    n = 1000
    pred, passes = cover_predict(
        lambda idx: np.tile(idx[:, None], (1, 4)) * 1.0,
        n,
        300,
        np.random.default_rng(0),
    )
    np.testing.assert_array_equal(pred[:, 0], np.arange(n))
    assert passes >= n // 300


def test_onecycle_matches_torch_one_cycle_lr():
    lr, b1 = onecycle(1000, PROTOCOL)  # evaluated in float32, hence rel=1e-5
    p = PROTOCOL
    end1 = p["pct_start"] * 1000 - 1
    assert float(lr(0)) == pytest.approx(p["lr"] / p["div_factor"], rel=1e-5)
    assert float(b1(0)) == pytest.approx(p["max_momentum"], rel=1e-5)
    assert float(lr(end1)) == pytest.approx(p["lr"], rel=1e-5)
    assert float(b1(end1)) == pytest.approx(p["base_momentum"], rel=1e-5)
    assert float(lr(999)) == pytest.approx(
        p["lr"] / p["div_factor"] / p["final_div_factor"], rel=1e-4
    )
    assert float(b1(999)) == pytest.approx(p["max_momentum"], rel=1e-5)


def test_region_losses_split_wall_and_volume():
    y = jnp.zeros((4, 4))
    out = jnp.array([[1.0] * 4, [1.0] * 4, [2.0] * 4, [2.0] * 4])
    lv, ls = region_losses(out, y, jnp.array([True, True, False, False]))
    assert float(ls) == pytest.approx(1.0)
    assert float(lv) == pytest.approx(4.0)


def test_published_transolver_size_and_official_init():
    run = RUNS["transolver"]
    spec = get_spec(run.model)
    model = spec.build(run.arch)
    params = template_params(spec, model, run.arch)
    n = int(count_model_params(params))
    assert n == 3_894_596
    assert n - PAPER_REFERENCE["n_params_used"] == 1_024  # two extra input features
    kernels = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        if jax.tree_util.keystr(path).endswith("['kernel']")
        and leaf.ndim == 2
        and leaf.size > 1000
    ]
    assert len(kernels) > 30
    for k in kernels:  # trunc_normal(0.02) everywhere, including in_project_slice
        assert 0.017 < float(jnp.std(k)) < 0.023


def test_layer_norm_epsilon_default_is_unchanged():
    kw = dict(
        space_dim=3,
        fun_dim=0,
        out_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=2,
        slice_num=4,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 50, 3))
    base = Transolver(**kw)
    params = base.init(jax.random.PRNGKey(1), x=x, fx=None)
    out = base.apply(params, x=x, fx=None)
    np.testing.assert_array_equal(
        out, Transolver(**kw, layer_norm_epsilon=1e-6).apply(params, x=x, fx=None)
    )
    assert not np.array_equal(
        out, Transolver(**kw, layer_norm_epsilon=1e-1).apply(params, x=x, fx=None)
    )


def test_checkpoint_round_trip(tmp_path):
    run = RunConfig(
        "tiny",
        "transolver",
        dict(n_hidden=16, n_layers=2, n_head=2, mlp_ratio=1, slice_num=4, ref=4),
    )
    spec = get_spec("transolver")
    model = spec.build(run.arch)
    params = template_params(spec, model, run.arch, n_nodes=64)
    cases = [synthetic_case(seed=i) for i in range(2)]
    stats, official = model_stats(cases), official_output_stats(cases)
    history = [
        dict(
            epoch=0,
            train_vol=1.0,
            train_surf=2.0,
            lr=1e-3,
            secs=1.0,
            val_vol=0.5,
            val_surf=0.6,
        )
    ]
    save_checkpoint(
        tmp_path / "tiny", run, params, stats, official, history, dict(note="test")
    )
    # the run is rebuilt from the stored config alone (it is not in RUNS)
    ck = load_checkpoint(tmp_path / "tiny")
    for a, b in zip(
        jax.tree_util.tree_leaves(ck["params"]),
        jax.tree_util.tree_leaves(params),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(ck["stats"].x_mean, stats.x_mean)
    np.testing.assert_array_equal(ck["official_out"][1], official[1])
    assert ck["history"] == history


DATA = _REPO_ROOT / DEFAULT_DATA_DIR


@pytest.mark.skipif(
    not (DATA / "manifest.json").exists(), reason="AirfRANS dataset not available"
)
def test_force_coefficients_of_a_real_case():
    from scripts.benchmarks.airfrans._data import build_case
    from scripts.benchmarks.airfrans._metrics import CaseGeometry

    cid = "airFoil2D_SST_31.812_1.334_0.371_3.287_0.0_19.548"
    geo = CaseGeometry(DATA, cid)
    cd, cl = geo.coefficients()
    assert cl == pytest.approx(0.149416, rel=1e-4)
    assert cd == pytest.approx(0.011619, rel=1e-3)
    # the true fields passed back in give the same coefficients
    case = build_case(DATA, cid)
    np.testing.assert_allclose(
        geo.coefficients(impose_wall(case["y"], case["surf"])), [cd, cl], rtol=1e-6
    )


def circle_segments(r=0.4, n=400, cx=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.stack([cx + r * np.cos(t), r * np.sin(t)], 1).astype(np.float32)
    return np.stack([p, np.roll(p, -1, axis=0)], axis=1)


def test_wno_grid_geometry_is_the_signed_distance_of_the_outline():
    from scripts.benchmarks.airfrans._models import MAX_SEGMENTS, grid_points

    spec = get_spec("wno")
    arch = dict(RUNS["wno"].arch, H=32, W=64)
    seg = circle_segments()
    geo = np.asarray(spec.extra(arch, dict(segments=seg)))
    assert geo.shape == (32, 64, 2)
    P = np.asarray(grid_points(32, 64)).reshape(32, 64, 2)
    dist = (
        np.linalg.norm(P - np.array([0.5, 0.0]), axis=-1) - 0.4
    )  # exact signed distance
    np.testing.assert_allclose(geo[..., 0], np.tanh(dist / 0.5), atol=2e-3)
    np.testing.assert_array_equal(geo[..., 1] == 1.0, dist < 0)
    # padding to MAX_SEGMENTS (done by extra) leaves the geometry unchanged
    from scripts.benchmarks.airfrans._models import grid_geometry

    unpadded = grid_geometry(jnp.asarray(seg), jnp.asarray(len(seg)), 32, 64)
    assert len(seg) < MAX_SEGMENTS
    np.testing.assert_allclose(geo, np.asarray(unpadded), atol=1e-6)


def test_published_wno_size():
    run = RUNS["wno"]
    spec = get_spec(run.model)
    params = template_params(spec, spec.build(run.arch), run.arch)
    n = int(count_model_params(params))
    assert n == 3_964_728
    assert abs(n / PAPER_REFERENCE["n_params_used"] - 1) < 0.02


def test_wno_forward_and_gradient_on_a_small_grid():
    spec = get_spec("wno")
    arch = dict(
        H=32, W=64, hidden=8, n_layers=1, level=2, latent=8, dec_hidden=16, dec_layers=2
    )
    model = spec.build(arch)
    extra = spec.extra(arch, dict(segments=circle_segments()))
    x = jax.random.normal(jax.random.PRNGKey(0), (300, N_FEATURES))
    pos = jax.random.uniform(jax.random.PRNGKey(1), (300, 2), minval=-3.0, maxval=5.0)
    params = spec.init(model, jax.random.PRNGKey(2), x, pos, extra)
    out = jax.jit(lambda p: spec.apply(model, p, x, pos, extra))(params)
    assert out.shape == (300, 4) and bool(jnp.isfinite(out).all())
    grads = jax.grad(lambda p: jnp.mean(spec.apply(model, p, x, pos, extra) ** 2))(
        params
    )
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))
