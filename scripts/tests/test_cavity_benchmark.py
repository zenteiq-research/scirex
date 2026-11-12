# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/cavity/.

Everything runs on small synthetic grids on the CPU, so no dataset or GPU is
needed: the data transforms, every model's output shape, the search
sampler, the train -> checkpoint -> load round trip, and the rollout and
baseline used by ``infer.py``.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys  # noqa: E402
from itertools import pairwise  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks.cavity import _data, infer  # noqa: E402
from scripts.benchmarks.cavity._checkpoint import (  # noqa: E402
    count_params,
    is_complete,
    load_checkpoint,
)
from scripts.benchmarks.cavity._config import (  # noqa: E402
    PROTOCOL,
    RUNS,
    RunConfig,
    sample_search_configs,
)
from scripts.benchmarks.cavity._models import (  # noqa: E402
    SPECS,
    build_model,
    gino_geometry,
)
from scripts.benchmarks.cavity.train import train_one  # noqa: E402

#: A deliberately tiny size for each architecture, so the shape and wiring
#: tests stay fast on the CPU. Keyed by ``SPECS``, so a new architecture
#: without an entry here fails the tests below rather than skipping them.
TINY_ARCH = {
    "fno": dict(modes=4, width=8, n_layers=1),
    "gino": dict(
        modes=4,
        width=8,
        n_layers=1,
        latent_grid=8,
        gno_channels=8,
        gno_mlp_hidden=(8, 8),
    ),
    # A shallower decomposition than the real run's, because these tests use a
    # 16x16 grid: level 5 on 16 pixels would leave almost nothing to weight.
    "wno": dict(hidden_channels=4, n_layers=1, level=2),
    # n_hidden has to stay at or above n_head, since the per-head width is
    # their integer quotient; 16 over the default 8 heads is the smallest
    # size that still gives every head more than a single channel.
    "transolver": dict(n_hidden=16, n_layers=1, slice_num=4),
}

#: The class from ``src/scirex/`` each adapter is required to be built on.
WRAPPED = {
    "fno": "FNOClassic2d",
    "gino": "GINO",
    "wno": "WNO",
    "transolver": "Transolver",
}

MODELS = sorted(SPECS)


def test_every_architecture_is_covered_by_these_tests() -> None:
    """Adding an architecture to SPECS must not silently skip its tests."""
    assert MODELS == sorted(TINY_ARCH) == sorted(WRAPPED)


def _write_cache(cache_dir: Path, cases, n_frames=10, side=16, seed=0) -> None:
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        grid = rng.standard_normal((n_frames, side, side, 3)).astype(np.float32)
        np.savez_compressed(cache_dir / f"{case}.npz", grid=grid)


def test_vertex_to_cell_averages_four_corners() -> None:
    v = np.arange(3 * 3 * 1, dtype=np.float32).reshape(3, 3, 1)
    cell = _data.vertex_to_cell(v)
    assert cell.shape == (2, 2, 1)
    assert cell[0, 0, 0] == pytest.approx(np.mean([0, 1, 3, 4]))
    assert cell[1, 1, 0] == pytest.approx(np.mean([4, 5, 7, 8]))


def test_vertex_sort_index_restores_raster_order() -> None:
    n = _data.N_VERT
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pos = np.stack([x.ravel(), y.ravel()], axis=-1) * _data.DX
    shuffle = np.random.default_rng(0).permutation(n * n)
    order = _data.vertex_sort_index(pos[shuffle])
    np.testing.assert_array_equal(shuffle[order], np.arange(n * n))


def test_make_windows_pairs_inputs_with_following_frames() -> None:
    grid = np.arange(8, dtype=np.float32)[:, None, None, None] * np.ones((8, 2, 2, 3))
    X, Y = _data.make_windows(grid, n_in=3, n_out=2)
    assert X.shape == (4, 3, 2, 2, 3) and Y.shape == (4, 2, 2, 2, 3)
    np.testing.assert_array_equal(X[1, :, 0, 0, 0], [1, 2, 3])
    np.testing.assert_array_equal(Y[1, :, 0, 0, 0], [4, 5])


def test_build_train_val_uses_training_cases_only(tmp_path: Path) -> None:
    # Only the training cases exist: the test case must never be read.
    _write_cache(tmp_path, _data.TRAIN_CASES)
    d = _data.build_train_val(tmp_path, n_in=3, n_out=1, val_frac=0.25, seed=0)
    n_windows = len(_data.TRAIN_CASES) * (10 - 3 - 1 + 1)
    assert d["X_train"].shape[0] + d["X_val"].shape[0] == n_windows
    assert d["X_val"].shape[0] == round(0.25 * n_windows)
    # Normalised with training-split statistics.
    np.testing.assert_allclose(d["X_train"].mean(axis=(0, 1, 2, 3)), 0, atol=1e-5)
    np.testing.assert_allclose(d["X_train"].std(axis=(0, 1, 2, 3)), 1, atol=1e-4)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("n_in,n_out", [(3, 1), (1, 1), (3, 3)])
def test_models_map_input_window_to_output_window(
    model: str, n_in: int, n_out: int
) -> None:
    """The three window shapes the reported runs use: 3 to 1, 1 to 1, 3 to 3."""
    net = build_model(model, n_out=n_out, **TINY_ARCH[model])
    x = jnp.zeros((2, n_in, 16, 16, 3))
    params = net.init(jax.random.PRNGKey(0), x)
    assert net.apply(params, x).shape == (2, n_out, 16, 16, 3)


@pytest.mark.parametrize("model", MODELS)
def test_search_is_deterministic_and_draws_distinct_sizes(model: str) -> None:
    runs = sample_search_configs(model, 10, seed=0)
    assert len({r.name for r in runs}) == 10
    assert [r.arch for r in runs] == [
        r.arch for r in sample_search_configs(model, 10, 0)
    ]


@pytest.mark.parametrize("model", MODELS)
def test_best_run_is_a_size_the_search_can_reach(model: str) -> None:
    """The reported ``best_*`` runs must be winners of this search space."""
    sizes = [arch for arch, _ in SPECS[model].enumerate_configs()]
    assert RUNS[f"best_{model}_3to1"].arch in sizes


@pytest.mark.parametrize("model", MODELS)
def test_every_model_has_a_default_and_a_searched_run(model: str) -> None:
    """``infer.py`` plots ``<model>_3to1`` against ``best_<model>_3to1``."""
    assert f"{model}_3to1" in RUNS
    assert f"best_{model}_3to1" in RUNS


@pytest.mark.parametrize("model", MODELS)
def test_search_refuses_more_sizes_than_exist(model: str) -> None:
    """Sampling without replacement means an impossible request fails fast
    rather than looping forever."""
    n_sizes = len(SPECS[model].enumerate_configs())
    with pytest.raises(ValueError, match="only"):
        sample_search_configs(model, n_sizes + 1, seed=0)


@pytest.mark.parametrize("model", MODELS)
def test_models_wrap_the_operators_package(model: str) -> None:
    """The benchmark adapts models from ``src/scirex/``; it does not carry its
    own copy of any network."""
    net = build_model(model, n_out=1, **TINY_ARCH[model])
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 16, 16, 3)))["params"]
    assert any(key.startswith(WRAPPED[model]) for key in params), sorted(params)


def test_train_saves_a_checkpoint_that_loads_back(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _write_cache(cache, _data.TRAIN_CASES)
    run = RunConfig("tiny_fno", "fno", 3, 1, TINY_ARCH["fno"], epochs=2)
    protocol = {**PROTOCOL, "batch_size": 4}
    run_dir = tmp_path / "ckpt" / run.name

    training = train_one(run, cache, run_dir, protocol)

    assert is_complete(run_dir)
    assert training["epochs_run"] == 2
    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert len(ck["history"]) == 2
    d = _data.build_train_val(cache, 3, 1, protocol["val_frac"], protocol["seed"])
    np.testing.assert_array_equal(ck["mean"], d["mean"])
    np.testing.assert_array_equal(ck["std"], d["std"])
    x = jnp.asarray(d["X_val"][:2])
    assert ck["model"].apply(ck["params"], x).shape == (2, 1, 16, 16, 3)


class _CopyLastFrame:
    """Stands in for a trained model: repeats the last input frame."""

    def __init__(self, n_out: int) -> None:
        self.n_out = n_out

    def apply(self, params, x):
        return jnp.repeat(x[:, -1:], self.n_out, axis=1)


@pytest.mark.parametrize("n_out", [1, 3])
def test_rollout_slides_the_window_like_the_baseline(n_out: int) -> None:
    grid = np.random.default_rng(0).standard_normal((20, 8, 8, 3)).astype(np.float32)
    ck = dict(
        model=_CopyLastFrame(n_out),
        params=None,
        run=RunConfig("copy", "fno", 3, n_out),
        normalizer=_data.normalizer_from_stats(
            np.zeros(3, np.float32), np.ones(3, np.float32)
        ),
    )
    got = infer.rollout(ck, grid, horizon=5, batch_size=4)
    want = infer.persistence(grid, n_in=3, horizon=5)["rollout"]
    assert got["n_sequences"] == want["n_sequences"] == 20 - 3 - 5 + 1
    np.testing.assert_allclose(got["per_step_rmse"], want["per_step_rmse"], rtol=1e-6)


def test_every_model_is_either_in_the_paper_or_declared_not_to_be() -> None:
    """``_reference`` lists the paper's numbers per model and names the models
    it has none for. A model in neither list would silently lose its
    comparison bar in ``infer.py``; a model in both would claim a number the
    paper never published."""
    from scripts.benchmarks.cavity import _reference as ref

    covered = set(ref.SINGLE_STEP_TARGET)
    absent = set(ref.NOT_IN_PAPER)
    assert covered | absent == set(MODELS)
    assert not covered & absent
    assert set(ref.TABLE1_50_EPOCHS) == covered


# ───────────────────────────── GINO geometry ───────────────────────────────
def test_cell_centres_match_the_field_layout() -> None:
    """Row ``h * W + w`` must be the centre of the cell at ``[h, w]``, or the
    point cloud GINO reads is a transposed copy of the field."""
    pts = _data.cell_centres(4, 4)
    assert pts.shape == (16, 2)
    # Field axes are (H = y, W = x), so x varies fastest along a row.
    np.testing.assert_allclose(pts[0], [0.125, 0.125])
    np.testing.assert_allclose(pts[1], [0.375, 0.125])  # next column, same row
    np.testing.assert_allclose(pts[4], [0.125, 0.375])  # next row, same column
    np.testing.assert_allclose(pts[-1], [0.875, 0.875])


def test_gino_geometry_reaches_every_point() -> None:
    """Every latent point must see some cell, and every cell some latent
    point. A query with no neighbours contributes nothing to the integral and
    would silently predict zero there."""
    geom = gino_geometry(16, 16, 8, 1.5, 1.0)
    n_latent, n_cells = 8 * 8, 16 * 16
    assert geom.neighbors_in["counts"].shape == (n_latent,)
    assert geom.neighbors_out["counts"].shape == (n_cells,)
    assert int(jnp.min(geom.neighbors_in["counts"])) > 0
    assert int(jnp.min(geom.neighbors_out["counts"])) > 0
    assert geom.input_geom.shape == (1, n_cells, 2)
    assert geom.latent_queries.shape == (1, 8, 8, 2)


def test_gino_geometry_rejects_a_radius_that_strands_points() -> None:
    """A too-small output radius leaves cells with no latent neighbour. That
    must fail loudly rather than train to a partly-zero field."""
    with pytest.raises(ValueError, match="no neighbor within radius"):
        gino_geometry(16, 16, 8, 1.5, 0.05)


def test_gino_geometry_is_cached_not_rebuilt() -> None:
    """The mesh never changes, so the KD-trees are built once. Flax retraces
    on every init, and repeating both searches per trace is pure waste."""
    assert gino_geometry(16, 16, 8, 1.5, 1.0) is gino_geometry(16, 16, 8, 1.5, 1.0)


def test_gino_radii_follow_the_latent_grid() -> None:
    """The radii are multiples of the latent spacing, so a finer latent grid
    stays connected without a second knob to remember."""
    coarse = gino_geometry(16, 16, 8, 1.5, 1.0)
    fine = gino_geometry(16, 16, 16, 1.5, 1.0)
    assert fine.out_radius < coarse.out_radius
    np.testing.assert_allclose(coarse.out_radius, 1.0 / 7)
    np.testing.assert_allclose(coarse.in_radius, 1.5 / 7)


def test_gino_reads_the_same_window_the_fno_does() -> None:
    """The point cloud must carry the window unchanged and in cell order.

    ``PointCloudGINO`` flattens ``(B, T, H, W, C)`` to ``(B, H * W, T * C)``
    by hand. If that transpose were wrong the model would still train, just on
    a scrambled field, so the layout is pinned here rather than trusted.
    """
    x = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.float32).reshape(2, 3, 4, 5, 3)
    flat = x.transpose(0, 2, 3, 1, 4).reshape(2, 4 * 5, 3 * 3)
    # Point p = h * W + w holds every (frame, field) value of cell (h, w).
    for h, w in ((0, 0), (2, 3), (3, 4)):
        np.testing.assert_array_equal(flat[1, h * 5 + w], x[1, :, h, w, :].ravel())


def test_wno_and_fno_fold_the_window_into_the_same_channels() -> None:
    """Both adapters must hand their operator the same window.

    The comparison between them is only about Fourier against wavelet, so the
    two must differ in nothing else. ``ChannelStackedFNO`` builds a
    channels-first ``(B, T * C, H, W)`` and ``GridWNO`` a channels-last
    ``(B, H, W, T * C)``; pinned here so that a change to one of those
    transposes cannot quietly make the models read different fields.
    """
    x = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.float32).reshape(2, 3, 4, 5, 3)
    fno_view = x.transpose(0, 1, 4, 2, 3).reshape(2, 3 * 3, 4, 5)
    wno_view = x.transpose(0, 2, 3, 1, 4).reshape(2, 4, 5, 3 * 3)
    np.testing.assert_array_equal(wno_view, fno_view.transpose(0, 2, 3, 1))
    # Channel t * C + c of cell (h, w) is field c of frame t, in both views.
    for h, w in ((0, 0), (2, 3), (3, 4)):
        np.testing.assert_array_equal(wno_view[1, h, w], x[1, :, h, w, :].ravel())


def test_wno_level_trades_resolution_for_parameters() -> None:
    """A deeper decomposition must cost fewer parameters, not more.

    ``WNOSpec`` justifies putting ``level`` where the FNO's ``modes`` sits by
    saying the two run in opposite directions: a deeper level leaves a smaller
    approximation band, so there are fewer coefficients to weight. The search
    grid's spread depends on that holding.
    """
    counts = [
        count_params(
            build_model(
                "wno", n_out=1, hidden_channels=4, n_layers=1, level=level
            ).init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 16, 16, 3)))
        )
        for level in (1, 2, 3)
    ]
    assert counts == sorted(counts, reverse=True), counts


def test_wno_trains_in_float32_like_every_other_architecture() -> None:
    """The WNO's own default is bfloat16; this benchmark overrides it.

    Measured, not assumed: at the FNO-matched size bfloat16 was slower per
    epoch on this GPU as well as less accurate, because the subband weights
    stay float32 either way. If the default is ever flipped back, the WNO
    would be the only model here not training in float32 and the comparison
    would no longer be like for like.
    """
    net = build_model("wno", n_out=1, **TINY_ARCH["wno"])
    assert net.compute_dtype is jnp.float32
    x = jnp.zeros((1, 3, 16, 16, 3))
    params = net.init(jax.random.PRNGKey(0), x)
    assert net.apply(params, x).dtype == jnp.float32


def test_transolver_reads_the_same_window_as_the_fno() -> None:
    """Transolver's tokens must carry the same values as the FNO's channels.

    ``GridTokenTransolver`` flattens the grid into one token per cell, so its
    view is ``(B, H * W, T * C)`` where the FNO's is ``(B, T * C, H, W)``. The
    comparison between them is about a learned decomposition against a fixed
    Fourier one, so, as for the WNO, nothing else may differ.
    """
    x = np.arange(2 * 3 * 4 * 5 * 3, dtype=np.float32).reshape(2, 3, 4, 5, 3)
    fno_view = x.transpose(0, 1, 4, 2, 3).reshape(2, 3 * 3, 4, 5)
    token_view = x.transpose(0, 2, 3, 1, 4).reshape(2, 4 * 5, 3 * 3)
    np.testing.assert_array_equal(
        token_view, fno_view.transpose(0, 2, 3, 1).reshape(2, 4 * 5, 3 * 3)
    )
    # Token h * W + w carries every (frame, field) value of cell (h, w).
    for h, w in ((0, 0), (2, 3), (3, 4)):
        np.testing.assert_array_equal(
            token_view[1, h * 5 + w], x[1, :, h, w, :].ravel()
        )


def test_transolver_token_order_survives_the_structured_attention_reshape() -> None:
    """The adapter's flatten must be the one the structured attention undoes.

    ``StructuredMesh2D`` reshapes ``(B, N, C)`` back to ``(B, H, W, C)``
    to run its convolutions, and takes on trust that ``N`` was laid out
    row-major over the grid. If the adapter ever flattened column-major, every
    convolution would read a transposed neighbourhood and nothing would fail
    loudly.
    """
    grid = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    flat = grid.reshape(2, 4 * 5, 3)
    np.testing.assert_array_equal(flat.reshape(2, 4, 5, 3), grid)
    for h, w in ((0, 0), (2, 3), (3, 4)):
        np.testing.assert_array_equal(flat[1, h * 5 + w], grid[1, h, w])


def _transolver_params(**arch: object) -> int:
    net = build_model("transolver", n_out=1, **arch)
    return count_params(net.init(jax.random.PRNGKey(0), jnp.zeros((1, 3, 16, 16, 3))))


def test_transolver_parameters_grow_quadratically_in_width() -> None:
    """Doubling ``n_hidden`` must roughly quadruple the count.

    This is why the search grid's widths are the FNO's own powers of two and
    why the FNO-matched size had to be solved for rather than picked: at three
    layers, one step of the width jumps the total by more than the whole
    budget's tolerance.
    """
    counts = [_transolver_params(n_hidden=d, n_layers=1) for d in (32, 64, 128)]
    assert counts == sorted(counts), counts
    for smaller, larger in pairwise(counts):
        assert 3.0 < larger / smaller < 4.2, counts


def test_transolver_slice_count_is_not_a_sizing_knob() -> None:
    """Changing ``slice_num`` must barely move the parameter count.

    ``TransolverSpec`` puts ``slices`` where the FNO's ``modes`` sits but says
    the two cost differently: a Fourier mode carries a weight matrix, a slice
    only one projection column per head. The FNO-matched size relies on that,
    since it fixes the slice count and solves for the width alone.
    """
    counts = [
        _transolver_params(n_hidden=32, n_layers=1, slice_num=s) for s in (16, 64)
    ]
    assert counts == sorted(counts), counts
    assert (counts[-1] - counts[0]) / counts[0] < 0.01, counts
