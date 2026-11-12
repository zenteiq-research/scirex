# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/car_cfd/.

No dataset or GPU needed: the run registry, the config each run builds, the
StepLR schedule, the checkpoint JSON round trip, the neighbour padding and
the Transolver and GIWO runs' size and inputs.
"""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# The benchmark scripts read meshes with open3d, which is in the optional "benchmark" extra.
pytest.importorskip("open3d", reason='needs the "benchmark" extra')

from scripts.benchmarks.car_cfd import _reference as ref  # noqa: E402
from scripts.benchmarks.car_cfd._checkpoint import count_params  # noqa: E402
from scripts.benchmarks.car_cfd._config import (  # noqa: E402
    PROTOCOL,
    RUNS,
    RunConfig,
    build_config,
    lr_at,
    run_config_from_dict,
)
from scripts.benchmarks.car_cfd._data import pad_neighbor_data  # noqa: E402
from scripts.benchmarks.car_cfd._models import (  # noqa: E402
    ARCHITECTURES,
    TransolverCFDDataProcessor,
    build_module,
)

#: The "paper" GINO run's parameter count, the budget "transolver" matches.
GINO_PAPER_PARAMS = 75_917_953


def test_paper_run_matches_the_paper_protocol() -> None:
    run = RUNS["paper"]
    config = build_config(run, PROTOCOL["n_train"], PROTOCOL["n_test"])
    assert config.data.sdf_query_resolution == 64
    assert config.model.gno_radius == 0.055
    assert config.model.use_in_gno is False
    assert config.opt.n_epochs == 100
    assert config.opt.learning_rate == 2.5e-4
    assert config.opt.weight_decay == 0.0
    assert config.model.fno_factorization is None
    assert config.model.fno_norm is None
    assert config.data.n_train == 500 and config.data.n_test == 111


@pytest.mark.parametrize("name", list(RUNS))
def test_every_run_builds_a_config(name: str) -> None:
    run = RUNS[name]
    config = build_config(run, 4, 2)
    assert config.data.sdf_query_resolution == run.sdf_query_resolution
    assert run.architecture in ARCHITECTURES
    if run.architecture != "gino":
        return
    for key, value in run.model.items():
        assert getattr(config.model, key) == value


def test_unknown_model_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="no field"):
        build_config(RunConfig("bad", {"not_a_field": 1}), 4, 2)


def test_modes_are_clamped_to_the_latent_grid() -> None:
    config = build_config(replace(RUNS["paper"], sdf_query_resolution=8), 4, 2)
    assert list(config.model.fno_n_modes) == [8, 8, 8]


def test_step_lr_halves_at_the_step() -> None:
    run = RUNS["paper"]
    assert lr_at(run, 0) == lr_at(run, 49) == 2.5e-4
    assert lr_at(run, 50) == lr_at(run, 99) == 1.25e-4
    assert lr_at(RUNS["demo"], 300) == 1e-3


@pytest.mark.parametrize("name", list(RUNS))
def test_run_config_round_trips_through_json(name: str) -> None:
    run = RUNS[name]
    assert run_config_from_dict(json.loads(json.dumps(asdict(run)))) == run


def test_every_gino_run_has_a_paper_row() -> None:
    for run in (r for r in RUNS.values() if r.architecture == "gino"):
        config = build_config(run, 4, 2)
        assert ref.paper_row(config.model.use_in_gno) in ref.TABLE2


def test_padding_gives_every_sample_the_same_edge_count() -> None:
    def nbrs(counts):
        splits = np.concatenate([[0], np.cumsum(counts)])
        return dict(neighbors_index=np.arange(splits[-1]), neighbors_row_splits=splits)

    a = {"neighbors_out": nbrs([1, 2, 0])}
    b = {"neighbors_out": nbrs([3, 3, 1])}
    max_in, max_out = pad_neighbor_data([[a], [b]])
    assert (max_in, max_out) == (0, 7)
    for s, n_real in ((a, 3), (b, 7)):
        out = s["neighbors_out"]
        assert out["neighbors_index"].shape == (7,)
        # Padded edges land in the overflow segment n_out=3, dropped by a
        # segment-sum over 3 segments.
        assert np.all(out["segment_ids"][n_real:] == 3)
        assert out["counts"].sum() == n_real


def test_transolver_run_shares_the_paper_protocol() -> None:
    gino, transolver = RUNS["paper"], RUNS["transolver"]
    for key in ("learning_rate", "lr_step", "lr_gamma", "weight_decay", "n_epochs"):
        assert getattr(transolver, key) == getattr(gino, key)


def test_transolver_matches_the_gino_parameter_budget() -> None:
    run = RUNS["transolver"]
    module = build_module(run, None)
    shapes = jax.eval_shape(
        lambda: module.init(jax.random.PRNGKey(0), x=jnp.zeros((1, 16, 6)))
    )
    n = count_params(shapes)
    assert abs(n - GINO_PAPER_PARAMS) / GINO_PAPER_PARAMS < 0.01


def test_transolver_processor_builds_position_and_normal_input() -> None:
    n_vert, n_press = 10, 7
    rng = np.random.default_rng(0)
    sample = dict(
        vertices=rng.random((1, n_vert, 3)),
        vertex_normals=rng.random((1, n_vert, 3)),
        press=rng.random((1, 1, n_press)),
    )
    batch = TransolverCFDDataProcessor(normalizer=None).preprocess(sample)
    assert batch["x"].shape == (1, n_press, 6)
    assert batch["y"].shape == (1, n_press, 1)
    np.testing.assert_array_equal(batch["x"][..., :3], sample["vertices"][:, :n_press])


def test_tiny_transolver_maps_points_to_pressure() -> None:
    run = replace(
        RUNS["transolver"],
        model=dict(n_hidden=16, n_layers=2, n_head=2, mlp_ratio=2, slice_num=4),
    )
    module = build_module(run, None)
    x = jnp.ones((1, 12, 6))
    params = module.init(jax.random.PRNGKey(0), x=x)
    assert module.apply(params, x=x).shape == (1, 12, 1)


def test_transolver_onecycle_follows_its_own_recipe() -> None:
    from scripts.benchmarks.car_cfd._config import lr_schedule

    run = RUNS["transolver_onecycle"]
    assert run.model == RUNS["transolver"].model
    assert run.n_epochs == RUNS["paper"].n_epochs
    steps = 10
    schedule = lr_schedule(run, steps)
    total = steps * run.n_epochs
    assert float(schedule(0)) == pytest.approx(1e-3 / 25)
    assert float(schedule(int(0.3 * total))) == pytest.approx(1e-3)
    assert float(schedule(total)) == pytest.approx(1e-3 / 25 / 1000)


def test_transolver_tuned_changes_only_the_optimiser() -> None:
    from scripts.benchmarks.car_cfd._config import lr_schedule

    base, tuned = RUNS["transolver"], RUNS["transolver_tuned"]
    assert tuned.model == base.model
    assert tuned.n_epochs == base.n_epochs == RUNS["paper"].n_epochs
    assert tuned.grad_clip is None and tuned.batch_size == 1
    schedule = lr_schedule(tuned, 10)
    assert float(schedule(0)) == 0.0
    assert float(schedule(10)) == pytest.approx(7.5e-5)
    assert float(schedule(10 * tuned.n_epochs)) == pytest.approx(0.0)


def test_transolver_val_differs_from_tuned_only_in_learning_rate() -> None:
    tuned, val = RUNS["transolver_tuned"], RUNS["transolver_val"]
    assert replace(val, name=tuned.name, learning_rate=tuned.learning_rate) == tuned
    assert val.learning_rate == 1e-4


def _gno_inputs(res: int, n_out: int) -> dict:
    """Decoder-only GINO/GIWO inputs on a ``res^3`` latent grid, with a
    dummy padded neighbour list (one edge per output point)."""
    return dict(
        input_geom=jnp.zeros((1, n_out, 3)),
        latent_queries=jnp.zeros((1, res, res, res, 3)),
        output_queries=jnp.zeros((1, n_out, 3)),
        latent_features=jnp.zeros((1, res, res, res, 1)),
        neighbors_out=dict(
            neighbors_index=jnp.zeros(n_out, jnp.int32),
            segment_ids=jnp.arange(n_out, dtype=jnp.int32),
            counts=jnp.ones(n_out, jnp.int32),
        ),
    )


def test_out_gno_radius_reaches_the_model() -> None:
    # gno_radius alone never reaches GINO's GNO; _PAPER_RADIUS does.
    paper = build_module(RUNS["paper"], build_config(RUNS["paper"], 4, 2))
    assert paper.out_gno_radius == 0.033
    run = RUNS["paper_r055"]
    assert build_module(run, build_config(run, 4, 2)).out_gno_radius == 0.055


def test_giwo_shares_the_paper_shell_and_protocol() -> None:
    gino, giwo = RUNS["paper"], RUNS["giwo"]
    assert {k: v for k, v in giwo.model.items() if k in gino.model} == gino.model
    assert replace(giwo, name=gino.name, model=gino.model, architecture="gino") == gino


def test_giwo_matches_the_gino_parameter_budget() -> None:
    run = RUNS["giwo"]
    module = build_module(run, build_config(run, 4, 2))
    shapes = jax.eval_shape(
        lambda: module.init(jax.random.PRNGKey(0), **_gno_inputs(64, 8))
    )
    n = count_params(shapes)
    # WNO's widths land ~7% apart at this grid (29: -2.27%, 30: +4.56%).
    assert abs(n - GINO_PAPER_PARAMS) / GINO_PAPER_PARAMS < 0.025


def test_tiny_giwo_maps_the_latent_sdf_to_surface_pressure() -> None:
    run = replace(
        RUNS["giwo"],
        sdf_query_resolution=16,
        model={**RUNS["giwo"].model, "fno_hidden_channels": 4, "wno_level": 1},
    )
    module = build_module(run, build_config(run, 4, 2))
    inputs = _gno_inputs(16, 5)
    params = module.init(jax.random.PRNGKey(0), **inputs)
    assert module.apply(params, **inputs).shape == (1, 5, 1)
    # GINO's own lifting/FNO blocks are declared but never called.
    assert "fno_blocks" not in params["params"]
    assert "wno" in params["params"]


@pytest.mark.parametrize(
    ("shape", "level"),
    [((2, 3, 32, 32, 32), 2), ((1, 2, 21, 17, 30), 2), ((2, 16, 16, 16), 1)],
)
def test_separable_3d_transforms_match_jaxwt(shape, level) -> None:
    import jaxwt

    from scripts.benchmarks.car_cfd import _fast_dwt

    x = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.float32)
    ref_c = jaxwt.wavedec3(x, "db4", mode="symmetric", level=level)
    fast_c = _fast_dwt.wavedec3(x, "db4", mode="symmetric", level=level)
    tree = jax.tree_util
    assert tree.tree_structure(ref_c) == tree.tree_structure(fast_c)
    for a, b in zip(
        jax.tree_util.tree_leaves(ref_c), jax.tree_util.tree_leaves(fast_c)
    ):
        np.testing.assert_allclose(b, a, atol=1e-4)
    # Reconstruct from altered coefficients, not just the round trip.
    coeffs = jax.tree_util.tree_map(lambda t: 1.3 * t + 0.1, ref_c)
    np.testing.assert_allclose(
        _fast_dwt.waverec3(coeffs, "db4"), jaxwt.waverec3(coeffs, "db4"), atol=1e-4
    )


def test_giwo_runs_on_the_separable_transform() -> None:
    from scirex.layers.blocks import wavelet_conv
    from scripts.benchmarks.car_cfd import _fast_dwt

    assert wavelet_conv.jaxwt.wavedec3 is _fast_dwt.wavedec3
    assert wavelet_conv.jaxwt.waverec3 is _fast_dwt.waverec3


def test_every_model_has_a_headline_run_of_its_own_architecture() -> None:
    from scripts.benchmarks.car_cfd._config import HEADLINE, MODEL_LABELS, MODELS, SEEDS

    assert set(HEADLINE) == set(MODELS) == set(MODEL_LABELS)
    for m, run in HEADLINE.items():
        assert RUNS[run].architecture == m
    assert set(SEEDS) <= set(HEADLINE.values())


def test_seed_spread_is_mean_and_sample_std_of_final_test_error() -> None:
    from scripts.benchmarks.car_cfd.infer import seed_spread

    def res(err, seed):
        return {"metrics": {"final_test_err": err, "seed": seed}}

    results = {
        "transolver_val": res(0.0676, 0),
        "transolver_val_seed1": res(0.0699, 1),
        "transolver_val_seed2": res(0.0681, 2),
        "giwo_val": res(0.119, 0),
    }
    spread = seed_spread(results, "transolver_val")
    assert spread["seeds"] == [0, 1, 2]
    assert spread["mean"] == pytest.approx(np.mean([0.0676, 0.0699, 0.0681]))
    assert spread["std"] == pytest.approx(np.std([0.0676, 0.0699, 0.0681], ddof=1))
    assert seed_spread(results, "giwo_val") is None


def test_benchmark_giwo_does_not_replace_the_library_giwo() -> None:
    from scirex.experimental.giwo import GIWO as library_giwo
    from scirex.models import BaseModel
    from scripts.benchmarks.car_cfd._giwo import GIWO

    assert BaseModel._models["giwo"] is library_giwo
    assert BaseModel._models["giwo_decoder"] is GIWO
