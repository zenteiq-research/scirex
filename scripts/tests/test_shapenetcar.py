# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for scripts/benchmarks/shapenetcar/.

Runs on tiny synthetic point clouds on the CPU (writing them straight into a
cache directory, so no VTK/raw dataset or GPU is needed -- see
``_write_fixture``): model shape/wiring, the split loss, the checkpoint round
trip, and the paper-reference coverage ``infer.py`` relies on.
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

from scripts.benchmarks.shapenetcar import _reference as ref  # noqa: E402
from scripts.benchmarks.shapenetcar import infer  # noqa: E402
from scripts.benchmarks.shapenetcar._checkpoint import (  # noqa: E402
    is_complete,
    load_checkpoint,
)
from scripts.benchmarks.shapenetcar._config import (  # noqa: E402
    MODELS,
    PROTOCOL,
    RunConfig,
)
from scripts.benchmarks.shapenetcar._data import MODEL_KWARGS  # noqa: E402
from scripts.benchmarks.shapenetcar._models import SPECS, build_model  # noqa: E402
from scripts.benchmarks.shapenetcar._physics import (  # noqa: E402
    make_loss,
    rel_l2,
    rel_l2_metrics,
)
from scripts.benchmarks.shapenetcar.train import train_one  # noqa: E402

#: Deliberately tiny, keyed by SPECS so a new architecture without an entry
#: here fails the tests below rather than skipping them. GINO's radii are
#: generous (not the production "avg 5-10 neighbors" sizing -- see
#: _config.py's _GINO_TRANSOLVER_MATCHED) so a handful of synthetic points
#: reliably find at least one neighbor against the fixed real-data-sized
#: latent domain (_gino_geometry._DOMAIN_MIN/_DOMAIN_MAX).
TINY_ARCH = {
    "transolver": dict(n_hidden=16, n_layers=1, n_head=2, slice_num=4, mlp_ratio=1),
    "gino": dict(
        latent_grid=3,
        modes=1,
        width=4,
        n_layers=1,
        gno_mlp_hidden=(4,),
        in_gno_radius=3.0,
        out_gno_radius=3.0,
    ),
    "giwo": dict(
        latent_grid=3,
        width=4,
        n_layers=1,
        level=1,
        wavelet="db2",
        gno_mlp_hidden=(4,),
        in_gno_radius=3.0,
        out_gno_radius=3.0,
    ),
}
WRAPPED = {
    "transolver": "Transolver",
    "gino": "GINOCarWrapper",
    "giwo": "GIWOCarWrapper",
}


def test_every_architecture_is_covered_by_these_tests() -> None:
    assert tuple(sorted(MODELS)) == tuple(sorted(TINY_ARCH)) == tuple(sorted(WRAPPED))


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("not-a-model", {}, MODEL_KWARGS)


def test_specs_hold_the_operators_class_directly() -> None:
    """Unlike lid_driven_cavity/darcy, this benchmark wraps no window/grid
    adapter around the model -- see _models.py's module docstring -- so the
    registry must hold scirex.models.Transolver itself, not a reimplementation."""
    from scirex.models import Transolver

    assert SPECS["transolver"].module is Transolver


def test_models_map_point_cloud_to_point_cloud() -> None:
    net = build_model("transolver", TINY_ARCH["transolver"], MODEL_KWARGS)
    x = jnp.zeros((1, 9, MODEL_KWARGS["space_dim"]))
    params = net.init(jax.random.PRNGKey(0), x=x, fx=None)
    out = net.apply(params, x=x, fx=None)
    assert out.shape == (1, 9, MODEL_KWARGS["out_dim"])


def test_gino_model_maps_point_cloud_to_point_cloud() -> None:
    """Unlike the generic test above, GINO's ``neighbors_in``/
    ``neighbors_out`` are left at their default ``None`` here -- the
    underlying GNOBlock computes them on the fly in that case (see
    src/scirex/layers/blocks/gno_block.py), which is fine outside a jit-compiled
    training step (this call isn't one); _gino_geometry.attach_neighbors is
    only needed to avoid retracing cost inside train.py's jitted loop."""
    net = build_model("gino", TINY_ARCH["gino"], MODEL_KWARGS)
    x = jnp.zeros((1, 9, MODEL_KWARGS["space_dim"]))
    params = net.init(jax.random.PRNGKey(0), x=x, fx=None)
    # GINOCarWrapper returns a 1-tuple (out[0],) whose array is already
    # batch-squeezed (unlike Transolver's bare (1, N, out_dim) array above)
    # -- see _models.py's GINOCarWrapper docstring: this [0]-indexed,
    # batch-free (N, out_dim) convention is exactly what _physics.py's
    # loss_fn and infer.py's evaluate() expect from module.apply(...)[0]
    # for either architecture.
    out = net.apply(params, x=x, fx=None)[0]
    assert out.shape == (9, MODEL_KWARGS["out_dim"])


def test_giwo_model_maps_point_cloud_to_point_cloud() -> None:
    """GIWO's own version of the GINO test above -- same ``neighbors_in``/
    ``neighbors_out`` default-None behavior (GNOBlock computes them on the
    fly outside a jitted training step) and the same already-batch-squeezed
    1-tuple return convention (see _models.py's GIWOCarWrapper docstring)."""
    net = build_model("giwo", TINY_ARCH["giwo"], MODEL_KWARGS)
    x = jnp.zeros((1, 9, MODEL_KWARGS["space_dim"]))
    params = net.init(jax.random.PRNGKey(0), x=x, fx=None)
    out = net.apply(params, x=x, fx=None)[0]
    assert out.shape == (9, MODEL_KWARGS["out_dim"])


def test_transolver_params_are_point_count_independent() -> None:
    """_checkpoint.py inits a template from a dummy point count to restore a
    checkpoint's weights; this only works if the point count never appears in
    a parameter's shape, the way it doesn't for a Transformer's sequence
    length. Verified here rather than assumed."""
    net = build_model("transolver", TINY_ARCH["transolver"], MODEL_KWARGS)
    shapes_by_n = [
        jax.tree_util.tree_map(
            lambda a: a.shape,
            net.init(jax.random.PRNGKey(0), x=jnp.zeros((1, n, 7)), fx=None),
        )
        for n in (3, 4, 11)
    ]
    assert shapes_by_n[0] == shapes_by_n[1] == shapes_by_n[2]


def test_gino_params_are_point_count_independent() -> None:
    """Same property as Transolver's, and for the same reason
    (_checkpoint.py's load_checkpoint inits a template from a dummy point
    count): GINO's weights live in the GNO kernel MLPs and the latent FNO,
    sized by latent_grid/gno_mlp_hidden/fno_hidden_channels, never by the
    car's own point count."""
    net = build_model("gino", TINY_ARCH["gino"], MODEL_KWARGS)
    shapes_by_n = [
        jax.tree_util.tree_map(
            lambda a: a.shape,
            net.init(jax.random.PRNGKey(0), x=jnp.zeros((1, n, 7)), fx=None),
        )
        for n in (3, 4, 11)
    ]
    assert shapes_by_n[0] == shapes_by_n[1] == shapes_by_n[2]


def test_giwo_params_are_point_count_independent() -> None:
    """Same property as GINO's own version above, for the same reason:
    GIWO's weights live in the GNO kernel MLPs and the latent WNO, sized by
    latent_grid/gno_mlp_hidden/width, never by the car's own point count."""
    net = build_model("giwo", TINY_ARCH["giwo"], MODEL_KWARGS)
    shapes_by_n = [
        jax.tree_util.tree_map(
            lambda a: a.shape,
            net.init(jax.random.PRNGKey(0), x=jnp.zeros((1, n, 7)), fx=None),
        )
        for n in (3, 4, 11)
    ]
    assert shapes_by_n[0] == shapes_by_n[1] == shapes_by_n[2]


# ─────────────────────────────────── loss ───────────────────────────────────
def test_split_loss_ignores_pressure_off_the_surface() -> None:
    """Pressure error at a non-surface point must not move loss_press --
    only loss_velo should see it."""
    net = build_model("transolver", TINY_ARCH["transolver"], MODEL_KWARGS)
    x = jnp.zeros((1, 4, 7))
    params = net.init(jax.random.PRNGKey(0), x=x, fx=None)
    loss_fn = make_loss(net, press_weight=0.5)

    y_a = jnp.zeros((4, 4))
    y_b = y_a.at[0, -1].set(100.0)  # huge pressure error at a non-surface point
    surf = jnp.array([False, True, True, True])

    _, (press_a, velo_a) = loss_fn(params, x, y_a, surf)
    _, (press_b, velo_b) = loss_fn(params, x, y_b, surf)
    # Point 0 is off the surface, so its pressure error must not move
    # loss_press; and only its pressure channel was touched, so loss_velo
    # (velocity channels only) must be unchanged too.
    assert press_a == pytest.approx(float(press_b))
    assert velo_a == pytest.approx(float(velo_b))

    # Sanity: the same perturbation at a SURFACE point DOES move loss_press.
    y_c = y_a.at[1, -1].set(100.0)  # point 1 is on the surface
    _, (press_c, _) = loss_fn(params, x, y_c, surf)
    assert press_c > press_a


def test_rel_l2_is_zero_for_a_perfect_prediction() -> None:
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert rel_l2(truth, truth) == pytest.approx(0.0, abs=1e-9)


def test_rel_l2_metrics_splits_volume_and_surface() -> None:
    # Velocity (cols :-1) matches everywhere, so "volume" should be ~0
    # regardless of the pressure column, which "surface" alone should see.
    out = np.array([[1.0, 1.0, 1.0, 5.0], [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0]])
    y = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    surf = np.array([False, True, True])
    m = rel_l2_metrics(out, y, surf)
    assert m["volume"] == pytest.approx(0.0, abs=1e-6)
    assert m["surface"] == pytest.approx(rel_l2(out[surf, -1], y[surf, -1]))
    assert m["surface"] > 0


# ───────────────────────────── checkpoint round trip ────────────────────────
def _write_fixture(
    data_dir: Path, cache_dir: Path, n_train: int, n_val: int, seed: int = 0
):
    """Synthetic samples written straight into the cache, with empty
    directories under data_dir so get_samples() enumerates the right
    sample_rel_paths -- load_shapenet_car never touches the raw VTK files
    when a cache hit exists (see _read_sample), so no dataset download is
    needed. Point counts vary per sample, exercising the same
    retrace-per-shape path production training hits."""
    rng = np.random.default_rng(seed)

    def _write(fold: str, sample_id: str, n_points: int) -> None:
        (data_dir / fold / sample_id).mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        x = rng.standard_normal((n_points, 7)).astype(np.float32)
        y = rng.standard_normal((n_points, 4)).astype(np.float32)
        pos = x[:, :3]
        surf = rng.random(n_points) > 0.5
        np.savez(cache_dir / f"{fold}_{sample_id}.npz", x=x, y=y, pos=pos, surf=surf)

    for i in range(n_val):
        _write("param0", f"val{i}", n_points=5 + i)
    for i in range(n_train):
        _write("param1", f"train{i}", n_points=6 + i)


def test_train_saves_a_checkpoint_that_loads_back(tmp_path: Path) -> None:
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=3, n_val=2)
    run = RunConfig(
        "tiny_transolver", "transolver", TINY_ARCH["transolver"], n_epochs=2
    )
    run_dir = tmp_path / "ckpt" / run.name

    training = train_one(
        run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL
    )

    assert is_complete(run_dir)
    assert training["n_epochs"] == 2
    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert len(ck["history"]) == 2
    x = jnp.zeros((1, 7, MODEL_KWARGS["space_dim"]))
    out = ck["model"].apply(ck["params"], x=x, fx=None)
    assert out.shape == (1, 7, MODEL_KWARGS["out_dim"])


def test_evaluate_reports_finite_metrics(tmp_path: Path) -> None:
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=2, n_val=2)
    run = RunConfig(
        "tiny_transolver", "transolver", TINY_ARCH["transolver"], n_epochs=1
    )
    run_dir = tmp_path / "ckpt" / run.name
    train_one(run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL)
    ck = load_checkpoint(run_dir)

    from scripts.benchmarks.shapenetcar._data import load_split

    val_samples = load_split(data_dir, 0, cache_dir, use_cache=True)["val"]
    metrics = infer.evaluate(ck, val_samples, data_dir)
    assert np.isfinite(metrics["val_loss"])
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    # No raw quadpress_smpl.vtk on disk (only cached arrays), so the drag
    # metrics must be skipped rather than crash.
    assert metrics["c_d_mean_relative_drag_error"] is None
    assert metrics["n_drag_eval_samples"] == 0


def test_gino_train_saves_a_checkpoint_that_loads_back(tmp_path: Path) -> None:
    """GINO's own version of the round trip above: exercises
    train_one's "if run.model == 'gino'" branch (attach_neighbors before
    model.init, and neighbors threaded through train_step/eval_step),
    not just the generic per-architecture shape checks."""
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=3, n_val=2)
    run = RunConfig("tiny_gino", "gino", TINY_ARCH["gino"], n_epochs=2)
    run_dir = tmp_path / "ckpt" / run.name

    training = train_one(
        run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL
    )

    assert is_complete(run_dir)
    assert training["n_epochs"] == 2
    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert len(ck["history"]) == 2
    x = jnp.zeros((1, 7, MODEL_KWARGS["space_dim"]))
    # GINOCarWrapper's 1-tuple output is already batch-squeezed -- see the
    # model-shape test above.
    out = ck["model"].apply(ck["params"], x=x, fx=None)[0]
    assert out.shape == (7, MODEL_KWARGS["out_dim"])


def test_gino_evaluate_reports_finite_metrics(tmp_path: Path) -> None:
    """GINO's own version of test_evaluate_reports_finite_metrics: exercises
    infer.evaluate's "if ck['run'].model == 'gino'" branch (attach_neighbors
    on val_samples before scoring)."""
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=2, n_val=2)
    run = RunConfig("tiny_gino", "gino", TINY_ARCH["gino"], n_epochs=1)
    run_dir = tmp_path / "ckpt" / run.name
    train_one(run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL)
    ck = load_checkpoint(run_dir)

    from scripts.benchmarks.shapenetcar._data import load_split

    val_samples = load_split(data_dir, 0, cache_dir, use_cache=True)["val"]
    metrics = infer.evaluate(ck, val_samples, data_dir)
    assert np.isfinite(metrics["val_loss"])
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    assert metrics["c_d_mean_relative_drag_error"] is None
    assert metrics["n_drag_eval_samples"] == 0


def test_giwo_train_saves_a_checkpoint_that_loads_back(tmp_path: Path) -> None:
    """GIWO's own version of the round trip above: exercises train_one's
    "if run.model in ('gino', 'giwo')" branch (attach_neighbors before
    model.init, and neighbors threaded through train_step/eval_step)."""
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=3, n_val=2)
    run = RunConfig("tiny_giwo", "giwo", TINY_ARCH["giwo"], n_epochs=2)
    run_dir = tmp_path / "ckpt" / run.name

    training = train_one(
        run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL
    )

    assert is_complete(run_dir)
    assert training["n_epochs"] == 2
    ck = load_checkpoint(run_dir)
    assert ck["run"] == run
    assert len(ck["history"]) == 2
    x = jnp.zeros((1, 7, MODEL_KWARGS["space_dim"]))
    out = ck["model"].apply(ck["params"], x=x, fx=None)[0]
    assert out.shape == (7, MODEL_KWARGS["out_dim"])


def test_giwo_evaluate_reports_finite_metrics(tmp_path: Path) -> None:
    """GIWO's own version of test_evaluate_reports_finite_metrics: exercises
    infer.evaluate's "if ck['run'].model in ('gino', 'giwo')" branch
    (attach_neighbors on val_samples before scoring)."""
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    _write_fixture(data_dir, cache_dir, n_train=2, n_val=2)
    run = RunConfig("tiny_giwo", "giwo", TINY_ARCH["giwo"], n_epochs=1)
    run_dir = tmp_path / "ckpt" / run.name
    train_one(run, data_dir, cache_dir, fold_id=0, run_dir=run_dir, p=PROTOCOL)
    ck = load_checkpoint(run_dir)

    from scripts.benchmarks.shapenetcar._data import load_split

    val_samples = load_split(data_dir, 0, cache_dir, use_cache=True)["val"]
    metrics = infer.evaluate(ck, val_samples, data_dir)
    assert np.isfinite(metrics["val_loss"])
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    assert metrics["c_d_mean_relative_drag_error"] is None
    assert metrics["n_drag_eval_samples"] == 0


# ───────────────────────────────── reference ────────────────────────────────
def test_every_model_is_either_in_the_paper_or_declared_not_to_be() -> None:
    assert set(ref.NOT_IN_PAPER).issubset(set(MODELS))
    for m in MODELS:
        expect = {} if m in ref.NOT_IN_PAPER else dict(ref.TABLE3)
        assert infer.paper_numbers(m) == expect


def test_paper_numbers_is_empty_for_a_model_declared_not_in_the_paper(
    monkeypatch,
) -> None:
    """Direct check that infer.paper_numbers actually reads NOT_IN_PAPER,
    rather than the coverage test above passing vacuously because
    NOT_IN_PAPER is empty today."""
    monkeypatch.setattr(ref, "NOT_IN_PAPER", ("transolver",))
    assert infer.paper_numbers("transolver") == {}
