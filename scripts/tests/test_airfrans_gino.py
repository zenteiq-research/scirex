# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data-free tests for scripts/benchmarks/airfrans/_gino.py: the GPU radius
search against a KD-tree, the reported model size, and one forward/backward pass."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.utils import count_model_params  # noqa: E402
from scripts.benchmarks.airfrans._gino import (  # noqa: E402
    _parser,
    build_model,
    latent_grid,
    make_forward,
    radius_neighbors,
)

REPORTED = [
    "--latent-sdf",
    "--decoder-point-features",
    "--out-gno-mlp",
    "512",
    "362",
    "--latent-res",
    "128",
    "64",
]


def _args(*extra):
    return _parser().parse_args(["--data-dir", "unused", *extra])


def _init(model, lq, args, n):
    pos = jax.random.uniform(jax.random.PRNGKey(0), (n, 2), minval=-1.0, maxval=1.0)
    lat = jnp.zeros((1, *lq.shape[:-1], 1)) if args.latent_sdf else None
    feats = jnp.zeros((1, n, 7)) if args.decoder_point_features else None
    return model.init(
        jax.random.PRNGKey(0),
        input_geom=pos[None],
        latent_queries=lq[None],
        output_queries=pos[None],
        x=jnp.zeros((1, n, 7)),
        latent_features=lat,
        neighbors_in=radius_neighbors(pos, lq.reshape(-1, 2), args.in_radius, 8),
        neighbors_out=radius_neighbors(lq.reshape(-1, 2), pos, args.out_radius, 8),
        output_query_features=feats,
    )


@pytest.mark.parametrize("radius, k", [(0.2, 8), (0.5, 32)])
def test_radius_neighbors_matches_kdtree(radius, k):
    """Same neighbour sets and counts as the k nearest KD-tree neighbours within
    the radius; empty slots carry segment id n_queries."""
    rng = np.random.default_rng(0)
    src = rng.uniform(-1, 1, (700, 2)).astype(np.float32)
    qry = rng.uniform(-1.2, 1.2, (300, 2)).astype(np.float32)
    got = radius_neighbors(jnp.asarray(src), jnp.asarray(qry), radius, k, chunk=128)
    dist, idx = cKDTree(src).query(qry, k=k)
    seg = np.asarray(got["segment_ids"]).reshape(300, k)
    nbr = np.asarray(got["neighbors_index"]).reshape(300, k)
    for q in range(300):
        want = set(idx[q][dist[q] <= radius].tolist())
        assert set(nbr[q][seg[q] == q].tolist()) == want
        assert int(got["counts"][q]) == len(want)
    assert set(np.unique(seg)) <= set(range(301))


def test_reported_model_size():
    """The reported configuration is matched to Transolver's 3,893,572 parameters."""
    args = _args(*REPORTED)
    lq = jnp.asarray(latent_grid(args.latent_res))
    shapes = jax.eval_shape(lambda: _init(build_model(args), lq, args, 512))
    assert int(count_model_params(shapes)) == 3_894_069


def test_forward_and_gradient_small_model():
    """One jitted forward and backward pass, with points outside the latent box."""
    args = _args(
        "--latent-sdf",
        "--decoder-point-features",
        "--latent-res",
        "16",
        "8",
        "--fno-modes",
        "4",
        "4",
        "--fno-hidden",
        "8",
        "--fno-layers",
        "1",
        "--fno-lifting-ratio",
        "1",
        "--out-gno-mlp",
        "16",
        "--in-k",
        "4",
        "--out-k",
        "4",
    )
    model = build_model(args)
    lq = jnp.asarray(latent_grid(args.latent_res))
    params = _init(model, lq, args, 256)
    forward = make_forward(model, lq, args)
    key = jax.random.PRNGKey(1)
    pos = jax.random.uniform(key, (256, 2), minval=-3.0, maxval=5.0)
    x = jax.random.normal(key, (256, 7))
    lat = jnp.ones((*lq.shape[:-1], 1))
    out = jax.jit(forward)(params, x, pos, lat)
    assert out.shape == (256, 4) and bool(jnp.isfinite(out).all())
    grads = jax.grad(lambda p: jnp.mean(forward(p, x, pos, lat) ** 2))(params)
    assert all(bool(jnp.isfinite(g).all()) for g in jax.tree_util.tree_leaves(grads))


def test_figures_from_needs_eval_only():
    """--figures-from without --eval-only would train a new model, so it is rejected
    before any data is read."""
    from scripts.benchmarks.airfrans import _gino

    with pytest.raises(SystemExit):
        _gino.main(["--data-dir", "unused", "--figures-from", "m"])


def test_published_args_are_the_reported_model():
    """train.py --run gino and infer.py pass _config.GINO_ARGS: the reported
    model, trained for 500 epochs with the loader's wall mask."""
    from scripts.benchmarks.airfrans._config import GINO_ARGS

    published = vars(_args(*GINO_ARGS))
    assert published == vars(_args(*REPORTED, "--epochs", "500", "--loader-surf-mask"))
