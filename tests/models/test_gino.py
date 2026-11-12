# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the GINO model (JAX/Flax).

Mirrors ``neuralop/models/tests/test_gino.py``. The reference ``test_gino``
sweeps ``gno_transform_type``, ``latent_feature_dim``, ``gno_coord_dim``,
``gno_pos_embed_type`` and ``batch_size`` through a forward + backward pass and
asserts (a) the output shape ``[batch, n_out, out_channels]``, (b) that every
parameter receives a gradient, and (c) gradient isolation across the batch (a
loss on sample 0 produces no gradient on samples ``1:``). Those checks are
reproduced here against the Flax port, adapting PyTorch idioms:

* ``model(...)`` -> ``model.apply(model.init(key, ...), ...)``;
* the geometry arguments (``input_geom`` / ``latent_queries`` /
  ``output_queries``) carry a leading singleton dim that the JAX GINO squeezes,
  so they are built with that extra axis;
* ``param.grad is None`` -> ``jax.grad`` over params with a zero-gradient leaf
  count; ``x.grad[1:]`` isolation -> ``jax.grad`` of ``out[0].sum()`` w.r.t. x;
* ``gno_use_torch_scatter`` is a torch-only neighbor-reduction path, so the JAX
  port uses its native reduction (``gno_use_open3d=False`` for the small CPU
  meshes here).

Tensor sizes are kept small (n_in = n_out = 40, latent grid 6**coord_dim) so the
neighbor search + FNO forward stay fast under XLA. Following the Phase 1
``layers/tests`` structure, the main forward test is parametrized over the
available CPU/GPU backends.

The reference ``test_gino_group_norm`` asserts ``fno_norm_groups`` propagation,
a parameter the JAX port does not expose; it is reproduced as a documented skip
(the portable part, ``fno_norm='group_norm'`` forward, is covered separately).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.models.gino import GINO

IN_CHANNELS = 3
OUT_CHANNELS = 2
N_IN = 40
N_OUT = 40
LATENT_DENSITY = 6


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _latent_queries(coord_dim: int) -> jnp.ndarray:
    """Latent grid on the unit cube, shape (1, density, ..., density, coord_dim)."""
    axes = [jnp.linspace(0, 1, LATENT_DENSITY)] * coord_dim
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="xy"))
    grid = jnp.transpose(mesh, (*range(1, coord_dim + 1), 0))  # (*grid, coord_dim)
    return grid[None]  # leading singleton dim (squeezed inside GINO)


def _build(
    coord_dim: int = 3,
    batch_size: int = 1,
    latent_feature_dim: int | None = None,
    pos_embed: str | None = "transformer",
    transform: str = "linear",
    fno_norm: str | None = None,
) -> tuple[GINO, dict[str, Any]]:
    model = GINO(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        latent_feature_channels=latent_feature_dim,
        in_gno_radius=0.3,  # large radius so neighborhoods overlap queries
        out_gno_radius=0.3,
        projection_channel_ratio=16,
        gno_coord_dim=coord_dim,
        in_gno_pos_embed_type=pos_embed,
        out_gno_pos_embed_type=pos_embed,
        in_gno_channel_mlp_hidden_layers=[16, 16],
        out_gno_channel_mlp_hidden_layers=[16, 16],
        in_gno_transform_type=transform,
        out_gno_transform_type=transform,
        fno_n_modes=(8, 8, 8)[:coord_dim],
        fno_hidden_channels=16,
        fno_n_layers=2,
        fno_lifting_channel_ratio=2,
        fno_norm=fno_norm,
        gno_use_open3d=False,
    )
    latent_queries = _latent_queries(coord_dim)
    inputs = dict(
        input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, coord_dim)),
        latent_queries=latent_queries,
        output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, coord_dim)),
        x=jax.random.normal(jax.random.PRNGKey(2), (batch_size, N_IN, IN_CHANNELS)),
        latent_features=(
            None
            if latent_feature_dim is None
            else jax.random.normal(
                jax.random.PRNGKey(3),
                (batch_size, *latent_queries.shape[1:-1], latent_feature_dim),
            )
        ),
        ada_in=None,
    )
    return model, inputs


def _init_apply(model: GINO, inputs: dict[str, Any]) -> tuple[Any, Any]:
    params = model.init(jax.random.PRNGKey(1), **inputs)
    return model.apply(params, **inputs), params


def _count_unused_params(model: GINO, params: Any, inputs: dict[str, Any]) -> int:
    def loss(p: Any) -> jnp.ndarray:
        out = cast(jnp.ndarray, model.apply(p, **inputs))
        return jnp.sum(jnp.abs(out))

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    return sum(1 for g in leaves if not bool((g != 0).any()))


# ---------------------------------------------------------------------------
# test_gino: forward shape, no unused params, batch gradient isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("gno_coord_dim", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_gino_forward_and_unused_params(
    gno_coord_dim: int, batch_size: int, backend: str
) -> None:
    """GINO forward gives [batch, n_out, out_channels]; all params get gradient."""
    with jax.default_device(jax.devices(backend)[0]):
        model, inputs = _build(coord_dim=gno_coord_dim, batch_size=batch_size)
        out, params = _init_apply(model, inputs)

        assert list(out.shape) == [batch_size, N_OUT, OUT_CHANNELS]
        assert bool(jnp.all(jnp.isfinite(out)))
        assert _count_unused_params(model, params, inputs) == 0

        # batch gradient isolation: a loss on sample 0 leaves x[1:] with no grad
        if batch_size > 1:

            def loss_x(x: jnp.ndarray) -> jnp.ndarray:
                local = dict(inputs)
                local["x"] = x
                return jnp.sum(model.apply(params, **local)[0])

            grad_x = jax.grad(loss_x)(inputs["x"])
            assert bool((grad_x[1:] == 0).all())
            assert bool((grad_x[0] != 0).any())


@pytest.mark.parametrize("transform", ["linear", "nonlinear_kernelonly", "nonlinear"])
def test_gino_transform_types(transform: str) -> None:
    """Each in/out GNO transform type produces the correct output shape."""
    model, inputs = _build(transform=transform)
    out, _ = _init_apply(model, inputs)
    assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


@pytest.mark.parametrize("pos_embed", [None, "transformer"])
def test_gino_pos_embed_type(pos_embed: str | None) -> None:
    """GNO positional embedding type (None / transformer) preserves output shape."""
    model, inputs = _build(pos_embed=pos_embed)
    out, _ = _init_apply(model, inputs)
    assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


@pytest.mark.parametrize("latent_feature_dim", [None, 2])
def test_gino_latent_features(latent_feature_dim: int | None) -> None:
    """An optional latent feature map is concatenated and forward still succeeds."""
    model, inputs = _build(latent_feature_dim=latent_feature_dim, batch_size=2)
    out, _ = _init_apply(model, inputs)
    assert list(out.shape) == [2, N_OUT, OUT_CHANNELS]


def test_gino_dict_output_queries() -> None:
    """output_queries may be a dict; GINO returns a dict of per-key outputs."""
    model, inputs = _build(coord_dim=3)
    single = inputs["output_queries"]
    inputs = dict(inputs)
    inputs["output_queries"] = {"a": single, "b": single}
    out, _ = _init_apply(model, inputs)
    assert set(out.keys()) == {"a", "b"}
    for value in out.values():
        assert list(value.shape) == [1, N_OUT, OUT_CHANNELS]


def test_gino_group_norm_forward() -> None:
    """The portable part of test_gino_group_norm: fno_norm='group_norm' forward."""
    model, inputs = _build(coord_dim=3, fno_norm="group_norm")
    out, _ = _init_apply(model, inputs)
    assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


@pytest.mark.skip(
    reason="Mirrors test_gino_group_norm, which asserts fno_norm_groups propagation "
    "to torch.nn.GroupNorm layers. The JAX GINO does not expose an `fno_norm_groups` "
    "parameter, so the propagation check has no analogue. The portable part "
    "(fno_norm='group_norm' forward) is covered by test_gino_group_norm_forward."
)
def test_gino_group_norm_norm_groups_not_ported() -> None:
    pass


# ---------------------------------------------------------------------------
# Warning paths in setup(): gno_coord_dim != 3 with open3d, FNO/GNO dim mismatch
# (source lines 146-151, 155)
# ---------------------------------------------------------------------------


def test_gino_setup_warns_open3d_non_3d_coord() -> None:
    """gno_coord_dim != 3 with gno_use_open3d=True triggers UserWarning (lines 146-151).

    Flax calls setup() lazily on the first init/apply, not at construction time,
    so we trigger the warning by calling model.init() with valid 2D inputs.
    """
    import warnings

    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=2,
            gno_use_open3d=True,  # triggers the non-3d warning (coord_dim != 3)
            fno_n_modes=(8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
        )
        key = jax.random.PRNGKey(0)
        latent_queries = _latent_queries(2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.init(
                key,
                input_geom=jax.random.normal(key, (1, N_IN, 2)),
                latent_queries=latent_queries,
                output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 2)),
                x=jax.random.normal(jax.random.PRNGKey(2), (1, N_IN, IN_CHANNELS)),
            )
        messages = [str(w.message) for w in caught if w.category == UserWarning]
        assert any(
            "Open3d" in m or "open3d" in m.lower() or "3-d" in m for m in messages
        )


def test_gino_setup_warns_fno_gno_dim_mismatch() -> None:
    """FNO modes dim != gno_coord_dim triggers UserWarning (line 155).

    Flax calls setup() lazily on the first init/apply. setup() emits the warning
    before __call__ runs, so we catch warnings even if init() later fails due to
    the intentional dimension mismatch.
    """
    import warnings

    with jax.default_device(jax.devices("cpu")[0]):
        # gno_coord_dim=2 but fno_n_modes has 3 entries → FNO/GNO dim mismatch
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=2,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),  # 3-d FNO but 2-d GNO
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
        )
        # latent_queries match the FNO dimensionality (3-d); input/output geom is 2-d
        latent_queries = _latent_queries(3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # setup() runs before __call__, issuing the warning. The forward pass
            # may then fail due to the intentional shape mismatch - that's expected.
            try:
                model.init(
                    jax.random.PRNGKey(0),
                    input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 2)),
                    latent_queries=latent_queries,
                    output_queries=jax.random.normal(
                        jax.random.PRNGKey(1), (1, N_OUT, 2)
                    ),
                    x=jax.random.normal(jax.random.PRNGKey(2), (1, N_IN, IN_CHANNELS)),
                )
            except Exception:
                pass  # expected: forward pass fails on incompatible coord dims
        messages = [str(w.message) for w in caught if w.category == UserWarning]
        assert any("FNO" in m or "GNO" in m or "expects" in m for m in messages)


# ---------------------------------------------------------------------------
# ada_in setup path: fno_norm="ada_in" (source lines 168-181)
# ---------------------------------------------------------------------------


def test_gino_adain_norm_setup_with_pos_embed() -> None:
    """fno_norm='ada_in' with fno_ada_in_features set creates adain_pos_embed
    (lines 172-178).

    ada_in must be non-None when fno_norm='ada_in' so that AdaIN receives its
    required embedding argument during the forward pass.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        model, inputs = _build(coord_dim=3, fno_norm="ada_in")
        # ada_in shape matches fno_ada_in_dim=1 (default)
        inputs = dict(inputs, ada_in=jax.random.normal(jax.random.PRNGKey(99), (1,)))
        params = model.init(jax.random.PRNGKey(0), **inputs)
        assert params is not None


def test_gino_adain_norm_no_pos_embed() -> None:
    """fno_norm='ada_in' with fno_ada_in_features=None uses raw ada_in_dim
    (lines 179-181).

    ada_in must be non-None when fno_norm='ada_in'; fno_ada_in_features=None
    skips SinusoidalEmbedding (else branch, lines 179-181) and passes ada_in
    directly as the embedding.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
            fno_norm="ada_in",
            fno_ada_in_features=None,  # triggers the else branch (lines 179-181)
            fno_ada_in_dim=1,
            out_gno_pos_embed_type="transformer",
        )
        latent_queries = _latent_queries(3)
        inputs: dict[str, Any] = dict(
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 3)),
            latent_queries=latent_queries,
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, N_IN, IN_CHANNELS)),
            latent_features=None,
            ada_in=jax.random.normal(jax.random.PRNGKey(99), (1,)),
        )
        params = model.init(jax.random.PRNGKey(0), **inputs)
        assert params is not None


# ---------------------------------------------------------------------------
# gno_weighting_function: dispatch_weighting_fn path (source line 243)
# ---------------------------------------------------------------------------


def test_gino_weighting_function_setup() -> None:
    """gno_weighting_function='bump' dispatches a weighting fn in gno_out (line 243)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
            gno_weighting_function="bump",
            gno_weight_function_scale=1.0,
        )
        latent_queries = _latent_queries(3)
        inputs = dict(
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 3)),
            latent_queries=latent_queries,
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, N_IN, IN_CHANNELS)),
            latent_features=None,
            ada_in=None,
        )
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


# ---------------------------------------------------------------------------
# ada_in forward path in latent_embedding (source lines 290-299)
# ---------------------------------------------------------------------------


def test_gino_adain_forward_with_ada_in() -> None:
    """Passing ada_in to forward triggers the latent_embedding ada_in branch
    (lines 290-299)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model, inputs = _build(coord_dim=3, fno_norm="ada_in")
        # ada_in shape: (fno_ada_in_dim,) = (1,) by default
        ada_in = jax.random.normal(jax.random.PRNGKey(99), (1,))
        inputs = dict(inputs, ada_in=ada_in)
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


def test_gino_adain_forward_2d_ada_in() -> None:
    """ada_in with ndim==2 is squeezed on axis=0 (line 290-291)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model, inputs = _build(coord_dim=3, fno_norm="ada_in")
        # 2-D ada_in triggers the squeeze branch
        ada_in = jax.random.normal(jax.random.PRNGKey(99), (1, 1))
        inputs = dict(inputs, ada_in=ada_in)
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


# ---------------------------------------------------------------------------
# Unexpected kwargs warning in __call__ (source line 322)
# ---------------------------------------------------------------------------


def test_gino_unexpected_kwargs_warns() -> None:
    """Passing unexpected kwargs to GINO.__call__ triggers a UserWarning (line 322)."""
    import warnings

    with jax.default_device(jax.devices("cpu")[0]):
        model, inputs = _build(coord_dim=3)
        params = model.init(jax.random.PRNGKey(0), **inputs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.apply(params, **inputs, unexpected_kwarg="oops")
        messages = [str(w.message) for w in caught if w.category == UserWarning]
        assert any(
            "unexpected" in m.lower() or "ignored" in m.lower() for m in messages
        )


# ---------------------------------------------------------------------------
# x=None path: batch_size defaults to 1 (source line 360)
# ---------------------------------------------------------------------------


def test_gino_output_query_features() -> None:
    """Per-query features reach the projection: right shape, every param gets a
    gradient, and the output changes with the features at each query."""
    model, inputs = _build(coord_dim=2)
    model = model.clone(out_query_feature_channels=3)
    feats = jax.random.normal(jax.random.PRNGKey(4), (1, N_OUT, 3))
    inputs = dict(inputs, output_query_features=feats)
    out, params = _init_apply(model, inputs)
    assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]
    assert _count_unused_params(model, params, inputs) == 0
    out2 = model.apply(params, **dict(inputs, output_query_features=feats + 1.0))
    assert bool((jnp.abs(out2 - out) > 0).all(axis=-1).all())

    with pytest.raises(AssertionError, match="output_query_features"):
        model.apply(params, **dict(inputs, output_query_features=None))


def test_gino_forward_x_none() -> None:
    """When x=None, batch_size defaults to 1 and forward still produces
    output (line 360)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model, inputs = _build(coord_dim=3)
        inputs = dict(inputs, x=None)
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


# ---------------------------------------------------------------------------
# latent_features batch tiling: shape[0]==1 broadcast to batch_size (lines 373-374)
# ---------------------------------------------------------------------------


def test_gino_latent_features_batch_tiling() -> None:
    """latent_features with batch dim=1 tiles to match batch_size>1 (lines 373-374)."""
    with jax.default_device(jax.devices("cpu")[0]):
        LATENT_FEATURE_DIM = 2
        BATCH = 3
        latent_queries = _latent_queries(3)
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
            latent_feature_channels=LATENT_FEATURE_DIM,
        )
        # latent_features has batch=1 but x has batch=BATCH → should trigger tiling
        inputs = dict(
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 3)),
            latent_queries=latent_queries,
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (BATCH, N_IN, IN_CHANNELS)),
            latent_features=jax.random.normal(
                jax.random.PRNGKey(3),
                (1, *latent_queries.shape[1:-1], LATENT_FEATURE_DIM),
            ),
            ada_in=None,
        )
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [BATCH, N_OUT, OUT_CHANNELS]


# ---------------------------------------------------------------------------
# out_gno_tanh path (source line 411)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tanh_mode", ["latent_embed", "both"])
def test_gino_out_gno_tanh(tanh_mode: str) -> None:
    """out_gno_tanh in ['latent_embed', 'both'] applies tanh to latent_embed
    (line 411)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.3,
            out_gno_radius=0.3,
            out_gno_tanh=tanh_mode,
        )
        latent_queries = _latent_queries(3)
        inputs = dict(
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, N_IN, 3)),
            latent_queries=latent_queries,
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, N_OUT, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, N_IN, IN_CHANNELS)),
            latent_features=None,
            ada_in=None,
        )
        out, _ = _init_apply(model, inputs)
        assert list(out.shape) == [1, N_OUT, OUT_CHANNELS]


# ---------------------------------------------------------------------------
# precompute_neighbors (source lines 463-545)
# ---------------------------------------------------------------------------


def test_gino_precompute_neighbors_basic() -> None:
    """precompute_neighbors populates neighbors_in/out for each dataset
    sample (lines 463-545)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.5,
            out_gno_radius=0.5,
        )

        # Build a tiny fake dataset with 2 samples
        n_pts = 8
        latent_grid = (
            np.random.RandomState(0).uniform(0, 1, (4, 4, 4, 3)).astype(np.float32)
        )
        sample_a = {
            "vertices": np.random.RandomState(1)
            .uniform(0, 1, (n_pts, 3))
            .astype(np.float32)
        }
        sample_b = {
            "vertices": np.random.RandomState(2)
            .uniform(0, 1, (n_pts, 3))
            .astype(np.float32)
        }

        dataset = SimpleNamespace(
            constant={"query_points": latent_grid},
            data_list=[sample_a, sample_b],
        )

        # Must be called inside model.bind (or via apply), but precompute_neighbors
        # only uses self.in_gno_radius, self.out_gno_radius, and imports
        # kdtree_neighbor_search.
        # We can bind the model to a dummy set of params and call it as a plain method.
        params = model.init(
            jax.random.PRNGKey(0),
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, n_pts, 3)),
            latent_queries=jnp.array(latent_grid)[None],
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, n_pts, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, n_pts, IN_CHANNELS)),
        )

        # precompute_neighbors is a regular method (not a Flax-transformed one)
        model.bind(params).precompute_neighbors(dataset)

        for sample in dataset.data_list:
            assert "neighbors_in" in sample
            assert "neighbors_out" in sample
            assert "neighbors_index" in sample["neighbors_in"]
            assert "neighbors_row_splits" in sample["neighbors_in"]


def test_gino_precompute_neighbors_with_log_dir(tmp_path: Path) -> None:
    """precompute_neighbors writes log files when log_dir is provided
    (lines 474-484)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.5,
            out_gno_radius=0.5,
        )

        n_pts = 8
        latent_grid = (
            np.random.RandomState(0).uniform(0, 1, (4, 4, 4, 3)).astype(np.float32)
        )
        sample = {
            "vertices": np.random.RandomState(1)
            .uniform(0, 1, (n_pts, 3))
            .astype(np.float32)
        }
        dataset = SimpleNamespace(
            constant={"query_points": latent_grid},
            data_list=[sample],
        )

        params = model.init(
            jax.random.PRNGKey(0),
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, n_pts, 3)),
            latent_queries=jnp.array(latent_grid)[None],
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, n_pts, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, n_pts, IN_CHANNELS)),
        )

        log_dir = tmp_path / "nbr_logs"
        model.bind(params).precompute_neighbors(dataset, log_dir=log_dir, split="train")

        assert (log_dir / "train.txt").exists()


def test_gino_precompute_neighbors_output_n_points() -> None:
    """precompute_neighbors respects output_n_points to limit out-GNO
    queries (lines 511-520)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.5,
            out_gno_radius=0.5,
        )

        n_pts = 8
        latent_grid = (
            np.random.RandomState(0).uniform(0, 1, (4, 4, 4, 3)).astype(np.float32)
        )
        sample: dict[str, Any] = {
            "vertices": np.random.RandomState(1)
            .uniform(0, 1, (n_pts, 3))
            .astype(np.float32)
        }
        dataset = SimpleNamespace(
            constant={"query_points": latent_grid},
            data_list=[sample],
        )

        params = model.init(
            jax.random.PRNGKey(0),
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, n_pts, 3)),
            latent_queries=jnp.array(latent_grid)[None],
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, n_pts, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, n_pts, IN_CHANNELS)),
        )

        # Use only 4 output points
        model.bind(params).precompute_neighbors(dataset, output_n_points=4)

        # The row_splits for neighbors_out should reflect 4 queries (5 entries)
        assert len(sample["neighbors_out"]["neighbors_row_splits"]) == 5


def test_gino_precompute_neighbors_torch_tensor_input() -> None:
    """precompute_neighbors handles torch-tensor-like inputs via .numpy()
    path (lines 466-468)."""
    with jax.default_device(jax.devices("cpu")[0]):
        model = GINO(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            gno_coord_dim=3,
            gno_use_open3d=False,
            fno_n_modes=(8, 8, 8),
            fno_hidden_channels=16,
            fno_n_layers=1,
            in_gno_channel_mlp_hidden_layers=[16],
            out_gno_channel_mlp_hidden_layers=[16],
            in_gno_radius=0.5,
            out_gno_radius=0.5,
        )

        n_pts = 8
        raw_grid = (
            np.random.RandomState(0).uniform(0, 1, (4, 4, 4, 3)).astype(np.float32)
        )

        # Create a mock "torch tensor" with a .numpy() method
        class _FakeTensor:
            def __init__(self, arr: np.ndarray) -> None:
                self._arr = arr

            def numpy(self) -> np.ndarray:
                return self._arr

            def detach(self) -> "_FakeTensor":
                return self

            def cpu(self) -> "_FakeTensor":
                return self

        fake_latent = _FakeTensor(raw_grid)
        sample = {
            "vertices": _FakeTensor(
                np.random.RandomState(1).uniform(0, 1, (n_pts, 3)).astype(np.float32)
            )
        }
        dataset = SimpleNamespace(
            constant={"query_points": fake_latent},
            data_list=[sample],
        )

        params = model.init(
            jax.random.PRNGKey(0),
            input_geom=jax.random.normal(jax.random.PRNGKey(0), (1, n_pts, 3)),
            latent_queries=jnp.array(raw_grid)[None],
            output_queries=jax.random.normal(jax.random.PRNGKey(1), (1, n_pts, 3)),
            x=jax.random.normal(jax.random.PRNGKey(2), (1, n_pts, IN_CHANNELS)),
        )

        model.bind(params).precompute_neighbors(dataset)

        assert "neighbors_in" in sample
        assert "neighbors_out" in sample
