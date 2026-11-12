# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the unified Transolver model (JAX/Flax),
``src/scirex/models/transolver.py``.

Covers: forward shape across the three geometry variants (irregular,
structured_2d, structured_3d) with fx given/None, the ``unified_pos``
positional-embedding replace-vs-concat behavior (``concat_pos``), custom
``grid_range`` bounds and a separate ``pos`` tensor (the Airfoil/Car usage
pattern), the ``time_input``/timestep-embedding path, registration in
``BaseModel``'s registry under the name "transolver", dropout train/eval
behavior, and gradient flow through every parameter (including the
``placeholder`` bias).
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from scirex.models.base import BaseModel, get_model_jax
from scirex.models.transolver import Transolver


def _init_and_apply(
    model: Transolver, key: jax.Array | None = None, **call_kwargs: Any
) -> tuple[jnp.ndarray, Any]:
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, **call_kwargs)
    return cast(jnp.ndarray, model.apply(variables, **call_kwargs)), variables


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_Transolver_registered_under_name() -> None:
    """Transolver self-registers in BaseModel._models under 'transolver'."""
    assert BaseModel._models.get("transolver") is Transolver


# ---------------------------------------------------------------------------
# Irregular geometry: fx given / fx None, unified_pos on/off
# ---------------------------------------------------------------------------


def test_Transolver_irregular_fx_given_no_unified_pos() -> None:
    key = jax.random.PRNGKey(0)
    B, N, space_dim, fun_dim, out_dim, hidden = 2, 15, 2, 3, 1, 16
    x = jax.random.normal(key, (B, N, space_dim), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, fun_dim), dtype=jnp.float32)
    model = Transolver(
        space_dim=space_dim,
        fun_dim=fun_dim,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
    )
    out, _ = _init_and_apply(model, key, x=x, fx=fx)
    assert out.shape == (B, N, out_dim)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_Transolver_irregular_fx_none() -> None:
    """fx=None: preprocess consumes x alone (fun_dim=0), placeholder still added."""
    key = jax.random.PRNGKey(0)
    B, N, space_dim, out_dim, hidden = 2, 15, 2, 1, 16
    x = jax.random.normal(key, (B, N, space_dim), dtype=jnp.float32)
    model = Transolver(
        space_dim=space_dim,
        fun_dim=0,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
    )
    out, _ = _init_and_apply(model, key, x=x, fx=None)
    assert out.shape == (B, N, out_dim)


def test_Transolver_irregular_unified_pos_replaces_x() -> None:
    """unified_pos, no concat: x is replaced by ref*ref distance features."""
    key = jax.random.PRNGKey(0)
    B, N, out_dim, hidden, ref = 2, 15, 1, 16, 4
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    model = Transolver(
        space_dim=ref * ref,
        fun_dim=0,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        unified_pos=True,
        ref=ref,
        concat_pos=False,
    )
    variables = model.init(key, x=x, fx=None)
    out = cast(jax.Array, model.apply(variables, x=x, fx=None))
    assert out.shape == (B, N, out_dim)
    # preprocess's first Dense kernel input dim should equal ref*ref (2 -> 16 replaced),
    # not 2 (the raw coordinate dim), proving replacement rather than concatenation.
    first_dense_kernel = variables["params"]["preprocess"]["Dense_0"]["kernel"]
    assert first_dense_kernel.shape[0] == ref * ref


def test_Transolver_irregular_unified_pos_3d() -> None:
    """unified_pos=True on 3D irregular coordinates (e.g. mesh vertex xyz, as
    used by car-CFD): _get_grid_irregular must build a 3D reference grid
    (ref**3 points), not hardcode a 2D one. Regression guard for a bug where
    this only worked for 2D irregular meshes (the only case exercised by the
    6 PDE-Solving datasets) and broke with a broadcasting TypeError on any
    3D irregular input.
    """
    key = jax.random.PRNGKey(0)
    B, N, out_dim, hidden, ref = 2, 12, 1, 16, 3
    x = jax.random.uniform(key, (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=ref**3,
        fun_dim=0,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        unified_pos=True,
        ref=ref,
        concat_pos=False,
    )
    variables = model.init(key, x=x, fx=None)
    out = cast(jax.Array, model.apply(variables, x=x, fx=None))
    assert out.shape == (B, N, out_dim)
    first_dense_kernel = variables["params"]["preprocess"]["Dense_0"]["kernel"]
    assert first_dense_kernel.shape[0] == ref**3


def test_Transolver_irregular_unified_pos_concat_appends_to_x() -> None:
    """unified_pos=True, concat_pos=True: x is concatenated with distance features.

    No separate `pos` is passed, so `x` itself doubles as the coordinate
    tensor fed to `_get_grid_irregular` (which builds a 2D reference grid) --
    so raw_dim must be 2 here, matching that assumption.
    """
    key = jax.random.PRNGKey(0)
    B, N, raw_dim, out_dim, hidden, ref = 2, 15, 2, 1, 16, 4
    x = jax.random.uniform(key, (B, N, raw_dim), dtype=jnp.float32)
    model = Transolver(
        space_dim=raw_dim + ref * ref,
        fun_dim=0,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        unified_pos=True,
        ref=ref,
        concat_pos=True,
    )
    variables = model.init(key, x=x, fx=None)
    out = cast(jax.Array, model.apply(variables, x=x, fx=None))
    assert out.shape == (B, N, out_dim)
    first_dense_kernel = variables["params"]["preprocess"]["Dense_0"]["kernel"]
    assert first_dense_kernel.shape[0] == raw_dim + ref * ref


def test_Transolver_irregular_unified_pos_uses_separate_pos_tensor() -> None:
    """concat_pos=True + separate `pos`: distances are computed from `pos`, not x."""
    key = jax.random.PRNGKey(0)
    B, N, feat_dim, out_dim, hidden, ref = 2, 10, 4, 1, 16, 3
    x = jax.random.normal(key, (B, N, feat_dim), dtype=jnp.float32)
    pos_a = jax.random.uniform(jax.random.PRNGKey(1), (B, N, 2), dtype=jnp.float32)
    pos_b = jax.random.uniform(jax.random.PRNGKey(2), (B, N, 2), dtype=jnp.float32)
    model = Transolver(
        space_dim=feat_dim + ref * ref,
        fun_dim=0,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        unified_pos=True,
        ref=ref,
        concat_pos=True,
        grid_range=((-1.0, 1.0), (-1.0, 1.0)),
    )
    variables = model.init(key, x=x, fx=None, pos=pos_a)
    out_a = cast(jax.Array, model.apply(variables, x=x, fx=None, pos=pos_a))
    out_b = cast(jax.Array, model.apply(variables, x=x, fx=None, pos=pos_b))
    assert not jnp.allclose(
        out_a, out_b
    ), "different pos tensors should yield different outputs"


# ---------------------------------------------------------------------------
# Structured 2D / 3D geometries
# ---------------------------------------------------------------------------


def test_Transolver_structured_2d_shape() -> None:
    key = jax.random.PRNGKey(0)
    H, W, fun_dim, out_dim, hidden, B = 4, 4, 2, 1, 16, 2
    x = jax.random.normal(key, (B, H * W, 2), dtype=jnp.float32)
    fx = jax.random.normal(
        jax.random.PRNGKey(1), (B, H * W, fun_dim), dtype=jnp.float32
    )
    model = Transolver(
        space_dim=2,
        fun_dim=fun_dim,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="structured_2d",
        H=H,
        W=W,
    )
    out, _ = _init_and_apply(model, key, x=x, fx=fx)
    assert out.shape == (B, H * W, out_dim)


def test_Transolver_stays_float32_under_jax_enable_x64() -> None:
    """Regression test: _get_grid_structured/_get_grid_irregular built their
    positional grids via jnp.linspace with no explicit dtype, and
    physics_attention.py's "temperature" param (plus this module's own
    "placeholder" param) were initialized the same way -- all four silently
    became float64 under jax_enable_x64=True (this repo's own tensorly
    import-time side effect, see spectral_convolution.py's docstring),
    and JAX's implicit type promotion then upcast every downstream
    activation that touched them. Measured at ~32x slower than float32 for
    this architecture. Covers both geometries the bug hit: structured_2d
    (positional grid + placeholder) and irregular (temperature, inside
    Physics_Attention_Irregular_Mesh).
    """

    def _assert_all_float32(geometry: str, model: Transolver) -> None:
        variables = model.init(key, x=x, fx=fx)
        leaf_dtypes = {
            jnp.dtype(leaf.dtype) for leaf in jax.tree_util.tree_leaves(variables)
        }
        assert leaf_dtypes == {
            jnp.dtype(jnp.float32)
        }, f"geometry={geometry!r}: expected all-float32 params, got {leaf_dtypes}"
        out = cast(jnp.ndarray, model.apply(variables, x=x, fx=fx))
        assert jnp.dtype(out.dtype) == jnp.dtype(
            jnp.float32
        ), f"geometry={geometry!r}: expected float32 output, got {out.dtype}"

    x64_was_enabled = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        key = jax.random.PRNGKey(0)
        H, W, fun_dim, out_dim, hidden, B = 4, 4, 2, 1, 16, 2
        x = jax.random.normal(key, (B, H * W, 2), dtype=jnp.float32)
        fx = jax.random.normal(
            jax.random.PRNGKey(1), (B, H * W, fun_dim), dtype=jnp.float32
        )
        _assert_all_float32(
            "structured_2d",
            Transolver(
                space_dim=2,
                fun_dim=fun_dim,
                out_dim=out_dim,
                n_hidden=hidden,
                n_layers=2,
                n_head=2,
                slice_num=4,
                unified_pos=True,
                geometry="structured_2d",
                H=H,
                W=W,
            ),
        )
        _assert_all_float32(
            "irregular",
            Transolver(
                space_dim=2,
                fun_dim=fun_dim,
                out_dim=out_dim,
                n_hidden=hidden,
                n_layers=2,
                n_head=2,
                slice_num=4,
                unified_pos=True,
                geometry="irregular",
            ),
        )
    finally:
        jax.config.update("jax_enable_x64", x64_was_enabled)


def test_Transolver_structured_2d_placeholder_only_added_when_fx_none() -> None:
    """Structured geometries add `placeholder` unconditionally in the fx=None branch,
    and never in the fx-given branch (unlike irregular, which always adds it)."""
    key = jax.random.PRNGKey(0)
    H, W, hidden, B = 4, 4, 16, 2
    x = jax.random.normal(key, (B, H * W, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, H * W, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=hidden,
        n_layers=1,
        n_head=2,
        slice_num=4,
        geometry="structured_2d",
        H=H,
        W=W,
    )
    variables = model.init(key, x=x, fx=fx)
    placeholder = variables["params"]["placeholder"]
    perturbed = jax.tree_util.tree_map(lambda leaf: leaf, variables)
    perturbed["params"]["placeholder"] = placeholder + 1000.0
    out_orig = cast(jax.Array, model.apply(variables, x=x, fx=fx))
    out_perturbed = cast(jax.Array, model.apply(perturbed, x=x, fx=fx))
    assert jnp.allclose(
        out_orig, out_perturbed
    ), "perturbing placeholder should not change output when fx is given (structured)"


def test_Transolver_structured_3d_shape() -> None:
    key = jax.random.PRNGKey(0)
    H, W, D, fun_dim, out_dim, hidden, B = 2, 2, 2, 1, 2, 16, 2
    x = jax.random.normal(key, (B, H * W * D, 3), dtype=jnp.float32)
    fx = jax.random.normal(
        jax.random.PRNGKey(1), (B, H * W * D, fun_dim), dtype=jnp.float32
    )
    model = Transolver(
        space_dim=3,
        fun_dim=fun_dim,
        out_dim=out_dim,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="structured_3d",
        H=H,
        W=W,
        D=D,
    )
    out, _ = _init_and_apply(model, key, x=x, fx=fx)
    assert out.shape == (B, H * W * D, out_dim)


def test_Transolver_structured_2d_unified_pos_ignores_real_x() -> None:
    """Structured unified_pos computes a fixed grid from H/W only; two different raw
    x give the same distance features (identical output when fx is None)."""
    key = jax.random.PRNGKey(0)
    H, W, hidden, ref = 3, 3, 16, 3
    B = 2
    x1 = jax.random.normal(key, (B, H * W, 5), dtype=jnp.float32)
    x2 = jax.random.normal(jax.random.PRNGKey(99), (B, H * W, 5), dtype=jnp.float32)
    model = Transolver(
        space_dim=ref * ref,
        fun_dim=0,
        out_dim=1,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="structured_2d",
        H=H,
        W=W,
        unified_pos=True,
        ref=ref,
        concat_pos=False,
    )
    variables = model.init(key, x=x1, fx=None)
    out1 = cast(jax.Array, model.apply(variables, x=x1, fx=None))
    out2 = cast(jax.Array, model.apply(variables, x=x2, fx=None))
    assert jnp.allclose(out1, out2)


def test_Transolver_structured_3d_custom_grid_range() -> None:
    key = jax.random.PRNGKey(0)
    H, W, D, hidden, ref = 2, 2, 2, 16, 3
    B = 2
    x = jax.random.normal(key, (B, H * W * D, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=ref**3,
        fun_dim=0,
        out_dim=1,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="structured_3d",
        H=H,
        W=W,
        D=D,
        unified_pos=True,
        ref=ref,
        concat_pos=False,
        grid_range=((0.0, 2.0), (-1.0, 1.0), (0.0, 5.0)),
    )
    out, _ = _init_and_apply(model, key, x=x, fx=None)
    assert out.shape == (B, H * W * D, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# geometry validation
# ---------------------------------------------------------------------------


def test_Transolver_unknown_geometry_raises() -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 5, 2), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=0,
        out_dim=1,
        n_hidden=8,
        n_layers=1,
        n_head=2,
        slice_num=2,
        geometry="bogus",
    )
    with pytest.raises(ValueError, match="Unknown geometry"):
        model.init(key, x=x, fx=None)


# ---------------------------------------------------------------------------
# time_input / timestep embedding path
# ---------------------------------------------------------------------------


def test_Transolver_time_input_changes_output() -> None:
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 10, 16
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        time_input=True,
    )
    T1 = jnp.zeros((B,), dtype=jnp.float32)
    T2 = jnp.ones((B,), dtype=jnp.float32) * 5.0
    variables = model.init(key, x=x, fx=fx, T=T1)
    out1 = cast(jax.Array, model.apply(variables, x=x, fx=fx, T=T1))
    out2 = cast(jax.Array, model.apply(variables, x=x, fx=fx, T=T2))
    assert out1.shape == (B, N, 1)
    assert not jnp.allclose(out1, out2)


def test_Transolver_no_time_input_ignores_T_kwarg_path() -> None:
    """When time_input=False, self.time_fc is never built; T is simply not consumed."""
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 10, 16
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=hidden,
        n_layers=1,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        time_input=False,
    )
    variables = model.init(key, x=x, fx=fx)
    assert "time_fc" not in variables["params"]


# ---------------------------------------------------------------------------
# Dropout train vs eval
# ---------------------------------------------------------------------------


def test_Transolver_dropout_rate_with_deterministic_warns() -> None:
    """dropout_rate > 0 has no effect under the default deterministic=True;
    __call__ must warn rather than silently train without dropout."""
    key = jax.random.PRNGKey(0)
    B, N = 2, 10
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=16,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        dropout_rate=0.5,
    )
    with pytest.warns(UserWarning, match="dropout_rate=0.5 has no effect"):
        model.init({"params": key, "dropout": jax.random.PRNGKey(1)}, x=x, fx=fx)


def test_Transolver_dropout_rate_zero_does_not_warn(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """dropout_rate=0.0 (the default) must not trigger the no-effect warning."""
    key = jax.random.PRNGKey(0)
    B, N = 2, 10
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=16,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
    )
    model.init(key, x=x, fx=fx)
    assert not any("has no effect" in str(w.message) for w in recwarn.list)


def test_Transolver_dropout_train_vs_eval() -> None:
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 10, 16
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=1,
        n_hidden=hidden,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
        dropout_rate=0.5,
    )
    variables = model.init(
        {"params": key, "dropout": jax.random.PRNGKey(1)},
        x=x,
        fx=fx,
        deterministic=True,
    )
    out1 = cast(
        jax.Array,
        model.apply(
            variables,
            x=x,
            fx=fx,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(2)},
        ),
    )
    out2 = cast(
        jax.Array,
        model.apply(
            variables,
            x=x,
            fx=fx,
            deterministic=True,
            rngs={"dropout": jax.random.PRNGKey(3)},
        ),
    )
    assert jnp.allclose(out1, out2)

    out3 = cast(
        jax.Array,
        model.apply(
            variables,
            x=x,
            fx=fx,
            deterministic=False,
            rngs={"dropout": jax.random.PRNGKey(2)},
        ),
    )
    out4 = cast(
        jax.Array,
        model.apply(
            variables,
            x=x,
            fx=fx,
            deterministic=False,
            rngs={"dropout": jax.random.PRNGKey(3)},
        ),
    )
    assert not jnp.allclose(out3, out4)


# ---------------------------------------------------------------------------
# Gradient flow (no unused params) across n_layers, including last_layer head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 3])
def test_Transolver_no_unused_params(n_layers: int) -> None:
    key = jax.random.PRNGKey(0)
    B, N, hidden = 2, 10, 16
    x = jax.random.normal(key, (B, N, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (B, N, 3), dtype=jnp.float32)
    model = Transolver(
        space_dim=2,
        fun_dim=3,
        out_dim=2,
        n_hidden=hidden,
        n_layers=n_layers,
        n_head=2,
        slice_num=4,
        geometry="irregular",
    )
    variables = model.init(key, x=x, fx=fx)

    def loss(params: dict) -> jnp.ndarray:
        out = cast(jax.Array, model.apply({"params": params}, x=x, fx=fx))
        return jnp.sum(out**2)

    grads = jax.grad(loss)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert all(bool((g != 0).any()) for g in leaves)


# ---------------------------------------------------------------------------
# get_model_jax: config-driven instantiation of Transolver specifically.
# Transolver has neither an in_channels nor an out_channels dataclass field
# (unlike FNO/GINO), so this exercises the branch of get_model_jax that must
# build the model *without* injecting in_channels or crashing on its absence.
# ---------------------------------------------------------------------------


class _Config(dict):
    """Bunch-like config: a dict (for ``config['patching']``) with a ``.model``."""

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model


def test_Transolver_get_model_jax_builds_without_in_channels() -> None:
    """get_model_jax builds a real Transolver from a config dict, with no
    data_channels/in_channels present at all -- the config-driven path the
    training scripts actually use."""
    config = _Config(
        model={
            "model_arch": "transolver",
            "space_dim": 2,
            "fun_dim": 3,
            "out_dim": 2,
            "n_hidden": 16,
            "n_layers": 2,
            "n_head": 2,
            "slice_num": 4,
            "geometry": "irregular",
        }
    )
    model = get_model_jax(config)
    assert isinstance(model, Transolver)
    assert model.space_dim == 2
    assert model.fun_dim == 3
    assert model.out_dim == 2
    assert not hasattr(model, "in_channels")

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 10, 2), dtype=jnp.float32)
    fx = jax.random.normal(jax.random.PRNGKey(1), (2, 10, 3), dtype=jnp.float32)
    out, _ = _init_and_apply(model, key, x=x, fx=fx)
    assert out.shape == (2, 10, 2)


def test_Transolver_get_model_jax_strips_config_name() -> None:
    """zencfg's "_config_name" bookkeeping key (re-added on CLI-override
    revalidation) must not reach Transolver's constructor."""
    config = _Config(
        model={
            "model_arch": "transolver",
            "space_dim": 1,
            "fun_dim": 1,
            "out_dim": 1,
            "n_hidden": 8,
            "n_layers": 1,
            "n_head": 2,
            "slice_num": 2,
            "geometry": "irregular",
            "_config_name": "configbase",
        }
    )
    model = get_model_jax(config)
    assert isinstance(model, Transolver)


def _structured_config(**geometry_keys: Any) -> _Config:
    return _Config(
        model={
            "model_arch": "transolver",
            "space_dim": 2,
            "fun_dim": 0,
            "out_dim": 1,
            "n_hidden": 8,
            "n_layers": 1,
            "n_head": 2,
            "slice_num": 4,
            "unified_pos": True,
            "geometry": "structured_2d",
            **geometry_keys,
        }
    )


def test_Transolver_get_model_jax_maps_height_width_depth_to_model_fields() -> None:
    """Config keys ``height``/``width``/``depth`` reach the model as its ``H``/
    ``W``/``D`` fields, so checkpoint ``_init_kwargs`` keep the old names."""
    model = get_model_jax(_structured_config(height=6, width=5, depth=None))
    assert (model.H, model.W, model.D) == (6, 5, None)
    kwargs = model._init_kwargs
    assert (kwargs["H"], kwargs["W"], kwargs["D"]) == (6, 5, None)
    for new_name in ("height", "width", "depth"):
        assert new_name not in kwargs

    x = jax.random.normal(jax.random.PRNGKey(0), (1, 30, 2), dtype=jnp.float32)
    out, _ = _init_and_apply(model, jax.random.PRNGKey(1), x=x, fx=None)
    assert out.shape == (1, 30, 1)


def test_Transolver_get_model_jax_height_width_depth_3d() -> None:
    config = _structured_config(height=3, width=2, depth=2)
    config.model["geometry"] = "structured_3d"
    model = get_model_jax(config)
    assert (model.H, model.W, model.D) == (3, 2, 2)


def test_Transolver_config_presets_use_snake_case_size_keys() -> None:
    from scirex.config.models import TransolverDarcy2dAligned, TransolverPipe

    pipe = TransolverPipe().to_dict()
    assert pipe["height"] == pipe["width"] == 129
    darcy = TransolverDarcy2dAligned().to_dict()
    assert darcy["height"] == darcy["width"] == 64
    assert not {"H", "W", "D"} & (set(pipe) | set(darcy))


def test_Transolver_get_model_jax_legacy_size_keys_warn() -> None:
    """Old-style ``config.model.H = ...`` still works, with a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="config.model.H is deprecated"):
        model = get_model_jax(_structured_config(H=4, W=4, height=None, width=None))
    assert (model.H, model.W) == (4, 4)
