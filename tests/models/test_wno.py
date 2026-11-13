# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the WNO (Wavelet Neural Operator) model (src/scirex/models/wno.py).

Data layout: NHWC - (batch, *spatial_dims, channels).

Covers:
  - Forward pass shape correctness - 1D / 2D / 3D
  - Output dtype preservation - compute_dtype propagation (bfloat16, float32, float64)
  - Backward pass - all parameter gradients finite (1D / 2D / 3D)
  - Loss function correctness - LpLoss relative L2, MSE
  - Edge cases - zero input, 1e6-scale input, batch_size=1
  - jax.jit smoke test
  - Grid embedding - use_grid on/off, explicit grid_boundaries
  - Padding - scalar (end / symmetric), per-dimension list
  - Multi-channel output
  - Constructor argument coverage - activation, projection_activation,
    compute_dtype, wavelet, mode, projection_hidden_channels, level, n_layers
  - Determinism - same params yield identical outputs
  - Error paths - invalid padding_side, padding length mismatch,
    grid_boundaries length mismatch

Run with:
    pytest tests/models/test_wno.py -v
"""

import jax
import jax.nn as jax_nn
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.wno_block import mish
from scirex.losses.data_losses import LpLoss
from scirex.models.wno import WNO
from tests import tolerances

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


def _wno_1d(**kwargs):
    defaults = dict(
        in_channels=3,
        hidden_channels=16,
        n_layers=2,
        level=2,
        size=64,
        out_channels=1,
        wavelet="db2",
    )
    defaults.update(kwargs)
    return WNO(**defaults)


def _wno_2d(**kwargs):
    defaults = dict(
        in_channels=2,
        hidden_channels=16,
        n_layers=2,
        level=2,
        size=(32, 32),
        out_channels=1,
        wavelet="db4",
    )
    defaults.update(kwargs)
    return WNO(**defaults)


def _wno_3d(**kwargs):
    defaults = dict(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16, 16),
        out_channels=1,
        wavelet="db2",
    )
    defaults.update(kwargs)
    return WNO(**defaults)


# ---------------------------------------------------------------------------
# Forward pass - output shape and finite values (1D / 2D / 3D)
# ---------------------------------------------------------------------------


def test_wno_forward_1d(rng):
    """WNO 1D forward pass produces (B, S, C_out) with finite values."""
    x = jax.random.normal(rng, (2, 64, 3))
    model = _wno_1d(out_channels=5)
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 64, 5)
    assert jnp.all(jnp.isfinite(out))


def test_wno_forward_2d(rng):
    """WNO 2D forward pass produces (B, H, W, C_out) with finite values."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = _wno_2d()
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


def test_wno_forward_3d(rng):
    """WNO 3D forward pass produces (B, D, H, W, C_out) with finite values."""
    x = jax.random.normal(rng, (1, 16, 16, 16, 2))
    model = _wno_3d()
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (1, 16, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


def test_wno_multi_channel_output(rng):
    """WNO correctly projects to out_channels > 1."""
    x = jax.random.normal(rng, (2, 16, 16, 3))
    model = WNO(
        in_channels=3,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=4,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 4)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Output dtype - compute_dtype propagation
# ---------------------------------------------------------------------------


def test_wno_output_dtype_bfloat16_default(rng):
    """WNO output dtype is bfloat16 when compute_dtype=bfloat16 (default)."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = _wno_2d(compute_dtype=jnp.bfloat16)
    out = model.apply(model.init(rng, x), x)
    assert out.dtype == jnp.bfloat16


def test_wno_output_dtype_float32(rng):
    """WNO output dtype is float32 when compute_dtype=float32."""
    x = jax.random.normal(rng, (2, 32, 32, 2), dtype=jnp.float32)
    model = _wno_2d(compute_dtype=jnp.float32)
    out = model.apply(model.init(rng, x), x)
    assert out.dtype == jnp.float32


@pytest.mark.needs_x64
def test_wno_output_dtype_float64(rng):
    """WNO output dtype is float64 when compute_dtype=float64."""
    x = jax.random.normal(rng, (2, 32, 32, 2), dtype=jnp.float64)
    model = _wno_2d(compute_dtype=jnp.float64, param_dtype=jnp.float64)
    out = model.apply(model.init(rng, x), x)
    assert out.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Backward pass - all parameter gradients finite (1D / 2D / 3D)
# ---------------------------------------------------------------------------


def test_wno_gradient_flows_1d(rng):
    """All WNO 1D parameter gradients are finite after one backward pass."""
    x = jax.random.normal(rng, (2, 64, 3))
    model = _wno_1d()
    params = model.init(rng, x)
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, x)))(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


def test_wno_gradient_flows_2d(rng):
    """All WNO 2D parameter gradients are finite after one backward pass."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
    )
    params = model.init(rng, x)
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, x)))(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


def test_wno_gradient_flows_3d(rng):
    """All WNO 3D parameter gradients are finite after one backward pass."""
    x = jax.random.normal(rng, (1, 16, 16, 16, 2))
    model = _wno_3d()
    params = model.init(rng, x)
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, x)))(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


def test_wno_lploss_2d(rng):
    """Relative L2 loss between WNO 2D output and a random target is finite and non-negative."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    y = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 32, 1))
    model = _wno_2d()
    y_pred = model.apply(model.init(rng, x), x)
    loss = LpLoss(d=2, p=2, reduction="mean").rel(y_pred, y)
    assert jnp.isfinite(loss)
    assert float(loss) >= 0.0


def test_wno_mse_self_is_zero(rng):
    """MSE between WNO 2D output and itself is exactly zero."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = _wno_2d()
    y_pred = model.apply(model.init(rng, x), x)
    assert float(jnp.mean((y_pred - y_pred) ** 2)) == pytest.approx(0.0, abs=1e-6)


def test_wno_mse_against_zeros_is_positive(rng):
    """MSE between WNO 2D output and all-zeros is positive and finite."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = _wno_2d()
    y_pred = model.apply(model.init(rng, x), x)
    mse = jnp.mean((y_pred - jnp.zeros_like(y_pred)) ** 2)
    assert float(mse) > 0.0
    assert jnp.isfinite(mse)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_wno_zero_input_2d(rng):
    """WNO 2D output is finite for all-zero input."""
    x = jnp.zeros((2, 32, 32, 2))
    model = _wno_2d()
    assert jnp.all(jnp.isfinite(model.apply(model.init(rng, x), x)))


def test_wno_large_input_2d(rng):
    """WNO 2D output is finite for very large magnitude input (×1e6)."""
    x = jax.random.normal(rng, (2, 32, 32, 2)) * 1e6
    model = _wno_2d()
    assert jnp.all(jnp.isfinite(model.apply(model.init(rng, x), x)))


def test_wno_batch_size_1_2d(rng):
    """WNO 2D handles batch_size=1 and returns correct shape."""
    x = jax.random.normal(rng, (1, 32, 32, 2))
    model = _wno_2d()
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (1, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# jax.jit smoke test
# ---------------------------------------------------------------------------


def test_wno_jit_1d(rng):
    """jax.jit(model.apply) compiles and runs without error for WNO 1D."""
    x = jax.random.normal(rng, (2, 64, 3))
    model = _wno_1d()
    out = jax.jit(model.apply)(model.init(rng, x), x)
    assert out.shape == (2, 64, 1)
    assert jnp.all(jnp.isfinite(out))


def test_wno_jit_2d(rng):
    """jax.jit(model.apply) compiles and runs without error for WNO 2D."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = _wno_2d()
    out = jax.jit(model.apply)(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Grid embedding
# ---------------------------------------------------------------------------


def test_wno_no_grid(rng):
    """WNO with use_grid=False skips coordinate appending and produces correct shape."""
    x = jax.random.normal(rng, (2, 32, 32, 3))
    model = WNO(
        in_channels=3,
        hidden_channels=16,
        n_layers=2,
        level=2,
        size=(32, 32),
        out_channels=1,
        use_grid=False,
    )
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (2, 32, 32, 1)


def test_wno_grid_matches_manual_channels(rng):
    """use_grid=True on 2-ch input and use_grid=False on 4-ch input give same output shape."""
    spatial = (32, 32)
    x_raw = jax.random.normal(rng, (1, *spatial, 2))
    x_pre = jax.random.normal(rng, (1, *spatial, 4))
    model_auto = WNO(
        in_channels=2,
        hidden_channels=16,
        n_layers=1,
        level=2,
        size=spatial,
        out_channels=1,
        use_grid=True,
    )
    model_manual = WNO(
        in_channels=4,
        hidden_channels=16,
        n_layers=1,
        level=2,
        size=spatial,
        out_channels=1,
        use_grid=False,
    )
    out_auto = model_auto.apply(model_auto.init(rng, x_raw), x_raw)
    out_manual = model_manual.apply(model_manual.init(rng, x_pre), x_pre)
    assert out_auto.shape == out_manual.shape == (1, *spatial, 1)


def test_wno_explicit_grid_boundaries(rng):
    """WNO accepts user-specified grid_boundaries and produces correct shape."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        use_grid=True,
        grid_boundaries=((0.0, 2.0), (0.0, 3.0)),
    )
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (2, 16, 16, 1)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def test_wno_padding_end(rng):
    """Scalar padding with padding_side='end'; output shape matches input spatial dims."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=2,
        size=(32, 32),
        out_channels=1,
        padding=2,
        padding_side="end",
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)


def test_wno_padding_symmetric(rng):
    """padding_side='symmetric' pads both sides equally; output shape unchanged."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=2,
        size=(32, 32),
        out_channels=1,
        padding=1,
        padding_side="symmetric",
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)


def _pad_apply_crop_reference(module, x, pad_amounts):
    """Explicit symmetric pad -> WNO blocks -> crop of the original region."""
    h = module.fc0(x)
    spatial_shape = h.shape[1:-1]
    h = jnp.pad(h, [(0, 0)] + [(p, p) for p in pad_amounts] + [(0, 0)])
    for block in module.wno_blocks:
        h = block(h)
    crop = tuple(slice(p, p + n) for p, n in zip(pad_amounts, spatial_shape))
    h = h[(slice(None), *crop, slice(None))]
    return module.fc2(module.projection_activation(module.fc1(h)))


@pytest.mark.parametrize(
    "shape, size, padding, pad_amounts",
    [
        ((2, 32, 1), 32, 8, [8]),
        ((2, 16, 32, 1), (16, 32), [3, 0.25], [3, 8]),
        ((2, 31, 1), 31, 0.1, [3]),
    ],
)
def test_wno_padding_symmetric_crops_original_region(
    rng, shape, size, padding, pad_amounts
):
    """Symmetric padding must crop [pad, pad + n) so outputs align with inputs."""
    x = jax.random.normal(rng, shape)
    model = WNO(
        in_channels=1,
        hidden_channels=8,
        n_layers=2,
        level=2,
        size=size,
        out_channels=1,
        wavelet="haar",
        use_grid=False,
        padding=padding,
        padding_side="symmetric",
        compute_dtype=jnp.float32,
    )
    params = model.init(rng, x)
    out = model.apply(params, x)
    expected = model.apply(params, x, pad_amounts, method=_pad_apply_crop_reference)
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, atol=tolerances.atol(1e-5))


def test_wno_padding_as_per_dim_list(rng):
    """Per-dimension padding list pads each spatial axis independently."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=2,
        size=(32, 32),
        out_channels=1,
        padding=[2, 0],
        padding_side="end",
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)


# ---------------------------------------------------------------------------
# Constructor argument coverage - activation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", [mish, jax_nn.gelu, jax_nn.relu])
def test_wno_activation_variants(rng, activation):
    """WNO forward pass is correct for different block activation functions."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=2,
        level=1,
        size=(16, 16),
        out_channels=1,
        activation=activation,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize(
    "projection_activation", [jax_nn.gelu, jax_nn.relu, jax_nn.tanh]
)
def test_wno_projection_activation_variants(rng, projection_activation):
    """WNO forward pass is correct for different projection activation functions."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        projection_activation=projection_activation,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Constructor argument coverage - wavelet and boundary mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wavelet", ["db1", "db2", "db4"])
def test_wno_wavelet_variants(rng, wavelet):
    """WNO forward + backward is correct for db1, db2, and db4 wavelets."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=2,
        size=(32, 32),
        out_channels=1,
        wavelet=wavelet,
        use_grid=False,
    )
    params = model.init(rng, x)
    out = model.apply(params, x)
    assert out.shape == (2, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))
    grads = jax.grad(lambda p: jnp.sum(model.apply(p, x)))(params)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero"])
def test_wno_boundary_mode_variants(rng, mode):
    """WNO forward pass is correct for each supported wavelet boundary mode."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=2,
        size=(32, 32),
        out_channels=1,
        mode=mode,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Constructor argument coverage - level and n_layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_wno_level_variants(rng, level):
    """WNO forward pass is correct for decomposition levels 1, 2, and 3."""
    x = jax.random.normal(rng, (2, 32, 32, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=level,
        size=(32, 32),
        out_channels=1,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 32, 32, 1)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("n_layers", [1, 2, 4])
def test_wno_n_layers_variants(rng, n_layers):
    """WNO forward pass is correct for n_layers = 1, 2, and 4."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=n_layers,
        level=1,
        size=(16, 16),
        out_channels=1,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Constructor argument coverage - projection_hidden_channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection_hidden_channels", [32, 64, 128])
def test_wno_projection_hidden_channels(rng, projection_hidden_channels):
    """WNO forward pass is correct for different projection hidden channel widths."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        projection_hidden_channels=projection_hidden_channels,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Constructor argument coverage - compute_dtype and param_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compute_dtype,param_dtype",
    [
        (jnp.bfloat16, jnp.float32),
        (jnp.float32, jnp.float32),
        pytest.param(jnp.float64, jnp.float64, marks=pytest.mark.needs_x64),
    ],
)
def test_wno_dtype_combinations(rng, compute_dtype, param_dtype):
    """WNO forward pass is correct and output dtype matches compute_dtype."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=8,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))
    assert out.dtype == compute_dtype


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_wno_deterministic_2d(rng):
    """Two forward passes with the same params produce identical outputs."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = _wno_2d(size=(16, 16))
    params = model.init(rng, x)
    out1 = model.apply(params, x)
    out2 = model.apply(params, x)
    assert jnp.allclose(
        out1, out2, atol=0.0
    ), "WNO is non-deterministic with identical params"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_wno_invalid_padding_side_raises(rng):
    """An unrecognised padding_side string should raise ValueError."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        padding=1,
        padding_side="left",
    )
    with pytest.raises(ValueError, match="Unsupported padding_side"):
        model.init(rng, x)


def test_wno_padding_length_mismatch_raises(rng):
    """A padding list whose length differs from the spatial rank should raise ValueError."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        padding=[1, 0, 0],
    )
    with pytest.raises(ValueError, match="Padding length mismatch"):
        model.init(rng, x)


def test_wno_invalid_grid_boundaries_length_raises(rng):
    """grid_boundaries with wrong number of entries should raise ValueError."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        use_grid=True,
        grid_boundaries=((0.0, 1.0),),
    )
    with pytest.raises(ValueError, match="grid_boundaries"):
        model.init(rng, x)


def test_wno_no_spatial_dims_raises(rng):
    """An input with no spatial dimensions (batch + channels only) should raise ValueError."""
    x = jax.random.normal(rng, (4, 2))  # ndim = 2 - 2 = 0
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=8,
        out_channels=1,
    )
    with pytest.raises(ValueError, match="at least one spatial dimension"):
        model.init(rng, x)


def test_wno_fractional_padding(rng):
    """Fractional padding values are interpreted as a ratio of each spatial dim."""
    x = jax.random.normal(rng, (2, 16, 16, 2))
    model = WNO(
        in_channels=2,
        hidden_channels=4,
        n_layers=1,
        level=1,
        size=(16, 16),
        out_channels=1,
        padding=0.25,
        padding_side="end",
        use_grid=False,
    )
    out = model.apply(model.init(rng, x), x)
    assert out.shape == (2, 16, 16, 1)
    assert jnp.all(jnp.isfinite(out))
