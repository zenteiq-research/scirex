# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for WaveletConv (JAX/Flax).

Covers forward-pass shape/dtype correctness across 1D/2D/3D, wavelet families,
and boundary modes; non-power-of-two padding/cropping; channel projection;
resolution generalization; gradient flow; and the layer's error paths - on
both CPU and GPU.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.wavelet_conv import (
    WaveletConv,
    _apply_weight_nchw,
    _from_jaxwt_array,
)


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _init_and_apply(model, x, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, x)
    return model.apply(variables, x), variables


# ---------------------------------------------------------------------------
# Test 1 - forward pass shape/dtype across dims, wavelets, and boundary modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("wavelet", ["db1", "db2", "db4"])
@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero", "constant"])
@pytest.mark.parametrize(
    "spatial_shape,size",
    [
        ((64,), 64),  # 1D
        ((32, 32), (32, 32)),  # 2D
        ((16, 16, 16), (16, 16, 16)),  # 3D
    ],
)
def test_WaveletConv(spatial_shape, size, mode, wavelet, backend) -> None:
    """WaveletConv forward pass across dims, wavelet families, and boundary modes.

    Checks output shape, float32 dtype (independent of jaxwt's internal
    float64 filter banks), and finiteness for 1D/2D/3D inputs.
    """
    batch, in_channels, out_channels = 2, 3, 5

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(
            key, (batch, *spatial_shape, in_channels), dtype=jnp.float32
        )

        model = WaveletConv(
            in_channels=in_channels,
            out_channels=out_channels,
            level=2,
            size=size,
            wavelet=wavelet,
            mode=mode,
        )
        y, _ = _init_and_apply(model, x, key)

        assert y.shape == (batch, *spatial_shape, out_channels)
        assert y.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Test 2 - non-power-of-two spatial sizes force real padding, cropped back
#           exactly, for every boundary mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero", "constant"])
def test_WaveletConv_padding(mode, backend) -> None:
    """WaveletConv pads non-power-of-two spatial sizes and crops back exactly.

    size=30 with level=2 forces real, non-trivial padding (target 32, pad=2)
    for every mode, so this exercises jnp.pad with actual pad_sizes > 0 - not
    just mode-string validation - including the "zero"/"constant" ->
    jnp.pad(mode="constant") translation.
    """
    batch, in_channels, out_channels = 2, 4, 8

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (batch, 30, 30, in_channels))

        model = WaveletConv(
            in_channels=in_channels,
            out_channels=out_channels,
            level=2,
            size=(30, 30),
            wavelet="db2",
            mode=mode,
        )
        y, _ = _init_and_apply(model, x, key)

        assert y.shape == (batch, 30, 30, out_channels)
        assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Test 3 - channel projection (expansion and contraction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("in_channels,out_channels", [(4, 16), (16, 4)])
def test_WaveletConv_channel_projection(in_channels, out_channels, backend) -> None:
    """WaveletConv projects correctly between differing in/out channel counts."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, in_channels))

        model = WaveletConv(
            in_channels=in_channels,
            out_channels=out_channels,
            level=2,
            size=32,
            wavelet="db2",
        )
        y, _ = _init_and_apply(model, x, key)

        assert y.shape == (2, 32, out_channels)


# ---------------------------------------------------------------------------
# Test 4 - resolution generalization (inference size != training size)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "input_size",
    [64, 16],  # 2x and 0.5x the training size (32)
)
def test_WaveletConv_resolution_generalization(input_size, backend) -> None:
    """WaveletConv adapts its effective decomposition level to inference-time
    resolutions that differ from the training size, without changing shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, input_size, 4))

        model = WaveletConv(
            in_channels=4, out_channels=4, level=2, size=32, wavelet="db2"
        )
        y, _ = _init_and_apply(model, x, key)

        assert y.shape == (2, input_size, 4)


# ---------------------------------------------------------------------------
# Test 5 - level=0 bypasses wavelet decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_level_zero_bypass(backend) -> None:
    """level=0 bypasses wavelet decomposition entirely and still returns the
    correct shape."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 4))

        model = WaveletConv(
            in_channels=4, out_channels=4, level=0, size=32, wavelet="db4"
        )
        y, _ = _init_and_apply(model, x, key)

        assert y.shape == (2, 32, 4)


# ---------------------------------------------------------------------------
# Test 6 - gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_gradient_flows(backend) -> None:
    """Gradients of a scalar loss w.r.t. all parameters are finite."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 4))

        model = WaveletConv(
            in_channels=4, out_channels=4, level=2, size=32, wavelet="db2"
        )
        variables = model.init(key, x)

        # model.apply's return type is a union (plain output vs. output+mutated
        # state, depending on `mutable`); mypy can't narrow it without a static
        # `mutable=False`, so this is a known false positive, not a real bug.
        def loss_fn(v):
            return jnp.sum(model.apply(v, x))  # type: ignore[arg-type]

        grads = jax.grad(loss_fn)(variables)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))


# ---------------------------------------------------------------------------
# Test 7 - error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_invalid_mode_raises(backend) -> None:
    """An unsupported boundary mode string raises ValueError."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 3))
        model = WaveletConv(
            in_channels=3, out_channels=3, level=1, size=32, mode="periodic"
        )
        with pytest.raises(ValueError, match="Unsupported wavelet boundary mode"):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_in_channels_mismatch_raises(backend) -> None:
    """in_channels != x.shape[-1] raises ValueError."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 3))  # 3 channels
        model = WaveletConv(in_channels=5, out_channels=5, level=1, size=32)
        with pytest.raises(ValueError, match="Input channels mismatch"):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_too_few_spatial_dims_raises(backend) -> None:
    """Input with no spatial dimension (batch + channel only) raises ValueError."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jnp.ones((4, 3))  # (batch, channels) - ndim = 2 - 2 = 0
        model = WaveletConv(in_channels=3, out_channels=3, level=1, size=8)
        with pytest.raises(ValueError):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_size_length_mismatch_raises(backend) -> None:
    """Providing a 3-tuple size for a 2D spatial input raises ValueError."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 3))  # 2D spatial
        model = WaveletConv(in_channels=3, out_channels=3, level=1, size=(16, 16, 16))
        with pytest.raises(ValueError):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("level", [0, 1])
def test_WaveletConv_too_many_dims_raises(level, backend) -> None:
    """4 spatial dims raise via the NCHW layout helper, for level=0 and level>0."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jnp.ones((1, 4, 4, 4, 4, 2))  # ndim = 6 - 2 = 4
        model = WaveletConv(in_channels=2, out_channels=2, level=level, size=4)
        with pytest.raises(ValueError, match="only 1D, 2D, and 3D"):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_reflect_overpad_raises(backend) -> None:
    """reflect-mode padding wider than the dimension itself raises ValueError."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1, 3, 2))
        model = WaveletConv(
            in_channels=2, out_channels=2, level=3, size=(3,), mode="reflect"
        )
        with pytest.raises(ValueError, match="reflect"):
            model.init(key, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_WaveletConv_resolution_forced_clamp_raises(backend) -> None:
    """Inference resolution far below training size forces effective_level to 0
    and raises instead of silently degrading to a pointwise mix."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 4, 4))
        model = WaveletConv(
            in_channels=4, out_channels=4, level=2, size=32, wavelet="db2"
        )
        with pytest.raises(ValueError, match="too small relative to"):
            model.init(key, x)


# ---------------------------------------------------------------------------
# Test 8 - internal helper functions (pure logic, no device dependency)
# ---------------------------------------------------------------------------


def test_from_jaxwt_array_too_many_dims_raises() -> None:
    """_from_jaxwt_array itself rejects ndim outside {1, 2, 3}."""
    x = jnp.ones((1, 2, 4, 4, 4, 4))
    with pytest.raises(ValueError, match="only 1D, 2D, and 3D"):
        _from_jaxwt_array(x, ndim=4)


def test_apply_weight_nchw_complex_coefficients() -> None:
    """Complex-valued coefficients are handled via the real/imag split path."""
    coeff = (jnp.ones((2, 3, 8)) + 1j * jnp.ones((2, 3, 8))).astype(jnp.complex64)
    weight = jnp.ones((3, 4, 8), dtype=jnp.float32)
    out = _apply_weight_nchw(coeff, weight)
    assert out.shape == (2, 4, 8)
    assert jnp.iscomplexobj(out)
    assert jnp.all(jnp.isfinite(out.real)) and jnp.all(jnp.isfinite(out.imag))
