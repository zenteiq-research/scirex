# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for WNOBlock (JAX/Flax).

Covers forward-pass shape/finiteness across dims, wavelet families, and
boundary modes; output dtype; activation options; channel projection;
gradient flow; jax.jit compatibility; determinism; MSE sanity; numerical
edge cases; and the standalone mish activation - on both CPU and GPU.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.wno_block import WNOBlock, mish


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
# Test 1 - forward pass shape/finiteness across dims, wavelets, boundary modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("wavelet", ["db1", "db2", "db4"])
@pytest.mark.parametrize("mode", ["symmetric", "reflect", "zero"])
@pytest.mark.parametrize(
    "spatial_shape,size",
    [
        ((64,), 64),
        ((32, 32), (32, 32)),
        ((16, 16, 16), (16, 16, 16)),
    ],
)
def test_WNOBlock(spatial_shape, size, mode, wavelet, backend) -> None:
    """WNOBlock maps to hidden_channels and preserves spatial shape across
    1D/2D/3D inputs, wavelet families, and boundary modes."""
    batch, in_channels, hidden_channels = 2, 3, 5

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (batch, *spatial_shape, in_channels))

        model = WNOBlock(
            hidden_channels=hidden_channels,
            level=2,
            size=size,
            wavelet=wavelet,
            mode=mode,
        )
        out, _ = _init_and_apply(model, x, key)

        assert out.shape == (batch, *spatial_shape, hidden_channels)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Test 2 - output dtype follows compute_dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "compute_dtype",
    [jnp.bfloat16, jnp.float32, pytest.param(jnp.float64, marks=pytest.mark.needs_x64)],
)
def test_WNOBlock_output_dtype(compute_dtype, backend) -> None:
    """WNOBlock output dtype follows input dtype when input is pre-cast to
    compute_dtype.

    WNOBlock does not cast its input (that is WNO's responsibility). When
    input is already in compute_dtype the output matches that dtype end-to-end.
    """
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 4)).astype(compute_dtype)

        model = WNOBlock(
            hidden_channels=4,
            level=1,
            size=(16, 16),
            wavelet="db2",
            compute_dtype=compute_dtype,
        )
        out, _ = _init_and_apply(model, x, key)

        assert out.dtype == compute_dtype


# ---------------------------------------------------------------------------
# Test 3 - activation options (mish, None/last-layer mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("activation", [mish, None])
def test_WNOBlock_activation_options(activation, backend) -> None:
    """WNOBlock accepts mish activation or None (last-layer mode) without error."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 10))

        model = WNOBlock(
            hidden_channels=10,
            level=1,
            size=(16, 16),
            wavelet="db2",
            activation=activation,
        )
        out, _ = _init_and_apply(model, x, key)

        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Test 4 - channel projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("hidden_channels", [2, 8, 32])
def test_WNOBlock_channel_projection(hidden_channels, backend) -> None:
    """WNOBlock projects in_channels to hidden_channels, whether that is a
    contraction or an expansion."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 32, 3))

        model = WNOBlock(
            hidden_channels=hidden_channels, level=2, size=(32, 32), wavelet="db4"
        )
        out, _ = _init_and_apply(model, x, key)

        assert out.shape == (2, 32, 32, hidden_channels)


# ---------------------------------------------------------------------------
# Test 5 - gradient flow across dims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "spatial_shape",
    [(64,), (16, 16), (8, 8, 8)],
)
def test_WNOBlock_gradient_flows(spatial_shape, backend) -> None:
    """All WNOBlock parameter gradients are finite after one backward pass,
    for 1D/2D/3D inputs."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        batch = 1 if len(spatial_shape) == 3 else 2
        x = jax.random.normal(key, (batch, *spatial_shape, 4))

        model = WNOBlock(hidden_channels=4, level=1, size=spatial_shape, wavelet="db2")
        params = model.init(key, x)

        # model.apply's return type is a union (plain output vs. output+mutated
        # state, depending on `mutable`); mypy can't narrow it without a static
        # `mutable=False`, so this is a known false positive, not a real bug.
        def loss_fn(p):
            return jnp.sum(model.apply(p, x))  # type: ignore[arg-type]

        grads = jax.grad(loss_fn)(params)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))


# ---------------------------------------------------------------------------
# Test 6 - level variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_WNOBlock_level_variants(level, backend) -> None:
    """WNOBlock forward pass is correct for decomposition levels 1, 2, and 3."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 32, 32, 8))

        model = WNOBlock(hidden_channels=8, level=level, size=(32, 32), wavelet="db2")
        out, _ = _init_and_apply(model, x, key)

        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Test 7 - MSE sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WNOBlock_mse_sanity(backend) -> None:
    """MSE(output, output) is zero; MSE(output, zeros) is positive and finite."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 4))

        model = WNOBlock(hidden_channels=4, level=1, size=(16, 16), wavelet="db2")
        y_pred, _ = _init_and_apply(model, x, key)

        assert float(jnp.mean((y_pred - y_pred) ** 2)) == pytest.approx(0.0, abs=1e-6)

        mse_against_zero = jnp.mean(y_pred**2)
        assert float(mse_against_zero) > 0.0
        assert jnp.isfinite(mse_against_zero)


# ---------------------------------------------------------------------------
# Test 8 - jax.jit smoke test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WNOBlock_jit(backend) -> None:
    """jax.jit(model.apply) compiles and runs without error."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 4))

        model = WNOBlock(hidden_channels=4, level=1, size=(16, 16), wavelet="db2")
        variables = model.init(key, x)
        out = jax.jit(model.apply)(variables, x)

        assert out.shape == (2, 16, 16, 4)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Test 9 - determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_WNOBlock_deterministic(backend) -> None:
    """Two forward passes with the same params produce identical outputs."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (2, 16, 16, 4))

        model = WNOBlock(hidden_channels=4, level=1, size=(16, 16), wavelet="db2")
        params = model.init(key, x)
        out1 = model.apply(params, x)
        out2 = model.apply(params, x)

        # model.apply's return type is a union (plain output vs. output+mutated
        # state, depending on `mutable`); mypy can't narrow it without a static
        # `mutable=False`, so this is a known false positive, not a real bug.
        assert jnp.allclose(out1, out2, atol=0.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 10 - numerical edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "case", ["zero_input", "large_input", "batch_size_1", "nonzero_residual"]
)
def test_WNOBlock_numerical_edge_cases(case, backend) -> None:
    """WNOBlock forward pass under edge-case inputs: all-zero, very large
    magnitude (x1e6), batch_size=1, and a non-trivial random input whose
    output must be non-zero."""
    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)
        model = WNOBlock(hidden_channels=4, level=1, size=(16, 16), wavelet="db2")

        if case == "zero_input":
            x = jnp.zeros((2, 16, 16, 4))
        elif case == "large_input":
            x = jax.random.normal(key, (2, 16, 16, 4)) * 1e6
        elif case == "batch_size_1":
            x = jax.random.normal(key, (1, 16, 16, 4))
        else:  # nonzero_residual
            x = jax.random.normal(key, (2, 16, 16, 4))

        out, _ = _init_and_apply(model, x, key)

        expected_batch = 1 if case == "batch_size_1" else 2
        assert out.shape == (expected_batch, 16, 16, 4)

        if case == "nonzero_residual":
            assert float(jnp.sum(jnp.abs(out))) > 0.0
        else:
            assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Test 11 - mish activation (pure function, no device dependency)
# ---------------------------------------------------------------------------


def test_mish_at_zero() -> None:
    """mish(0) should equal exactly 0 because 0 * tanh(...) = 0."""
    assert float(mish(jnp.array(0.0))) == pytest.approx(0.0, abs=1e-7)


def test_mish_positive_input() -> None:
    """mish(x) should be strictly positive for positive x."""
    assert float(mish(jnp.array(1.0))) > 0.0


def test_mish_large_negative_approaches_zero() -> None:
    """mish(x) should approach 0 from below for large negative x (bounded, not
    diverging)."""
    val = float(mish(jnp.array(-10.0)))
    assert val > -0.5
    assert jnp.isfinite(mish(jnp.array(-10.0)))


def test_mish_output_is_finite() -> None:
    """mish produces finite values for a range of random inputs."""
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    assert jnp.all(jnp.isfinite(mish(x)))


def test_mish_gradient_is_finite() -> None:
    """Gradient of mish is finite for a range of random inputs."""
    x = jax.random.normal(jax.random.PRNGKey(0), (100,))
    grads = jax.grad(lambda v: jnp.sum(mish(v)))(x)
    assert jnp.all(jnp.isfinite(grads))
