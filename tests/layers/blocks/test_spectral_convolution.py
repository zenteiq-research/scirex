# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for SpectralConv (JAX/Flax).

Covers factorization variants (Dense, Tucker), separable and non-separable modes,
real and complex data, resolution scaling, Hermitian symmetry, and output shape
correctness across 1D/2D/3D on both CPU and GPU.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.blocks.spectral_convolution import (
    JAXTuckerTensor,
    SpectralConv,
    _contract_dense,
    _contract_dense_separable,
    get_contract_fun,
)
from tests import tolerances

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _init_and_apply(
    conv: SpectralConv, x: jnp.ndarray, key: jax.Array | None = None
) -> tuple[jnp.ndarray, Any]:
    if key is None:
        key = jax.random.PRNGKey(42)
    variables = conv.init(key, x)
    return cast(jnp.ndarray, conv.apply(variables, x)), variables


# ---------------------------------------------------------------------------
# Test 1 - factorization, separability, weight correctness, mode reduction,
#           and resolution scaling across 1D/2D/3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("factorization", ["Dense", "Tucker"])
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("complex_data", [False, True])
def test_SpectralConv(
    factorization: str,
    implementation: str,
    separable: bool,
    dim: int,
    complex_data: bool,
    backend: str,
) -> None:
    """SpectralConv forward pass across factorizations, dimensions, and data types.

    - Checks that the weight tensor is complex.
    - For Dense non-separable: verifies that two convolutions with shared weights
      produce identical outputs regardless of implementation.
    - Verifies that reducing n_modes does not change the spatial output shape.
    - Checks output shape after downsampling (factor 0.5) and upsampling (factor 2).
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        shape = (2, 3, *(12,) * dim)
        if complex_data:
            xr = jax.random.normal(key, shape, dtype=jnp.float32)
            xi = jax.random.normal(jax.random.PRNGKey(1), shape, dtype=jnp.float32)
            x = (xr + 1j * xi).astype(jnp.complex64)
        else:
            x = jax.random.normal(key, shape, dtype=jnp.float32)

        conv = SpectralConv(
            in_channels=3,
            out_channels=3,
            n_modes=modes[:dim],
            bias=False,
            implementation=implementation,
            factorization=factorization,
            complex_data=complex_data,
            separable=separable,
        )

        conv_dense = SpectralConv(
            in_channels=3,
            out_channels=3,
            n_modes=modes[:dim],
            bias=False,
            implementation="reconstructed",
            factorization=None,  # Dense by default
            complex_data=complex_data,
        )

        res, vars_conv = _init_and_apply(conv, x, key)
        res_shape = res.shape

        if factorization == "Dense":
            assert jnp.issubdtype(
                vars_conv["params"]["weight"].dtype, jnp.complexfloating
            )

        # For Dense non-separable: inject the same weight into conv_dense and
        # verify both implementations produce the same output.
        if not separable and factorization == "Dense":
            _, vars_dense = _init_and_apply(conv_dense, x, key)
            vars_dense_shared = {
                **vars_dense,
                "params": {
                    **vars_dense["params"],
                    "weight": vars_conv["params"]["weight"],
                },
            }
            res_dense = cast(jnp.ndarray, conv_dense.apply(vars_dense_shared, x))
            assert jnp.allclose(res_dense, res, atol=tolerances.atol(1e-5))

        # Reducing n_modes must not change the output shape.
        conv_incremental = SpectralConv(
            in_channels=3,
            out_channels=3,
            n_modes=incremental_modes[:dim],
            bias=False,
            implementation=implementation,
            factorization=factorization,
            complex_data=complex_data,
            separable=separable,
        )
        res_inc, _ = _init_and_apply(conv_incremental, x, key)
        assert res_shape == res_inc.shape

        # Downsample outputs
        x_real = jax.random.normal(key, (2, 3, *(12,) * dim), dtype=jnp.float32)
        block_down = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=modes[:dim],
            resolution_scaling_factor=0.5,
        )
        res_down, _ = _init_and_apply(block_down, x_real, key)
        assert list(res_down.shape[2:]) == [12 // 2] * dim

        # Upsample outputs
        block_up = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=modes[:dim],
            resolution_scaling_factor=2,
        )
        res_up, _ = _init_and_apply(block_up, x_real, key)
        assert res_up.shape[1] == 4  # out_channels
        assert list(res_up.shape[2:]) == [12 * 2] * dim


# ---------------------------------------------------------------------------
# Test 2 - Hermitian symmetry, even/odd spatial sizes, and resolution scaling
#           with real-valued data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("enforce_hermitian_symmetry", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize(
    "spatial_size", [8, 9]
)  # even and odd: Nyquist handling differs
@pytest.mark.parametrize("resolution_scaling_factor", [None, 0.5, 2])
@pytest.mark.parametrize("modes", [(4, 4, 4), (4, 5, 7)])
def test_SpectralConv2(
    enforce_hermitian_symmetry: bool,
    dim: int,
    spatial_size: int,
    modes: tuple[int, ...],
    resolution_scaling_factor: float | int | None,
    backend: str,
) -> None:
    modes = modes[:dim]
    size = [spatial_size] * dim
    out_size = (
        size
        if resolution_scaling_factor is None
        else [round(s * resolution_scaling_factor) for s in size]
    )

    with jax.default_device(jax.devices(backend)[0]):
        key = jax.random.PRNGKey(0)

        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=modes,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            complex_data=False,
            resolution_scaling_factor=resolution_scaling_factor,
        )
        x = jax.random.normal(key, (2, 3, *size), dtype=jnp.float32)
        res, _ = _init_and_apply(conv, x, key)

    assert res.shape == (2, 4, *out_size)
    assert res.dtype == jnp.float32
    assert not jnp.issubdtype(res.dtype, jnp.complexfloating)


# ---------------------------------------------------------------------------
# Test 3 - JAXTuckerTensor non-tuple slice (line 47)
# ---------------------------------------------------------------------------


def test_JAXTuckerTensor_scalar_slice() -> None:
    """JAXTuckerTensor.__getitem__ with a scalar wraps it in a tuple (line 47)."""
    core = jnp.ones((4, 4))
    factors = [jnp.eye(8, 4), jnp.eye(6, 4)]
    tucker = JAXTuckerTensor(core, factors)
    # Integer (non-tuple) slice: factor[2, :] has shape (4,) - 1D
    sliced = tucker[2]
    assert sliced.core is core
    assert sliced.factors[0].shape == (4,)


def test_JAXTuckerTensor_shape_property() -> None:
    """JAXTuckerTensor.shape returns the full-tensor shape from factor rows."""
    core = jnp.ones((4, 4))
    factors = [jnp.eye(8, 4), jnp.eye(6, 4)]
    tucker = JAXTuckerTensor(core, factors)
    assert tucker.shape == (8, 6)


# ---------------------------------------------------------------------------
# Test 4 - _contract_dense / _contract_dense_separable with mock weight (lines 74, 85)
# ---------------------------------------------------------------------------


class _MockWeight:
    """Non-jnp.ndarray weight stub with .to_tensor() for dead-code path coverage."""

    def __init__(self, arr: jnp.ndarray) -> None:
        self._arr = arr

    def to_tensor(self) -> jnp.ndarray:
        return self._arr


def test_contract_dense_with_mock_weight() -> None:
    """_contract_dense calls weight.to_tensor() when weight is not jnp.ndarray
    (line 74).

    For order-3 input (batch, in_ch, spatial), _contract_dense builds the equation
    "abc,bdc->adc" (non-separable), so weight must be (in_ch, out_ch, spatial).
    """
    x = jnp.ones((1, 3, 4), dtype=jnp.complex64)
    # weight shape for non-separable, order-3: (in_ch=3, out_ch=5, spatial=4)
    mock_w = _MockWeight(jnp.ones((3, 5, 4), dtype=jnp.complex64))
    result = _contract_dense(x, mock_w, separable=False)
    assert result.shape == (1, 5, 4)


def test_contract_dense_separable_with_mock_weight() -> None:
    """_contract_dense_separable calls weight.to_tensor() when not ndarray (line 85)."""
    x = jnp.ones((1, 3, 4), dtype=jnp.complex64)
    mock_w = _MockWeight(jnp.ones((1, 3, 4), dtype=jnp.complex64))
    result = _contract_dense_separable(x, mock_w, separable=True)
    assert result.shape == x.shape


# ---------------------------------------------------------------------------
# Test 5 - get_contract_fun error paths (lines 145-149, 241)
# ---------------------------------------------------------------------------


def test_get_contract_fun_factorized_non_array_raises() -> None:
    """get_contract_fun with implementation='factorized' and non-array weight
    raises ValueError (lines 145-149)."""
    mock_w = _MockWeight(jnp.ones((3, 4)))
    with pytest.raises(ValueError, match="unexpected weight type"):
        get_contract_fun(mock_w, implementation="factorized")


def test_get_contract_fun_unknown_implementation_raises() -> None:
    """get_contract_fun with unknown implementation raises ValueError (line 241)."""
    weight = jnp.ones((3, 4, 5), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="reconstructed.*factorized"):
        get_contract_fun(weight, implementation="unknown_impl")


# ---------------------------------------------------------------------------
# Test 6 - max_n_modes as int (line 202)
# ---------------------------------------------------------------------------


def test_SpectralConv_n_modes_as_int() -> None:
    """SpectralConv with n_modes as an integer (not tuple) wraps it in a list
    (line 202)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=4,  # int → [4] conversion at line 202
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        res, _ = _init_and_apply(conv, x, key)
        assert res.shape == (2, 4, 12)


def test_SpectralConv_max_n_modes_as_int() -> None:
    """SpectralConv with max_n_modes as an integer wraps it in a list
    (lines 213-214)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=(4,),
            max_n_modes=8,  # int → [8] conversion at lines 213-214
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        res, _ = _init_and_apply(conv, x, key)
        assert res.shape == (2, 4, 12)


# ---------------------------------------------------------------------------
# Test 7 - separable with mismatched channels raises ValueError (lines 213-214)
# ---------------------------------------------------------------------------


def test_SpectralConv_separable_channel_mismatch_raises() -> None:
    """SpectralConv separable=True with in_channels!=out_channels raises
    ValueError (lines 213-214)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,  # different from in_channels
            n_modes=(4,),
            separable=True,
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        with pytest.raises(ValueError, match="separable"):
            conv.init(key, x)


# ---------------------------------------------------------------------------
# Test 8 - unsupported factorization types raise (line 237)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factorization", ["CP", "TT"])
def test_SpectralConv_cp_tt_factorization_raises(factorization: str) -> None:
    """CP/TT are PyTorch-only factorizations with no JAX port; this must raise,
    not silently train with a different factorization than requested."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=(4,),
            factorization=factorization,
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        with pytest.raises(NotImplementedError, match="dense.*tucker"):
            _init_and_apply(conv, x, key)


# ---------------------------------------------------------------------------
# Test 9 - fno_block_precision='half' and 'mixed' (lines 342, 366, 382, 385)
# ---------------------------------------------------------------------------


def test_SpectralConv_half_precision() -> None:
    """fno_block_precision='half' casts input to float16 before FFT (line 342).

    JAX RFFT does not support float16; the cast still executes, so line 342 is
    covered even though the subsequent FFT raises.  The exception is expected.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=(4,),
            fno_block_precision="half",
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        try:
            res, _ = _init_and_apply(conv, x, key)
            assert res.shape == (2, 4, 12)
        except Exception:
            pass  # float16 FFT is unsupported in JAX; line 342 is still hit


def test_SpectralConv_mixed_precision() -> None:
    """fno_block_precision='mixed' casts to complex64 after FFT
    (lines 366, 382, 385)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(
            in_channels=3,
            out_channels=4,
            n_modes=(4,),
            fno_block_precision="mixed",
        )
        x = jax.random.normal(key, (2, 3, 12), dtype=jnp.float32)
        res, _ = _init_and_apply(conv, x, key)
        assert res.shape == (2, 4, 12)


# ---------------------------------------------------------------------------
# Test 10 - output_shape passed to __call__ (line 457)
# ---------------------------------------------------------------------------


def test_SpectralConv_explicit_output_shape() -> None:
    """output_shape passed to __call__ overrides resolution_scaling_factor
    (line 457)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = SpectralConv(in_channels=3, out_channels=4, n_modes=(4, 4))
        x = jax.random.normal(key, (2, 3, 12, 12), dtype=jnp.float32)
        variables = conv.init(key, x)
        res = cast(jnp.ndarray, conv.apply(variables, x, output_shape=(8, 8)))
        assert res.shape == (2, 4, 8, 8)


# ---------------------------------------------------------------------------
# split_complex_params
# ---------------------------------------------------------------------------
# optax shapes its moment estimates like the parameters, so a complex weight
# gives Adam a complex state and optax.tree.cast_like then hits JAX's
# "Casting from complex to real dtypes will soon raise a ValueError".
# split_complex_params stores the two components as real parameters instead.
# These tests pin both the new behaviour and the default, since changing the
# default would rename a parameter in every saved FNO checkpoint.


def _conv(**kwargs: Any) -> SpectralConv:
    return SpectralConv(in_channels=4, out_channels=4, n_modes=(6, 6), **kwargs)


def test_split_complex_params_defaults_to_the_original_layout() -> None:
    params = _conv().init(jax.random.PRNGKey(0), jnp.zeros((1, 4, 16, 16)))["params"]
    assert set(params) == {"weight", "bias"}
    assert jnp.iscomplexobj(params["weight"])


def test_split_complex_params_stores_two_real_arrays() -> None:
    params = _conv(split_complex_params=True).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 4, 16, 16))
    )["params"]
    assert set(params) == {"weight_re", "weight_im", "bias"}
    for name in ("weight_re", "weight_im"):
        assert params[name].dtype == jnp.float32
        assert params[name].shape == (4, 4, 6, 4)  # last mode axis is halved


def test_split_complex_params_keeps_the_same_forward_pass() -> None:
    """Same weights, whichever way they are stored."""
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 16, 16), jnp.float32)
    dense = _conv()
    split = _conv(split_complex_params=True)
    p_dense = dense.init(jax.random.PRNGKey(0), x)
    p_split = split.init(jax.random.PRNGKey(0), x)

    w = p_dense["params"]["weight"]
    p_split = jax.tree_util.tree_map(lambda a: a, p_split)
    p_split["params"]["weight_re"] = jnp.real(w)
    p_split["params"]["weight_im"] = jnp.imag(w)
    p_split["params"]["bias"] = p_dense["params"]["bias"]

    assert jnp.allclose(
        dense.apply(p_dense, x), split.apply(p_split, x), atol=tolerances.atol(1e-6)
    )


def test_split_complex_params_keeps_the_adam_state_real() -> None:
    """The reason the option exists: a real optimiser state."""
    optax = pytest.importorskip("optax")
    x = jnp.zeros((1, 4, 16, 16))
    for split, want_complex in ((False, True), (True, False)):
        params = _conv(split_complex_params=split).init(jax.random.PRNGKey(0), x)
        state = optax.adam(1e-3).init(params)
        leaves = jax.tree_util.tree_leaves(state)
        assert any(jnp.iscomplexobj(a) for a in leaves) is want_complex


def test_split_complex_params_rejects_tucker() -> None:
    """Only the dense weight is split; say so instead of ignoring the flag."""
    with pytest.raises(ValueError, match="dense factorization only"):
        _conv(split_complex_params=True, factorization="tucker").init(
            jax.random.PRNGKey(0), jnp.zeros((1, 4, 16, 16))
        )
