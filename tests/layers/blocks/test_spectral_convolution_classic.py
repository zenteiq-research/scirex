# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for ClassicSpectralConv2d."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scirex.layers.blocks.spectral_convolution_classic import ClassicSpectralConv2d
from tests import tolerances


@pytest.mark.parametrize("in_channels,out_channels", [(1, 1), (3, 4)])
@pytest.mark.parametrize("n_modes1,n_modes2", [(4, 4), (6, 3)])
def test_ClassicSpectralConv2d_shape(
    in_channels: int, out_channels: int, n_modes1: int, n_modes2: int
) -> None:
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = ClassicSpectralConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes1=n_modes1,
            n_modes2=n_modes2,
        )
        x = jax.random.normal(key, (2, in_channels, 16, 16), dtype=jnp.float32)
        variables = conv.init(key, x)
        out = conv.apply(variables, x)
        assert out.shape == (2, out_channels, 16, 16)
        assert jnp.iscomplexobj(variables["params"]["weights1"])
        assert jnp.iscomplexobj(variables["params"]["weights2"])
        assert variables["params"]["weights1"].shape == (
            in_channels,
            out_channels,
            n_modes1,
            n_modes2,
        )


def test_ClassicSpectralConv2d_output_is_real() -> None:
    """Real-valued input must produce real-valued output (irfft2 round-trip)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = ClassicSpectralConv2d(
            in_channels=2, out_channels=2, n_modes1=4, n_modes2=4
        )
        x = jax.random.normal(key, (1, 2, 16, 16), dtype=jnp.float32)
        variables = conv.init(key, x)
        out = conv.apply(variables, x)
        assert jnp.isrealobj(out)


def _finite_diff_complex_grad(
    loss_fn: Any, variables: dict, name: str, eps: float = 1e-2
) -> jnp.ndarray:
    """Numerically differentiate the real-valued ``loss_fn(variables)``
    w.r.t. the complex leaf ``variables["params"][name]``, perturbing its
    real and imaginary parts as independent real scalars -- the
    convention-agnostic ground truth for "the correct steepest-descent
    direction for a complex parameter of a real-valued loss." """
    w = variables["params"][name]

    def _with_leaf(new_leaf: jnp.ndarray) -> dict:
        return {**variables, "params": {**variables["params"], name: new_leaf}}

    grad = jnp.zeros_like(w)
    for idx in np.ndindex(w.shape):
        loss_re_plus = float(loss_fn(_with_leaf(w.at[idx].add(eps))))
        loss_re_minus = float(loss_fn(_with_leaf(w.at[idx].add(-eps))))
        loss_im_plus = float(loss_fn(_with_leaf(w.at[idx].add(1j * eps))))
        loss_im_minus = float(loss_fn(_with_leaf(w.at[idx].add(-1j * eps))))
        d_re = (loss_re_plus - loss_re_minus) / (2 * eps)
        d_im = (loss_im_plus - loss_im_minus) / (2 * eps)
        grad = grad.at[idx].set(d_re + 1j * d_im)
    return grad


def test_ClassicSpectralConv2d_conjugated_gradient_matches_finite_difference() -> None:
    """Documents the correct usage contract for this layer's complex
    weights: ``jax.grad`` of a real-valued loss returns a value that must
    be *conjugated* before it is a valid steepest-descent direction (a
    general JAX-vs-naive-real-gradient convention fact, true for any
    complex JAX parameter, not specific to this layer -- confirmed
    directly against a finite-difference ground truth that perturbs the
    weight's real and imaginary parts as independent real scalars).
    ``scirex.training.optim.adamw.adamw_cx`` already performs this
    conjugation for every complex parameter it optimizes; this test
    exists so that contract stays correct for this layer's weights too,
    without baking a conjugation into the layer itself (which would
    double up with adamw_cx's own conjugation and reintroduce the wrong
    gradient direction -- see the same lesson recorded for
    ``SpectralConv``'s weight)."""
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(0)
        conv = ClassicSpectralConv2d(
            in_channels=1, out_channels=1, n_modes1=3, n_modes2=3
        )
        x = jax.random.normal(key, (1, 1, 8, 8), dtype=jnp.float32)
        variables = conv.init(key, x)

        def loss_fn(v: dict) -> jnp.ndarray:
            out = conv.apply(v, x)
            return jnp.sum(out**2)

        analytic_grads = jax.grad(loss_fn)(variables)

        for name in ("weights1", "weights2"):
            g_analytic = jnp.conj(analytic_grads["params"][name])
            g_fd = _finite_diff_complex_grad(loss_fn, variables, name)
            denom = float(jnp.linalg.norm(g_fd))
            rel_err = float(jnp.linalg.norm(g_analytic - g_fd)) / max(denom, 1e-8)
            assert rel_err < 0.05, (
                f"{name}: conj(jax.grad) does not match the finite-difference "
                f"gradient (rel_err={rel_err:.4f})"
            )


# ---------------------------------------------------------------------------
# Real-valued parameter storage
# ---------------------------------------------------------------------------


def test_split_complex_params_stores_real_parameters():
    """With split_complex_params the layer holds four real tensors instead of
    two complex ones, so a moment-based optimizer's state stays real."""
    layer = ClassicSpectralConv2d(
        in_channels=2,
        out_channels=3,
        n_modes1=4,
        n_modes2=4,
        split_complex_params=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16))
    params = layer.init(jax.random.PRNGKey(0), x)["params"]

    assert sorted(params) == [
        "weights1_im",
        "weights1_re",
        "weights2_im",
        "weights2_re",
    ]
    for leaf in jax.tree_util.tree_leaves(params):
        assert leaf.dtype == jnp.float32
    assert layer.apply({"params": params}, x).shape == (1, 3, 16, 16)


def test_split_complex_params_trains_without_a_complex_optimizer_state():
    """One Adam step: the moment estimates are real, which is the whole point
    of the split storage (a complex state trips JAX's complex-to-real cast)."""
    import optax

    layer = ClassicSpectralConv2d(
        in_channels=2,
        out_channels=2,
        n_modes1=4,
        n_modes2=4,
        split_complex_params=True,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16))
    params = layer.init(jax.random.PRNGKey(0), x)
    tx = optax.adam(1e-3)
    state = tx.init(params)
    for leaf in jax.tree_util.tree_leaves(state):
        assert not jnp.issubdtype(leaf.dtype, jnp.complexfloating)

    grads = jax.grad(lambda p: jnp.sum(layer.apply(p, x) ** 2))(params)
    updates, _ = tx.update(grads, state, params)
    updated = optax.apply_updates(params, updates)
    assert jnp.all(jnp.isfinite(layer.apply(updated, x)))


def test_default_and_split_storage_agree_for_the_same_weights():
    """The two storage modes are one parameterisation of the same layer."""
    shape = (2, 3, 4, 4)
    real = jax.random.uniform(jax.random.PRNGKey(1), shape, dtype=jnp.float32)
    imag = jax.random.uniform(jax.random.PRNGKey(2), shape, dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 16, 16))
    kwargs = dict(in_channels=2, out_channels=3, n_modes1=4, n_modes2=4)

    complex_out = ClassicSpectralConv2d(**kwargs).apply(
        {
            "params": {
                "weights1": jax.lax.complex(real, imag),
                "weights2": jax.lax.complex(real, -imag),
            }
        },
        x,
    )
    split_out = ClassicSpectralConv2d(**kwargs, split_complex_params=True).apply(
        {
            "params": {
                "weights1_re": real,
                "weights1_im": imag,
                "weights2_re": real,
                "weights2_im": -imag,
            }
        },
        x,
    )
    assert jnp.allclose(complex_out, split_out, atol=tolerances.atol(1e-6))


def test_modes_larger_than_the_grid_raise():
    """Overlapping row bands would silently clobber each other."""
    layer = ClassicSpectralConv2d(in_channels=1, out_channels=1, n_modes1=9, n_modes2=4)
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 16, 16))
    with pytest.raises(ValueError, match="exceed what a 16x16 grid supports"):
        layer.init(jax.random.PRNGKey(0), x)
