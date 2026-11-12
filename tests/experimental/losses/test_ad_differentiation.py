# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for ADDiff (JAX), the autodiff/sum-trick differentiation
primitive for point-cloud/query-coordinate fields.

* ``test_addiff_first_derivative_matches_analytic`` /
  ``test_addiff_second_derivative_matches_analytic`` -- correctness against
  a known analytic function, at *fixed* (not increasing) resolution: unlike
  FiniteDiff (convergence-only) or FourierDiff (exact only in the limit of a
  well-resolved periodic grid), AD differentiation is exact up to
  floating-point precision regardless of how many/few points are sampled,
  since there is no discretization at all.
* ``test_addiff_matches_vmap_grad`` -- cross-checks the sum-trick result
  against the more obviously-correct (but O(N) rather than O(1) backward
  passes) ``jax.vmap(jax.grad(f_single))`` reference, confirming the
  block-diagonal cancellation argument actually holds for a plain pointwise
  function.
* ``test_addiff_gradient_matches_per_dim_derivative`` -- the multi-dim
  convenience wrapper agrees with calling ``derivative`` once per dim.
* ``test_addiff_rejects_bad_order`` -- validation.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.experimental.losses.ad_differentiation import ADDiff
from tests import tolerances


def _f(coords: jnp.ndarray) -> jnp.ndarray:
    """u(x, y) = sin(x) * cos(y), a plain pointwise (block-diagonal) field."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y)


def _random_coords(n: int, seed: int = 0) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)


def test_addiff_first_derivative_matches_analytic() -> None:
    coords = _random_coords(37)
    ad = ADDiff()

    du_dx = ad.derivative(_f, coords, dim=0, order=1)
    du_dy = ad.derivative(_f, coords, dim=1, order=1)

    x, y = coords[:, 0], coords[:, 1]
    assert jnp.allclose(du_dx, jnp.cos(x) * jnp.cos(y), atol=tolerances.atol(1e-5))
    assert jnp.allclose(du_dy, -jnp.sin(x) * jnp.sin(y), atol=tolerances.atol(1e-5))


def test_addiff_second_derivative_matches_analytic() -> None:
    coords = _random_coords(37, seed=1)
    ad = ADDiff()

    d2u_dx2 = ad.derivative(_f, coords, dim=0, order=2)
    d2u_dy2 = ad.derivative(_f, coords, dim=1, order=2)

    x, y = coords[:, 0], coords[:, 1]
    expected = -jnp.sin(x) * jnp.cos(y)
    assert jnp.allclose(d2u_dx2, expected, atol=tolerances.atol(1e-5))
    assert jnp.allclose(d2u_dy2, expected, atol=tolerances.atol(1e-5))


def test_addiff_matches_vmap_grad() -> None:
    """The O(1)-backward-pass sum trick agrees with the O(N) vmap(grad)
    per-point reference -- confirming the block-diagonal cancellation."""
    coords = _random_coords(23, seed=2)
    ad = ADDiff()
    sum_trick = ad.derivative(_f, coords, dim=0, order=1)

    def f_single(c: jnp.ndarray) -> jnp.ndarray:
        return _f(c[None, :])[0]

    per_point = jax.vmap(jax.grad(f_single))(coords)[:, 0]
    assert jnp.allclose(sum_trick, per_point, atol=tolerances.atol(1e-6))


def test_addiff_gradient_matches_per_dim_derivative() -> None:
    coords = _random_coords(19, seed=3)
    ad = ADDiff()

    du_dx = ad.derivative(_f, coords, dim=0, order=1)
    du_dy = ad.derivative(_f, coords, dim=1, order=1)
    grads = ad.gradient(_f, coords, dims=[0, 1])

    assert jnp.allclose(grads[0], du_dx)
    assert jnp.allclose(grads[1], du_dy)


def test_addiff_rejects_bad_order() -> None:
    coords = _random_coords(5, seed=4)
    with pytest.raises(ValueError):
        ADDiff().derivative(_f, coords, dim=0, order=3)


def test_addiff_rejects_batched_output() -> None:
    """f(coords) must be unbatched (n_points,) -- a (batch, n_points) output
    would silently mix gradient contributions across samples if allowed
    through (see the module docstring); this must raise instead."""
    coords = _random_coords(5, seed=5)

    def f_batched(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([_f(c), _f(c) * 2.0])  # fake (batch=2, n_points)

    with pytest.raises(ValueError):
        ADDiff().derivative(f_batched, coords, dim=0, order=1)


def test_addiff_batching_via_vmap_matches_per_sample() -> None:
    """The correct way to handle a batch: vmap the whole per-sample closure
    (including the ADDiff call) over the batch axis, rather than batching f
    itself. Confirms this gives each sample its own, uncontaminated
    derivative -- distinct per-sample scale factors must come out distinct.
    """
    coords = _random_coords(11, seed=6)
    scales = jnp.array([1.0, 3.0, -2.0])  # 3 "samples" sharing one coords set
    ad = ADDiff()

    def per_sample_dudx(scale: jnp.ndarray) -> jnp.ndarray:
        def f_sample(c: jnp.ndarray) -> jnp.ndarray:
            return scale * _f(c)

        return ad.derivative(f_sample, coords, dim=0, order=1)

    batched = jax.vmap(per_sample_dudx)(scales)  # (batch, n_points)

    x, y = coords[:, 0], coords[:, 1]
    expected_unit = jnp.cos(x) * jnp.cos(y)
    for b in range(scales.shape[0]):
        assert jnp.allclose(
            batched[b], scales[b] * expected_unit, atol=tolerances.atol(1e-5)
        )


def test_nested_differentiator_is_self() -> None:
    """ADDiff has no state tied to a specific coordinate array, so it is
    always safe to reuse for a nested derivative call -- unlike
    PatchQuadratureADDiff, see that class's own test of this property."""
    ad = ADDiff()
    assert ad.nested_differentiator is ad


def test_owns_coords_is_false() -> None:
    """ADDiff is stateless -- every call needs coords supplied, unlike
    PatchQuadratureADDiff, see that class's own test of this flag."""
    assert ADDiff().owns_coords is False
