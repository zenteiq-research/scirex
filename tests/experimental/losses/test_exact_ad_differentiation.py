# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for ExactPointwiseADDiff (JAX), the exact-but-O(n) fallback for
non-block-diagonal ``f(coords) -> u``.

* ``test_exactpointwiseaddiff_matches_analytic_*`` -- correctness against a
  known analytic function, block-diagonal case (mirrors ADDiff's own tests).
* ``test_exactpointwiseaddiff_matches_jacrev_diagonal_on_mixing_field`` --
  the key test: on a genuinely mixing (non-block-diagonal) field, this
  primitive is checked against an independent ground truth (the diagonal of
  a full ``jax.jacrev`` Jacobian, computed with no stop_gradient/vmap trick
  at all), not merely self-consistency.
* ``test_addiff_gives_contaminated_result_on_mixing_field`` -- demonstrates,
  on the same mixing field, that :class:`ADDiff`'s sum trick (which assumes
  block-diagonality) gives a *wrong*, contaminated answer -- the failure
  mode :class:`ExactPointwiseADDiff` exists to avoid.
* ``test_exactpointwiseaddiff_gradient_matches_per_dim_derivative`` /
  ``test_exactpointwiseaddiff_rejects_bad_order`` /
  ``test_exactpointwiseaddiff_rejects_batched_output`` /
  ``test_exactpointwiseaddiff_batching_via_vmap_matches_per_sample`` --
  mirror ADDiff's own coverage of the same behaviors.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.experimental.losses.ad_differentiation import ADDiff
from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from tests import tolerances


def _f(coords: jnp.ndarray) -> jnp.ndarray:
    """u(x, y) = sin(x) * cos(y), a plain pointwise (block-diagonal) field."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y)


def _f_mixing(coords: jnp.ndarray) -> jnp.ndarray:
    """u_i = sin(x_i)*cos(y_i) + mean_j(x_j) -- every point's output also
    depends on every other point's x-coordinate through the shared mean, so
    this is deliberately NOT block-diagonal (unlike ``_f`` above)."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y) + jnp.mean(x)


def _random_coords(n: int, seed: int = 0) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)


def test_exactpointwiseaddiff_matches_analytic_first_derivative() -> None:
    coords = _random_coords(17)
    ad = ExactPointwiseADDiff()

    du_dx = ad.derivative(_f, coords, dim=0, order=1)
    du_dy = ad.derivative(_f, coords, dim=1, order=1)

    x, y = coords[:, 0], coords[:, 1]
    assert jnp.allclose(du_dx, jnp.cos(x) * jnp.cos(y), atol=tolerances.atol(1e-5))
    assert jnp.allclose(du_dy, -jnp.sin(x) * jnp.sin(y), atol=tolerances.atol(1e-5))


def test_exactpointwiseaddiff_matches_analytic_second_derivative() -> None:
    coords = _random_coords(13, seed=1)
    ad = ExactPointwiseADDiff()

    d2u_dx2 = ad.derivative(_f, coords, dim=0, order=2)
    x, y = coords[:, 0], coords[:, 1]
    expected = -jnp.sin(x) * jnp.cos(y)
    assert jnp.allclose(d2u_dx2, expected, atol=tolerances.atol(1e-5))


def test_exactpointwiseaddiff_matches_addiff_on_block_diagonal_field() -> None:
    """On a genuinely block-diagonal field, the exact O(n) fallback and the
    O(1) sum trick must agree exactly -- the precondition ADDiff needs
    actually holds here."""
    coords = _random_coords(21, seed=2)
    exact = ExactPointwiseADDiff().derivative(_f, coords, dim=0, order=1)
    sum_trick = ADDiff().derivative(_f, coords, dim=0, order=1)
    assert jnp.allclose(exact, sum_trick, atol=tolerances.atol(1e-6))


def test_exactpointwiseaddiff_matches_jacrev_diagonal_on_mixing_field() -> None:
    """Ground truth computed an entirely different way: the full Jacobian
    d(u_i)/d(coords[j, k]) via jax.jacrev (no stop_gradient, no vmap(grad)
    per-point trick), with the i==j diagonal pulled out by hand. Kept small
    (n=6) since jacrev's cost is O(n) times the mixing field's own cost."""
    coords = _random_coords(6, seed=3)

    full_jac = jax.jacrev(_f_mixing)(coords)  # shape (n, n, coord_dim)
    diag = jnp.diagonal(full_jac, axis1=0, axis2=1)  # shape (coord_dim, n)
    expected_dx = diag[0]
    expected_dy = diag[1]

    exact = ExactPointwiseADDiff()
    got_dx = exact.derivative(_f_mixing, coords, dim=0, order=1)
    got_dy = exact.derivative(_f_mixing, coords, dim=1, order=1)

    assert jnp.allclose(got_dx, expected_dx, atol=tolerances.atol(1e-5))
    assert jnp.allclose(got_dy, expected_dy, atol=tolerances.atol(1e-5))


def test_addiff_gives_contaminated_result_on_mixing_field() -> None:
    """The failure mode ExactPointwiseADDiff exists to avoid: ADDiff's sum
    trick silently gives a WRONG answer on a mixing field, contaminated by
    every other point's contribution through the shared mean term."""
    coords = _random_coords(9, seed=4)
    n = coords.shape[0]

    exact = ExactPointwiseADDiff().derivative(_f_mixing, coords, dim=0, order=1)
    contaminated = ADDiff().derivative(_f_mixing, coords, dim=0, order=1)

    x, y = coords[:, 0], coords[:, 1]
    # true d(u_i)/d(x_i) = cos(x_i)*cos(y_i) + 1/n (only i's own share of
    # the mean's derivative)
    expected_exact = jnp.cos(x) * jnp.cos(y) + 1.0 / n
    assert jnp.allclose(exact, expected_exact, atol=tolerances.atol(1e-5))

    # sum trick instead picks up every other point's 1/n share too:
    # cos(x_i)*cos(y_i) + 1/n + (n-1)/n = cos(x_i)*cos(y_i) + 1
    expected_contaminated = jnp.cos(x) * jnp.cos(y) + 1.0
    assert jnp.allclose(contaminated, expected_contaminated, atol=tolerances.atol(1e-5))

    assert not jnp.allclose(exact, contaminated, atol=1e-3)


def test_exactpointwiseaddiff_gradient_matches_per_dim_derivative() -> None:
    coords = _random_coords(11, seed=5)
    ad = ExactPointwiseADDiff()

    du_dx = ad.derivative(_f, coords, dim=0, order=1)
    du_dy = ad.derivative(_f, coords, dim=1, order=1)
    grads = ad.gradient(_f, coords, dims=[0, 1])

    assert jnp.allclose(grads[0], du_dx)
    assert jnp.allclose(grads[1], du_dy)


def test_exactpointwiseaddiff_rejects_bad_order() -> None:
    coords = _random_coords(5, seed=6)
    with pytest.raises(ValueError):
        ExactPointwiseADDiff().derivative(_f, coords, dim=0, order=3)


def test_exactpointwiseaddiff_rejects_batched_output() -> None:
    coords = _random_coords(5, seed=7)

    def f_batched(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([_f(c), _f(c) * 2.0])  # fake (batch=2, n_points)

    with pytest.raises(ValueError):
        ExactPointwiseADDiff().derivative(f_batched, coords, dim=0, order=1)


def test_exactpointwiseaddiff_batching_via_vmap_matches_per_sample() -> None:
    coords = _random_coords(7, seed=8)
    scales = jnp.array([1.0, 3.0, -2.0])
    ad = ExactPointwiseADDiff()

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
    """Like ADDiff, ExactPointwiseADDiff has no state tied to a specific
    coordinate array, so it is always safe to reuse for a nested derivative
    call -- unlike PatchQuadratureADDiff, see that class's own test."""
    ad = ExactPointwiseADDiff()
    assert ad.nested_differentiator is ad


def test_owns_coords_is_false() -> None:
    """Stateless, like ADDiff -- every call needs coords supplied."""
    assert ExactPointwiseADDiff().owns_coords is False
