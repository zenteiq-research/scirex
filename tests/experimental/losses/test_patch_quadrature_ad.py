# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for PatchQuadratureADDiff (JAX), the bounded-cost exact
differentiator for globally-mixing ``f(coords) -> u`` (e.g. Transolver).

* ``test_construction_*`` -- ``self.coords``/``self.weights`` match a manual
  assembly via ``make_patches`` + ``gauss_legendre_points_weights`` +
  ``patch_quadrature_coords`` (the same primitives, called directly).
* ``test_derivative_matches_analytic_*`` -- correctness against a known
  analytic function, same convergence pattern as the other differentiators'
  own tests.
* ``test_single_patch_matches_exactpointwiseaddiff_on_mixing_field`` -- the
  key anchor: with exactly one patch (covering the whole domain), no actual
  patching happens, so the result must agree with
  ``ExactPointwiseADDiff`` run directly on the same coords, bit for bit --
  isolating "is the per-patch loop wired correctly" from "is the patch
  approximation itself accurate" (a separate, harder question this test
  deliberately does not try to answer).
* ``test_multi_patch_only_sees_within_patch_mixing`` -- documents the
  intentional cost/accuracy tradeoff: with more than one patch, a field that
  mixes *across* patch boundaries is only ever seen mixing *within* each
  patch by this primitive, so its result differs from the fully exact,
  whole-domain ``ExactPointwiseADDiff`` -- this is expected, not a bug.
* ``test_gradient_matches_per_dim_derivative`` -- mirrors the other
  differentiators' own coverage of the same behavior.
* ``test_derivative_with_omitted_coords_matches_explicit_self_coords`` /
  ``test_gradient_with_omitted_coords_matches_explicit_self_coords`` -- the
  recommended usage (omit ``coords``) gives exactly the same result as
  passing ``self.coords`` explicitly, which now also emits a warning
  pointing back at the recommended usage.
* ``test_omitted_coords_emits_no_warning`` -- the negative control: the
  recommended usage itself must stay silent.
* ``test_omitted_coords_cannot_be_crossed_between_same_shaped_instances`` --
  the scenario the optional-``coords`` form exists for: two instances with
  the same total point count (same ``n``, different patch layout) are
  indistinguishable by shape alone, but omitting ``coords`` removes the
  possibility of crossing them entirely, since each instance always uses
  its own.
* ``test_explicit_coords_from_same_shaped_instance_is_not_caught`` --
  documents, rather than overstates, what passing ``coords`` explicitly
  still cannot catch: a same-shaped different instance's coords slip
  through uncaught. Not a regression -- this is the residual gap the
  previous test's recommended usage sidesteps.
* ``test_derivative_rejects_wrong_shape_coords`` /
  ``test_gradient_rejects_wrong_shape_coords`` /
  ``test_derivative_rejects_wrong_total_point_count`` -- when ``coords`` is
  passed explicitly with a genuinely different total point count, it's
  still rejected by shape, so an unrelated array is not silently sliced
  with the wrong patch boundaries.
* ``test_nested_differentiator_is_the_internal_exact_worker`` -- a PDE term
  needing a second, nested derivative (differentiating a coefficient
  function, see
  :class:`~scirex.experimental.losses.ad_pde_terms.DiffusionTermAD`'s flux-form
  path) must go through ``nested_differentiator`` rather than calling back
  into this instance -- confirms that reference is the internal
  ``ExactPointwiseADDiff`` worker, usable directly on a single patch's
  coordinates with no further splitting.
* ``test_owns_coords_is_true`` -- this class is stateful, unlike
  ``ADDiff``/``ExactPointwiseADDiff`` (see their own tests of this flag) --
  see :mod:`~scirex.experimental.losses.ad_pde_terms` for how PDE terms use it.
* ``test_construction_with_explicit_patches_*`` -- the ``patches``
  constructor argument (see
  :func:`~scirex.experimental.losses.domain_membership.cull_excluded_patches`): a
  given patch list is used as-is instead of the full uniform grid, and a
  patch with the wrong ``dim`` is rejected.
* ``test_derivative_with_reduced_patch_list_matches_manual_per_patch_exact``
  -- the reduced-patch-list path still produces exactly the same per-point
  derivatives as calling ``ExactPointwiseADDiff`` on each surviving patch
  by hand, confirming the shorter :attr:`_patch_slices` loop is wired
  correctly, not just that construction accepts the argument.
* ``test_patch_batch_size_*`` / ``test_derivative_with_patch_batch_size_*`` /
  ``test_gradient_with_patch_batch_size_*`` -- the ``patch_batch_size``
  constructor argument (bounded-memory ``jax.lax.map`` path, see the module
  docstring's "Bounding memory: patch_batch_size" section): defaults to
  ``None`` (today's unchunked ``jax.vmap`` behavior), rejects a non-positive
  value, and produces results identical to the unchunked path whether or not
  ``n_patches`` divides evenly by the chunk size (``jax.lax.map`` handles the
  remainder as a separate chunk internally -- both cases are exercised).
"""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.experimental.losses.quadrature import (
    Patch,
    gauss_legendre_points_weights,
    make_patches,
    patch_quadrature_coords,
)
from tests import tolerances

_UNIT_SQUARE = [(0.0, 1.0), (0.0, 1.0)]


def _f(coords: jnp.ndarray) -> jnp.ndarray:
    """u(x, y) = sin(x) * cos(y), a plain pointwise (non-mixing) field."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y)


def _f_mixing(coords: jnp.ndarray) -> jnp.ndarray:
    """u_i = sin(x_i)*cos(y_i) + mean_j(x_j) over whatever coords are
    actually passed in -- deliberately mixes every point it's called with,
    same construction as the exact-AD test suite's mixing field."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y) + jnp.mean(x)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_coords_and_weights_match_manual_assembly() -> None:
    bounds = [(0.0, 1.0), (2.0, 3.0)]
    n_patches_per_axis = [2, 2]
    n = 3

    pq = PatchQuadratureADDiff(bounds, n_patches_per_axis, n)

    patches = make_patches(bounds, n_patches_per_axis)
    ref_points, ref_weights = gauss_legendre_points_weights(n, dim=2)
    per_patch = [patch_quadrature_coords(p, ref_points, ref_weights) for p in patches]
    expected_coords = jnp.concatenate([c for c, _ in per_patch], axis=0)
    expected_weights = jnp.concatenate([w for _, w in per_patch], axis=0)

    assert jnp.allclose(pq.coords, expected_coords)
    assert jnp.allclose(pq.weights, expected_weights)


def test_construction_weight_sum_equals_domain_volume() -> None:
    bounds = [(0.0, 2.0), (-1.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds, n_patches_per_axis=[4, 3], n=3)

    domain_volume = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
    assert jnp.sum(pq.weights) == pytest.approx(domain_volume, abs=1e-5)


def test_construction_total_points_matches_patches_times_quad_points() -> None:
    n_patches_per_axis = [2, 3]
    n = 4
    pq = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=n_patches_per_axis, n=n
    )
    n_patches = n_patches_per_axis[0] * n_patches_per_axis[1]
    assert pq.coords.shape == (n_patches * n**2, 2)
    assert pq.weights.shape == (n_patches * n**2,)


def test_construction_with_explicit_patches_uses_them_instead_of_make_patches() -> None:
    """A reduced patch list (as cull_excluded_patches would produce) is used
    as-is -- the resulting coords/weights match a manual assembly of just
    those patches, not the full uniform grid bounds/n_patches_per_axis would
    otherwise give."""
    n = 3
    all_patches = make_patches(_UNIT_SQUARE, n_patches_per_axis=[2, 2])
    reduced = all_patches[:2]

    pq = PatchQuadratureADDiff(
        _UNIT_SQUARE, n_patches_per_axis=[2, 2], n=n, patches=reduced
    )

    ref_points, ref_weights = gauss_legendre_points_weights(n, dim=2)
    per_patch = [patch_quadrature_coords(p, ref_points, ref_weights) for p in reduced]
    expected_coords = jnp.concatenate([c for c, _ in per_patch], axis=0)
    expected_weights = jnp.concatenate([w for _, w in per_patch], axis=0)

    assert pq.patches == reduced
    assert jnp.allclose(pq.coords, expected_coords)
    assert jnp.allclose(pq.weights, expected_weights)


def test_construction_with_explicit_patches_rejects_dim_mismatch() -> None:
    wrong_dim_patch = Patch(center=(0.5, 0.5, 0.5), half_width=(0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="patches must have dim=2"):
        PatchQuadratureADDiff(
            _UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3, patches=[wrong_dim_patch]
        )


def test_derivative_with_reduced_patch_list_matches_manual_per_patch_exact() -> None:
    """The shorter _patch_slices loop that a reduced patch list produces is
    wired correctly: its derivative matches calling ExactPointwiseADDiff on
    each surviving patch's own coordinates by hand."""
    n = 3
    all_patches = make_patches(_UNIT_SQUARE, n_patches_per_axis=[2, 2])
    reduced = [all_patches[0], all_patches[3]]

    pq = PatchQuadratureADDiff(
        _UNIT_SQUARE, n_patches_per_axis=[2, 2], n=n, patches=reduced
    )
    got = pq.derivative(_f_mixing, dim=0, order=1)

    ref_points, ref_weights = gauss_legendre_points_weights(n, dim=2)
    exact = ExactPointwiseADDiff()
    expected_list = []
    for patch in reduced:
        coords, _ = patch_quadrature_coords(patch, ref_points, ref_weights)
        expected_list.append(exact.derivative(_f_mixing, coords, dim=0, order=1))
    expected = jnp.concatenate(expected_list, axis=0)

    assert jnp.allclose(got, expected)


# ---------------------------------------------------------------------------
# derivative / gradient correctness
# ---------------------------------------------------------------------------


def test_derivative_matches_analytic_first_derivative() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    du_dx = pq.derivative(_f, dim=0, order=1)
    du_dy = pq.derivative(_f, dim=1, order=1)

    x, y = pq.coords[:, 0], pq.coords[:, 1]
    assert jnp.allclose(du_dx, jnp.cos(x) * jnp.cos(y), atol=tolerances.atol(1e-5))
    assert jnp.allclose(du_dy, -jnp.sin(x) * jnp.sin(y), atol=tolerances.atol(1e-5))


def test_derivative_matches_analytic_second_derivative() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    d2u_dx2 = pq.derivative(_f, dim=0, order=2)
    x, y = pq.coords[:, 0], pq.coords[:, 1]
    assert jnp.allclose(d2u_dx2, -jnp.sin(x) * jnp.cos(y), atol=tolerances.atol(1e-5))


def test_single_patch_matches_exactpointwiseaddiff_on_mixing_field() -> None:
    """One patch == the whole domain, so no actual patching happens -- the
    per-patch loop must reduce to exactly one ExactPointwiseADDiff call."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 1], n=4)

    got = pq.derivative(_f_mixing, dim=0, order=1)
    expected = ExactPointwiseADDiff().derivative(_f_mixing, pq.coords, dim=0, order=1)

    assert jnp.allclose(got, expected, atol=tolerances.atol(1e-6))


def test_multi_patch_only_sees_within_patch_mixing() -> None:
    """With more than one patch, a field mixing across the whole domain is
    only ever seen mixing within each patch -- this is the intentional
    cost/accuracy tradeoff, not a bug, so the two results must differ."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    patched = pq.derivative(_f_mixing, dim=0, order=1)
    full_exact = ExactPointwiseADDiff().derivative(_f_mixing, pq.coords, dim=0, order=1)

    assert not jnp.allclose(patched, full_exact, atol=1e-3)


def test_gradient_matches_per_dim_derivative() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    du_dx = pq.derivative(_f, dim=0, order=1)
    du_dy = pq.derivative(_f, dim=1, order=1)
    grads = pq.gradient(_f, dims=[0, 1])

    assert jnp.allclose(grads[0], du_dx)
    assert jnp.allclose(grads[1], du_dy)


# ---------------------------------------------------------------------------
# default (omitted) coords
# ---------------------------------------------------------------------------


def test_derivative_with_omitted_coords_matches_explicit_self_coords() -> None:
    """The recommended usage (omit coords) must give the exact same result
    as passing self.coords explicitly -- it is the same array either way,
    just without a separate argument a caller could get wrong."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    omitted = pq.derivative(_f, dim=0, order=1)
    with pytest.warns(UserWarning, match="Omit it instead"):
        explicit = pq.derivative(_f, pq.coords, dim=0, order=1)

    assert jnp.allclose(omitted, explicit)


def test_gradient_with_omitted_coords_matches_explicit_self_coords() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    omitted = pq.gradient(_f, dims=[0, 1])
    with pytest.warns(UserWarning, match="Omit it instead"):
        explicit = pq.gradient(_f, pq.coords, dims=[0, 1])

    assert jnp.allclose(omitted[0], explicit[0])
    assert jnp.allclose(omitted[1], explicit[1])


def test_omitted_coords_emits_no_warning() -> None:
    """The negative control for the previous two tests: the recommended
    (omitted) usage must not trigger the explicit-coords warning at all."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pq.derivative(_f, dim=0, order=1)
        pq.gradient(_f, dims=[0, 1])


def test_omitted_coords_cannot_be_crossed_between_same_shaped_instances() -> None:
    """The scenario the optional-coords form exists for: two instances with
    the same total point count (same n, different patch layout -- e.g. one
    instance per sample in a dataset like ShapeNetCar, all sharing the same
    quadrature resolution but a different domain each) are indistinguishable
    by shape alone. Passing coords explicitly cannot catch a caller crossing
    them (see test_explicit_coords_from_same_shaped_instance_is_not_caught
    below -- this is the residual, documented gap). Omitting coords removes
    the possibility entirely: each instance always uses its own coords, so
    there is no array left to cross."""
    pq_a = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 4], n=3)
    pq_b = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq_a.coords.shape == pq_b.coords.shape  # the ambiguous case

    result_a = pq_a.derivative(_f, dim=0, order=1)
    result_b = pq_b.derivative(_f, dim=0, order=1)

    x_a, y_a = pq_a.coords[:, 0], pq_a.coords[:, 1]
    x_b, y_b = pq_b.coords[:, 0], pq_b.coords[:, 1]
    assert jnp.allclose(
        result_a, jnp.cos(x_a) * jnp.cos(y_a), atol=tolerances.atol(1e-5)
    )
    assert jnp.allclose(
        result_b, jnp.cos(x_b) * jnp.cos(y_b), atol=tolerances.atol(1e-5)
    )


def test_explicit_coords_from_same_shaped_instance_is_not_caught() -> None:
    """Documents the residual gap the optional-coords form (previous test)
    was added to sidestep, rather than overstating what the shape check
    itself can do: a same-shaped different instance's coords, passed
    explicitly, is NOT rejected -- shape alone cannot tell them apart. This
    is expected, not a regression; see the module docstring's "Why the
    optional form exists" section."""
    pq_a = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 4], n=3)
    pq_b = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq_a.coords.shape == pq_b.coords.shape

    # No exception here -- this is the documented, still-open limitation of
    # passing coords explicitly, not something this test claims is fixed.
    with pytest.warns(UserWarning, match="Omit it instead"):
        wrong = pq_a.derivative(_f, pq_b.coords, dim=0, order=1)
    assert wrong.shape == pq_a.coords.shape[:1]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_derivative_rejects_wrong_shape_coords() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    wrong_coords = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="own self"):
        pq.derivative(_f, wrong_coords, dim=0, order=1)


def test_gradient_rejects_wrong_shape_coords() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    wrong_coords = jnp.zeros((5, 2))
    with pytest.raises(ValueError, match="own self"):
        pq.gradient(_f, wrong_coords, dims=[0, 1])


def test_derivative_rejects_wrong_total_point_count() -> None:
    """A caller mistake with a different total point count (a different
    n_patches_per_axis) must still be rejected -- the case that matters in
    practice, distinct from the genuinely-ambiguous same-shape case a
    shape-only check cannot resolve (see the module docstring)."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    other_pq = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[3, 3], n=3
    )
    with pytest.raises(ValueError, match="own self"):
        pq.derivative(_f, other_pq.coords, dim=0, order=1)


def test_nested_differentiator_is_the_internal_exact_worker() -> None:
    """``nested_differentiator`` is how a PDE term gets a second, nested
    derivative from inside a closure this instance is already
    differentiating, without calling back into this instance itself (which
    would fail -- see the module docstring and
    ``test_derivative_rejects_wrong_shape_coords``). It must be usable
    directly on a patch-sized (or smaller) array, with no per-patch
    splitting applied."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq.nested_differentiator is pq._exact

    one_patch_coords = pq.coords[: pq._patch_slices[0][1]]
    got = pq.nested_differentiator.derivative(_f, one_patch_coords, dim=0, order=1)
    expected = ExactPointwiseADDiff().derivative(_f, one_patch_coords, dim=0, order=1)
    assert jnp.allclose(got, expected)


def test_owns_coords_is_true() -> None:
    """Stateful -- always uses its own coords, unlike ADDiff/
    ExactPointwiseADDiff. See those classes' own tests of this flag, and
    scirex.experimental.losses.ad_pde_terms for how PDE terms use it."""
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq.owns_coords is True


# ---------------------------------------------------------------------------
# patch_batch_size (bounded-memory jax.lax.map path)
# ---------------------------------------------------------------------------


def test_patch_batch_size_none_by_default() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq.patch_batch_size is None


def test_patch_batch_size_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PatchQuadratureADDiff(
            bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3, patch_batch_size=0
        )


def test_derivative_with_patch_batch_size_matches_unchunked_evenly_divisible() -> None:
    """4 patches, batch_size=2 -- divides evenly, exercises only the
    lax.scan-over-vmapped-chunks path (no remainder chunk)."""
    pq_chunked = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3, patch_batch_size=2
    )
    pq_full = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    chunked = pq_chunked.derivative(_f_mixing, dim=0, order=1)
    full = pq_full.derivative(_f_mixing, dim=0, order=1)

    assert jnp.allclose(chunked, full, atol=tolerances.atol(1e-6))


def test_derivative_with_patch_batch_size_matches_unchunked_with_remainder() -> None:
    """3 patches, batch_size=2 -- does not divide evenly, exercises
    jax.lax.map's separate remainder-chunk path too."""
    pq_chunked = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 3], n=3, patch_batch_size=2
    )
    pq_full = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 3], n=3)

    chunked = pq_chunked.derivative(_f_mixing, dim=0, order=1)
    full = pq_full.derivative(_f_mixing, dim=0, order=1)

    assert jnp.allclose(chunked, full, atol=tolerances.atol(1e-6))


# ---------------------------------------------------------------------------
# checkpoint_patches (jax.checkpoint under an outer differentiation)
# ---------------------------------------------------------------------------


def test_checkpoint_patches_false_by_default() -> None:
    pq = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)
    assert pq.checkpoint_patches is False


def test_checkpoint_patches_does_not_change_forward_value() -> None:
    """jax.checkpoint is a no-op on the forward computation -- it only
    changes what an *outer* differentiation retains versus recomputes. The
    plain forward derivative value must therefore be identical whether or
    not checkpoint_patches is set."""
    pq_checkpointed = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE,
        n_patches_per_axis=[2, 2],
        n=3,
        checkpoint_patches=True,
    )
    pq_plain = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3
    )

    checkpointed = pq_checkpointed.derivative(_f_mixing, dim=0, order=1)
    plain = pq_plain.derivative(_f_mixing, dim=0, order=1)

    assert jnp.allclose(checkpointed, plain, atol=tolerances.atol(1e-6))


def test_checkpoint_patches_does_not_change_outer_gradient() -> None:
    """The scenario checkpoint_patches actually exists for: this whole
    derivative call sits inside a further, outer jax.grad (e.g. w.r.t. a
    model parameter that f itself depends on, mirroring real training). The
    resulting gradient must be identical with checkpoint_patches True or
    False -- checkpointing only changes when the forward pass gets
    recomputed during that outer backward sweep, never what is computed."""

    def f_theta(coords: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        x, y = coords[..., 0], coords[..., 1]
        return theta * jnp.sin(x) * jnp.cos(y) + jnp.mean(theta * x)

    def make_loss(pq: PatchQuadratureADDiff) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def loss(theta: jnp.ndarray) -> jnp.ndarray:
            du_dx = pq.derivative(lambda c: f_theta(c, theta), dim=0, order=1)
            return jnp.mean(du_dx**2)

        return loss

    pq_checkpointed = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE,
        n_patches_per_axis=[2, 2],
        n=3,
        checkpoint_patches=True,
    )
    pq_plain = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3
    )

    theta = jnp.array(1.3)
    grad_checkpointed = jax.grad(make_loss(pq_checkpointed))(theta)
    grad_plain = jax.grad(make_loss(pq_plain))(theta)

    assert jnp.allclose(grad_checkpointed, grad_plain, atol=tolerances.atol(1e-6))


def test_checkpoint_patches_combined_with_patch_batch_size_matches_plain() -> None:
    """checkpoint_patches and patch_batch_size are independent knobs -- both
    set together must still agree with neither set."""

    def f_theta(coords: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        x, y = coords[..., 0], coords[..., 1]
        return theta * jnp.sin(x) * jnp.cos(y) + jnp.mean(theta * x)

    def make_loss(pq: PatchQuadratureADDiff) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def loss(theta: jnp.ndarray) -> jnp.ndarray:
            du_dx = pq.derivative(lambda c: f_theta(c, theta), dim=0, order=1)
            return jnp.mean(du_dx**2)

        return loss

    pq_both = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE,
        n_patches_per_axis=[1, 3],
        n=3,
        patch_batch_size=2,
        checkpoint_patches=True,
    )
    pq_plain = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 3], n=3
    )

    theta = jnp.array(0.7)
    grad_both = jax.grad(make_loss(pq_both))(theta)
    grad_plain = jax.grad(make_loss(pq_plain))(theta)

    assert jnp.allclose(grad_both, grad_plain, atol=tolerances.atol(1e-6))


def test_gradient_with_patch_batch_size_matches_unchunked() -> None:
    pq_chunked = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 3], n=3, patch_batch_size=2
    )
    pq_full = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[1, 3], n=3)

    chunked = pq_chunked.gradient(_f_mixing, dims=[0, 1])
    full = pq_full.gradient(_f_mixing, dims=[0, 1])

    assert jnp.allclose(chunked[0], full[0], atol=tolerances.atol(1e-6))
    assert jnp.allclose(chunked[1], full[1], atol=tolerances.atol(1e-6))


def test_patch_batch_size_of_one_matches_unchunked() -> None:
    """The most memory-bounded extreme (one patch resident at a time,
    closest in spirit to the original Python loop this module replaced)
    must still agree with the unchunked result."""
    pq_chunked = PatchQuadratureADDiff(
        bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3, patch_batch_size=1
    )
    pq_full = PatchQuadratureADDiff(bounds=_UNIT_SQUARE, n_patches_per_axis=[2, 2], n=3)

    chunked = pq_chunked.derivative(_f_mixing, dim=0, order=1)
    full = pq_full.derivative(_f_mixing, dim=0, order=1)

    assert jnp.allclose(chunked, full, atol=tolerances.atol(1e-6))
