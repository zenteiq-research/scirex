# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the AD-based PDE-term primitives (JAX): DiffusionTermAD,
AdvectionTermAD, ReactionTermAD, SourceTermAD, and reuse of the existing
PDEResidualLoss composite (from pde_terms's FDM side) with these
AD-backed terms.

* ``test_*termad_*`` -- each term tested standalone via a plain context
  dict (``{"u_fn": ..., "coords": ..., ...}``), against manually computed
  analytic derivatives -- mirrors test_pde_terms.py's structure.
* ``test_*_ignores_wrong_ctx_coords_with_patch_quadrature`` -- with a
  differentiator that owns its coords (``PatchQuadratureADDiff``), a term
  must ignore ``ctx["coords"]`` entirely, confirmed by feeding it a
  different, same-shaped instance's coords (the scenario
  ``owns_coords``/``_coords_kwarg`` exist for) and checking the result is
  unaffected -- also asserts no warning fires, confirming the term never
  reaches the explicit-``coords`` path at all for this differentiator.
* ``test_pderesidualloss_reused_with_ad_terms_*`` -- confirms
  PDEResidualLoss (unmodified) composes AD-backed terms correctly, and that
  jax.grad flows through to a model parameter, the way it needs to for
  actual training.
"""

import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from scirex.experimental.losses.ad_differentiation import ADDiff
from scirex.experimental.losses.ad_pde_terms import (
    AdvectionTermAD,
    DiffusionTermAD,
    ReactionTermAD,
    SourceTermAD,
)
from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.losses.equation_losses import PDEResidualLoss
from tests import tolerances


def _f(coords: jnp.ndarray) -> jnp.ndarray:
    """u(x, y) = sin(x) * cos(y)."""
    x, y = coords[..., 0], coords[..., 1]
    return jnp.sin(x) * jnp.cos(y)


def _random_coords(n: int, seed: int = 0) -> jnp.ndarray:
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)


def _analytic_laplacian(coords: jnp.ndarray) -> jnp.ndarray:
    """d2/dx2 + d2/dy2 of sin(x)*cos(y) = -2*sin(x)*cos(y), computed
    independently of ADDiff/DiffusionTermAD, as a ground truth to check
    against."""
    x, y = coords[:, 0], coords[:, 1]
    return -2.0 * jnp.sin(x) * jnp.cos(y)


# ---------------------------------------------------------------------------
# DiffusionTermAD
# ---------------------------------------------------------------------------


def test_diffusiontermad_matches_coeff_times_laplacian() -> None:
    coords = _random_coords(20)
    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=3.0)
    result = term({"u_fn": _f, "coords": coords})

    expected = 3.0 * _analytic_laplacian(coords)
    assert jnp.allclose(result, expected, atol=tolerances.atol(1e-5))


def test_diffusiontermad_matches_manual_sum_of_second_derivatives() -> None:
    """Cross-checks against ADDiff.derivative(order=2) called directly per
    dim and summed by hand, independent of DiffusionTermAD's own internals."""
    coords = _random_coords(20, seed=8)
    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=1.0)
    result = term({"u_fn": _f, "coords": coords})

    ad = ADDiff()
    expected = ad.derivative(_f, coords, dim=0, order=2) + ad.derivative(
        _f, coords, dim=1, order=2
    )
    assert jnp.allclose(result, expected)


def test_diffusiontermad_rejects_empty_dims() -> None:
    """An empty dims list would silently give an all-zero Laplacian instead
    of raising."""
    with pytest.raises(AssertionError):
        DiffusionTermAD(dims=[], field="u_fn", coeff=1.0)


def test_diffusiontermad_rejects_repeated_dims() -> None:
    """A repeated dim index would silently double-count that dimension's
    contribution to the Laplacian instead of raising."""
    with pytest.raises(AssertionError):
        DiffusionTermAD(dims=[0, 0], field="u_fn", coeff=1.0)


def test_diffusiontermad_rejects_array_coeff() -> None:
    """A precomputed array -- even one with a shape matching the query
    points -- is no longer accepted for coeff. AD can't differentiate a
    fixed array, and there's no jit-safe way to check whether it's
    actually spatially uniform, so this must raise instead of silently
    using it (which would be wrong whenever the array isn't uniform)."""
    coords = _random_coords(10, seed=9)
    coeff_array = jnp.full((coords.shape[0],), 2.0)  # correctly shaped, still rejected
    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff="a")
    with pytest.raises(TypeError):
        term({"u_fn": _f, "coords": coords, "a": coeff_array})


def test_diffusiontermad_flux_form_matches_fast_path_for_constant_coeff_fn() -> None:
    """When coeff_fn happens to be spatially uniform, the flux-form path
    (nested AD) must agree exactly with the cheap coeff*laplacian(u) path,
    confirming the two formulas really do coincide in the constant case,
    not just that both run without error."""
    coords = _random_coords(15, seed=1)

    def coeff_fn(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.full((c.shape[0],), 2.0)

    flux_form_term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff="nu")
    fast_path_term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=2.0)

    result_flux = flux_form_term({"u_fn": _f, "coords": coords, "nu": coeff_fn})
    result_fast = fast_path_term({"u_fn": _f, "coords": coords})

    assert jnp.allclose(result_flux, result_fast, atol=tolerances.atol(1e-5))


def test_diffusiontermad_flux_form_matches_manufactured_solution() -> None:
    """Correctness test for a genuinely non-constant coeff_fn (varies with
    x). Cross-checked against jax.vmap(jax.jacfwd(...)) computing the true
    flux-form divergence per point directly -- completely independent of
    ADDiff/DiffusionTermAD's own sum-trick machinery, the same way
    test_addiff_matches_vmap_grad independently verifies ADDiff itself."""
    coords = _random_coords(20, seed=13)

    def coeff_fn(c: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + 0.5 * c[:, 0]

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=coeff_fn)
    result = term({"u_fn": _f, "coords": coords})

    def coeff_single(c: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + 0.5 * c[0]

    def flux_single(c: jnp.ndarray) -> jnp.ndarray:
        grad_u = jax.grad(lambda cc: _f(cc[None, :])[0])(c)
        return coeff_single(c) * grad_u

    def divergence_single(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.trace(jax.jacfwd(flux_single)(c))

    expected = jax.vmap(divergence_single)(coords)
    assert jnp.allclose(result, expected, atol=1e-4)


def test_diffusiontermad_flux_form_with_patch_quadrature_differentiator() -> None:
    """Regression test: the flux-form path (non-constant coeff) nests a
    second call from inside a closure that is already evaluating one
    patch's coordinates (see ``DiffusionTermAD._flux_form``'s ``flux_d``).
    With ``PatchQuadratureADDiff`` and more than one patch, that nested call
    is handed a single patch's worth of coordinates, not the full array --
    before this term routed the nested call through
    ``differentiator.nested_differentiator`` instead of calling back into
    ``differentiator`` itself, this raised instead of returning a result. A
    single patch (``n_patches_per_axis=[1, 1]``) would not have caught this,
    since a lone patch's own coordinates are shape-identical to the full
    array -- multiple patches are required so the nested call actually
    receives a smaller shape.
    """
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[2, 2], n=3)

    def coeff_fn(c: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + 0.5 * c[:, 0]

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=coeff_fn, differentiator=pq)
    result = term({"u_fn": _f, "coords": pq.coords})

    reference_term = DiffusionTermAD(
        dims=[0, 1], field="u_fn", coeff=coeff_fn, differentiator=ExactPointwiseADDiff()
    )
    expected = reference_term({"u_fn": _f, "coords": pq.coords})
    assert jnp.allclose(result, expected, atol=1e-4)


def test_diffusiontermad_laplacian_ignores_wrong_ctx_coords_with_patch_quadrature() -> (
    None
):
    """The scenario ``PatchQuadratureADDiff.owns_coords`` exists for, at the
    PDE-term level: a batch-loop mistake could build ``ctx["coords"]`` from
    a *different*, same-shaped PatchQuadratureADDiff instance (see that
    class's module docstring). Because this term now only forwards
    ``ctx["coords"]`` to the differentiator when it does not own its
    coords, the wrong array has no effect at all here -- confirmed by
    comparing against the same term called with the correct ``ctx["coords"]``."""
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[2, 2], n=3)
    wrong_pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[1, 4], n=3)
    assert pq.coords.shape == wrong_pq.coords.shape  # the ambiguous case

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=1.0, differentiator=pq)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_with_wrong_ctx_coords = term({"u_fn": _f, "coords": wrong_pq.coords})
    result_with_correct_ctx_coords = term({"u_fn": _f, "coords": pq.coords})

    assert jnp.allclose(result_with_wrong_ctx_coords, result_with_correct_ctx_coords)


def test_diffusiontermad_flux_form_ignores_wrong_ctx_coords_with_patch_quadrature() -> (
    None
):
    """Same guarantee as the Laplacian-path test above, for the flux-form
    (callable coeff) path's outer derivative call."""
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[2, 2], n=3)
    wrong_pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[1, 4], n=3)
    assert pq.coords.shape == wrong_pq.coords.shape

    def coeff_fn(c: jnp.ndarray) -> jnp.ndarray:
        return 1.0 + 0.5 * c[:, 0]

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=coeff_fn, differentiator=pq)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_with_wrong_ctx_coords = term({"u_fn": _f, "coords": wrong_pq.coords})
    result_with_correct_ctx_coords = term({"u_fn": _f, "coords": pq.coords})

    assert jnp.allclose(result_with_wrong_ctx_coords, result_with_correct_ctx_coords)


def test_diffusiontermad_flux_form_rejects_batched_coeff_fn() -> None:
    """coeff_fn(coords) must be unbatched, same as field -- a batched
    output would otherwise mix gradient contributions across samples
    inside the flux, the same failure mode ADDiff guards against for the
    field itself."""
    coords = _random_coords(6, seed=14)

    def coeff_fn_batched(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([1.0 + 0.5 * c[:, 0], 2.0 + 0.5 * c[:, 0]])  # (2, n)

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=coeff_fn_batched)
    with pytest.raises(ValueError):
        term({"u_fn": _f, "coords": coords})


def test_diffusiontermad_flux_form_rejects_wrong_shape_coeff_fn() -> None:
    """coeff_fn(coords) must return one value per query point -- a
    mismatched shape would otherwise silently broadcast against the flux
    instead of raising."""
    coords = _random_coords(6, seed=15)

    def coeff_fn_wrong_shape(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([1.0])  # shape (1,), not (n_points,)

    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=coeff_fn_wrong_shape)
    with pytest.raises(ValueError):
        term({"u_fn": _f, "coords": coords})


# ---------------------------------------------------------------------------
# AdvectionTermAD
# ---------------------------------------------------------------------------


def test_advectiontermad_self_advection() -> None:
    """velocity=None: field advects itself, u * sum_d du/d(dim)."""
    coords = _random_coords(18, seed=2)
    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity=None)
    result = term({"u_fn": _f, "coords": coords})

    ad = ADDiff()
    du_dx, du_dy = ad.gradient(_f, coords, dims=[0, 1])
    u = _f(coords)
    expected = u * du_dx + u * du_dy
    assert jnp.allclose(result, expected)


def test_advectiontermad_constant_velocity() -> None:
    coords = _random_coords(18, seed=3)
    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity=2.0)
    result = term({"u_fn": _f, "coords": coords})

    ad = ADDiff()
    du_dx, du_dy = ad.gradient(_f, coords, dims=[0, 1])
    expected = 2.0 * du_dx + 2.0 * du_dy
    assert jnp.allclose(result, expected)


def test_advectiontermad_per_dim_velocity_components() -> None:
    coords = _random_coords(18, seed=4)
    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity=("vx", "vy"))
    vx = jnp.full((coords.shape[0],), 1.5)
    vy = jnp.full((coords.shape[0],), -0.5)
    result = term({"u_fn": _f, "coords": coords, "vx": vx, "vy": vy})

    ad = ADDiff()
    du_dx, du_dy = ad.gradient(_f, coords, dims=[0, 1])
    expected = vx * du_dx + vy * du_dy
    assert jnp.allclose(result, expected)


def test_advectiontermad_rejects_empty_dims() -> None:
    """An empty dims list would silently give an all-zero term instead of
    raising."""
    with pytest.raises(AssertionError):
        AdvectionTermAD(dims=[], field="u_fn")


def test_advectiontermad_rejects_repeated_dims() -> None:
    """A repeated dim index would silently double-count that dimension's
    contribution instead of raising."""
    with pytest.raises(AssertionError):
        AdvectionTermAD(dims=[0, 0], field="u_fn")


def test_advectiontermad_rejects_collapsed_velocity_shape() -> None:
    """A velocity field reduced/collapsed to a different shape than the
    query points would otherwise silently broadcast instead of raising."""
    coords = _random_coords(10, seed=10)
    v_collapsed = jnp.array([1.0])  # shape (1,), not (10,)
    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity="v")
    with pytest.raises(ValueError):
        term({"u_fn": _f, "coords": coords, "v": v_collapsed})


# ---------------------------------------------------------------------------
# ReactionTermAD / SourceTermAD
# ---------------------------------------------------------------------------


def test_reactiontermad_constant_and_field_coeff() -> None:
    coords = _random_coords(10, seed=5)
    u = _f(coords)

    result_const = ReactionTermAD(field="u_fn", coeff=2.0)(
        {"u_fn": _f, "coords": coords}
    )
    assert jnp.allclose(result_const, 2.0 * u)

    k = jnp.full((coords.shape[0],), -1.0)
    result_field = ReactionTermAD(field="u_fn", coeff="k")(
        {"u_fn": _f, "coords": coords, "k": k}
    )
    assert jnp.allclose(result_field, k * u)


def test_reactiontermad_rejects_collapsed_coeff_shape() -> None:
    """A coeff field reduced/collapsed to a different shape than the query
    points would otherwise silently broadcast instead of raising."""
    coords = _random_coords(10, seed=11)
    coeff_collapsed = jnp.array([1.0])  # shape (1,), not (10,)
    term = ReactionTermAD(field="u_fn", coeff="r")
    with pytest.raises(ValueError):
        term({"u_fn": _f, "coords": coords, "r": coeff_collapsed})


def test_reactiontermad_rejects_batched_field() -> None:
    """Unlike DiffusionTermAD/AdvectionTermAD, ReactionTermAD never calls
    ADDiff, so it must check the unbatched-field invariant itself instead
    of silently accepting a batched f(coords)."""
    coords = _random_coords(10, seed=12)

    def f_batched(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([_f(c), 2.0 * _f(c)])  # shape (2, n_points)

    term = ReactionTermAD(field="u_fn", coeff=1.0)
    with pytest.raises(ValueError):
        term({"u_fn": f_batched, "coords": coords})


def test_sourcetermad_constant_and_field() -> None:
    assert SourceTermAD(1.0)({}) == 1.0
    src = jnp.array([1.0, 2.0])
    assert jnp.allclose(SourceTermAD("f")({"f": src}), src)
    assert SourceTermAD()({}) == 0.0


# ---------------------------------------------------------------------------
# PDEResidualLoss reused unmodified with AD-backed terms
# ---------------------------------------------------------------------------


def test_pderesidualloss_reused_with_ad_terms_matches_manual_residual() -> None:
    """-nu*Laplacian(u) - f, assembled via PDEResidualLoss + AD terms,
    matches computing the same residual by hand."""
    coords = _random_coords(25, seed=6)
    nu = 0.5
    forcing = 1.0

    pde = PDEResidualLoss(
        terms={
            "diffusion": (-1.0, DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=nu)),
            "source": (-1.0, SourceTermAD(forcing)),
        },
        reduction="mean",
        name="PoissonAD",
    )
    ctx = {"u_fn": _f, "coords": coords}
    residual = pde.residual(ctx)

    expected = -nu * _analytic_laplacian(coords) - forcing
    assert jnp.allclose(residual, expected, atol=tolerances.atol(1e-5))


def test_pderesidualloss_reused_with_ad_terms_grad_flows_to_params() -> None:
    """jax.grad through PDEResidualLoss + AD terms reaches a model
    parameter -- the actual thing training needs to work."""
    coords = _random_coords(16, seed=7)

    def make_f(theta: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
        def f(c: jnp.ndarray) -> jnp.ndarray:
            x, y = c[..., 0], c[..., 1]
            return theta * jnp.sin(x) * jnp.cos(y)

        return f

    pde = PDEResidualLoss(
        terms={
            "diffusion": (-1.0, DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=1.0)),
            "source": (-1.0, SourceTermAD(1.0)),
        },
        reduction="mean",
    )

    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        ctx = {"u_fn": make_f(theta), "coords": coords}
        return pde.abs(ctx, take_root=True)

    theta0 = jnp.array(2.0)
    grad = jax.grad(loss_fn)(theta0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_pderesidualloss_rejects_bad_reduction_still_works() -> None:
    """Sanity check: PDEResidualLoss's own validation is untouched by AD terms."""
    with pytest.raises(AssertionError):
        PDEResidualLoss(terms={}, reduction="median")


# ---------------------------------------------------------------------------
# injectable differentiator (ADDiff / ExactPointwiseADDiff / PatchQuadratureADDiff)
# ---------------------------------------------------------------------------


def test_diffusiontermad_default_differentiator_is_addiff() -> None:
    """No differentiator passed -> ADDiff, preserving prior behavior."""
    term = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=1.0)
    assert isinstance(term._ad, ADDiff)


def test_diffusiontermad_with_exact_differentiator_matches_addiff() -> None:
    """On a non-mixing field, injecting ExactPointwiseADDiff instead of the
    default ADDiff must give the same result -- both are exact when the
    block-diagonal precondition genuinely holds, only their cost differs."""
    coords = _random_coords(20, seed=20)

    term_default = DiffusionTermAD(dims=[0, 1], field="u_fn", coeff=1.0)
    term_exact = DiffusionTermAD(
        dims=[0, 1], field="u_fn", coeff=1.0, differentiator=ExactPointwiseADDiff()
    )

    result_default = term_default({"u_fn": _f, "coords": coords})
    result_exact = term_exact({"u_fn": _f, "coords": coords})
    assert jnp.allclose(result_default, result_exact, atol=tolerances.atol(1e-5))


def test_advectiontermad_with_patch_quadrature_differentiator() -> None:
    """Injecting PatchQuadratureADDiff (single patch, so no approximation)
    into AdvectionTermAD reaches the same analytic answer as the default
    ADDiff path, confirming the injected differentiator is actually used."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[1, 1], n=4)

    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity=None, differentiator=pq)
    result = term({"u_fn": _f, "coords": pq.coords})

    x, y = pq.coords[:, 0], pq.coords[:, 1]
    u = _f(pq.coords)
    du_dx = jnp.cos(x) * jnp.cos(y)
    du_dy = -jnp.sin(x) * jnp.sin(y)
    expected = u * du_dx + u * du_dy
    assert jnp.allclose(result, expected, atol=tolerances.atol(1e-5))


def test_advectiontermad_ignores_wrong_ctx_coords_with_patch_quadrature() -> None:
    """Same guarantee as DiffusionTermAD's equivalent tests, for
    AdvectionTermAD's gradient call."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[2, 2], n=3)
    wrong_pq = PatchQuadratureADDiff(bounds=bounds, n_patches_per_axis=[1, 4], n=3)
    assert pq.coords.shape == wrong_pq.coords.shape

    term = AdvectionTermAD(dims=[0, 1], field="u_fn", velocity=None, differentiator=pq)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_with_wrong_ctx_coords = term({"u_fn": _f, "coords": wrong_pq.coords})
    result_with_correct_ctx_coords = term({"u_fn": _f, "coords": pq.coords})

    assert jnp.allclose(result_with_wrong_ctx_coords, result_with_correct_ctx_coords)
