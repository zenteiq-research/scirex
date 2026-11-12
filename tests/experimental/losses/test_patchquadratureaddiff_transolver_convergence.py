# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Validation step 2 of the physics-loss roadmap's Phase 3b: manufactured
convergence check for
:class:`~scirex.experimental.losses.patch_quadrature_ad.PatchQuadratureADDiff`
(patched, bounded-cost) against
:class:`~scirex.experimental.losses.exact_ad_differentiation.ExactPointwiseADDiff`
(fully exact, whole-domain, ground truth) on a real
:class:`~scirex.models.transolver.Transolver` -- the globally-mixing
architecture this whole mechanism exists for (attention over every point,
so :class:`~scirex.experimental.losses.ad_differentiation.ADDiff`'s sum trick
does not apply here -- see that module's docstring).

For a *fixed* patch/quadrature point set, ``PatchQuadratureADDiff`` and
``ExactPointwiseADDiff`` differ only in how much of the model's mixing each
one's derivative captures: patched only sees mixing *within* a patch,
exact sees all of it. Coarsening the partition (fewer, bigger patches)
can only ever recover cross-terms a finer partition dropped -- it never
loses any -- so the (weighted, quadrature-consistent) relative error
between the two must shrink monotonically as patches merge, hitting
exactly zero at a single patch covering the whole domain (already proven
algebraically in ``test_patch_quadrature_ad.py``'s
``test_single_patch_matches_exactpointwiseaddiff_on_mixing_field``; this
test is the empirical version, on a real attention-mixing model instead of
a synthetic mixing field).

Note the three ``n_patches_per_axis`` settings below are compared via a
relative-error *metric*, not point-by-point -- each setting's quadrature
nodes are an independent Gauss-Legendre tensor product, not a nested
refinement of the others, so their coordinate sets don't coincide.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp

from scirex.experimental.losses.exact_ad_differentiation import ExactPointwiseADDiff
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.models.transolver import Transolver

_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]

# The single-patch error is zero algebraically, but patched and exact reach
# it through different float32 op sequences (the Transolver computes in
# float32 since #307), so it lands at rounding level (~2e-7), not exactly 0.
_SINGLE_PATCH_TOL = 1e-6


def _build_transolver_closure() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """A small, real, untrained Transolver (irregular geometry, plain attention
    mixing every point) wrapped as an unbatched ``f(coords) -> (n_points,)``
    closure -- same convention every AD differentiator in this repo expects.
    """
    model = Transolver(
        space_dim=2,
        n_layers=2,
        n_hidden=16,
        n_head=2,
        fun_dim=0,
        out_dim=1,
        slice_num=4,
        unified_pos=False,
        geometry="irregular",
    )
    params = model.init(jax.random.PRNGKey(0), x=jnp.zeros((1, 4, 2)))

    def f(coords: jnp.ndarray) -> jnp.ndarray:
        out = model.apply(params, x=coords[None, :, :])
        return cast(jnp.ndarray, out)[0, :, 0]

    return f


def _relative_patched_vs_exact_error(
    f: Callable[[jnp.ndarray], jnp.ndarray], n_patches_per_axis: list[int]
) -> float:
    """Weighted relative L2 difference between PatchQuadratureADDiff and
    ExactPointwiseADDiff, evaluated at the same (this config's own)
    quadrature coordinates."""
    pq = PatchQuadratureADDiff(
        bounds=_BOUNDS, n_patches_per_axis=n_patches_per_axis, n=3
    )
    patched = pq.derivative(f, dim=0, order=1)
    exact = ExactPointwiseADDiff().derivative(f, pq.coords, dim=0, order=1)

    num = jnp.sqrt(jnp.sum(pq.weights * (patched - exact) ** 2))
    den = jnp.sqrt(jnp.sum(pq.weights * exact**2)) + 1e-12
    return float(num / den)


def test_patchquadratureaddiff_matches_exact_at_single_patch_on_transolver() -> None:
    """Sanity floor for this test module specifically: on a real Transolver
    (not the earlier tests' synthetic mixing field), a single patch spanning
    the whole domain must still reduce to zero relative error, up to float32
    rounding."""
    f = _build_transolver_closure()
    err = _relative_patched_vs_exact_error(f, n_patches_per_axis=[1, 1])
    assert err < _SINGLE_PATCH_TOL


def test_patchquadratureaddiff_error_shrinks_as_patches_merge_on_transolver() -> None:
    """The actual convergence claim: error strictly decreases as the
    partition coarsens from many small patches toward the single-patch
    (fully exact) limit."""
    f = _build_transolver_closure()

    err_4x4 = _relative_patched_vs_exact_error(f, n_patches_per_axis=[4, 4])
    err_2x2 = _relative_patched_vs_exact_error(f, n_patches_per_axis=[2, 2])
    err_1x1 = _relative_patched_vs_exact_error(f, n_patches_per_axis=[1, 1])

    assert err_4x4 > err_2x2 > err_1x1
    assert err_1x1 < _SINGLE_PATCH_TOL
    # not a vacuously-tiny difference -- confirms this model genuinely mixes
    # enough for coarser patches to matter
    assert err_4x4 > 0.05
