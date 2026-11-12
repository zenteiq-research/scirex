# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Autodiff-based (exact) differentiation for point-cloud/query-coordinate fields.

Companion to `differentiation` (`FiniteDiff`
/ `FourierDiff`, which differentiate a field *sampled on a regular
grid*). This module differentiates a field defined by a *function of
continuous coordinates* -- `f(coords) -> u` -- with respect to those
coordinates directly, via `jax.grad`. This is the right tool whenever the
model doesn't produce a regular grid at all (point-cloud/mesh operators such
as GINO queried at arbitrary points, or a PINN evaluated at scattered
collocation points), so `FiniteDiff`/`FourierDiff` don't apply.

Model-agnostic by design: `ADDiff` only ever sees a plain callable
`f: coords -> u` and the `coords` array -- it never sees model parameters
or any model-specific structure. The caller builds the closure (e.g.
`f = lambda q: pinn.apply(params, q)` or
`f = lambda q: gino.apply(params, input_geom, latent_queries, q, x=x_data)`),
so the same primitive works for a PINN, GINO, or any other coordinate-in
model without modification.

The "sum trick"
----------------
`jax.grad` requires a scalar output, but we want the *per-point* derivative
`du_i/d(coords_i)` for every point `i` in a batch. Summing the output to
a scalar and taking one `jax.grad` gives, at point `i`::

    d(sum_j u_j) / d(coords_i) = sum_j d(u_j) / d(coords_i)

This only equals the per-point derivative `d(u_i)/d(coords_i)` we actually
want if every other term in that sum is zero -- i.e. if `f` is
**block-diagonal**: point `i`'s output must not depend on any other
point's coordinates (`d(u_j)/d(coords_i) = 0` for `j != i`). This holds
for a PINN (a plain pointwise MLP, no cross-point mixing at all) and for
GINO's decode step (each output query reads only its own coordinate plus a
*fixed*, query-independent latent embedding -- see `gino.py`'s
`gno_out` call). It does **not** hold for an architecture with genuine
cross-point mixing (e.g. attention over the full point cloud, as in
Transolver) -- using `ADDiff` there would silently produce wrong
(contaminated) derivatives rather than raising an error, since there is no
cheap way to detect non-block-diagonality from inside this primitive. This
is a precondition on `f`, not something `ADDiff` checks.

Cost: one backward pass total, independent of the number of points -- unlike
`jax.vmap(jax.grad(f_single))` (also valid whenever the same
block-diagonal condition holds), which costs `N` backward passes for `N`
points.

No batch dimension -- compose with `jax.vmap` instead
----------------------------------------------------------
`f` must return a single **unbatched** field, shape `(n_points,)` -- the
same cancellation argument the sum trick relies on breaks in a different way
if `f` returns a batch of samples (`(batch, n_points)`): summing over
the batch axis too would mix gradient contributions from *different
samples* at the same query point into one number (this matters because, in
e.g. GINO, the query coordinates themselves are shared/unbatched --
`output_queries` is one fixed set of points reused for every sample --
while the decoded output is batched). Rather than teach this primitive
about a batch axis, the idiomatic JAX fix is composition: wrap the whole
per-sample closure (including the `ADDiff` call) in `jax.vmap` over
the batch axis at the call site. `derivative` enforces this by raising
if `f(coords)` isn't 1-D.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

FieldFn = Callable[[jnp.ndarray], jnp.ndarray]


class ADDiff:
    """Exact per-point derivatives of `f(coords)` via the AD sum trick.

    !!! note
        `dim` and `order` must be static (plain Python values, not traced
        arrays), matching the convention used by
        `FiniteDiff` /
        `FourierDiff`.

    !!! warning
        `f` must be block-diagonal in `coords` -- see the module
        docstring. This is a precondition, not something checked here.
    """

    owns_coords = False
    """Stateless: every call needs its own coordinates, since this is meant
    to be reused with different or resampled coordinates across calls (a
    PINN resampling collocation points, GINO getting different query points
    per batch). See
    `owns_coords`
    for the differentiator where this is `True` instead, and
    `ad_pde_terms` for how PDE terms use this
    flag to decide whether to pass `coords` along at all."""

    @property
    def nested_differentiator(self) -> "ADDiff":
        """The differentiator a PDE term should use for a *further*,
        nested differentiation call made from inside a closure already
        being differentiated by this one (e.g.
        `DiffusionTermAD`'s
        flux-form path, which differentiates a coefficient function and the
        field it multiplies separately). `ADDiff` has no internal state
        tied to a specific coordinate array, so it is always safe to call
        again on any input -- this simply returns `self`. See
        `nested_differentiator`
        for the differentiator where this is *not* true.
        """
        return self

    def derivative(
        self,
        f: FieldFn,
        coords: jnp.ndarray,
        dim: int,
        order: int = 1,
    ) -> jnp.ndarray:
        """`d^order u / d(coords[..., dim])^order` at every point.

        Parameters
        ----------
        f : Callable[[jnp.ndarray], jnp.ndarray]
            maps `coords` (shape `(n_points, coord_dim)`) to the field
            value at each point (shape `(n_points,)`)
        coords : jnp.ndarray
            query coordinates, shape `(n_points, coord_dim)`
        dim : int
            which coordinate dimension to differentiate along (0 for x,
            1 for y, ...)
        order : int, optional
            derivative order, 1 or 2, by default 1. Order 2 differentiates
            along `dim` twice -- a diagonal second derivative
            (`d^2u/dx^2`), not a full Hessian. The two passes
            deliberately use different differentiation modes: see
            `_first_fwd` for why.

        Returns
        -------
        jnp.ndarray
            derivative at each point, shape `(n_points,)`
        """
        if order not in (1, 2):
            raise ValueError(f"ADDiff only supports order in (1, 2), got {order}")
        if order == 1:
            return self._first(f, coords, dim)

        def first_fn(c: jnp.ndarray) -> jnp.ndarray:
            return self._first(f, c, dim)

        return self._first_fwd(first_fn, coords, dim)

    @staticmethod
    def _first(f: FieldFn, coords: jnp.ndarray, dim: int) -> jnp.ndarray:
        """One sum-trick backward pass, sliced to a single coordinate dim."""

        def scalar_sum(c: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            u = f(c)
            _check_unbatched(u)
            return jnp.sum(u), u

        grad_coords, _ = jax.grad(scalar_sum, has_aux=True)(coords)
        return grad_coords[..., dim]

    @staticmethod
    def _first_fwd(f: FieldFn, coords: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Forward-mode (`jax.jvp`) counterpart of `_first`, used
        only for order 2's *outer* differentiation step.

        Deliberately not the same mode as the inner step it wraps -- see
        `_diag_via_jvp`
        for the full reasoning (forward-over-reverse composes far more
        robustly than reverse-over-reverse for anything not perfectly
        smooth, without changing what is mathematically being computed).

        Still a single sweep, exploiting the same block-diagonal
        precondition `_first` relies on: perturbing every point's
        `dim`-th coordinate by the same unit tangent simultaneously, via
        one `jax.jvp` call, gives `sum_j d(f_i)/d(coords[j, dim])` for
        each output `i`. Block-diagonal `f` means every term with
        `j != i` is exactly zero, so this collapses to exactly
        `d(f_i)/d(coords[i, dim])` -- the forward-mode mirror of
        `_first`'s reverse-mode sum trick, at the same `O(1)`-sweep
        cost, not the `O(n)` sweeps
        `_diag_via_jvp`
        needs for a field that isn't block-diagonal.
        """
        tangent = jnp.zeros_like(coords).at[:, dim].set(1.0)
        primal, out_tangent = jax.jvp(f, (coords,), (tangent,))
        _check_unbatched(primal)
        return out_tangent

    def gradient(
        self, f: FieldFn, coords: jnp.ndarray, dims: Sequence[int]
    ) -> list[jnp.ndarray]:
        """First derivative of `f(coords)` along each of `dims`, from a
        single backward pass (all coordinate dims come out of one
        `jax.grad` call already -- this just slices them out), rather than
        one call to `derivative` per dim.

        Parameters
        ----------
        f : Callable[[jnp.ndarray], jnp.ndarray]
            as in `derivative`
        coords : jnp.ndarray
            query coordinates, shape `(n_points, coord_dim)`
        dims : Sequence[int]
            coordinate dimensions to differentiate along

        Returns
        -------
        list[jnp.ndarray]
            `[du/d(coords[...,dims[0]]), du/d(coords[...,dims[1]]), ...]`,
            each shape `(n_points,)`
        """

        def scalar_sum(c: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            u = f(c)
            _check_unbatched(u)
            return jnp.sum(u), u

        grad_coords, _ = jax.grad(scalar_sum, has_aux=True)(coords)
        return [grad_coords[..., d] for d in dims]


def _check_unbatched(u: jnp.ndarray) -> None:
    """Guard against the batch-mixing failure mode described in the module
    docstring: `f` must return one unbatched field, not a batch of them."""
    if u.ndim != 1:
        raise ValueError(
            "ADDiff requires f(coords) to return an unbatched field, shape "
            f"(n_points,), got ndim={u.ndim} (shape={u.shape}). If you have "
            "a batch of samples, wrap the whole per-sample closure "
            "(including the ADDiff call) in jax.vmap over the batch axis "
            "instead of batching f itself -- see the module docstring."
        )
