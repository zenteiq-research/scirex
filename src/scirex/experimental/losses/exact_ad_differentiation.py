# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Exact per-point autodiff for non-block-diagonal `f(coords) -> u`.

Companion to `ad_differentiation` (`ADDiff`),
which gets every point's derivative from *one* backward pass, but only when
`f` is block-diagonal in `coords` (see that module's docstring: point
`i`'s output must not depend on point `j`'s coordinates). This module
lifts that precondition: `ExactPointwiseADDiff` computes
`d(u_i)/d(coords_i)` exactly for every point `i`, regardless of how much
`f` mixes points together -- e.g. a self-attention layer, where every
point's output genuinely depends on every other point's coordinates.

The mechanism
--------------
`f` is evaluated **once** -- `jax.vjp(f, coords)` runs a single forward
pass and returns `(u, vjp_fn)`, where `vjp_fn` is a *linear* function
that reuses that one forward pass's saved computation to turn any cotangent
vector into a gradient, without ever re-running `f`. Calling
`vjp_fn(e_i)` with the one-hot vector `e_i` (1 at position `i`, 0
elsewhere) gives exactly the `i`-th row of the full Jacobian --
`d(u_i)/d(coords_j)` for every `j`, including every cross-point term::

    u, vjp_fn = jax.vjp(f, coords)
    (grad_row_i,) = vjp_fn(e_i)   # d(u_i)/d(coords_j) for every j

Running this once per point (`jax.vmap` over all `n` one-hot vectors,
one vectorised XLA computation rather than a Python loop) builds the full
`(n, n, coord_dim)` Jacobian; keeping only the `i == j` diagonal gives
exactly `d(u_i)/d(coords_i)` for every point -- correct for any `f`,
since nothing here assumes points don't interact (unlike the sum trick,
which relies on that assumption to cancel every `j != i` term). The
off-diagonal terms this computes along the way are simply discarded, not
avoided -- see the cost note below for what that does and doesn't save.

Cost, and why this is the thing the patch mechanism exists to bound
----------------------------------------------------------------------
This still costs `O(n)` *backward* passes for `n` points -- reverse-mode
AD gives one Jacobian row per cotangent, and `n` independent rows are
needed, so that part is unavoidable for an exact per-point result from a
globally-mixing `f`. What sharing one `jax.vjp` call *does* save,
versus computing each row via its own independent `jax.grad` call (which
would silently re-run `f`'s forward pass every single time), is exactly
that redundant forward-pass repetition: one forward pass total instead of
`n`. If `f` costs `O(n^2)` per evaluation (e.g. full self-attention
over `n` points) and each backward sweep costs about the same order as
one forward pass, total cost here is `~(n + 1) * O(n^2)` versus
`~2n * O(n^2)` for `n` independent forward+backward pairs -- roughly a
constant factor of 2, **not** a change in the `O(n^3)` scaling. That
`O(n^3)` is the concrete, quantified reason exact AD is still expensive
for a globally-mixing model at full domain scale, and exactly what the
patch+quadrature mechanism (`patch_quadrature_ad.py`) exists to
bound, by applying this same exact machinery to a small patch (size
`p << n`) instead of the whole point cloud -- a different, complementary
saving (asymptotic cost control) from the constant-factor one here.

This module is the **unconditionally correct fallback and reference**: use
it as ground truth to validate any cheaper approximation against.

Same conventions as `ADDiff`:
`f(coords) -> u` must be unbatched (shape `(n_points,)`); compose with
`jax.vmap` outside for a batch dimension. `dim`/`order` must be
static.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .ad_differentiation import FieldFn, _check_unbatched


class ExactPointwiseADDiff:
    """Exact per-point derivatives of `f(coords)` via one shared
    `jax.vjp` call, correct even when `f` is not block-diagonal.

    Same `.derivative`/`.gradient` interface as
    `ADDiff`, so it is a
    drop-in alternative wherever block-diagonality doesn't hold, or can't
    be assumed.

    !!! note
        `dim` and `order` must be static (plain Python values, not
        traced arrays), matching `ADDiff`'s convention.

    !!! warning
        Order 2 nests this exact primitive on top of itself (see
        `derivative`), so it costs roughly the *square* of order 1's
        already-`O(n)` cost. Prefer order 1 (or the patch+quadrature
        mechanism) wherever possible.

    !!! warning
        `_all_dims` materializes the full `(n, n, coord_dim)`
        Jacobian before keeping only the diagonal, so memory use is
        `O(n^2)`, not `O(n)`. Fine at the patch sizes this class is
        actually meant for (tens of points), but a real out-of-memory risk
        if this is reached for as an unpatched, full-domain ground truth
        (as this class's own docstring recommends) on a mesh with `n` in
        the thousands. Prefer the patch+quadrature mechanism, or a
        deliberately small subsample, for anything at that scale.
    """

    owns_coords = False
    """Stateless -- see
    `owns_coords`."""

    @property
    def nested_differentiator(self) -> "ExactPointwiseADDiff":
        """See
        `nested_differentiator`.
        `ExactPointwiseADDiff` has no state tied to a specific coordinate
        array either, so this also just returns `self`.
        """
        return self

    def derivative(
        self,
        f: FieldFn,
        coords: jnp.ndarray,
        dim: int,
        order: int = 1,
    ) -> jnp.ndarray:
        """`d^order u / d(coords[..., dim])^order` at every point, exact.

        Parameters
        ----------
        f : Callable[[jnp.ndarray], jnp.ndarray]
            maps `coords` (shape `(n_points, coord_dim)`) to the field
            value at each point (shape `(n_points,)`); may mix points
            arbitrarily
        coords : jnp.ndarray
            query coordinates, shape `(n_points, coord_dim)`
        dim : int
            which coordinate dimension to differentiate along
        order : int, optional
            derivative order, 1 or 2, by default 1. Order 2 differentiates
            `d(u_i)/d(coords[i, dim])` (itself a function of `coords`
            that may still mix points, since `f` did) a second time --
            a diagonal second derivative, not a full Hessian, same
            semantics as `ADDiff.derivative`. The two differentiation
            passes are deliberately *not* the same mode: see
            `_diag_via_jvp` for why.

        Returns
        -------
        jnp.ndarray
            derivative at each point, shape `(n_points,)`
        """
        if order not in (1, 2):
            raise ValueError(
                f"ExactPointwiseADDiff only supports order in (1, 2), got {order}"
            )
        if order == 1:
            return self._all_dims(f, coords)[..., dim]

        def first_fn(c: jnp.ndarray) -> jnp.ndarray:
            return self._all_dims(f, c)[..., dim]

        return self._diag_via_jvp(first_fn, coords, dim)

    def gradient(
        self, f: FieldFn, coords: jnp.ndarray, dims: Sequence[int]
    ) -> list[jnp.ndarray]:
        """First derivative of `f(coords)` along each of `dims`, exact,
        from a single `O(n)` sweep (one `vmap(grad)` call already
        produces the derivative with respect to every coordinate dimension
        simultaneously -- this just slices out the ones asked for), rather
        than one call to `derivative` per dim.

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
        grad_all = self._all_dims(f, coords)
        return [grad_all[..., d] for d in dims]

    @staticmethod
    def _all_dims(f: FieldFn, coords: jnp.ndarray) -> jnp.ndarray:
        """One shared forward pass (`jax.vjp`), then `n` backward-only
        sweeps reusing it -- giving every point's exact gradient w.r.t. its
        own coordinate, all dims at once, without re-running `f` per point
        (unlike `n` independent `jax.grad` calls, which would each
        re-run the forward pass from scratch -- see the module docstring's
        cost note for what this does and doesn't save).

        Returns
        -------
        jnp.ndarray, shape (n_points, coord_dim)
        """
        n = coords.shape[0]
        u, vjp_fn = jax.vjp(f, coords)
        _check_unbatched(u)

        def jacobian_row(cotangent: jnp.ndarray) -> jnp.ndarray:
            (grad_coords,) = vjp_fn(cotangent)
            return grad_coords

        # full_jacobian[i, j, :] = d(u_i)/d(coords_j); only the i == j
        # diagonal (each point's derivative w.r.t. its OWN coordinate) is
        # wanted here -- the off-diagonal entries are the exact cross-point
        # contributions ADDiff's sum trick would (wrongly) include instead.
        full_jacobian = jax.vmap(jacobian_row)(jnp.eye(n, dtype=u.dtype))
        return jnp.diagonal(full_jacobian, axis1=0, axis2=1).T

    @staticmethod
    def _diag_via_jvp(f: FieldFn, coords: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Forward-mode (`jax.jvp`) counterpart of `_all_dims`,
        used only for order 2's *outer* differentiation step.

        Deliberately not the same mode as the inner step it wraps: order 2
        needs one differentiation pass applied on top of another that is
        already reverse-mode (`_all_dims` itself uses `jax.vjp`).
        Reverse-mode differentiating *through* an already-reverse-mode
        computation (reverse-over-reverse) is a real correctness risk for
        anything not perfectly smooth -- a `stop_gradient`, a
        `custom_vjp` rule not meant for a second differentiation pass, or
        a genuine singularity (e.g. a pairwise distance evaluated at zero)
        can all behave unpredictably under it. Forward-mode composes far
        more robustly with an inner computation that already contains a
        backward pass, which is why forward-over-reverse (this) rather
        than reverse-over-reverse is the standard way to get a second
        derivative -- see the module docstring. This does not change what
        is being computed (forward-mode and reverse-mode compute the
        identical mathematical derivative, just via different mechanics),
        and does not remove risk from a field with a genuine mathematical
        singularity in its second derivative -- no differentiation method
        can, since the true derivative does not exist there either.

        Same cost shape as `_all_dims`: one `jax.vmap` over `n`
        one-hot tangent vectors materializes the full `(n, n)` structure
        before keeping only the diagonal, so this is not a cost
        optimization over the reverse-mode version, purely a robustness
        one.

        Returns
        -------
        jnp.ndarray, shape (n_points,)
        """
        n = coords.shape[0]

        def one_hot_tangent(i: jnp.ndarray) -> jnp.ndarray:
            return jnp.zeros_like(coords).at[i, dim].set(1.0)

        def jvp_column(tangent: jnp.ndarray) -> jnp.ndarray:
            primal, out_tangent = jax.jvp(f, (coords,), (tangent,))
            _check_unbatched(primal)
            return out_tangent

        tangents = jax.vmap(one_hot_tangent)(jnp.arange(n))
        # full_jvp[i, j] = d(f_j)/d(coords[i, dim]); only the i == j
        # diagonal (each point's derivative w.r.t. its OWN coordinate) is
        # wanted, same reasoning as _all_dims's diagonal.
        full_jvp = jax.vmap(jvp_column)(tangents)
        return jnp.diagonal(full_jvp)
