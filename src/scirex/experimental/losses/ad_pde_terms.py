# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reusable PDE-term building blocks for AD-based (point-cloud) physics losses.

Autodiff counterpart to `pde_terms`, which builds
its terms on top of grid-sampled arrays (`FiniteDiff`/
`FourierDiff`). Here, the field being differentiated is a *function*
of continuous query coordinates -- `f(coords) -> u` -- differentiated via
`ADDiff`'s sum trick, so it
applies to any model that produces such a function (a point-cloud/mesh
operator such as GINO queried at arbitrary points, or a PINN), not just
grid-native models.

Scope of this module (steady-state only)
------------------------------------------
`AdvectionTermAD`'s `velocity` and `ReactionTermAD`'s `coeff` are
never differentiated, so they can be a constant, a plain sampled
`(n_points,)` array, or a coordinate function -- `v(x)*grad(u)` and
`coeff(x)*u(x)` are exactly correct either way, since neither term takes
a further derivative of the coefficient.

`DiffusionTermAD.coeff` is different, because diffusion's true governing
term is `div(coeff * grad(u))`, which only collapses to the cheaper
`coeff * laplacian(u)` when `coeff` is genuinely spatially constant
(`div(coeff*grad(u)) = coeff*laplacian(u) + grad(coeff)-dot-grad(u)`, and
the second term is dropped by that shortcut). So `DiffusionTermAD.coeff`
accepts either:

* a plain Python `float` -- trivially constant, uses the cheap
  `coeff * laplacian(u)` path;
* a coordinate function `coeff_fn(coords) -> jnp.ndarray` -- computed via
  genuine flux-form nested AD (see `DiffusionTermAD`), correct
  whether or not the coefficient actually varies, since JAX's autodiff
  applies the product rule automatically when the flux is differentiated a
  second time.

A precomputed `(n_points,)` *array* (as opposed to a function) is
deliberately **not** accepted for `coeff` here: AD can only differentiate
a function of `coords`, not a fixed array with no traced relationship to
`coords`, so there is no way to run flux-form on one, and no jit-safe way
to check at trace time whether it happens to be uniform. Pass a
`coeff_fn` instead (a function that returns a constant is a valid
`coeff_fn` too, and takes the flux-form path just like a genuinely
varying one -- see the cross-check test confirming the two paths agree
exactly in that case).

There is no `TransientTerm` counterpart here: this module only supports
steady-state equations (no time axis) -- see project decisions.

Context-dict interface
-----------------------
Every term is called as `term(ctx)`, where `ctx` is a
`dict[str, Any]` carrying (at least):

* `"u_fn"` -- the callable field, `coords -> jnp.ndarray` (unbatched,
  shape `(n_points,)`; see
  `ad_differentiation` for the batching
  convention -- the caller is responsible for `jax.vmap`-ing over any
  batch axis outside of this)
* `"coords"` -- the query coordinates, shape `(n_points, coord_dim)`
* any named coefficient/source fields (constants or `(n_points,)` arrays)
  a specific term needs

This mirrors `pde_terms`'s `ctx` convention
closely enough that the *same*
`PDEResidualLoss` composite is
reused unchanged -- it only ever calls `sign * term(ctx)` and sums, and
never looks inside `ctx` itself.
"""

from collections.abc import Sequence
from typing import Any, cast

import jax.numpy as jnp

from scirex.losses.pde_terms import _require_shape

from .ad_differentiation import ADDiff, FieldFn, _check_unbatched
from .exact_ad_differentiation import ExactPointwiseADDiff
from .patch_quadrature_ad import PatchQuadratureADDiff

Context = dict[str, Any]

ADDifferentiator = ADDiff | ExactPointwiseADDiff | PatchQuadratureADDiff
"""Any of the three AD-based differentiators, all sharing the same
`.derivative`/`.gradient`/`.nested_differentiator`/`.owns_coords`
interface -- see their respective modules for which one is appropriate for
a given model:
`ADDiff` (cheap, requires
block-diagonal `f`),
`ExactPointwiseADDiff`
(exact for any `f`, `O(n)`), or
`PatchQuadratureADDiff`
(exact within each patch, bounded cost, for a globally-mixing `f` at
full-domain scale)."""


def _resolve(ctx: Context, value: str | float) -> jnp.ndarray | float:
    """A term's coefficient/source config is either a `ctx` key (str) or a
    literal constant -- resolve it against `ctx` if it's a key. Mirrors
    `scirex.losses.pde_terms._resolve`."""
    return ctx[value] if isinstance(value, str) else value


def _field_fn(ctx: Context, key: str) -> FieldFn:
    """Fetch a `ctx` entry that must be the callable field, e.g. the `u_fn`
    being differentiated."""
    fn: FieldFn = ctx[key]
    return fn


def _coords_kwarg(
    differentiator: ADDifferentiator, coords: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """Whether to pass `coords` along to a `.derivative`/`.gradient` call at
    all, decided once per call site rather than every consumer needing to
    know which concrete differentiator it was given.

    A differentiator with `owns_coords` (currently only
    `PatchQuadratureADDiff`)
    always uses its own coordinates regardless of what's passed, so passing
    `ctx["coords"]` along would be redundant at best -- and, if a caller
    ever built `ctx["coords"]` from the wrong sample (a batch-loop
    mistake, see that class's own module docstring), silently wrong at
    worst, since a same-shaped different instance's coords aren't
    distinguishable by the shape check alone. Omitting it here removes that
    risk at the one place a PDE term actually talks to the differentiator,
    rather than leaving it up to every term implementation to remember.
    """
    return {} if differentiator.owns_coords else {"coords": coords}


class DiffusionTermAD:
    """`div(coeff * grad(field))`, in flux form.

    See the module docstring for why `coeff` accepts a plain `float` or
    a coordinate function `coeff_fn(coords) -> jnp.ndarray`, but not a
    precomputed array. A `float` (trivially constant) takes the cheap
    `coeff * laplacian(field)` path; a callable takes the true flux-form
    path, computed with nested AD: ``flux_d(c) = coeff_fn(c) *
    d(field)/d(c[..., d])`` for each dimension `d`, differentiated once
    more via the sum trick and summed over `dims` -- exactly the AD
    analogue of what
    `DiffusionTerm` does with
    `FiniteDiff`. Because `flux_d` is a genuine function of `c`,
    JAX's autodiff applies the product rule automatically when it's
    differentiated the second time, so this is correct whether or not
    `coeff_fn` actually varies -- it is not merely a fallback for the
    varying case.

    `d(field)/d(c[..., d])` inside `flux_d` is computed via
    `differentiator.nested_differentiator`, not `differentiator`
    itself -- this inner call happens from *inside* a closure
    `differentiator` is already differentiating (the outer
    `flux_d` derivative below it), and for
    `PatchQuadratureADDiff`
    specifically, calling back into the same instance for that would (and
    once did) fail, since by that point `c` is only one patch's worth of
    coordinates, not the full array that instance expects. Every
    `ADDifferentiator` exposes `.nested_differentiator` precisely so
    this term never needs to know which concrete differentiator was
    injected -- see
    `PatchQuadratureADDiff`'s
    own `nested_differentiator`.

    Similarly, `ctx["coords"]` is only forwarded to the *outer*
    `differentiator.derivative` call (the Laplacian path and the outer
    step of the flux-form path) when `differentiator.owns_coords` is
    `False` -- see `_coords_kwarg`. For a differentiator that owns
    its coordinates, `ctx["coords"]` is used only for the output array's
    shape/dtype, never actually passed in, so this term is unaffected even
    if `ctx["coords"]` does not match that differentiator's own coords.

    !!! warning
        A `coeff_fn` must be block-diagonal in `coords`, the exact same
        precondition `ADDiff`
        already documents for `field` -- point `i`'s coefficient must
        not depend on point `j`'s coordinates. This isn't checked here,
        for the same reason it isn't checked for `field`: there is no
        cheap way to detect non-block-diagonality from inside the
        primitive. `coeff_fn(coords)` is however checked for the same
        unbatched-output and shape requirements `field` gets.

    Parameters
    ----------
    dims : Sequence[int]
        coordinate dimensions to include in the divergence
    field : str, optional
        `ctx` key of the callable field being diffused, by default "u_fn"
    coeff : str, float, or Callable[[jnp.ndarray], jnp.ndarray], optional
        a diffusion coefficient -- a constant, a coordinate function, or a
        `ctx` key resolving to either of those -- by default 1.0. A
        `ctx` key resolving to a precomputed array raises; see the module
        docstring for why.
    differentiator : ADDifferentiator, optional
        which AD backend to differentiate `field` (and, for the flux-form
        path, `coeff_fn`) with, by default `ADDiff()`. Pass
        `ExactPointwiseADDiff`
        or `PatchQuadratureADDiff`
        instead whenever `field`/`coeff_fn` mix points (e.g. Transolver's
        attention) -- see `ADDifferentiator`.
    """

    def __init__(
        self,
        dims: Sequence[int],
        field: str = "u_fn",
        coeff: str | float | FieldFn = 1.0,
        differentiator: ADDifferentiator | None = None,
    ) -> None:
        self.dims = dims
        self.field = field
        self.coeff = coeff
        self._ad = differentiator if differentiator is not None else ADDiff()
        # The flux-form path (see _flux_form) needs a *second*, nested
        # derivative computed from inside a closure self._ad is already
        # differentiating. Calling back into self._ad itself for that would
        # break for a differentiator with state tied to one specific
        # coordinate array (PatchQuadratureADDiff) -- nested_differentiator
        # is the explicit, safe reference for that inner call instead. See
        # PatchQuadratureADDiff's module docstring for why this must be an
        # explicit reference rather than something inferred at call time.
        self._nested_ad = self._ad.nested_differentiator
        assert len(self.dims) > 0, "DiffusionTermAD requires at least one dim"
        assert len(set(self.dims)) == len(self.dims), (
            f"DiffusionTermAD got a repeated dim index in {list(self.dims)} -- "
            "each dimension must appear at most once, or the Laplacian "
            "silently double-counts it"
        )

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate the diffusion term at `ctx['coords']` using autodiff."""
        f = _field_fn(ctx, self.field)
        coords = ctx["coords"]
        coeff = _resolve(ctx, self.coeff)  # type: ignore[arg-type]

        if callable(coeff):
            return self._flux_form(f, coeff, coords)

        if not isinstance(coeff, int | float):
            raise TypeError(
                f"DiffusionTermAD coeff ({self.coeff!r}) resolved to a "
                f"precomputed array (type {type(coeff).__name__}), which "
                "isn't accepted -- AD can only differentiate a function of "
                "coords, not a fixed array with no traced relationship to "
                "coords, so flux-form can't run on one, and there's no "
                "jit-safe way to check whether it happens to be uniform. "
                "Pass a coeff_fn(coords) -> array callable instead (a "
                "function that returns a constant is a valid coeff_fn too)."
            )

        coords_kwarg = _coords_kwarg(self._ad, coords)
        laplacian = jnp.zeros(coords.shape[0], dtype=coords.dtype)
        for d in self.dims:
            laplacian = laplacian + self._ad.derivative(
                f, dim=d, order=2, **coords_kwarg
            )
        return coeff * laplacian

    def _flux_form(
        self, f: FieldFn, coeff_fn: FieldFn, coords: jnp.ndarray
    ) -> jnp.ndarray:
        """`div(coeff_fn(coords) * grad(field))` via nested AD, one
        dimension at a time."""
        coords_kwarg = _coords_kwarg(self._ad, coords)
        divergence = jnp.zeros(coords.shape[0], dtype=coords.dtype)
        for d in self.dims:

            def flux_d(c: jnp.ndarray, d: int = d) -> jnp.ndarray:
                du = self._nested_ad.derivative(f, c, dim=d, order=1)
                coeff_vals = coeff_fn(c)
                _check_unbatched(coeff_vals)
                _require_shape(coeff_vals, du.shape, "DiffusionTermAD coeff_fn(coords)")
                return coeff_vals * du

            divergence = divergence + self._ad.derivative(
                flux_d, dim=d, order=1, **coords_kwarg
            )
        return divergence


class AdvectionTermAD:
    """`velocity . grad(field)`.

    Like `DiffusionTermAD`, `ctx["coords"]` is only forwarded to
    `differentiator.gradient` when `differentiator.owns_coords` is
    `False` -- see `_coords_kwarg`.

    Parameters
    ----------
    dims : Sequence[int]
        coordinate dimensions to include
    field : str, optional
        `ctx` key of the callable field being advected, by default "u_fn"
    velocity : str, Sequence[str], float, or None, optional
        the advecting velocity, by default None

        * `None` -- self-advection: `field` advects itself along every
          dim (e.g. steady viscous Burgers' `u * u_x`), using `u_fn`
          evaluated (not differentiated) at `coords`
        * a single `ctx` key or constant -- that value advects along
          every dim
        * a sequence of `ctx` keys/constants, one per dim in `dims`
    differentiator : ADDifferentiator, optional
        which AD backend to differentiate `field` with, by default
        `ADDiff()` -- see `ADDifferentiator` and
        `DiffusionTermAD`'s `differentiator` parameter.
    """

    def __init__(
        self,
        dims: Sequence[int],
        field: str = "u_fn",
        velocity: str | Sequence[str | float] | float | None = None,
        differentiator: ADDifferentiator | None = None,
    ) -> None:
        self.dims = dims
        self.field = field
        self.velocity = velocity
        self._ad = differentiator if differentiator is not None else ADDiff()
        assert len(self.dims) > 0, "AdvectionTermAD requires at least one dim"
        assert len(set(self.dims)) == len(self.dims), (
            f"AdvectionTermAD got a repeated dim index in {list(self.dims)} -- "
            "each dimension must appear at most once, or its contribution "
            "silently double-counts"
        )

    def _velocity_components(
        self, ctx: Context, f: FieldFn, coords: jnp.ndarray
    ) -> list[jnp.ndarray | float]:
        if self.velocity is None:
            return [f(coords)] * len(self.dims)
        if isinstance(self.velocity, str) or isinstance(self.velocity, int | float):
            v = _resolve(ctx, self.velocity)
            return [v] * len(self.dims)
        return [_resolve(ctx, v) for v in self.velocity]

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate the advection term at `ctx['coords']` using autodiff."""
        f = _field_fn(ctx, self.field)
        coords = ctx["coords"]
        # Self-advection (velocity=None) evaluates f directly at coords, not
        # through the differentiator -- if the differentiator owns its
        # coords, derivs below is computed at self._ad.coords regardless of
        # ctx["coords"], so the self-advection velocity must be evaluated at
        # the same points, or the two would silently combine values from
        # different locations as if they were at the same one.
        eval_coords = (
            cast(PatchQuadratureADDiff, self._ad).coords
            if self._ad.owns_coords
            else coords
        )
        velocities = self._velocity_components(ctx, f, eval_coords)

        derivs = self._ad.gradient(f, dims=self.dims, **_coords_kwarg(self._ad, coords))
        total = jnp.zeros(eval_coords.shape[0], dtype=eval_coords.dtype)
        for d, v, du in zip(self.dims, velocities, derivs, strict=True):
            _require_shape(
                v, (eval_coords.shape[0],), f"AdvectionTermAD velocity for dim {d}"
            )
            total = total + v * du
        return total


class ReactionTermAD:
    """`coeff * field` -- direct evaluation of the callable field at
    `coords`, no differentiation at all.

    Parameters
    ----------
    field : str, optional
        `ctx` key of the callable field, by default "u_fn"
    coeff : str or float, optional
        `ctx` key of the reaction coefficient, or a constant, by default 1.0
    """

    def __init__(self, field: str = "u_fn", coeff: str | float = 1.0) -> None:
        self.field = field
        self.coeff = coeff

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate `coeff * field` at `ctx['coords']`."""
        f = _field_fn(ctx, self.field)
        coords = ctx["coords"]
        u = f(coords)
        _check_unbatched(u)
        coeff = _resolve(ctx, self.coeff)
        _require_shape(coeff, u.shape, f"ReactionTermAD coeff ({self.coeff!r})")
        return coeff * u


class SourceTermAD:
    """A constant or sampled forcing term `f` -- identical in spirit to
    `SourceTerm`; no function needed
    since the source is never itself differentiated.

    Parameters
    ----------
    value : str or float, optional
        `ctx` key of the (possibly per-point) source values, or a
        constant, by default 0.0
    """

    def __init__(self, value: str | float = 0.0) -> None:
        self.value = value

    def __call__(self, ctx: Context) -> jnp.ndarray | float:
        """Return the forcing value, resolved from `ctx` if it is a key."""
        return _resolve(ctx, self.value)
