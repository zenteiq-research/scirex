# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Patch + Gauss-quadrature AD: bounded-cost exact derivatives for
globally-mixing models (e.g. Transolver).

`ExactPointwiseADDiff`
is exact for any `f`, including one that mixes every point together (full
self-attention), but costs `O(n)` backward passes, each a *full*
evaluation of `f` -- `O(n^3)` total if `f` itself costs `O(n^2)` per
call. That is not affordable at full-domain scale.

This module bounds that cost with a geometric trick, not an approximation
in the differentiation itself: split the domain into small patches
(`quadrature`), and call the *full* model closure
`f` separately on each patch's own Gauss-Legendre quadrature coordinates
only. Every patch's `f` call now mixes only over that patch's own (few)
points, so `ExactPointwiseADDiff` is affordable *within* a patch even
though it would not be affordable over the whole domain. Because
`integral(Omega) = sum_e integral(Omega_e)` exactly, for any partition
(see `quadrature`'s module docstring), assembling
per-patch results this way loses nothing beyond each patch's own quadrature
error -- the same exactness argument that module already relies on.

Collocation-only scope
------------------------
`PatchQuadratureADDiff` owns its own quadrature coordinates
(`self.coords`, `self.weights`), built once at construction time from
`bounds`/`n_patches_per_axis`/`n`. This only makes sense in the
physics-loss-only (self-supervised, collocation-style) training regime,
where query coordinates are ours to choose freely -- a patch's quadrature
nodes are real, literal Gauss-Legendre points fed directly as the model's
input coordinates for that patch. The hybrid case (physics loss blended
with a data loss at fixed dataset point locations, which can't be relocated
to quadrature nodes) is not supported here.

Skipping fully-excluded patches
-----------------------------------
For an irregular-geometry domain (see
`domain_membership`), a patch entirely inside the
excluded region contributes nothing once its weights are masked, but by
default it is still built and evaluated through the model before that
happens. A patch that never appears in `patches` in the first place
never costs a forward pass at all. Pass a pre-culled list via the
`patches` constructor argument (see
`cull_excluded_patches`) to
skip those patches entirely, rather than building the full uniform grid and
relying on masking alone.

`jax.vmap` over patches, not a Python loop
--------------------------------------------------
`derivative`/`gradient` call `f` once per patch, but do so via
a single `jax.vmap` over a `(n_patches, points_per_patch, coord_dim)`
reshape of `coords`, not a Python `for` loop unrolled at trace time.
This matters for compile time, not runtime or correctness: a Python loop
over patches makes `jax.jit` trace and compile `n_patches` separately
unrolled copies of the full model's forward+backward graph, and that
compile cost was found to grow sharply non-linearly with patch count, not
just proportionally -- on the real ShapeNetCar-scale Transolver config
(`scirex.config.models.TransolverShapeNetCar`, ~3.86M parameters), compiling 8
patches this way took 90.9 seconds; the identical computation (same 8
patches, same points, same model, same parameters) restructured as one
`jax.vmap` call instead took 5.2 seconds, about 17.5x faster, verified to
produce bit-identical results (max abs difference ~2e-7, floating-point
noise) against the loop-based version it replaces. Every patch having the
same `points_per_patch` (fixed by construction, since `n` is one shared
quadrature order for every patch, culled or not) is exactly what
`jax.vmap` needs and what a Python loop over unevenly-shaped patches
would not have allowed.

Bounding memory: `patch_batch_size`
----------------------------------------
The `jax.vmap` above fixes compile time, but real GPU measurements (real
ShapeNetCar-scale Transolver, `scirex.config.models.TransolverShapeNetCar`)
surfaced a cost it does not remove: batching `n_patches` patches together
also batches `ExactPointwiseADDiff`'s own internal
`jax.vmap` over `points_per_patch` one-hot cotangents (see
`exact_ad_differentiation.py`'s `_all_dims`). Nesting these means the
*effective* batch size pushed through the whole model's backward pass at
once is `n_patches * points_per_patch`, not `n_patches` -- a Python loop
never pays this (it fully finishes and frees one patch before starting the
next), so this cost was invisible until measured with vmap on a real model
on GPU: at 125 patches (27 points/patch, 3375 effective batch) the step
itself was already the slowest part measured (9.3s per step, versus 0.47s
at 8 patches); at 216 patches (5832 effective batch) it did not just get
slower, XLA requested 100GB and the run died with
`RESOURCE_EXHAUSTED`. This is the real, opposite-direction tradeoff of the
compile-time fix above: unrolling a Python loop is slow to compile but cheap
in peak memory (one patch resident at a time); one flat `jax.vmap` is fast
to compile but has peak memory proportional to the *whole* batch at once.

`patch_batch_size` (optional, `None` by default) lets a caller bound
that peak without giving up the compile-time fix: when set, `derivative`/
`gradient` route the per-patch call through ``jax.lax.map(..., batch_size=
patch_batch_size)`` instead of a bare `jax.vmap`. `jax.lax.map` with a
`batch_size` still only traces/compiles the per-chunk computation *once*
(via `jax.vmap` internally) and then ``lax.scan``s over chunks -- so
compile cost stays governed by chunk size, not total patch count, same as
the plain `jax.vmap` path. Leaving it at the default `None` keeps
today's behavior (a single, unchunked `jax.vmap` over every patch at
once) exactly as before -- this is purely opt-in, chosen by the caller based
on how much device memory is actually available, not a value this module
can pick correctly on its own (it depends on the model size, the accelerator,
and how much other memory is in use for that training run).

`patch_batch_size` alone bounds the *forward* pass's peak batch size, but
does not by itself bound memory when this whole computation sits inside a
further, outer differentiation w.r.t. the model's own parameters -- exactly
the normal training setup (`jax.grad` of a physics loss w.r.t. params, see
the benchmark scripts). Reverse-mode autodiff through `jax.lax.scan`
(what a chunked `jax.lax.map` uses internally) does not discard each
chunk's forward residuals once used the way a true streaming/sequential
loop would -- by default it stacks them, one slot per scan step, so the
outer `jax.grad` can still recover them during its backward sweep. That
stacked storage scales with *total* chunk count, not per-chunk batch size,
so on its own, chunking bounds the size of any *one* batched forward/backward
call but not the cumulative memory the outer gradient needs to hold onto
across every chunk -- confirmed empirically: reducing `patch_batch_size`
from unset down to 16 did not resolve a real GPU `RESOURCE_EXHAUSTED` at
216 patches (the real ShapeNetCar-scale Transolver config), because the
outer `jax.grad` over model params was still the one driving retention.

`checkpoint_patches` (optional, `False` by default) fixes this the
standard way: gradient checkpointing (`jax.checkpoint`, a.k.a.
`jax.remat`). When `True`, the per-patch call is wrapped in
`jax.checkpoint` before being handed to `jax.vmap`/`jax.lax.map`, so
the outer differentiation does not retain that call's internal forward
residuals at all -- it *recomputes* the forward pass during its own
backward sweep instead of having saved it the first time. This trades
additional forward-pass compute (each patch's model forward effectively
runs twice: once during the ordinary forward pass, once again during the
outer backward pass) for peak memory that no longer accumulates across
chunks/patches. This is the standard JAX technique for exactly this
situation (a nested per-example/per-patch computation living inside an
outer training-step gradient) -- see JAX's own gradient-checkpointing
documentation -- not something specific to quadrature or patches. Combine
it with `patch_batch_size` for the full effect: chunking bounds any one
batched call's size, checkpointing stops the outer gradient from retaining
every chunk's residuals simultaneously. Verified here to produce
bit-identical results to leaving it `False` (checkpointing changes *when*
values get recomputed, never *what* is computed); the memory reduction
itself was reasoned from JAX's documented scan/checkpoint semantics and has
not been independently profiled against the exact GPU OOM it targets.

Because the coordinates are owned by this object rather than supplied
per-call, `coords` is optional on `derivative`/`gradient` --
omit it (the recommended usage) and `self.coords` is used directly, with
nothing for a caller to get wrong. Passing `coords` explicitly is still
supported, for interface parity with
`ADDiff` and
`ExactPointwiseADDiff`
(both stateless, so *they* require it every call) and for existing call
sites -- see `_check_coords` for what gets validated in that case, and
why it can only ever be a shape check, not an identity one.

Why the optional form exists, not just the check
-----------------------------------------------------
The precomputed per-patch slice boundaries (`_patch_slices`) are only
valid for the exact array this instance built. A shape check alone cannot
tell "the wrong array" apart from "a different instance's own coords that
happens to have the same total point count" -- e.g. two
`PatchQuadratureADDiff` instances built with the same `n`/patch
count but different `bounds` (a realistic setup: one instance per sample
in a dataset like ShapeNetCar, all sharing the same quadrature resolution,
each with its own domain). A caller-side mistake (a batch loop that
associates the wrong sample's coordinates with the wrong instance) would
silently compute derivatives sliced with the wrong patch boundaries -- no
exception, wrong numbers. Fixing this by making the check itself smarter is
not possible while staying `jax.jit`-compatible: identity (`is`) breaks
under tracing (see `_check_coords`), and shape is the only thing that
survives it. Omitting `coords` sidesteps the check entirely instead of
trying to make it stricter -- there is no separate array for a caller to
mismatch, in any context (eager, `jit`, `grad`, `vmap`), since none is
ever passed. This only closes the gap for direct calls to this class,
though; `ad_pde_terms`'s PDE terms call
whichever differentiator they were given the same way regardless of type
(so that they never need to know which concrete differentiator they have),
which means they still pass `coords` explicitly today, to stay compatible
with the stateless differentiators too. Removing `coords` from this
class's signature entirely, rather than defaulting it, would close the gap
there as well, but needs that shared calling convention reworked first --
tracked as a follow-up, not done here.

Nested differentiation without recursing back through this class
--------------------------------------------------------------------
A coefficient that must itself be differentiated (e.g. Darcy's
`div(a(x) * grad(u))`) needs a *second*, nested derivative computed from
*inside* a closure that a PDE term is already differentiating (see
`DiffusionTermAD`'s flux-form
path). That closure only ever runs on one patch's own coordinates -- a
smaller array than `self.coords` -- so calling straight back into this
same instance's `derivative`/`gradient` for that nested step
would (and, before this was fixed, did) trip the shape check above, since
those methods only ever expect the full array.

The fix is not to loosen that check -- a shape-only comparison cannot tell
"this is a legitimate nested call" apart from "this is simply the wrong
array, coincidentally the same shape" (e.g. a second
`PatchQuadratureADDiff` instance's own coordinates), and silently
guessing wrong there would compute confidently incorrect derivatives with
no error at all. Instead, `nested_differentiator` gives any caller
that needs a nested derivative step an explicit, separate reference to use
for it: `_exact` here, bypassing the per-patch wrapper entirely,
since a single patch's points are already a complete, self-contained input
for the exact differentiator, no further splitting needed. Callers ask for
this by name (`differentiator.nested_differentiator`) rather than this
class trying to infer it from whatever array shows up, so
`derivative`/`gradient` themselves never need to guess, and
keep rejecting anything that isn't genuinely `self.coords`.

Weights are deliberately not consumed here
---------------------------------------------
`weights` (`ref_weights * patch.jacobian` for every point, from
`patch_quadrature_coords`) is exposed
for the caller to use when reducing a per-point residual into the scalar
domain-integral loss (`jnp.sum(weights * residual**2)`, or similar). This
module never touches `weights` itself -- computing `d(u_i)/d(coords_i)`
has nothing to do with how the result is later weighted and summed, so
keeping that reduction out of the differentiator keeps this module's
responsibility limited to exactly what its name says: derivatives.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from .ad_differentiation import FieldFn
from .exact_ad_differentiation import ExactPointwiseADDiff
from .quadrature import (
    Patch,
    gauss_legendre_points_weights,
    make_patches,
    patch_quadrature_coords,
)


class PatchQuadratureADDiff:
    """Exact per-point derivatives of `f(coords)`, bounded-cost via
    per-patch evaluation, for models that mix points arbitrarily (e.g.
    Transolver's attention).

    Same `.derivative`/`.gradient`/`.nested_differentiator` interface
    as `ADDiff` and
    `ExactPointwiseADDiff`,
    so it is a drop-in alternative wherever the cost of the fully exact,
    full-domain `ExactPointwiseADDiff` is not affordable.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float]]
        `(lo, hi)` per dimension of the rectangular domain, e.g.
        `[(0.0, 1.0), (0.0, 1.0)]` for the unit square -- forwarded to
        `make_patches`. The number of
        dimensions (`dim`) is inferred from `len(bounds)`.
    n_patches_per_axis : Sequence[int]
        how many equal-width patches to cut each axis into -- forwarded to
        `make_patches`
    n : int
        quadrature points per axis, per patch -- forwarded to
        `gauss_legendre_points_weights`
    patches : Sequence[Patch], optional
        use this exact patch list instead of the full uniform grid
        `bounds`/`n_patches_per_axis` would otherwise produce --
        `bounds`/`n_patches_per_axis` are still required (to fix
        `dim` and validate consistency) but `make_patches` is not
        called when this is given. Meant for
        `cull_excluded_patches`'s
        output: a patch entirely inside an excluded region never needs its
        own quadrature points built at all, let alone evaluated through the
        model, so building the full grid first and dropping some of it here
        avoids the forward passes those patches would otherwise cost. Every
        patch must have the same `dim` as `len(bounds)`; nothing else
        about `patches` is checked against `bounds` (e.g. a culled
        patch's own extent is still whatever `make_patches` gave it)
    patch_batch_size : int, optional
        if given, bounds peak memory in `derivative`/`gradient`
        by routing the per-patch call through ``jax.lax.map(..., batch_size=
        patch_batch_size)`` instead of one unchunked `jax.vmap` over every
        patch at once -- see the module docstring's "Bounding memory:
        patch_batch_size" section for why this is needed and what it costs.
        `None` (the default) keeps the original unchunked `jax.vmap`
        behavior. Must be a positive integer if given.
    checkpoint_patches : bool, optional
        if `True`, wraps the per-patch call in `jax.checkpoint` so an
        *outer* differentiation (e.g. `jax.grad` of a training loss w.r.t.
        model params) recomputes each patch's forward pass during its
        backward sweep instead of retaining every patch/chunk's residuals at
        once -- see the module docstring's "Bounding memory:
        patch_batch_size" section for why `patch_batch_size` alone does
        not bound this. `False` by default (today's behavior).

    Attributes
    ----------
    coords : jnp.ndarray, shape (total_points, dim)
        every patch's physical-space quadrature coordinates, concatenated in
        patch order -- feed this directly as the model's query coordinates
        (e.g. Transolver's `pos`/`x`)
    weights : jnp.ndarray, shape (total_points,)
        the matching quadrature weight for each row of `coords`
        (`ref_weights * patch.jacobian`, already patch-Jacobian-scaled --
        see `patch_quadrature_coords`),
        for the caller to use when reducing a per-point residual into a
        scalar domain-integral loss
    """

    owns_coords = True
    """Stateful: this instance always uses its own `coords`, so a
    caller-agnostic consumer (e.g.
    `ad_pde_terms`'s PDE terms) can check this
    flag once and skip passing `coords` at all for this differentiator,
    rather than every consumer needing to special-case which concrete
    differentiator it was given. See
    `owns_coords` for
    the stateless differentiators, where this is `False` and `coords`
    genuinely must be supplied every call."""

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        n_patches_per_axis: Sequence[int],
        n: int,
        patches: Sequence[Patch] | None = None,
        patch_batch_size: int | None = None,
        checkpoint_patches: bool = False,
    ) -> None:
        if patch_batch_size is not None and patch_batch_size < 1:
            raise ValueError(
                "PatchQuadratureADDiff: patch_batch_size must be a positive "
                f"integer if given, got {patch_batch_size}"
            )
        self.patch_batch_size = patch_batch_size
        self.checkpoint_patches = checkpoint_patches
        dim = len(bounds)
        if patches is None:
            self.patches = make_patches(bounds, n_patches_per_axis)
        else:
            for patch in patches:
                if patch.dim != dim:
                    raise ValueError(
                        f"PatchQuadratureADDiff: patches must have dim={dim} "
                        f"(from len(bounds)), got a patch with dim={patch.dim}"
                    )
            self.patches = list(patches)
        ref_points, ref_weights = gauss_legendre_points_weights(n, dim)

        coords_list = []
        weights_list = []
        patch_slices = []
        start = 0
        for patch in self.patches:
            coords, weights = patch_quadrature_coords(patch, ref_points, ref_weights)
            coords_list.append(coords)
            weights_list.append(weights)
            end = start + coords.shape[0]
            patch_slices.append((start, end))
            start = end

        self.coords = jnp.concatenate(coords_list, axis=0)
        self.weights = jnp.concatenate(weights_list, axis=0)
        self._patch_slices = patch_slices
        self._exact = ExactPointwiseADDiff()

    @property
    def nested_differentiator(self) -> ExactPointwiseADDiff:
        """The differentiator to use for a *further* nested differentiation
        call made from inside a closure that this instance is already
        differentiating (see the module docstring's "Nested differentiation
        without recursing back through this class" section) -- `self._exact`,
        bypassing the per-patch wrapper entirely, since by the time such a
        call happens, `coords` is already a single patch's worth of
        points, a complete input for the exact differentiator with nothing
        left to split.
        """
        return self._exact

    def _check_coords(self, coords: jnp.ndarray) -> None:
        """Only reached when a caller passes `coords` explicitly instead
        of omitting it (see `derivative`) -- `coords` must then be
        (shape-compatible with) `self.coords`, since the precomputed
        per-patch slice boundaries are only valid for the exact array this
        instance built. A nested differentiation call must go through
        `nested_differentiator` instead of calling back into this
        instance -- see the module docstring.

        Checked by shape, not by Python object identity (`is`): under
        `jax.jit`, an array closed over as `self.coords` and the *same*
        array also passed in as an argument stop being the same object --
        the argument gets replaced by a tracer at the trace boundary, so an
        identity check would spuriously fail for the single most common way
        this class is actually used (inside a jitted loss function). Shape
        alone cannot distinguish this instance's own coords from a
        different `PatchQuadratureADDiff` instance's coords that
        happen to have the same total point count -- omitting `coords`
        (the recommended usage) avoids this ambiguity entirely rather than
        trying to detect it; see the module docstring's "Why the optional
        form exists" section.
        """
        if coords.shape != self.coords.shape:
            raise ValueError(
                "PatchQuadratureADDiff.derivative/gradient requires coords "
                f"to be this instance's own self.coords, shape "
                f"{self.coords.shape}, got shape {coords.shape} -- the "
                "precomputed per-patch slicing is only valid for the "
                "coordinates this instance generated. A nested "
                "differentiation call (e.g. a variable diffusion "
                "coefficient's own gradient) should use "
                "differentiator.nested_differentiator instead of calling "
                "back into this instance."
            )

    def _resolve_coords(self, coords: jnp.ndarray | None) -> jnp.ndarray:
        """Shared by `derivative`/`gradient`: omitted `coords`
        (the recommended usage) resolves straight to `self.coords`, no
        check needed since nothing was passed to get wrong. Explicit
        `coords` still gets today's shape check, plus a warning -- shape
        alone cannot tell this instance's own coords apart from a different,
        same-shaped instance's (see `_check_coords`), so passing
        anything here is inherently the weaker path, even on the calls
        where what was passed happens to be correct.
        """
        if coords is None:
            return self.coords
        self._check_coords(coords)
        warnings.warn(
            "PatchQuadratureADDiff.derivative/gradient was called with "
            "coords passed explicitly. Omit it instead (the recommended "
            "usage) so self.coords is used directly -- passing it explicitly "
            "is only checked by shape, which cannot distinguish this "
            "instance's own coords from a different PatchQuadratureADDiff "
            "instance's coords that happen to have the same total point "
            "count (e.g. one instance per sample in a dataset like "
            "ShapeNetCar, all built with the same n_patches_per_axis/n). "
            "See the module docstring's 'Why the optional form exists' "
            "section.",
            stacklevel=3,
        )
        return coords

    def _map_over_patches(
        self, per_patch: Callable[[jnp.ndarray], Any], reshaped: jnp.ndarray
    ) -> Any:
        """Applies `per_patch` (a function of one patch's own coordinates,
        shape `(points_per_patch, coord_dim)`) across the leading
        `n_patches` axis of `reshaped`, stacking results exactly like
        `jax.vmap` would -- but via ``jax.lax.map(..., batch_size=
        self.patch_batch_size)`` when that is set, to bound peak memory
        instead of batching every patch at once. See the module docstring's
        "Bounding memory: patch_batch_size" section for why this exists and
        what it costs relative to the unchunked default. When
        `checkpoint_patches` is set, `per_patch` is additionally
        wrapped in `jax.checkpoint` first, so an outer differentiation
        recomputes rather than retains each call's forward residuals -- see
        the same docstring section for why `patch_batch_size` alone does
        not bound memory under an outer `jax.grad`.
        """
        if self.checkpoint_patches:
            per_patch = jax.checkpoint(per_patch)
        if self.patch_batch_size is None:
            return jax.vmap(per_patch)(reshaped)
        return jax.lax.map(per_patch, reshaped, batch_size=self.patch_batch_size)

    def derivative(
        self,
        f: FieldFn,
        coords: jnp.ndarray | None = None,
        *,
        dim: int,
        order: int = 1,
    ) -> jnp.ndarray:
        """`d^order u / d(coords[..., dim])^order` at every point, exact
        within each patch, at bounded cost.

        `f` is called once per patch, on only that patch's own rows of
        `coords` -- never on the full domain at once -- so its cost is
        governed by the patch size, not the total number of points.

        Parameters
        ----------
        f : Callable[[jnp.ndarray], jnp.ndarray]
            maps a *patch's* coordinates (shape `(patch_points, coord_dim)`)
            to the field value at each of those points (shape
            `(patch_points,)`); may mix points within the patch arbitrarily
        coords : jnp.ndarray, optional
            omit this (recommended) to use `self.coords` directly -- there
            is then no separate array a caller could mismatch, in any
            context. If given explicitly, it must be this instance's own
            `coords` (checked by shape -- see `_check_coords`);
            for a nested differentiation call, use
            `nested_differentiator` instead of passing anything here
        dim : int
            which coordinate dimension to differentiate along
        order : int, optional
            derivative order, 1 or 2, by default 1 -- same semantics as
            `derivative`

        Returns
        -------
        jnp.ndarray, shape (total_points,)
            derivative at every point, in the same order as `coords`
        """
        coords = self._resolve_coords(coords)
        n_patches = len(self._patch_slices)
        pts_per_patch = self._patch_slices[0][1] - self._patch_slices[0][0]
        reshaped = coords.reshape(n_patches, pts_per_patch, coords.shape[-1])

        def per_patch(patch_coords: jnp.ndarray) -> jnp.ndarray:
            return self._exact.derivative(f, patch_coords, dim=dim, order=order)

        return self._map_over_patches(per_patch, reshaped).reshape(-1)

    def gradient(
        self, f: FieldFn, coords: jnp.ndarray | None = None, *, dims: Sequence[int]
    ) -> list[jnp.ndarray]:
        """First derivative of `f` along each of `dims`, exact within
        each patch, at bounded cost -- one
        `gradient`
        call per patch (all requested dims from one sweep, per patch),
        rather than one call to `derivative` per dim.

        Parameters
        ----------
        f : Callable[[jnp.ndarray], jnp.ndarray]
            as in `derivative`
        coords : jnp.ndarray, optional
            omit this (recommended) to use `self.coords` directly -- see
            `derivative` for the full explanation; if given explicitly
            it must be this instance's own `coords`
        dims : Sequence[int]
            coordinate dimensions to differentiate along

        Returns
        -------
        list[jnp.ndarray]
            `[du/d(coords[...,dims[0]]), du/d(coords[...,dims[1]]), ...]`,
            each shape `(total_points,)`
        """
        coords = self._resolve_coords(coords)
        n_patches = len(self._patch_slices)
        pts_per_patch = self._patch_slices[0][1] - self._patch_slices[0][0]
        reshaped = coords.reshape(n_patches, pts_per_patch, coords.shape[-1])

        def per_patch(patch_coords: jnp.ndarray) -> list[jnp.ndarray]:
            return self._exact.gradient(f, patch_coords, dims=dims)

        per_dim = self._map_over_patches(per_patch, reshaped)
        return [g.reshape(-1) for g in per_dim]
