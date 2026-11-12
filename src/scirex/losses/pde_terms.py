# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Reusable PDE-term building blocks for grid-based physics-informed losses.

A PDE residual is a signed sum of a handful of recurring term *kinds* --
transient, advection, diffusion, reaction, source -- with some terms absent
(zero) depending on the equation. Rather than one bespoke class per PDE
re-deriving its own flux/derivative logic, each term kind here is a small,
independent, reusable building block on top of
`FiniteDiff` /
`FourierDiff`. A specific PDE
loss (e.g. `DarcyEqnLoss`) is then just a *particular combination* of
these terms with signs -- see
`PDEResidualLoss`.

Context-dict interface
-----------------------
Every term is called as `term(ctx)`, where `ctx` is a dict of named
fields (e.g. `{"u": predicted_field, "a": coefficient_field}`) assembled
by the composite loss that owns the terms. This keeps a term agnostic to
which specific PDE loss is using it -- a term only needs to know the
*names* of the fields it reads from `ctx`, not where they came from.
Every term in this module only ever reads `jnp.ndarray | float` values
out of `ctx`, but `Context` itself is typed as `dict[str, Any]`,
since `PDEResidualLoss` is deliberately reused unmodified by
`ad_pde_terms`, whose context dicts also carry
callables (the field being differentiated).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .differentiation import FiniteDiff, FourierDiff

Context = dict[str, Any]


@dataclass(frozen=True)
class AxisSpec:
    """Everything a term needs to differentiate along one axis of a field.

    Parameters
    ----------
    axis : int
        axis of the field to differentiate along
    differentiator : FiniteDiff or FourierDiff
        which differentiation primitive to use for this axis -- `FiniteDiff`
        for non-periodic and/or non-smooth axes (e.g. Darcy's spatial axes),
        `FourierDiff` for periodic, smooth axes (e.g. Burgers'/Navier-Stokes'
        spatial axis)
    measure : float, optional
        physical domain length along this axis, by default 1.0, must be > 0
    periodic : bool, optional
        whether this axis is periodic, by default True. Only meaningful for
        a `FiniteDiff` axis (`FourierDiff` always assumes periodic).
    """

    axis: int
    differentiator: FiniteDiff | FourierDiff
    measure: float = 1.0
    periodic: bool = True

    def __post_init__(self) -> None:
        if self.measure <= 0:
            raise ValueError(
                f"AxisSpec requires measure > 0, got {self.measure} for axis "
                f"{self.axis} -- a non-positive measure makes spacing() "
                "return 0 (or negative), which FiniteDiff/FourierDiff would "
                "then silently turn into inf/nan derivatives"
            )

    def spacing(self, u: jnp.ndarray) -> float:
        """Grid spacing along this axis, given a field sampled on it.

        A periodic axis with `n` points spans its full `measure` without
        a duplicated endpoint (`h = measure / n`); a non-periodic axis
        samples both endpoints (`h = measure / (n - 1)`).
        """
        n = u.shape[self.axis]
        return self.measure / n if self.periodic else self.measure / (n - 1)

    def derivative(
        self, u: jnp.ndarray, order: int = 1, scheme: str = "central"
    ) -> jnp.ndarray:
        """Differentiate `u` along this axis, dispatching to the configured
        differentiator with this axis's spacing/periodicity.
        """
        h = self.spacing(u)
        if isinstance(self.differentiator, FourierDiff):
            if self.periodic is False:
                raise ValueError(
                    "FourierDiff assumes periodic axes -- set `periodic=True` "
                    f"for axis {self.axis} or use FiniteDiff instead"
                )
            if scheme != "central":
                raise ValueError(
                    "FourierDiff has no directional bias -- scheme must be "
                    f"'central' for a spectral axis, got {scheme!r}"
                )
            return self.differentiator.derivative(
                u, axis=self.axis, order=order, spacing=h
            )
        return self.differentiator.derivative(
            u,
            axis=self.axis,
            order=order,
            spacing=h,
            periodic=self.periodic,
            scheme=scheme,
        )


def _resolve(ctx: Context, value: str | float) -> jnp.ndarray | float:
    """A term's coefficient/field config is either a `ctx` key (str) or a
    literal constant -- resolve it against `ctx` if it's a key.
    """
    return ctx[value] if isinstance(value, str) else value


def _field(ctx: Context, key: str) -> jnp.ndarray:
    """Fetch a `ctx` entry that must be a full array (not a scalar
    constant), e.g. the field being differentiated -- as opposed to
    `_resolve`, which a coefficient/source is allowed to be a plain float.
    """
    return jnp.asarray(ctx[key])


def _require_shape(value: object, expected_shape: tuple[int, ...], label: str) -> None:
    """Raise if `value` is array-valued (has a `.shape`, including a 0-d
    array) and its shape doesn't exactly match `expected_shape`.

    A genuine Python `int`/`float` (no `.shape` attribute at all) is
    exempt and always allowed to broadcast -- matching `Context`'s own
    `dict[str, jnp.ndarray | float]` convention, where a `jnp.ndarray`
    entry is expected to be a real field aligned with the rest of the
    computation, and a `float` entry is expected to be a genuine scalar
    constant.

    Without this, a field accidentally reduced/collapsed along an axis
    (even all the way down to a 0-d array) silently broadcasts against a
    correctly-shaped field instead of raising -- broadcasting a size-1 (or
    0-d) axis is valid by design in JAX/NumPy, so nothing else in the
    chain would ever catch it, and the result comes out wrong with no
    exception anywhere.
    """
    shape = getattr(value, "shape", None)
    if shape is not None and shape != expected_shape:
        raise ValueError(
            f"{label} has shape {shape}, expected {expected_shape} -- "
            "broadcasting would silently combine them anyway, which "
            "usually means the field was accidentally reduced/collapsed "
            "along an axis"
        )


_DIRECTIONAL_SCHEMES = ("central", "forward", "backward")


def _validate_directional_scheme(order: int, scheme: str, label: str) -> None:
    """Shared fail-fast check for the terms that forward `order`/`scheme`
    straight through to `AxisSpec.derivative` (as opposed to
    `AdvectionTerm`, which accepts the term-specific "upwind" instead
    of a raw scheme). Mirrors `FiniteDiff.derivative`'s own two checks, so an
    invalid combination is caught at construction time instead of on the
    first call, deep inside `FiniteDiff`.
    """
    if scheme not in _DIRECTIONAL_SCHEMES:
        raise ValueError(
            f"{label} scheme must be one of {_DIRECTIONAL_SCHEMES}, got {scheme!r}"
        )
    if scheme != "central" and order != 1:
        raise ValueError(
            f"{label} scheme={scheme!r} is only supported for order=1, "
            f"got order={order}"
        )


class DiffusionTerm:
    """`div(coeff * grad(field))`, in flux form.

    `flux = coeff * grad(field)` is computed first (one derivative per
    axis), and only `flux` -- not `coeff` -- is differentiated again to
    get the divergence. This avoids ever taking a finite difference of
    `coeff` directly, which is important when `coeff` is a rough or
    discontinuous field (e.g. Darcy's permeability).

    Parameters
    ----------
    axes : Sequence[AxisSpec]
        one entry per spatial axis to include in the divergence, must be non-empty
    field : str, optional
        `ctx` key of the field being diffused, by default "u"
    coeff : str or float, optional
        `ctx` key of the (possibly spatially-varying) diffusion
        coefficient, or a constant, by default 1.0
    order : int, optional
        derivative order applied at both the flux step (`grad(field)`)
        and the divergence step, by default 1 -- together giving the usual
        2nd-order-overall diffusion operator. Passed straight through to
        each axis's `derivative` call, so any order the configured
        differentiator supports works here too -- e.g. `order=2` with a
        `FourierDiff` axis gives, per axis, ``d^2(coeff * d^2 field /
        d(axis)^2) / d(axis)^2``, summed over `axes`. This is a per-axis
        higher-order damping term, not the standard isotropic
        biharmonic/hyperdiffusion operator -- it has no cross-axis terms
        (e.g. no `d^4/dx^2dy^2`), so it's directionally anisotropic
        whenever more than one axis is included.
    scheme : str, optional
        finite-difference scheme applied at both the flux step and the
        divergence step, one of `"central"`, `"forward"`, `"backward"`,
        by default `"central"`. Passed straight through to each axis's
        `derivative` call -- `"forward"`/`"backward"` only combine with
        `order=1`, since `FiniteDiff` itself rejects a directional
        scheme at any other order, and a `FourierDiff` axis only accepts
        `"central"` (see `AxisSpec`).
    """

    def __init__(
        self,
        axes: Sequence[AxisSpec],
        field: str = "u",
        coeff: str | float = 1.0,
        order: int = 1,
        scheme: str = "central",
    ) -> None:
        _validate_directional_scheme(order, scheme, "DiffusionTerm")
        self.axes = axes
        self.field = field
        self.coeff = coeff
        self.order = order
        self.scheme = scheme
        assert len(self.axes) > 0, "DiffusionTerm requires at least one axis"
        axis_indices = [ax.axis for ax in self.axes]
        assert len(set(axis_indices)) == len(axis_indices), (
            f"DiffusionTerm got a repeated axis index in {axis_indices} -- "
            "each spatial axis must appear at most once, or the divergence "
            "silently double-counts it"
        )

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate `div(coeff * grad(field))` for the fields in `ctx`."""
        u = _field(ctx, self.field)
        coeff = _resolve(ctx, self.coeff)
        _require_shape(coeff, u.shape, f"DiffusionTerm coeff ({self.coeff!r})")
        divergence = jnp.zeros_like(u)
        for ax in self.axes:
            grad_i = ax.derivative(u, order=self.order, scheme=self.scheme)
            flux_i = coeff * grad_i
            divergence = divergence + ax.derivative(
                flux_i, order=self.order, scheme=self.scheme
            )
        return divergence


class AdvectionTerm:
    """`velocity . grad(field)`.

    Parameters
    ----------
    axes : Sequence[AxisSpec]
        one entry per spatial axis to include, must be non-empty
    field : str, optional
        `ctx` key of the field being advected, by default "u"
    velocity : str, Sequence[str], or None, optional
        the advecting velocity, by default None

        * `None` -- self-advection: `field` advects itself along every
          axis (e.g. Burgers' `u * u_x`)
        * a single `ctx` key -- that field advects along every axis
        * a sequence of `ctx` keys, one per axis in `axes` -- e.g. the
          two velocity components `(u, v)` of a 2D advection term
    scheme : str, optional
        `"central"` (2nd-order accurate, can oscillate on
        advection-dominated/near-shock terms) or `"upwind"` (1st-order
        accurate, picks the `FiniteDiff` "backward" stencil where the
        local velocity is >= 0 and "forward" where it's negative --
        biasing the stencil toward where the advected quantity is
        physically coming *from*, which is what makes it stable), by
        default "central". `"upwind"` requires every axis in `axes` to
        use a `FiniteDiff`
        axis (`FourierDiff` has no directional stencil to pick between).
    order : int, optional
        derivative order applied along each axis, by default 1 (the usual
        `velocity . grad(field)` meaning of "advection"). Passed straight
        through to each axis's `derivative` call, so any order the
        configured differentiator supports works here too -- e.g. a
        dispersion-like term `v * u_xxx` via `order=3` with a
        `FourierDiff` axis. `scheme="upwind"` only combines with
        `order=1` -- `FiniteDiff` itself rejects a directional scheme at
        any other order.
    """

    def __init__(
        self,
        axes: Sequence[AxisSpec],
        field: str = "u",
        velocity: str | Sequence[str] | None = None,
        scheme: str = "central",
        order: int = 1,
    ) -> None:
        if scheme not in ("central", "upwind"):
            raise ValueError(
                f'scheme must be one of ("central", "upwind"), got {scheme!r}'
            )
        if scheme == "upwind" and order != 1:
            raise ValueError(
                f"AdvectionTerm scheme='upwind' is only supported for order=1, "
                f"got order={order}"
            )
        self.axes = axes
        self.field = field
        self.velocity = velocity
        self.scheme = scheme
        self.order = order
        assert len(self.axes) > 0, "AdvectionTerm requires at least one axis"
        axis_indices = [ax.axis for ax in self.axes]
        assert len(set(axis_indices)) == len(axis_indices), (
            f"AdvectionTerm got a repeated axis index in {axis_indices} -- "
            "each spatial axis must appear at most once, or its "
            "contribution silently double-counts"
        )

    def _velocity_components(
        self, ctx: Context, u: jnp.ndarray
    ) -> list[jnp.ndarray | float]:
        if self.velocity is None:
            return [u] * len(self.axes)
        if isinstance(self.velocity, str):
            return [ctx[self.velocity]] * len(self.axes)
        return [ctx[name] for name in self.velocity]

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate `velocity . grad(field)` for the fields in `ctx`."""
        u = _field(ctx, self.field)
        velocities = self._velocity_components(ctx, u)

        total = jnp.zeros_like(u)
        for ax, v in zip(self.axes, velocities, strict=True):
            _require_shape(v, u.shape, f"AdvectionTerm velocity for axis {ax.axis}")
            if self.scheme == "central":
                du = ax.derivative(u, order=self.order, scheme="central")
            else:
                forward = ax.derivative(u, order=self.order, scheme="forward")
                backward = ax.derivative(u, order=self.order, scheme="backward")
                du = jnp.where(v >= 0, backward, forward)
            total = total + v * du
        return total


class GradientTerm:
    """`d(field)/d(axis)` -- one component of a scalar field's gradient.

    Deliberately just one axis at a time (e.g. one call for `p_x`, another
    for `p_y`), rather than returning a full gradient vector, since a PDE
    residual needs each component summed into a *different* equation (the
    x-momentum residual gets `p_x`, y-momentum gets `p_y`) -- unlike
    `DiffusionTerm`/`DivergenceTerm`, which sum every axis's
    contribution into one residual.

    Mathematically identical to `TransientTerm` (both are just
    `axis.derivative(field, order=1)`); kept as a separate class so a
    term's name still tells you which PDE role it plays -- a time
    derivative vs. a spatial pressure-gradient component.

    Parameters
    ----------
    axis : AxisSpec
        the axis to differentiate along
    field : str, optional
        `ctx` key of the field being differentiated, by default "u"
    order : int, optional
        derivative order, by default 1 (`d(field)/d(axis)`). Passed
        straight through to `axis.derivative`, so it accepts whatever
        order the configured differentiator supports.
    scheme : str, optional
        finite-difference scheme, one of `"central"`, `"forward"`,
        `"backward"`, by default `"central"`. Passed straight through to
        `axis.derivative` -- `"forward"`/`"backward"` only combine with
        `order=1`, since `FiniteDiff` itself rejects a directional
        scheme at any other order, and a `FourierDiff` axis only accepts
        `"central"` (see `AxisSpec`).
    """

    def __init__(
        self,
        axis: AxisSpec,
        field: str = "u",
        order: int = 1,
        scheme: str = "central",
    ) -> None:
        _validate_directional_scheme(order, scheme, "GradientTerm")
        self.axis = axis
        self.field = field
        self.order = order
        self.scheme = scheme

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate the derivative of `field` along the term's axis."""
        return self.axis.derivative(
            _field(ctx, self.field), order=self.order, scheme=self.scheme
        )


class DivergenceTerm:
    """`sum_i d(field_i)/d(axis_i)` -- divergence of a vector field given
    componentwise, e.g. the incompressible-flow continuity residual
    `div(velocity) = u_x + v_y`.

    Unlike `DiffusionTerm`, there is no coefficient here: each
    component's derivative is summed directly, since continuity has no
    coefficient to multiply by.

    Parameters
    ----------
    fields : Sequence[str]
        `ctx` key of the vector field's component along each axis in
        `axes` (e.g. `["u", "v"]` for a 2D velocity field), must be the
        same length as `axes`
    axes : Sequence[AxisSpec]
        one entry per component in `fields`, must be non-empty
    order : int, optional
        derivative order applied to each component, by default 1 (the
        usual `sum_i d(field_i)/d(axis_i)` meaning of "divergence").
        Passed straight through to each axis's `derivative` call.
    scheme : str, optional
        finite-difference scheme applied to each component, one of
        `"central"`, `"forward"`, `"backward"`, by default
        `"central"`. Passed straight through to each axis's
        `derivative` call -- `"forward"`/`"backward"` only combine
        with `order=1`, since `FiniteDiff` itself rejects a directional
        scheme at any other order, and a `FourierDiff` axis only accepts
        `"central"` (see `AxisSpec`).
    """

    def __init__(
        self,
        fields: Sequence[str],
        axes: Sequence[AxisSpec],
        order: int = 1,
        scheme: str = "central",
    ) -> None:
        _validate_directional_scheme(order, scheme, "DivergenceTerm")
        assert len(axes) > 0, "DivergenceTerm requires at least one axis"
        assert len(fields) == len(axes), (
            f"DivergenceTerm got {len(fields)} fields but {len(axes)} axes -- "
            "each field needs exactly one axis to differentiate it along"
        )
        axis_indices = [ax.axis for ax in axes]
        assert len(set(axis_indices)) == len(axis_indices), (
            f"DivergenceTerm got a repeated axis index in {axis_indices} -- "
            "each spatial axis must appear at most once, or the divergence "
            "silently double-counts it"
        )
        self.fields = fields
        self.axes = axes
        self.order = order
        self.scheme = scheme

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate the divergence of the vector field given by `fields`."""
        u0 = _field(ctx, self.fields[0])
        total = jnp.zeros_like(u0)
        for field, ax in zip(self.fields, self.axes, strict=True):
            component = _field(ctx, field)
            _require_shape(component, u0.shape, f"DivergenceTerm field {field!r}")
            total = total + ax.derivative(
                component, order=self.order, scheme=self.scheme
            )
        return total


class TransientTerm:
    """`d(field)/dt` along a single (typically non-periodic) time axis.

    Parameters
    ----------
    axis : AxisSpec
        the time axis
    field : str, optional
        `ctx` key of the field, by default "u"
    order : int, optional
        how many times the field is differentiated in time, by default 1
        (`du/dt`). Set to 2 for `d^2u/dt^2` -- the time derivative a wave
        equation needs, since it contains a second time derivative rather
        than a first one. This is the differentiation order, not the
        numerical accuracy of the stencil approximating it (that's
        `scheme` below). Passed straight through to `axis.derivative`,
        so it accepts whatever order the configured differentiator supports.
    scheme : str, optional
        finite-difference scheme, one of `"central"`, `"forward"`,
        `"backward"`, by default `"central"`. Passed straight through
        to `axis.derivative` -- `"forward"`/`"backward"` only combine
        with `order=1`, since `FiniteDiff` itself rejects a directional
        scheme at any other order, and a `FourierDiff` axis only accepts
        `"central"` (see `AxisSpec`).
    """

    def __init__(
        self,
        axis: AxisSpec,
        field: str = "u",
        order: int = 1,
        scheme: str = "central",
    ) -> None:
        _validate_directional_scheme(order, scheme, "TransientTerm")
        self.axis = axis
        self.field = field
        self.order = order
        self.scheme = scheme

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate the time derivative of `field`."""
        return self.axis.derivative(
            _field(ctx, self.field), order=self.order, scheme=self.scheme
        )


class ReactionTerm:
    """`coeff * field`.

    Parameters
    ----------
    field : str, optional
        `ctx` key of the field, by default "u"
    coeff : str or float, optional
        `ctx` key of the (possibly spatially-varying) reaction
        coefficient, or a constant, by default 1.0
    """

    def __init__(self, field: str = "u", coeff: str | float = 1.0) -> None:
        self.field = field
        self.coeff = coeff

    def __call__(self, ctx: Context) -> jnp.ndarray:
        """Evaluate `coeff * field` for the fields in `ctx`."""
        u = _field(ctx, self.field)
        coeff = _resolve(ctx, self.coeff)
        _require_shape(coeff, u.shape, f"ReactionTerm coeff ({self.coeff!r})")
        return coeff * u


class SourceTerm:
    """A constant or spatially-varying forcing term `f`.

    Parameters
    ----------
    value : str or float, optional
        `ctx` key of the (possibly spatially-varying) source field, or a
        constant, by default 0.0
    """

    def __init__(self, value: str | float = 0.0) -> None:
        self.value = value

    def __call__(self, ctx: Context) -> jnp.ndarray | float:
        """Return the forcing value, resolved from `ctx` if it is a key."""
        return _resolve(ctx, self.value)


Term = Callable[[Context], "jnp.ndarray | float"]
