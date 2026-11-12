# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Grid-based differentiation utilities for physics-informed losses.

Provides two ways to differentiate a field that is sampled on a regular
grid, matching the two "discretized" derivative methods described for PINO
(Li et al., 2021, https://arxiv.org/abs/2111.03794):

* `FiniteDiff` -- central-difference derivatives, with either
  periodic (`jnp.roll`-based) or non-periodic (one-sided boundary
  stencil) handling. Appropriate for non-periodic domains and/or
  non-smooth coefficient fields, e.g. the Darcy-flow permeability field.
* `FourierDiff` -- spectral derivatives via FFT. Appropriate for
  periodic, smooth domains, e.g. the Burgers and Navier-Stokes benchmarks,
  where it is exact up to floating-point precision rather than merely
  convergent with grid refinement.

Both classes operate purely on grid-valued arrays -- they do not know or
care which model produced the array. Any model can be paired with either
class as long as it can produce a regular-grid-shaped output (natively, for
grid models such as FNO/WNO, or by querying at grid coordinates, for
point-cloud models such as GINO/Transolver).

Input/output convention
------------------------
Every `derivative` call takes an array `u` of arbitrary rank and an
`axis` argument selecting *which* axis of `u` is the spatial (or
temporal) axis to differentiate along -- e.g. for a Burgers field of shape
`(batch, n_x, n_t)`, pass `axis=1` for `d/dx` and `axis=2` for
`d/dt`. The returned array has exactly the same shape as `u`: the
derivative at each point is computed from its neighbors along `axis`, and
no points are dropped.

There is no assumption about what any other axis represents -- `u` need
not carry a channels axis at all (none of the current callers in
`equation_losses` pass one; `DarcyEqnLoss` and
`NavierStokesEqnLoss` both slice their channel-last public input down to
one channel-less field per physical quantity, e.g. `y_pred[..., 0]`,
before it ever reaches these classes). If `u` does carry extra axes
(batch, channels, or otherwise), each slice along those axes is
differentiated independently and identically along `axis` -- the stencil
only ever combines neighboring values along `axis` itself, so nothing
about batch size or channel count changes the result per point.

Internally, `axis` is moved to the last position with `jnp.moveaxis`
before differencing, so the stencil logic is written once against axis
`-1` and generalizes to any axis/rank without a per-axis code path; the
result is moved back to its original position before returning.
"""

from collections.abc import Sequence

import jax.numpy as jnp


class FiniteDiff:
    """Central-difference derivatives of a field sampled on a regular grid.

    !!! note
        `axis`, `order`, and `periodic` must be static (plain Python
        values, not traced arrays) since they select which code path runs
        at trace time -- this matches how `d`/`p`/`reduction` are used
        as static configuration in
        `LpLoss`. If used inside
        `jax.jit`, mark them as `static_argnames`.
    """

    _NON_PERIODIC_MIN_POINTS = 4

    def derivative(
        self,
        u: jnp.ndarray,
        axis: int,
        order: int = 1,
        spacing: float = 1.0,
        periodic: bool = True,
        scheme: str = "central",
    ) -> jnp.ndarray:
        """Differentiate `u` along `axis`.

        Parameters
        ----------
        u : jnp.ndarray
            field values on a regular grid, arbitrary rank
        axis : int
            axis of `u` to differentiate along
        order : int, optional
            derivative order, 1 or 2, by default 1
        spacing : float, optional
            grid spacing `h` along `axis`, by default 1.0

            !!! note
                For a periodic axis sampled at `n` points over a domain
                of length `L` (no duplicated endpoint), use
                `h = L / n`. For a non-periodic axis sampled at `n`
                points including both endpoints, use `h = L / (n - 1)`.
        periodic : bool, optional
            whether `axis` is periodic, by default True

            * `periodic=True` uses a `jnp.roll`-based stencil at
              every point (wrapping around the boundary).
            * `periodic=False` uses the interior stencil in the interior
              and a matched-order one-sided stencil at the two boundary
              points along `axis`.
        scheme : str, optional
            one of `"central"`, `"forward"`, `"backward"`, by default
            `"central"`. Only `"central"` is supported for `order=2`
            (diffusion-type terms don't need directional biasing).
            `"forward"`/`"backward"` are 1st-order-accurate directional
            derivatives -- the building blocks for an upwind advection
            scheme (pick `"backward"` where the advecting velocity is
            positive, `"forward"` where it's negative), which trades
            central differencing's 2nd-order accuracy for stability on
            advection-dominated terms. At a non-periodic domain boundary,
            the direction that would need an out-of-domain neighbor falls
            back to the other one-sided difference instead (there is no
            other choice at the edge).

        Returns
        -------
        jnp.ndarray
            derivative of `u` along `axis`, same shape as `u`
        """
        if order not in (1, 2):
            raise ValueError(f"FiniteDiff only supports order in (1, 2), got {order}")
        if scheme not in ("central", "forward", "backward"):
            raise ValueError(
                'scheme must be one of ("central", "forward", "backward"), '
                f"got {scheme!r}"
            )
        if scheme != "central" and order != 1:
            raise ValueError(
                f"scheme={scheme!r} is only supported for order=1, got order={order}"
            )

        h = spacing
        u_axis = jnp.moveaxis(u, axis, -1)
        n = u_axis.shape[-1]

        if periodic:
            if scheme == "central":
                d_axis = self._periodic(u_axis, order, h)
            else:
                d_axis = self._periodic_directional(u_axis, h, scheme)
        else:
            if n < self._NON_PERIODIC_MIN_POINTS:
                raise ValueError(
                    "FiniteDiff with periodic=False needs at least "
                    f"{self._NON_PERIODIC_MIN_POINTS} points along `axis`, got {n}"
                )
            if scheme == "central":
                d_axis = self._non_periodic(u_axis, order, h)
            else:
                d_axis = self._non_periodic_directional(u_axis, h, scheme)

        return jnp.moveaxis(d_axis, -1, axis)

    @staticmethod
    def _periodic_directional(
        u_axis: jnp.ndarray, h: float, scheme: str
    ) -> jnp.ndarray:
        """1st-order forward/backward difference at every point, wrapping via roll."""
        if scheme == "forward":
            return (jnp.roll(u_axis, -1, axis=-1) - u_axis) / h
        return (u_axis - jnp.roll(u_axis, 1, axis=-1)) / h

    @staticmethod
    def _non_periodic_directional(
        u_axis: jnp.ndarray, h: float, scheme: str
    ) -> jnp.ndarray:
        """1st-order forward/backward difference; falls back to the other
        one-sided difference at whichever boundary point lacks the neighbor
        the requested direction needs.
        """
        diffs = (u_axis[..., 1:] - u_axis[..., :-1]) / h  # length n-1
        if scheme == "forward":
            # index i uses u[i+1]-u[i] for i=0..n-2; index n-1 has no i+1,
            # so it falls back to the backward difference u[n-1]-u[n-2].
            return jnp.concatenate([diffs, diffs[..., -1:]], axis=-1)
        # backward: index i uses u[i]-u[i-1] for i=1..n-1; index 0 has no
        # i-1, so it falls back to the forward difference u[1]-u[0].
        return jnp.concatenate([diffs[..., :1], diffs], axis=-1)

    @staticmethod
    def _periodic(u_axis: jnp.ndarray, order: int, h: float) -> jnp.ndarray:
        """Central difference at every point, wrapping via `jnp.roll`."""
        u_fwd = jnp.roll(u_axis, -1, axis=-1)
        u_bwd = jnp.roll(u_axis, 1, axis=-1)
        if order == 1:
            return (u_fwd - u_bwd) / (2.0 * h)
        return (u_fwd - 2.0 * u_axis + u_bwd) / (h**2)

    @staticmethod
    def _non_periodic(u_axis: jnp.ndarray, order: int, h: float) -> jnp.ndarray:
        """Central difference in the interior; one-sided stencil at the boundary.

        The one-sided stencils match the order of accuracy of the interior
        central difference (2nd-order accurate), so accuracy doesn't drop
        at the domain boundary.
        """
        if order == 1:
            interior = (u_axis[..., 2:] - u_axis[..., :-2]) / (2.0 * h)
            left = (
                -11 * u_axis[..., 0]
                + 18 * u_axis[..., 1]
                - 9 * u_axis[..., 2]
                + 2 * u_axis[..., 3]
            ) / (6.0 * h)
            right = (
                -2 * u_axis[..., -4]
                + 9 * u_axis[..., -3]
                - 18 * u_axis[..., -2]
                + 11 * u_axis[..., -1]
            ) / (6.0 * h)
        else:
            interior = (
                u_axis[..., 2:] - 2.0 * u_axis[..., 1:-1] + u_axis[..., :-2]
            ) / (h**2)
            left = (
                2 * u_axis[..., 0]
                - 5 * u_axis[..., 1]
                + 4 * u_axis[..., 2]
                - u_axis[..., 3]
            ) / (h**2)
            right = (
                2 * u_axis[..., -1]
                - 5 * u_axis[..., -2]
                + 4 * u_axis[..., -3]
                - u_axis[..., -4]
            ) / (h**2)

        return jnp.concatenate(
            [left[..., jnp.newaxis], interior, right[..., jnp.newaxis]], axis=-1
        )


class FourierDiff:
    """Spectral derivatives (via FFT) of a field sampled on a periodic grid.

    Differentiation in physical space becomes multiplication by `(i*k)` per
    Fourier mode, where `k` is the angular wavenumber -- exact for smooth
    periodic functions up to floating-point precision, rather than merely
    convergent with grid refinement as `FiniteDiff` is.

    !!! note
        `axis` and `order` must be static (plain Python values, not
        traced arrays); mark them as `static_argnames` if used inside
        `jax.jit`.
    """

    def derivative(
        self,
        u: jnp.ndarray,
        axis: int,
        order: int = 1,
        spacing: float = 1.0,
    ) -> jnp.ndarray:
        """Differentiate `u` along a periodic `axis` via FFT.

        Parameters
        ----------
        u : jnp.ndarray
            field values on a regular, periodic grid, arbitrary rank
        axis : int
            axis of `u` to differentiate along
        order : int, optional
            derivative order, by default 1. Any non-negative integer is
            supported (unlike `FiniteDiff`, which only supports 1
            and 2), since spectral differentiation is just repeated
            multiplication by `(i*k)` in frequency space.
        spacing : float, optional
            grid spacing `h` along `axis`, by default 1.0. For `n`
            points sampled over a periodic domain of length `L` (no
            duplicated endpoint), use `h = L / n`.

        Returns
        -------
        jnp.ndarray
            derivative of `u` along `axis`, same shape as `u`, dtype
            matching `u` (the imaginary part left over from floating-point
            round-off in the inverse FFT is discarded)
        """
        if order < 0:
            raise ValueError(f"FourierDiff requires order >= 0, got {order}")
        if not jnp.issubdtype(u.dtype, jnp.floating):
            raise ValueError(
                f"FourierDiff requires a floating-point dtype, got {u.dtype}. "
                "The result is cast back to u's dtype, so an integer dtype "
                "would silently truncate the derivative -- cast u to float "
                "first."
            )

        u_axis = jnp.moveaxis(u, axis, -1)
        n = u_axis.shape[-1]

        # angular wavenumbers k, ordered to match jnp.fft's frequency layout
        # dtype pinned per CONTRIBUTING.md -- without it, fftfreq falls back
        # to JAX's default float dtype, which silently becomes float64 if
        # jax_enable_x64 is ever active on the import path (e.g. flipped on
        # by tensorly, as spectral_convolution.py already guards against)
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=spacing, dtype=jnp.float32)

        u_hat = jnp.fft.fft(u_axis, axis=-1)
        d_hat = u_hat * (1j * k) ** order
        d_axis = jnp.fft.ifft(d_hat, axis=-1).real.astype(u_axis.dtype)

        return jnp.moveaxis(d_axis, -1, axis)

    def gradient(
        self,
        u: jnp.ndarray,
        axes: Sequence[int],
        spacing: Sequence[float] | float = 1.0,
    ) -> list[jnp.ndarray]:
        """First derivative of `u` along each of `axes`.

        Convenience wrapper for computing e.g. `(du/dx, du/dy)` in one
        call -- used by multi-dimensional PDE residuals (Phase 1) that need
        several partial derivatives of the same field.

        Parameters
        ----------
        u : jnp.ndarray
            field values on a regular, periodic grid
        axes : Sequence[int]
            axes of `u` to differentiate along
        spacing : float or Sequence[float], optional
            grid spacing per axis in `axes`, by default 1.0 (either a
            single value applied to every axis, or one value per axis)

        Returns
        -------
        list[jnp.ndarray]
            `[du/d(axes[0]), du/d(axes[1]), ...]`, each the same shape as `u`
        """
        if isinstance(spacing, int | float):
            spacing = [spacing] * len(axes)
        return [
            self.derivative(u, axis=ax, order=1, spacing=h)
            for ax, h in zip(axes, spacing, strict=True)
        ]
