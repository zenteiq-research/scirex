# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""PDE-residual ("physics-informed") losses for grid-based neural operators.

Each loss in this module measures how well a model's *prediction* satisfies
a governing PDE, using the `FiniteDiff`
/ `FourierDiff` primitives -- as
opposed to `LpLoss` /
`H1Loss`, which measure agreement
with a ground-truth *target*.

Input/output convention
------------------------
Unlike the rest of the `scirex` package (which follows the reference
PyTorch `neuralop` channel-*first* convention, e.g.
`(batch, channels, H, W)`), losses in this module fix their own,
model-independent convention: **channels-last**,
`(batch, s_1, ..., s_d, channels)`. This lets the same loss be reused
across models with different native layouts (e.g. FNO's channel-first
output vs. WNO's channel-last output) -- the caller is responsible for
transposing a model's raw output into this layout before calling the loss.
"""

from collections.abc import Sequence

import jax.numpy as jnp

from .differentiation import FiniteDiff
from .pde_terms import (
    AdvectionTerm,
    AxisSpec,
    Context,
    DiffusionTerm,
    DivergenceTerm,
    GradientTerm,
    SourceTerm,
    Term,
    _require_shape,
)


def _is_scalar(value: object) -> bool:
    """True for a plain Python int/float, or a 0-d array-like (e.g. a 0-d
    jax/numpy array). `isinstance(value, int | float)` alone misses the
    latter, even though a 0-d array behaves like a scalar everywhere else.
    """
    if isinstance(value, int | float):
        return True
    return getattr(value, "ndim", None) == 0


class PDEResidualLoss:
    """A PDE residual assembled as a signed sum of reusable term primitives.

    Rather than each PDE loss re-deriving its own flux/derivative logic,
    `pde_terms` provides a handful of reusable
    term *kinds* (diffusion, advection, transient, reaction, source); a
    specific PDE loss (e.g. `DarcyEqnLoss`) is just a particular
    signed combination of them -- terms that are absent for a given PDE are
    simply not included, rather than computed and zeroed out.

    Parameters
    ----------
    terms : dict[str, tuple[float, Term]]
        name -> (sign, term) for each active term, must be non-empty. The
        residual is `sum(sign * term(ctx) for sign, term in terms.values())`.
    reduction : str, optional
        whether to reduce the squared residual across all remaining axes by
        summing ("sum") or averaging ("mean"), by default "mean"
    name : str, optional
        human-readable identifier, by default "PDEResidualLoss"
    """

    def __init__(
        self,
        terms: dict[str, tuple[float, Term]],
        reduction: str = "mean",
        name: str = "PDEResidualLoss",
    ) -> None:
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions, (
            f"error: expected `reduction` to be one of {allowed_reductions}, "
            f"got {reduction}"
        )
        if not terms:
            raise ValueError(
                "PDEResidualLoss requires at least one term in `terms` -- an "
                "empty dict would silently give a residual of 0.0 everywhere "
                "(a loss that always reports the PDE as satisfied)"
            )
        self.terms = terms
        self.reduction = reduction
        self._name = name

    @property
    def name(self) -> str:
        """str: Human-readable identifier."""
        return self._name

    def residual(self, ctx: Context) -> jnp.ndarray:
        """Pointwise residual: the signed sum of every active term.

        A genuine Python `float`/`int` term (e.g. a constant
        `SourceTerm`) is allowed to broadcast against an
        array-valued term -- that's intentional. But two *array*-valued
        terms with different shapes -- including a term collapsed all the
        way down to a 0-d array -- broadcasting against each other almost
        always means one term's output is the wrong shape (e.g. a
        missing/extra batch dim, or a field accidentally reduced along an
        axis), so that case raises via `_require_shape` instead of
        silently broadcasting.
        """
        total: jnp.ndarray | float = 0.0
        ref_shape: tuple[int, ...] | None = None
        ref_name: str | None = None
        for name, (sign, term) in self.terms.items():
            value = term(ctx)
            shape = getattr(value, "shape", None)
            if shape is not None:
                if ref_shape is None:
                    ref_shape, ref_name = shape, name
                else:
                    _require_shape(
                        value,
                        ref_shape,
                        f"PDEResidualLoss term {name!r} (reference: term {ref_name!r})",
                    )
            total = total + sign * value
        return total  # type: ignore[return-value]

    def _reduce_all(self, val: jnp.ndarray) -> jnp.ndarray:
        """Reduce across all remaining axes according to `self.reduction`."""
        if self.reduction == "sum":
            return jnp.sum(val)
        return jnp.mean(val)

    def abs(self, ctx: Context, take_root: bool = True) -> jnp.ndarray:
        """Scalar residual norm: the (optionally root-ed) reduced squared residual."""
        val = self.residual(ctx) ** 2
        val = self._reduce_all(val)
        if take_root:
            val = val**0.5
        return jnp.squeeze(val)


class DarcyEqnLoss:
    """PDE-residual loss for the 2D Darcy-flow equation.

    Measures how well a predicted pressure field `u` satisfies

    $$
    -\\nabla \\cdot (a \\nabla u) = f, \\quad x \\in (0, 1)^2,
    \\quad u = 0 \\text{ on } \\partial(0,1)^2,
    $$

    the standard Darcy-flow benchmark (Li et al., *Fourier Neural Operator
    for Parametric Partial Differential Equations*), where `a` is the
    (possibly rough/discontinuous) permeability field and `f` is a
    constant forcing term (`f = 1` by convention).

    The divergence is computed in **flux form**: `flux = a * grad(u)` is
    computed first (one `FiniteDiff` call per spatial axis), and only
    `flux` -- not `a` itself -- is differentiated again to get the
    divergence. This avoids ever taking a finite difference of the rough
    coefficient field `a` directly, which would be numerically poor for a
    non-smooth field.

    !!! note
        **Dirichlet boundary handling.** By default (`enforce_dirichlet=True`),
        the boundary rows/columns of the predicted field are overwritten with
        the known exact value (`u=0` on every edge, the standard Darcy-flow
        convention) before differentiating, rather than trusting the model's
        own (as-yet-unconstrained) boundary output. This makes the PDE
        residual numerically correct near the boundary, but it also means
        **no gradient flows back to the model's boundary predictions through
        this loss** (the overwritten values are constants, not a function of
        the model's output). This is only safe when some *other* loss in the
        training objective (e.g.
        `LpLoss` /
        `H1Loss` against paired
        ground-truth data, which already encodes the true boundary values)
        supervises the boundary directly. If this loss is ever used without
        such a data loss (e.g. training data-free), set
        `enforce_dirichlet=False`, or the model will never learn the
        boundary condition.

        If the exact boundary value isn't `0` (a non-standard Darcy
        variant, or a manufactured-solution test with a nonzero boundary),
        set `boundary_value` -- either one constant applied to all four
        edges, or a per-edge dict (see below) if the edges genuinely differ.

    Parameters
    ----------
    measure : float, 0-d array, or Sequence[float], optional
        physical domain length per spatial dimension, by default 1.0 (the
        standard unit-square Darcy domain), either a single (scalar or 0-d
        array) value applied to both spatial dims, or a length-2 sequence
        with one value per dim
    reduction : str, optional
        whether to reduce the squared residual across the batch and spatial
        dims by summing ("sum") or averaging ("mean"), by default "mean"
    forcing : float, optional
        the (constant) forcing term `f`, by default 1.0 (the standard
        Darcy-flow benchmark convention)
    enforce_dirichlet : bool, optional
        whether to hard-constrain the boundary of the predicted field to
        `boundary_value` before differentiating, by default True. See the
        note above.
    boundary_value : float, 0-d array, or dict[str, float], optional
        the known Dirichlet value(s) enforced when `enforce_dirichlet=True`,
        by default 0.0 (the standard Darcy-flow convention). Either a single
        (scalar or 0-d array) constant applied to all four edges, or a dict
        giving a separate constant per edge with exactly the keys
        `"x_min"`, `"x_max"`, `"y_min"`, `"y_max"` (`u`'s axis 1 is
        x, axis 2 is y, matching the `AxisSpec` convention used elsewhere
        in this class). At the four corners, where an x-edge and a y-edge
        overlap, the y-edge value is the one that ends up applied.
    diffusion_order : int, optional
        derivative order used for both the flux (`grad(u)`) and the
        divergence step of the diffusion term, by default 1 -- together
        giving the standard 2nd-order-overall Darcy diffusion operator
        `div(a * grad(u))`. Forwarded straight to
        `DiffusionTerm`; only change
        this for a non-standard, higher-order flux variant of the equation.
    diffusion_scheme : str, optional
        finite-difference scheme used for both the flux and the divergence
        step of the diffusion term, one of `"central"`, `"forward"`,
        `"backward"`, by default `"central"` (2nd-order accurate, the
        standard choice for a diffusion term, which has no preferred
        direction to bias toward). Forwarded straight to
        `DiffusionTerm`.
        `"forward"`/`"backward"` only combine with
        `diffusion_order=1`.
    """

    _MIN_GRID_POINTS = 4
    _EDGE_KEYS = ("x_min", "x_max", "y_min", "y_max")

    def __init__(
        self,
        measure: float | jnp.ndarray | Sequence[float] = 1.0,
        reduction: str = "mean",
        forcing: float = 1.0,
        enforce_dirichlet: bool = True,
        boundary_value: float | jnp.ndarray | dict[str, float] = 0.0,
        diffusion_order: int = 1,
        diffusion_scheme: str = "central",
    ) -> None:
        """Configure the Darcy equation loss. See the class docstring for parameters."""
        self.d = 2

        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions, (
            f"error: expected `reduction` to be one of {allowed_reductions}, "
            f"got {reduction}"
        )
        self.reduction = reduction

        self.measure: Sequence[float]
        if _is_scalar(measure):
            self.measure = [float(measure)] * self.d  # type: ignore[arg-type]
        else:
            if len(measure) != self.d:  # type: ignore[arg-type]
                raise ValueError(
                    f"DarcyEqnLoss measure must have exactly {self.d} entries "
                    f"(one per spatial dim), got {len(measure)}"  # type: ignore[arg-type]
                )
            self.measure = [float(m) for m in measure]  # type: ignore[union-attr]

        self.forcing = forcing
        self.enforce_dirichlet = enforce_dirichlet
        self.boundary_value = self._resolve_boundary_value(boundary_value)
        self.diffusion_order = diffusion_order
        self.diffusion_scheme = diffusion_scheme
        self._fd = FiniteDiff()

        axes = [
            AxisSpec(
                axis=1, differentiator=self._fd, measure=self.measure[0], periodic=False
            ),
            AxisSpec(
                axis=2, differentiator=self._fd, measure=self.measure[1], periodic=False
            ),
        ]
        self._pde = PDEResidualLoss(
            terms={
                "diffusion": (
                    -1.0,
                    DiffusionTerm(
                        axes=axes,
                        field="u",
                        coeff="a",
                        order=diffusion_order,
                        scheme=diffusion_scheme,
                    ),
                ),
                "source": (-1.0, SourceTerm("f")),
            },
            reduction=reduction,
            name="DarcyEqnLoss",
        )

    @property
    def name(self) -> str:
        """str: Human-readable identifier."""
        return self._pde.name

    @staticmethod
    def _to_channel_last(field: jnp.ndarray) -> jnp.ndarray:
        if field.ndim == 4 and field.shape[1] < field.shape[-1]:
            return jnp.moveaxis(field, 1, -1)
        return field

    def prepare_eval_inputs(
        self, y_pred: jnp.ndarray, sample: dict
    ) -> tuple[jnp.ndarray, dict]:
        """Prepare Trainer eval inputs for this loss's channel-last contract."""
        prepared = dict(sample)
        y_pred = self._to_channel_last(y_pred)
        if "x" in prepared:
            x = self._to_channel_last(prepared["x"])
            prepared["x"] = x[..., :1]
        return y_pred, prepared

    def eval_batch_loss(
        self, y_pred: jnp.ndarray, x: jnp.ndarray | None = None, **kwargs: object
    ) -> jnp.ndarray:
        """Return unrooted `sum(residual**2)` for cross-batch eval aggregation."""
        del kwargs
        if x is None:
            raise ValueError("DarcyEqnLoss requires the permeability field `x`.")
        return jnp.sum(self.residual(y_pred, x) ** 2)

    def finalize_eval_metric(
        self, accumulated_loss: float | jnp.ndarray, n_samples: int
    ) -> float:
        """Convert accumulated squared residuals to one global per-sample RMS."""
        return float((jnp.asarray(accumulated_loss) / max(n_samples, 1)) ** 0.5)

    def _grid_spacing(self, u: jnp.ndarray) -> tuple[float, float]:
        """Grid spacing per spatial axis, assuming both endpoints are sampled
        (non-periodic Dirichlet grid): `h = measure / (n_points - 1)`.
        """
        n_x, n_y = u.shape[1], u.shape[2]
        for n in (n_x, n_y):
            if n < self._MIN_GRID_POINTS:
                raise ValueError(
                    "DarcyEqnLoss needs at least "
                    f"{self._MIN_GRID_POINTS} grid points per spatial axis, got {n}"
                )
        return self.measure[0] / (n_x - 1), self.measure[1] / (n_y - 1)

    @classmethod
    def _resolve_boundary_value(
        cls, boundary_value: float | jnp.ndarray | dict[str, float]
    ) -> dict[str, float]:
        """A single constant applies to all four edges; a dict must give
        exactly the four edges explicitly (no silent per-edge default), so a
        forgotten edge fails loudly instead of quietly staying at 0.
        """
        if _is_scalar(boundary_value):
            return dict.fromkeys(cls._EDGE_KEYS, float(boundary_value))  # type: ignore[arg-type]

        got = set(boundary_value)  # type: ignore[arg-type]
        expected = set(cls._EDGE_KEYS)
        if got != expected:
            raise ValueError(
                "DarcyEqnLoss boundary_value dict must have exactly the keys "
                f"{cls._EDGE_KEYS}, got {sorted(got)}"
            )
        return {k: float(v) for k, v in boundary_value.items()}  # type: ignore[union-attr]

    def _apply_dirichlet(self, u: jnp.ndarray) -> jnp.ndarray:
        """Overwrite the boundary rows/columns of `u` with the configured
        Dirichlet value(s). x-edges are applied before y-edges, so at the
        four corners the y-edge value is the one that sticks.
        """
        bv = self.boundary_value
        u = u.at[:, 0, :].set(bv["x_min"])
        u = u.at[:, -1, :].set(bv["x_max"])
        u = u.at[:, :, 0].set(bv["y_min"])
        u = u.at[:, :, -1].set(bv["y_max"])
        return u

    def _context(self, y_pred: jnp.ndarray, x: jnp.ndarray) -> Context:
        """Validate shapes, apply the Dirichlet constraint, and assemble the
        context dict the underlying PDEResidualLoss's terms read from.
        """
        if y_pred.shape[-1] != 1 or x.shape[-1] != 1:
            raise ValueError(
                "DarcyEqnLoss expects single-channel (batch, n_x, n_y, 1) "
                f"inputs, got y_pred.shape={y_pred.shape}, x.shape={x.shape}"
            )
        if y_pred.shape[0] != x.shape[0]:
            raise ValueError(
                "DarcyEqnLoss expects y_pred and x to share the same batch "
                f"size, got y_pred.shape={y_pred.shape}, x.shape={x.shape} "
                "(a size-1 batch would otherwise silently broadcast against "
                "the other instead of raising)"
            )

        u = y_pred[..., 0]
        a = x[..., 0]

        if self.enforce_dirichlet:
            u = self._apply_dirichlet(u)

        self._grid_spacing(u)  # validates grid size; AxisSpec derives its own spacing

        return {"u": u, "a": a, "f": self.forcing}

    def residual(self, y_pred: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Pointwise Darcy PDE residual, `-div(a * grad(u)) - f`.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted pressure field `u`, shape `(batch, n_x, n_y, 1)`
        x : jnp.ndarray
            permeability field `a` (the model's own input), shape
            `(batch, n_x, n_y, 1)`

        Returns
        -------
        jnp.ndarray
            residual, shape `(batch, n_x, n_y)`
        """
        return self._pde.residual(self._context(y_pred, x))

    def abs(
        self, y_pred: jnp.ndarray, x: jnp.ndarray, take_root: bool = True
    ) -> jnp.ndarray:
        """Scalar PDE-residual norm: the (optionally root-ed) reduced squared residual.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted pressure field, shape `(batch, n_x, n_y, 1)`
        x : jnp.ndarray
            permeability field, shape `(batch, n_x, n_y, 1)`
        take_root : bool, optional
            whether to take the square root of the reduced squared residual,
            by default True
        """
        return self._pde.abs(self._context(y_pred, x), take_root=take_root)

    def __call__(
        self,
        y_pred: jnp.ndarray,
        x: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
        **kwargs: object,
    ) -> jnp.ndarray:
        """Return the Darcy PDE-residual loss, `abs(y_pred, x)`.

        Matches the `loss(out, **sample)` calling convention used by
        `Trainer` -- `x` (the permeability field) is read from the batch
        dict; `y` (the ground-truth target) is accepted for interface
        compatibility but unused, since this loss doesn't need a target.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted pressure field, shape `(batch, n_x, n_y, 1)`
        x : jnp.ndarray
            permeability field (required)
        y : jnp.ndarray, optional
            unused; accepted for interface compatibility with `Trainer`

        Returns
        -------
        jnp.ndarray
            scalar PDE-residual loss
        """
        del y, kwargs
        if x is None:
            raise ValueError("DarcyEqnLoss requires the permeability field `x`.")
        return self.abs(y_pred, x)


class NavierStokesEqnLoss:
    """PDE-residual loss for steady, incompressible 2D Navier-Stokes in
    primitive (velocity-pressure) form -- the lid-driven-cavity benchmark.

    Measures how well a predicted `(u, v, p)` field satisfies, on the unit
    square with a no-slip wall on three sides and a moving lid on the
    fourth:

    $$
    \\begin{aligned}
    u u_x + v u_y + p_x - \\tfrac{1}{Re}(u_{xx} + u_{yy}) &= 0 \\\\
    u v_x + v v_y + p_y - \\tfrac{1}{Re}(v_{xx} + v_{yy}) &= 0 \\\\
    u_x + v_y &= 0
    \\end{aligned}
    $$

    Unlike `DarcyEqnLoss` (one equation, one `PDEResidualLoss`),
    this is three *coupled* equations, so it holds three independent
    `PDEResidualLoss` instances internally -- momentum-x, momentum-y,
    continuity -- and combines their three scalar norms into one loss value.

    !!! note
        **No-slip + lid boundary handling.** By default
        (`enforce_dirichlet=True`), the boundary rows/columns of the
        predicted `u`/`v` fields are overwritten before differentiating:
        `u=lid_velocity, v=0` on the lid (the `y_max` edge -- axis 2's
        last index, matching `DarcyEqnLoss`'s axis convention:
        `u`'s axis 1 is x, axis 2 is y), and `u=v=0` on the other three
        edges. As with `DarcyEqnLoss`, this means no gradient flows
        back to the model's own boundary predictions through this loss, so
        it is only safe when some other loss (e.g. a data loss against
        paired ground truth) already supervises the boundary, or when
        `enforce_dirichlet=False` is set instead.

        **Pressure pinning.** Pressure only ever appears through its
        gradient (`p_x`, `p_y`), so it is determined only up to an
        additive constant -- the discrete analogue of the classical
        pure-Neumann pressure null space. By default (`pin_pressure=True`),
        a single grid point (`p[:, 0, 0]`) is hard-set to
        `pressure_pin_value` before differentiating, breaking that
        indeterminacy the same way the velocity Dirichlet values break
        theirs. Leaving both `enforce_dirichlet` and `pin_pressure` off
        is possible but not recommended without another loss constraining
        the boundary and the pressure level.

    Parameters
    ----------
    reynolds : float
        the Reynolds number `Re`, required (there is no standard default
        the way Darcy's `forcing=1.0` is -- the lid-driven-cavity
        benchmark is defined *by* which `Re` is used, e.g. 100, 400, 1000)
    measure : float, 0-d array, or Sequence[float], optional
        physical domain length per spatial dimension, by default 1.0 (the
        standard unit-square cavity domain)
    reduction : str, optional
        whether to reduce the squared residual across the batch and spatial
        dims by summing ("sum") or averaging ("mean"), by default "mean"
    lid_velocity : float, optional
        the moving lid's horizontal speed (`u` on the `y_max` edge), by
        default 1.0 (the standard lid-driven-cavity convention)
    enforce_dirichlet : bool, optional
        whether to hard-constrain `u`/`v` at the four walls before
        differentiating, by default True. See the note above.
    pin_pressure : bool, optional
        whether to hard-constrain `p` at one grid point before
        differentiating, by default True. See the note above.
    pressure_pin_value : float, optional
        the value `p` is pinned to at `(0, 0)` when `pin_pressure=True`,
        by default 0.0
    weights : dict[str, float], optional
        per-equation weight applied before summing the three residual norms
        into one scalar, by default ``{"momentum_x": 1.0, "momentum_y": 1.0,
        "continuity": 1.0}`` -- must have exactly these three keys if given,
        since momentum and continuity residuals are not generally the same
        scale and an unweighted sum can let one equation dominate training.
    boundary_margin : int, optional
        number of outermost grid rings excluded from the residual norm, by
        default 0 (use the whole field). See `_crop` -- on a
        wall-bounded flow the one-sided boundary stencils dominate the norm
        with discretization error the true solution cannot satisfy either,
        so a small non-zero margin (4 on this repo's 129x129 cavity) is
        recommended whenever this loss is *trained on* rather than only
        reported.
    """

    _MIN_GRID_POINTS = 4
    _EQUATIONS = ("momentum_x", "momentum_y", "continuity")

    def __init__(
        self,
        reynolds: float,
        measure: float | jnp.ndarray | Sequence[float] = 1.0,
        reduction: str = "mean",
        lid_velocity: float = 1.0,
        enforce_dirichlet: bool = True,
        pin_pressure: bool = True,
        pressure_pin_value: float = 0.0,
        weights: dict[str, float] | None = None,
        boundary_margin: int = 0,
    ) -> None:
        """Configure the Navier-Stokes equation loss. See the class docstring
        for parameters.
        """
        self.d = 2

        if not isinstance(boundary_margin, int) or boundary_margin < 0:
            raise ValueError(
                "NavierStokesEqnLoss boundary_margin must be a non-negative "
                f"int, got {boundary_margin!r}"
            )
        self.boundary_margin = boundary_margin

        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions, (
            f"error: expected `reduction` to be one of {allowed_reductions}, "
            f"got {reduction}"
        )
        self.reduction = reduction

        self.measure: Sequence[float]
        if _is_scalar(measure):
            self.measure = [float(measure)] * self.d  # type: ignore[arg-type]
        else:
            if len(measure) != self.d:  # type: ignore[arg-type]
                raise ValueError(
                    f"NavierStokesEqnLoss measure must have exactly {self.d} "
                    f"entries (one per spatial dim), got {len(measure)}"  # type: ignore[arg-type]
                )
            self.measure = [float(m) for m in measure]  # type: ignore[union-attr]

        if reynolds <= 0:
            raise ValueError(
                f"NavierStokesEqnLoss requires reynolds > 0, got {reynolds}"
            )
        self.reynolds = float(reynolds)
        self.lid_velocity = lid_velocity
        self.enforce_dirichlet = enforce_dirichlet
        self.pin_pressure = pin_pressure
        self.pressure_pin_value = pressure_pin_value

        if weights is None:
            self.weights = dict.fromkeys(self._EQUATIONS, 1.0)
        else:
            got = set(weights)
            expected = set(self._EQUATIONS)
            if got != expected:
                raise ValueError(
                    "NavierStokesEqnLoss weights dict must have exactly the "
                    f"keys {self._EQUATIONS}, got {sorted(got)}"
                )
            self.weights = dict(weights)

        self._u_boundary = {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": self.lid_velocity,
        }
        self._v_boundary = {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0}

        self._fd = FiniteDiff()
        axes = [
            AxisSpec(
                axis=1, differentiator=self._fd, measure=self.measure[0], periodic=False
            ),
            AxisSpec(
                axis=2, differentiator=self._fd, measure=self.measure[1], periodic=False
            ),
        ]
        ax_x, ax_y = axes
        nu = 1.0 / self.reynolds

        advection_velocity = ("u", "v")
        self._momentum_x = PDEResidualLoss(
            terms={
                "advection": (
                    1.0,
                    AdvectionTerm(axes=axes, field="u", velocity=advection_velocity),
                ),
                "pressure_grad": (1.0, GradientTerm(axis=ax_x, field="p")),
                "diffusion": (-1.0, DiffusionTerm(axes=axes, field="u", coeff=nu)),
            },
            reduction=reduction,
            name="NavierStokesEqnLoss/momentum_x",
        )
        self._momentum_y = PDEResidualLoss(
            terms={
                "advection": (
                    1.0,
                    AdvectionTerm(axes=axes, field="v", velocity=advection_velocity),
                ),
                "pressure_grad": (1.0, GradientTerm(axis=ax_y, field="p")),
                "diffusion": (-1.0, DiffusionTerm(axes=axes, field="v", coeff=nu)),
            },
            reduction=reduction,
            name="NavierStokesEqnLoss/momentum_y",
        )
        self._continuity = PDEResidualLoss(
            terms={"divergence": (1.0, DivergenceTerm(fields=["u", "v"], axes=axes))},
            reduction=reduction,
            name="NavierStokesEqnLoss/continuity",
        )

    @property
    def name(self) -> str:
        """str: Human-readable identifier."""
        return "NavierStokesEqnLoss"

    def _grid_spacing(self, u: jnp.ndarray) -> tuple[float, float]:
        """Grid spacing per spatial axis, assuming both endpoints are sampled
        (non-periodic Dirichlet grid): `h = measure / (n_points - 1)`.
        """
        n_x, n_y = u.shape[1], u.shape[2]
        for n in (n_x, n_y):
            if n < self._MIN_GRID_POINTS:
                raise ValueError(
                    "NavierStokesEqnLoss needs at least "
                    f"{self._MIN_GRID_POINTS} grid points per spatial axis, got {n}"
                )
            if 2 * self.boundary_margin >= n:
                raise ValueError(
                    f"NavierStokesEqnLoss boundary_margin={self.boundary_margin} "
                    f"leaves no interior points on an axis of length {n} -- it "
                    "must satisfy 2 * boundary_margin < n_points"
                )
        return self.measure[0] / (n_x - 1), self.measure[1] / (n_y - 1)

    @staticmethod
    def _apply_dirichlet(
        field: jnp.ndarray, boundary_value: dict[str, float]
    ) -> jnp.ndarray:
        """Overwrite the boundary rows/columns of a single 2D field with the
        given per-edge Dirichlet value(s). x-edges are applied before
        y-edges, so at the four corners the y-edge value is the one that
        sticks -- matching DarcyEqnLoss's convention.
        """
        field = field.at[:, 0, :].set(boundary_value["x_min"])
        field = field.at[:, -1, :].set(boundary_value["x_max"])
        field = field.at[:, :, 0].set(boundary_value["y_min"])
        field = field.at[:, :, -1].set(boundary_value["y_max"])
        return field

    def _context(self, y_pred: jnp.ndarray) -> Context:
        """Validate shapes, apply the Dirichlet/pinning constraints, and
        assemble the context dict the underlying PDEResidualLoss's terms
        read from.
        """
        if y_pred.shape[-1] != 3:
            raise ValueError(
                "NavierStokesEqnLoss expects a 3-channel (batch, n_x, n_y, 3) "
                f"(u, v, p) prediction, got y_pred.shape={y_pred.shape}"
            )

        u = y_pred[..., 0]
        v = y_pred[..., 1]
        p = y_pred[..., 2]

        if self.enforce_dirichlet:
            u = self._apply_dirichlet(u, self._u_boundary)
            v = self._apply_dirichlet(v, self._v_boundary)

        if self.pin_pressure:
            p = p.at[:, 0, 0].set(self.pressure_pin_value)

        self._grid_spacing(u)  # validates grid size; AxisSpec derives its own spacing

        return {"u": u, "v": v, "p": p}

    def residual(self, y_pred: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Pointwise residual for each of the three coupled equations.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted `(u, v, p)` field, shape `(batch, n_x, n_y, 3)`

        Returns
        -------
        dict[str, jnp.ndarray]
            `{"momentum_x": ..., "momentum_y": ..., "continuity": ...}`,
            each shape `(batch, n_x, n_y)`
        """
        ctx = self._context(y_pred)
        return {
            "momentum_x": self._momentum_x.residual(ctx),
            "momentum_y": self._momentum_y.residual(ctx),
            "continuity": self._continuity.residual(ctx),
        }

    def _crop(self, val: jnp.ndarray) -> jnp.ndarray:
        """Drop the outermost `boundary_margin` rings of a pointwise
        residual field, so the reduction sees interior points only.

        The finite-difference stencils fall back to one-sided formulas on
        the boundary rows/columns, and on a wall-bounded flow whose near-wall
        layer is not grid-resolved those one-sided values carry truncation
        error far larger than the residual itself -- large enough to dominate
        the reduced norm. Measured on this repo's lid-driven-cavity *ground
        truth* (129x129, correctly-scaled `dt`), the full-field residual is
        ~1.1 while the residual over the interior with `boundary_margin=4`
        is ~0.05: over 95% of what the loss would otherwise be minimizing is
        boundary discretization error, not physics. Minimizing it drives the
        model away from the true solution, so a wall-bounded problem should
        set a small non-zero margin.

        Left at 0 (the default) the full field is used, matching the
        behaviour of every caller written before this option existed.
        """
        m = self.boundary_margin
        if m == 0:
            return val
        return val[:, m:-m, m:-m]

    def _reduce_equations(
        self, res: dict[str, jnp.ndarray], take_root: bool
    ) -> jnp.ndarray:
        """Weighted sum of the three equations' (optionally root-ed) reduced
        squared residuals -- the shared tail of `abs` and
        `abs_unsteady`, so both apply the crop/weights identically.
        """
        pdes = {
            "momentum_x": self._momentum_x,
            "momentum_y": self._momentum_y,
            "continuity": self._continuity,
        }
        total: jnp.ndarray | float = 0.0
        for eq_name, val in res.items():
            reduced = pdes[eq_name]._reduce_all(self._crop(val) ** 2)
            if take_root:
                reduced = reduced**0.5
            total = total + self.weights[eq_name] * jnp.squeeze(reduced)
        return total  # type: ignore[return-value]

    def abs(self, y_pred: jnp.ndarray, take_root: bool = True) -> jnp.ndarray:
        """Scalar PDE-residual loss: the weighted sum of the three
        equations' (optionally root-ed) reduced squared residuals.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted `(u, v, p)` field, shape `(batch, n_x, n_y, 3)`
        take_root : bool, optional
            whether to take the square root of each equation's reduced
            squared residual before weighting/summing, by default True
        """
        return self._reduce_equations(self.residual(y_pred), take_root)

    def residual_unsteady(
        self, y_pred: jnp.ndarray, y_pred_next: jnp.ndarray, dt: float
    ) -> dict[str, jnp.ndarray]:
        """Pointwise residual for the transient (unsteady) equations.

        Adds `du/dt`/`dv/dt` -- estimated by finite-differencing this
        prediction against `y_pred_next` (the same model's own prediction
        at the same spatial grid and same case, `dt` later in time) -- to
        the steady momentum residuals from `residual`. Continuity has
        no transient term: mass conservation (`div(u) = 0`) is an
        instantaneous constraint, independent of time.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted `(u, v, p)` field at time `t`, shape
            `(batch, n_x, n_y, 3)`
        y_pred_next : jnp.ndarray
            predicted `(u, v, p)` field at time `t + dt`, same shape --
            same case (Reynolds number / lid velocity) and same spatial grid
            as `y_pred`, differing only in the time input that produced
            each of the two forward passes.
        dt : float
            time step between the two predictions, in whatever time units
            the caller's own time input uses -- this class has no opinion
            on time units, it just finite-differences whatever two
            predictions it's given.

        Returns
        -------
        dict[str, jnp.ndarray]
            `{"momentum_x": ..., "momentum_y": ..., "continuity": ...}`,
            each shape `(batch, n_x, n_y)`
        """
        ctx = self._context(y_pred)
        ctx_next = self._context(y_pred_next)
        du_dt = (ctx_next["u"] - ctx["u"]) / dt
        dv_dt = (ctx_next["v"] - ctx["v"]) / dt
        return {
            "momentum_x": self._momentum_x.residual(ctx) + du_dt,
            "momentum_y": self._momentum_y.residual(ctx) + dv_dt,
            "continuity": self._continuity.residual(ctx),
        }

    def abs_unsteady(
        self,
        y_pred: jnp.ndarray,
        y_pred_next: jnp.ndarray,
        dt: float,
        take_root: bool = True,
    ) -> jnp.ndarray:
        """Scalar unsteady PDE-residual loss.

        Same weighted-sum-of-equations reduction as `abs`, applied to
        the transient residuals from `residual_unsteady` instead of
        the steady ones.

        Parameters
        ----------
        y_pred, y_pred_next, dt : see `residual_unsteady`
        take_root : bool, optional
            whether to take the square root of each equation's reduced
            squared residual before weighting/summing, by default True
        """
        res = self.residual_unsteady(y_pred, y_pred_next, dt)
        return self._reduce_equations(res, take_root)

    def __call__(
        self,
        y_pred: jnp.ndarray,
        x: jnp.ndarray | None = None,
        y: jnp.ndarray | None = None,
        **kwargs: object,
    ) -> jnp.ndarray:
        """Return the Navier-Stokes PDE-residual loss, `abs(y_pred)`.

        Matches the `loss(out, **sample)` calling convention used by
        `Trainer`. Neither `x` nor `y` is needed -- `reynolds` is a
        fixed configuration constant (like Burgers' `nu`), not a per-sample
        field -- both are accepted only for interface compatibility.

        Parameters
        ----------
        y_pred : jnp.ndarray
            predicted `(u, v, p)` field, shape `(batch, n_x, n_y, 3)`
        x : jnp.ndarray, optional
            unused; accepted for interface compatibility with `Trainer`
        y : jnp.ndarray, optional
            unused; accepted for interface compatibility with `Trainer`

        Returns
        -------
        jnp.ndarray
            scalar PDE-residual loss
        """
        del x, y, kwargs
        return self.abs(y_pred)
