# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Q4 (bilinear quadrilateral) Galerkin weak-form gradient machinery, a
sibling of `scirex.experimental.losses.galerkin_t6`'s `T6Galerkin` --
same reasoning, different reference element.

Why this needs the T6-style quadrature machinery, not T3's closed form
--------------------------------------------------------------------------
A Q4 element's shape functions are bilinear (`N_i = (1+-xi)(1+-eta)/4`),
so a nodal field's gradient is *linear within the element* (varies with
the other coordinate), not constant like T3's linear-triangle case. There
is no single-point closed form for the stiffness integral here either --
this module follows `galerkin_t6.py`'s pattern exactly: precompute
physical shape-function gradients and the Jacobian determinant at a set
of quadrature points once (isoparametric map), then assemble the weak
residual from those.

Reference element and quadrature
-----------------------------------
Reference square `[-1,1] x [-1,1]`, node order matching
`build_grid_aligned_q4_mesh`: node 1 = `(xi,eta)=(-1,-1)` (bottom-left
physical corner), 2 = `(1,-1)` (bottom-right), 3 = `(1,1)`
(top-right), 4 = `(-1,1)` (top-left) -- standard CCW bilinear-quad
convention. Quadrature is the standard 2x2 Gauss-Legendre rule (points at
`+-1/sqrt(3)`, weight 1 each) -- degree-3-exact, more than enough for
the stiffness integrand (a bilinear field's gradient is linear in one
coordinate, so `grad(u).grad(N_i)` is at most quadratic per axis before
`kappa`/the Jacobian enter). Since the 4 Gauss weights already sum to
4 (the reference square's own area), no extra reference-area
normalization constant is needed here (unlike `galerkin_t6.py`'s
`_REF_TRIANGLE_AREA`, needed there because that module's weights were
normalized to sum to 1 first).

For this repo's actual use case -- a uniform, axis-aligned grid where
every element is an identical `h x h` square -- the isoparametric
Jacobian collapses to the same constant matrix everywhere. The general
per-quadrature-point machinery here is used anyway (not special-cased to
axis-aligned rectangles), the same design choice `galerkin.py`
makes for evaluating `kappa` via a 3-point rule even where a single
point would technically be exact: reusing one general, already-validated
code path is simpler than adding a special case, and this generality
would matter (not just be technically unused) on a non-uniform
quadrilateral mesh in the future.

Same sign convention and `kappa_fn`/`source_fn` calling convention as
`scirex.losses.galerkin` -- see that module's docstring for
the strong-form derivation.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from scirex.losses.galerkin import element_scatter_add

_G = 1.0 / jnp.sqrt(3.0)
_QUAD_POINTS = jnp.array(
    [
        [-_G, -_G],
        [_G, -_G],
        [_G, _G],
        [-_G, _G],
    ]
)
_QUAD_WEIGHTS = jnp.array([1.0, 1.0, 1.0, 1.0])


def _shape_functions(xi: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """Bilinear shape functions at one reference point, shape (4,). Node
    order: (-1,-1), (1,-1), (1,1), (-1,1) -- bottom-left, bottom-right,
    top-right, top-left.
    """
    return 0.25 * jnp.stack(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


def _shape_function_ref_grad(xi: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """`(dN/dxi, dN/deta)` at one reference point, each shape (4,), via
    autodiff on `_shape_functions`.
    """
    jac = jax.jacfwd(_shape_functions, argnums=(0, 1))(xi, eta)
    return jnp.stack(jac, axis=-1)  # (4, 2): d/dxi, d/deta per shape function


# Precomputed once at import time -- identical for every element, since
# these live in reference (xi, eta) space, not physical space.
_N_AT_QUAD = jax.vmap(_shape_functions)(_QUAD_POINTS[:, 0], _QUAD_POINTS[:, 1])
_DN_REF_AT_QUAD = jax.vmap(_shape_function_ref_grad)(
    _QUAD_POINTS[:, 0], _QUAD_POINTS[:, 1]
)  # (Q, 4, 2)


class Q4Galerkin:
    """Generic bilinear (Q4) quadrilateral-element Galerkin machinery,
    precomputed once for a fixed mesh and shared by every instance in a
    training batch -- the Q4 analogue of
    `T6Galerkin`.

    Parameters
    ----------
    xy : array, shape (N, 2)
        Node coordinates.
    quad : array, shape (T, 4), int
        Quadrilateral connectivity, 0-indexed, CCW node order (see module
        docstring).
    """

    def __init__(self, xy: jnp.ndarray, quad: jnp.ndarray) -> None:
        xy = jnp.asarray(xy)
        quad = jnp.asarray(quad, dtype=jnp.int32)
        if quad.shape[-1] != 4:
            raise ValueError(
                f"Q4Galerkin expects 4-node quad connectivity (T, 4), "
                f"got quad.shape={quad.shape}."
            )
        self.xy = xy
        self.quad = quad
        self.n_nodes = xy.shape[0]
        self.n_elements = quad.shape[0]

        dN_dx, dN_dy, det_j = self._compute_physical_grad_and_jacobian()
        edge_xy = self.xy[self.quad]  # (T, 4, 2)
        e01_sq = jnp.sum((edge_xy[:, 1] - edge_xy[:, 0]) ** 2, axis=-1)
        e12_sq = jnp.sum((edge_xy[:, 2] - edge_xy[:, 1]) ** 2, axis=-1)
        edge_scale_sq = jnp.maximum(e01_sq, e12_sq)  # (T,)
        if jnp.any(jnp.abs(det_j) < 1e-8 * edge_scale_sq[None, :] + 1e-12):
            raise ValueError(
                "Q4Galerkin: one or more elements are degenerate (near-zero "
                "Jacobian determinant, relative to the element's own size, "
                "at a quadrature point) -- check the mesh's node ordering/"
                "connectivity."
            )
        self.dN_dx, self.dN_dy, self.det_j = dN_dx, dN_dy, det_j
        self.quad_phys_pts = self._compute_quadrature_points_physical()

    def _element_coords(self) -> jnp.ndarray:
        """Node coordinates gathered per element, shape (T, 4, 2)."""
        return self.xy[self.quad]

    def _compute_physical_grad_and_jacobian(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        elem_xy = self._element_coords()  # (T, 4, 2)
        x_nodes, y_nodes = elem_xy[..., 0], elem_xy[..., 1]  # (T, 4) each

        def at_one_quad_point(
            dn_ref: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            dn_dxi, dn_deta = dn_ref[:, 0], dn_ref[:, 1]  # (4,) each
            dx_dxi = jnp.einsum(
                "i,ti->t", dn_dxi, x_nodes, precision=jax.lax.Precision.HIGHEST
            )
            dy_dxi = jnp.einsum(
                "i,ti->t", dn_dxi, y_nodes, precision=jax.lax.Precision.HIGHEST
            )
            dx_deta = jnp.einsum(
                "i,ti->t", dn_deta, x_nodes, precision=jax.lax.Precision.HIGHEST
            )
            dy_deta = jnp.einsum(
                "i,ti->t", dn_deta, y_nodes, precision=jax.lax.Precision.HIGHEST
            )
            det_j = dx_dxi * dy_deta - dx_deta * dy_dxi  # (T,)

            inv_det = 1.0 / det_j
            dxi_dx, dxi_dy = dy_deta * inv_det, -dx_deta * inv_det
            deta_dx, deta_dy = -dy_dxi * inv_det, dx_dxi * inv_det

            dn_dx = (
                dn_dxi[None, :] * dxi_dx[:, None] + dn_deta[None, :] * deta_dx[:, None]
            )
            dn_dy = (
                dn_dxi[None, :] * dxi_dy[:, None] + dn_deta[None, :] * deta_dy[:, None]
            )
            return dn_dx, dn_dy, det_j

        dN_dx, dN_dy, det_j = jax.vmap(at_one_quad_point)(_DN_REF_AT_QUAD)
        return dN_dx, dN_dy, det_j  # each (Q, T, 4) or (Q, T)

    def physical_grad_and_jacobian(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Physical-space shape-function gradients and the Jacobian
        determinant at every quadrature point, for every element.

        Returns
        -------
        dN_dx, dN_dy : array, shape (Q, T, 4)
        det_j : array, shape (Q, T)
        """
        return self.dN_dx, self.dN_dy, self.det_j

    def _compute_quadrature_points_physical(self) -> jnp.ndarray:
        elem_xy = self._element_coords()  # (T, 4, 2)

        def field_at_quad(n_q: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum(
                "i,tid->td", n_q, elem_xy, precision=jax.lax.Precision.HIGHEST
            )

        return jax.vmap(field_at_quad)(_N_AT_QUAD)  # (Q, T, 2)

    def quadrature_points_physical(self) -> jnp.ndarray:
        """Physical coordinates of every quadrature point, every element,
        shape `(Q, T, 2)`.
        """
        return self.quad_phys_pts

    def element_gradient_at_quad(
        self, u: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """`u`, shape (N,) -> `(du_dx, du_dy)`, each shape (Q, T): the
        gradient of `u`'s Q4 (bilinear) interpolant, evaluated at every
        quadrature point of every element (linear within an element, not
        constant).
        """
        u_elem = u[self.quad]  # (T, 4)
        du_dx = jnp.einsum(
            "qti,ti->qt", self.dN_dx, u_elem, precision=jax.lax.Precision.HIGHEST
        )
        du_dy = jnp.einsum(
            "qti,ti->qt", self.dN_dy, u_elem, precision=jax.lax.Precision.HIGHEST
        )
        return du_dx, du_dy

    def grad_energy(
        self,
        u: jnp.ndarray,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scalar `int kappa * |grad(u)|^2 dOmega`, the Q4 analogue of
        `T6Galerkin.grad_energy`.
        """
        du_dx, du_dy = self.element_gradient_at_quad(u)  # (Q, T)
        kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn(params, pt))(pts))(
            self.quad_phys_pts
        )  # (Q, T)
        quad_scale = _QUAD_WEIGHTS[:, None] * jnp.abs(self.det_j)  # (Q, T)
        return jnp.sum(quad_scale * kappa_q * (du_dx**2 + du_dy**2))

    def scatter_to_nodes(self, elem_contrib: jnp.ndarray) -> jnp.ndarray:
        """`elem_contrib`, shape (T, 4) -- one value per element per
        local node (0..3) -- scattered/accumulated into a nodal array,
        shape (N,).
        """
        return element_scatter_add(self.quad, self.n_nodes, elem_contrib)


def q4_weak_residual(
    galerkin: Q4Galerkin,
    u: jnp.ndarray,
    kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Per-node Galerkin weak-form residual, shape (N,), for
    `div(kappa * grad(u)) - source = 0` on a Q4 mesh -- the Q4 analogue
    of `t6_weak_residual`. Same sign convention: a **plus** between the
    two element terms.

    Parameters
    ----------
    galerkin : Q4Galerkin
    u : array, shape (N,)
        Nodal field, evaluated at `galerkin.xy`.
    kappa_fn, source_fn : callable
        `(params, point) -> scalar`, evaluated at every quadrature point
        of every element.
    params : array
        This instance's parameter vector, passed through to `kappa_fn`/
        `source_fn`.
    """
    g = galerkin
    dN_dx, dN_dy, det_j = g.dN_dx, g.dN_dy, g.det_j  # (Q,T,4), (Q,T,4), (Q,T)
    du_dx, du_dy = g.element_gradient_at_quad(u)  # (Q, T)
    quad_phys_pts = g.quad_phys_pts  # (Q, T, 2)

    def eval_fn_at_quad(
        fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], pts_q: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.vmap(lambda pt: fn(params, pt))(pts_q)  # (T,)

    kappa_q = jax.vmap(lambda pts: eval_fn_at_quad(kappa_fn, pts))(quad_phys_pts)
    source_q = jax.vmap(lambda pts: eval_fn_at_quad(source_fn, pts))(quad_phys_pts)

    grad_dot_integrand = kappa_q[:, :, None] * (
        du_dx[:, :, None] * dN_dx + du_dy[:, :, None] * dN_dy
    )  # (Q, T, 4)
    src_integrand = source_q[:, :, None] * _N_AT_QUAD[:, None, :]  # (Q, T, 4)

    quad_scale = _QUAD_WEIGHTS[:, None, None] * jnp.abs(det_j)[:, :, None]  # (Q,T,1)
    elem_contrib = jnp.sum(
        quad_scale * (grad_dot_integrand + src_integrand), axis=0
    )  # (T, 4)

    return g.scatter_to_nodes(elem_contrib)


class BatchedVariableCoefficientGalerkinResidualLossQ4:
    """Q4 analogue of
    `BatchedVariableCoefficientGalerkinResidualLossT3`
    -- identical calling convention, including the same optional
    `energy_floor_weight`/`energy_floor` amplitude-collapse mitigation;
    only the element type differs.
    """

    def __init__(
        self,
        galerkin: Q4Galerkin,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        interior_mask: jnp.ndarray,
        energy_floor_weight: float = 0.0,
        energy_floor: float = 0.0,
    ) -> None:
        if not bool(jnp.any(interior_mask)):
            raise ValueError(
                "BatchedVariableCoefficientGalerkinResidualLossQ4: "
                "interior_mask has no True entries -- every node is on the "
                "boundary, so the residual mean would divide by zero. Check "
                "the mesh's boundary detection."
            )
        self.galerkin = galerkin
        self.kappa_fn = kappa_fn
        self.source_fn = source_fn
        self.interior_mask = interior_mask
        self.energy_floor_weight = energy_floor_weight
        self.energy_floor = energy_floor

    def __call__(
        self,
        u_batch: jnp.ndarray,
        params_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        """`mean_b mean_i(weak_residual_{b,i}^2)` over the instance batch
        and interior nodes, plus the optional energy-floor penalty.
        """
        g = self.galerkin

        residual_batch = jax.vmap(
            lambda u, params: q4_weak_residual(
                g, u, self.kappa_fn, self.source_fn, params
            )
        )(u_batch, params_batch)  # (B, N)

        masked = jnp.where(self.interior_mask[None, :], residual_batch, 0.0)
        n_instances = params_batch.shape[0]
        loss = jnp.sum(masked**2) / (jnp.sum(self.interior_mask) * n_instances)

        if self.energy_floor_weight != 0.0:
            e_grad = jax.vmap(
                lambda u, params: g.grad_energy(u, self.kappa_fn, params)
            )(u_batch, params_batch)
            floor_penalty = jnp.mean(jax.nn.relu(self.energy_floor - e_grad) ** 2)
            loss = loss + self.energy_floor_weight * floor_penalty
        return loss
