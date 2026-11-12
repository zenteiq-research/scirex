# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""T6 (quadratic triangle) Galerkin weak-form gradient machinery, the
higher-order sibling of `scirex.losses.galerkin`'s
`T3Galerkin`.

Why this is a separate module, not a T3 extension
---------------------------------------------------
A T3 element's shape functions are linear, so a nodal field's gradient is
*constant per triangle* -- `T3Galerkin`
gets away with closed-form algebra (the `b`/`c` coefficients) and never
needs a quadrature loop for the stiffness term itself, only for evaluating
`kappa`/`source`. A T6 element's shape functions are quadratic, so a
nodal field's gradient is *linear within the element* -- there is no
single-point closed form for `int grad(u).grad(N_i) dOmega` any more; it
must be assembled from genuine multi-point Gauss quadrature, evaluating
shape-function gradients (mapped through the element's own Jacobian) at
each point. This is a real structural difference, not just "more nodes",
which is why T6 gets its own class rather than a branch inside T3Galerkin.

Isoparametric, not subparametric
-----------------------------------
Both the field *and the geometry* are represented with the same quadratic
shape functions (`x(xi,eta) = sum_i N_i(xi,eta) * x_i` over all 6 nodes,
not just the 3 corners). For a straight-sided mesh (e.g. this repo's
square/L-shape T6 meshes, whose mid-edge nodes sit at exact chord
midpoints), this map reduces algebraically to the same affine map T3 would
give -- so the extra generality costs nothing there. It matters on a
curved boundary (a T6 disk mesh, whose order-2 elevation pulls boundary
mid-edge nodes onto the true circle, not the chord): there, the isoparametric
map is genuinely non-affine, and only this general per-quadrature-point
Jacobian approach represents that curvature correctly.

Quadrature rule
------------------
A 6-point, degree-4-accurate symmetric rule (Hammer-Marlowe-Stroud /
Dunavant), the standard choice for P2 elements -- adequate since the
stiffness integrand (grad(u), itself linear, dotted with grad(N_i), also
linear) is already quadratic before `kappa`/the Jacobian even enter, one
degree higher than what T3's 3-point rule needed to integrate `kappa`/
`source` alone.

Same sign convention and kappa_fn/source_fn calling convention as
`scirex.losses.galerkin` -- see that module's docstring for the
strong-form derivation and why source is a function, not a precomputed
nodal array.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from scirex.losses.galerkin import element_scatter_add

# 6-point, degree-4-accurate symmetric quadrature rule on the reference
# triangle (vertices (0,0), (1,0), (0,1)), in (xi, eta) coordinates.
# Weights sum to 1 (a fraction of the reference triangle's own area, 0.5);
# see _quadrature_integral below for where that 0.5 enters.
_G1 = 0.0915762135098
_G2 = 0.4459484909160
_QUAD_POINTS = jnp.array(
    [
        [_G1, _G1],
        [1.0 - 2.0 * _G1, _G1],
        [_G1, 1.0 - 2.0 * _G1],
        [_G2, _G2],
        [1.0 - 2.0 * _G2, _G2],
        [_G2, 1.0 - 2.0 * _G2],
    ]
)
_W_A = 0.1099517436553
_W_B = 0.2233815896780
_QUAD_WEIGHTS = jnp.array([_W_A, _W_A, _W_A, _W_B, _W_B, _W_B])
_REF_TRIANGLE_AREA = 0.5


def _shape_functions(xi: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """P2 shape functions at one reference point, shape (6,). Node order
    matches gmsh's own T6 convention: 3 corners then 3 mid-edge nodes
    (mid(1,2), mid(2,3), mid(3,1)) -- see gmsh_mesh.py's own docstring."""
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    return jnp.stack(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ]
    )


def _shape_function_ref_grad(xi: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    """`(dN/dxi, dN/deta)` at one reference point, each shape (6,), via
    autodiff on `_shape_functions` -- avoids hand-transcribing the
    (error-prone) closed-form derivative formulas."""
    jac = jax.jacfwd(_shape_functions, argnums=(0, 1))(xi, eta)
    return jnp.stack(jac, axis=-1)  # (6, 2): d/dxi, d/deta per shape function


# Precomputed once at import time: reference-space shape values/gradients
# at every quadrature point -- identical for every element, since these
# live in reference (xi, eta) space, not physical space.
_N_AT_QUAD = jax.vmap(_shape_functions)(_QUAD_POINTS[:, 0], _QUAD_POINTS[:, 1])
_DN_REF_AT_QUAD = jax.vmap(_shape_function_ref_grad)(
    _QUAD_POINTS[:, 0], _QUAD_POINTS[:, 1]
)  # (Q, 6, 2)


class T6Galerkin:
    """Generic quadratic (T6) triangular-element Galerkin machinery,
    precomputed once for a fixed mesh and shared by every instance in a
    PINO batch -- the T6 analogue of
    `T3Galerkin`.

    Unlike T3, per-element quantities here (physical shape-function
    gradients, the Jacobian determinant) vary *within* an element and are
    only ever available at the quadrature points, not in closed form --
    `physical_grad_and_jacobian` returns them at all 6 quadrature
    points at once, shape `(Q, ...)`.

    Parameters
    ----------
    xy : array, shape (N, 2)
        Node coordinates (corners and mid-edge nodes together).
    tri : array, shape (T, 6), int
        Triangle connectivity, 0-indexed, gmsh T6 node order (3 corners,
        then mid(1,2), mid(2,3), mid(3,1)).
    """

    def __init__(self, xy: jnp.ndarray, tri: jnp.ndarray) -> None:
        xy = jnp.asarray(xy)
        tri = jnp.asarray(tri, dtype=jnp.int32)
        if tri.shape[-1] != 6:
            raise ValueError(
                f"T6Galerkin expects 6-node triangle connectivity (T, 6), "
                f"got tri.shape={tri.shape} -- for a 3-node (T3) mesh, use "
                f"scirex.losses.galerkin.T3Galerkin instead."
            )
        self.xy = xy
        self.tri = tri
        self.n_nodes = xy.shape[0]
        self.n_elements = tri.shape[0]

        dN_dx, dN_dy, det_j = self._compute_physical_grad_and_jacobian()
        # det_j scales as length^2 (same reasoning as T3Galerkin's own
        # guard), so compare against each element's own corner-triangle
        # edge length rather than a fixed absolute floor -- scale-invariant
        # across mesh units/resolutions.
        corner_xy = self.xy[self.tri[:, :3]]  # (T, 3, 2)
        x1, x2, x3 = corner_xy[:, 0], corner_xy[:, 1], corner_xy[:, 2]
        e12_sq = jnp.sum((x2 - x1) ** 2, axis=-1)
        e23_sq = jnp.sum((x3 - x2) ** 2, axis=-1)
        e31_sq = jnp.sum((x1 - x3) ** 2, axis=-1)
        edge_scale_sq = jnp.maximum(jnp.maximum(e12_sq, e23_sq), e31_sq)  # (T,)
        if jnp.any(jnp.abs(det_j) < 1e-8 * edge_scale_sq[None, :] + 1e-12):
            raise ValueError(
                "T6Galerkin: one or more elements are degenerate (near-zero "
                "Jacobian determinant, relative to the element's own size, "
                "at a quadrature point) -- check the mesh's node ordering/"
                "connectivity, including the mid-edge nodes' placement."
            )
        self.dN_dx, self.dN_dy, self.det_j = dN_dx, dN_dy, det_j
        self.quad_phys_pts = self._compute_quadrature_points_physical()

    def _element_coords(self) -> jnp.ndarray:
        """Node coordinates gathered per element, shape (T, 6, 2)."""
        return self.xy[self.tri]

    def _compute_physical_grad_and_jacobian(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        elem_xy = self._element_coords()  # (T, 6, 2)
        x_nodes, y_nodes = elem_xy[..., 0], elem_xy[..., 1]  # (T, 6) each

        def at_one_quad_point(
            dn_ref: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            dn_dxi, dn_deta = dn_ref[:, 0], dn_ref[:, 1]  # (6,) each
            dx_dxi = jnp.einsum("i,ti->t", dn_dxi, x_nodes)
            dy_dxi = jnp.einsum("i,ti->t", dn_dxi, y_nodes)
            dx_deta = jnp.einsum("i,ti->t", dn_deta, x_nodes)
            dy_deta = jnp.einsum("i,ti->t", dn_deta, y_nodes)
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
        return dN_dx, dN_dy, det_j  # each (Q, T, 6) or (Q, T)

    def physical_grad_and_jacobian(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Physical-space shape-function gradients and the Jacobian
        determinant at every quadrature point, for every element.
        Precomputed once in `__init__` (and validated away from a
        degenerate/near-zero determinant there) since it depends only on
        the mesh, not on any per-instance field.

        Returns
        -------
        dN_dx, dN_dy : array, shape (Q, T, 6)
        det_j : array, shape (Q, T)
        """
        return self.dN_dx, self.dN_dy, self.det_j

    def _compute_quadrature_points_physical(self) -> jnp.ndarray:
        elem_xy = self._element_coords()  # (T, 6, 2)

        def field_at_quad(n_q: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("i,tid->td", n_q, elem_xy)

        return jax.vmap(field_at_quad)(_N_AT_QUAD)  # (Q, T, 2)

    def quadrature_points_physical(self) -> jnp.ndarray:
        """Physical coordinates of every quadrature point, every element,
        shape `(Q, T, 2)`, from the isoparametric map. Precomputed once
        in `__init__` since it depends only on the mesh."""
        return self.quad_phys_pts

    def element_gradient_at_quad(
        self, u: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """`u`, shape (N,) -> `(du_dx, du_dy)`, each shape (Q, T): the
        gradient of `u`'s T6 (piecewise-quadratic) interpolant, evaluated
        at every quadrature point of every element (linear *within* an
        element, unlike T3's constant-per-element gradient)."""
        u_elem = u[self.tri]  # (T, 6)
        du_dx = jnp.einsum("qti,ti->qt", self.dN_dx, u_elem)
        du_dy = jnp.einsum("qti,ti->qt", self.dN_dy, u_elem)
        return du_dx, du_dy

    def grad_energy(
        self,
        u: jnp.ndarray,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scalar `int kappa * |grad(u)|^2 dOmega`, the T6 analogue of
        `grad_energy`, used
        by `BatchedVariableCoefficientGalerkinResidualLossT6`'s own
        amplitude-collapse mitigation."""
        du_dx, du_dy = self.element_gradient_at_quad(u)  # (Q, T)
        kappa_q = jax.vmap(lambda pts: jax.vmap(lambda pt: kappa_fn(params, pt))(pts))(
            self.quad_phys_pts
        )  # (Q, T)
        quad_scale = (_QUAD_WEIGHTS * _REF_TRIANGLE_AREA)[:, None] * jnp.abs(
            self.det_j
        )  # (Q, T)
        return jnp.sum(quad_scale * kappa_q * (du_dx**2 + du_dy**2))

    def scatter_to_nodes(self, elem_contrib: jnp.ndarray) -> jnp.ndarray:
        """`elem_contrib`, shape (T, 6) -- one value per triangle per
        local node (0..5) -- scattered/accumulated into a nodal array,
        shape (N,)."""
        return element_scatter_add(self.tri, self.n_nodes, elem_contrib)


def t6_weak_residual(
    galerkin: T6Galerkin,
    u: jnp.ndarray,
    kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """Per-node Galerkin weak-form residual, shape (N,), for
    `div(kappa * grad(u)) - source = 0` on a T6 mesh -- the T6 analogue
    of `t3_weak_residual`. Same sign
    convention: a **plus** between the two element terms (see that
    function's / the module docstring's derivation).

    Parameters
    ----------
    galerkin : T6Galerkin
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
    dN_dx, dN_dy, det_j = g.dN_dx, g.dN_dy, g.det_j  # (Q,T,6), (Q,T,6), (Q,T)
    du_dx, du_dy = g.element_gradient_at_quad(u)  # (Q, T)
    quad_phys_pts = g.quad_phys_pts  # (Q, T, 2)

    def eval_fn_at_quad(
        fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], pts_q: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.vmap(lambda pt: fn(params, pt))(pts_q)  # (T,)

    kappa_q = jax.vmap(lambda pts: eval_fn_at_quad(kappa_fn, pts))(
        quad_phys_pts
    )  # (Q,T)
    source_q = jax.vmap(lambda pts: eval_fn_at_quad(source_fn, pts))(
        quad_phys_pts
    )  # (Q,T)

    # int kappa*grad(u).grad(N_i) dOmega, per local node i, summed over
    # quadrature points: weight_q * ref_area * |det_j| * kappa_q *
    # (du_dx*dN_dx_i + du_dy*dN_dy_i).
    grad_dot_integrand = kappa_q[:, :, None] * (
        du_dx[:, :, None] * dN_dx + du_dy[:, :, None] * dN_dy
    )  # (Q, T, 6)
    src_integrand = source_q[:, :, None] * _N_AT_QUAD[:, None, :]  # (Q, T, 6)

    quad_scale = (_QUAD_WEIGHTS * _REF_TRIANGLE_AREA)[:, None, None] * jnp.abs(det_j)[
        :, :, None
    ]  # (Q, T, 1)
    elem_contrib = jnp.sum(
        quad_scale * (grad_dot_integrand + src_integrand), axis=0
    )  # (T, 6) -- '+', matching the strong-form sign convention

    return g.scatter_to_nodes(elem_contrib)


class BatchedVariableCoefficientGalerkinResidualLossT6:
    """T6 analogue of
    `BatchedVariableCoefficientGalerkinResidualLossT3`
    -- identical calling convention, including the same optional
    `energy_floor_weight`/`energy_floor` amplitude-collapse mitigation
    (via `T6Galerkin.grad_energy`); only the element type (and
    therefore the quadrature machinery underneath) differs. See that
    class's docstring for the full parameter rationale.
    """

    def __init__(
        self,
        galerkin: T6Galerkin,
        kappa_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        source_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        interior_mask: jnp.ndarray,
        energy_floor_weight: float = 0.0,
        energy_floor: float = 0.0,
    ) -> None:
        if not bool(jnp.any(interior_mask)):
            raise ValueError(
                "BatchedVariableCoefficientGalerkinResidualLossT6: "
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
        and interior nodes, plus the optional energy-floor penalty. See
        `BatchedVariableCoefficientGalerkinResidualLossT3.__call__` for
        the parameter shapes and why there is no separate `xy` argument
        -- identical here."""
        g = self.galerkin

        residual_batch = jax.vmap(
            lambda u, params: t6_weak_residual(
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
