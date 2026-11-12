# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
End-to-end demo for the irregular-geometry extension of the patch +
quadrature AD mechanism: physics-loss-only training on a domain that is
NOT a plain box -- the unit square with a circular hole excised from its
interior (a stand-in for the solid interior of a car or airfoil in an
external-flow problem), composing:

* :func:`~scirex.experimental.losses.quadrature.make_patches` /
  :class:`~scirex.experimental.losses.patch_quadrature_ad.PatchQuadratureADDiff` --
  quadrature over the *bounding box* (unaware of the hole)
* :func:`~scirex.experimental.losses.domain_membership.mask_quadrature_weights` --
  zeroes the weight of every quadrature point that lands inside the
  excluded hole (``exclude="inside"``, the external-flow convention)
* :class:`~scirex.models.transolver.Transolver` -- a real,
  globally-mixing model, trained with a Darcy/Poisson-style residual

Deliberately synthetic (a circle, not a real dataset): the whole point of
:mod:`~scirex.experimental.losses.domain_membership` is that it doesn't care where
the boundary point cloud came from, so this test proves the mechanism works
without needing any dataset-specific loader -- a real dataset's boundary
point cloud (from AirFoil, ShapeNetCar, or anything else) would plug into
exactly the same ``mask_quadrature_weights`` call with no other change.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from scirex.experimental.losses.ad_pde_terms import DiffusionTermAD, SourceTermAD
from scirex.experimental.losses.domain_membership import mask_quadrature_weights
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.losses.equation_losses import PDEResidualLoss
from scirex.models.transolver import Transolver

_CENTER = (0.5, 0.5)
_RADIUS = 0.2


def _circle_boundary(n_boundary: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_boundary, endpoint=False)
    points = np.stack(
        [_CENTER[0] + _RADIUS * np.cos(theta), _CENTER[1] + _RADIUS * np.sin(theta)],
        axis=-1,
    )
    normals = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return jnp.asarray(points), jnp.asarray(normals)


def test_masking_excludes_only_points_inside_the_hole() -> None:
    """Structural check before the training claim: the hole is actually
    being excluded, not the whole domain or nothing at all."""
    boundary_points, boundary_normals = _circle_boundary(n_boundary=200)
    pq = PatchQuadratureADDiff(
        bounds=[(0.0, 1.0), (0.0, 1.0)], n_patches_per_axis=[2, 2], n=3
    )

    masked_weights = mask_quadrature_weights(
        pq.coords, pq.weights, boundary_points, boundary_normals, exclude="inside"
    )

    n_masked = int(jnp.sum(masked_weights == 0.0))
    assert 0 < n_masked < pq.coords.shape[0]
    # every masked-out point must genuinely be inside the circular hole
    dist_to_center = jnp.linalg.norm(pq.coords - jnp.asarray(_CENTER), axis=-1)
    masked_out = masked_weights == 0.0
    assert bool(jnp.all(dist_to_center[masked_out] < _RADIUS))


def test_smoke_training_transolver_on_domain_with_excluded_hole() -> None:
    boundary_points, boundary_normals = _circle_boundary(n_boundary=200)

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

    pq = PatchQuadratureADDiff(
        bounds=[(0.0, 1.0), (0.0, 1.0)], n_patches_per_axis=[2, 2], n=3
    )
    masked_weights = mask_quadrature_weights(
        pq.coords, pq.weights, boundary_points, boundary_normals, exclude="inside"
    )

    diffusion = DiffusionTermAD(dims=[0, 1], coeff=1.0, differentiator=pq)
    pde = PDEResidualLoss(
        terms={"diffusion": (-1.0, diffusion), "source": (-1.0, SourceTermAD(1.0))},
        reduction="mean",
    )

    def loss_fn(params: Any) -> jnp.ndarray:
        def f(coords: jnp.ndarray) -> jnp.ndarray:
            out = model.apply(params, x=coords[None, :, :])
            return cast(jnp.ndarray, out)[0, :, 0]

        ctx = {"u_fn": f, "coords": pq.coords}
        residual = pde.residual(ctx)
        # weighted reduction using the hole-masked weights, so points inside
        # the excluded hole contribute nothing to the loss
        return jnp.sqrt(jnp.sum(masked_weights * residual**2) / jnp.sum(masked_weights))

    step_fn = jax.jit(jax.value_and_grad(loss_fn))

    lr = 1e-2
    losses = []
    for step in range(8):
        loss_val, grads = step_fn(params)
        losses.append(float(loss_val))

        flat_grads = jax.tree_util.tree_leaves(grads)
        assert all(
            bool(jnp.all(jnp.isfinite(g))) for g in flat_grads
        ), f"non-finite gradient at step {step}"

        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    assert losses[-1] < losses[0]
