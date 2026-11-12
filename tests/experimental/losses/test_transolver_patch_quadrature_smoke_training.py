# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Validation step 3 of the physics-loss roadmap's Phase 3b: an end-to-end
smoke-training test -- a physics-loss-only (collocation-style, no paired
data) training run on a real :class:`~scirex.models.transolver.Transolver`,
using :class:`~scirex.experimental.losses.patch_quadrature_ad.PatchQuadratureADDiff`
as the AD backend for a Darcy/Poisson-style residual
(``-Laplacian(u) = f``), assembled from the same
:class:`~scirex.experimental.losses.ad_pde_terms.DiffusionTermAD` /
:class:`~scirex.experimental.losses.ad_pde_terms.SourceTermAD` primitives
already unit-tested against ``ADDiff`` elsewhere -- this test's only job is
confirming the *whole stack* (model -> patch quadrature coords ->
PatchQuadratureADDiff -> PDEResidualLoss -> jax.grad -> a parameter update)
actually works together, not re-deriving per-term correctness.

Collocation-only, by design (see
:mod:`~scirex.experimental.losses.patch_quadrature_ad`'s module docstring):
:attr:`PatchQuadratureADDiff.coords` -- fixed Gauss-Legendre nodes on
``[0, 1]^2`` -- are themselves the model's query coordinates every step;
there is no paired dataset here at all.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp

from scirex.experimental.losses.ad_pde_terms import DiffusionTermAD, SourceTermAD
from scirex.experimental.losses.patch_quadrature_ad import PatchQuadratureADDiff
from scirex.losses.equation_losses import PDEResidualLoss
from scirex.models.transolver import Transolver


def _build_model_and_params() -> tuple[Transolver, Any]:
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
    return model, params


def test_smoke_training_transolver_patch_quadrature_darcy_style_residual() -> None:
    model, params = _build_model_and_params()
    pq = PatchQuadratureADDiff(
        bounds=[(0.0, 1.0), (0.0, 1.0)], n_patches_per_axis=[2, 2], n=3
    )

    diffusion = DiffusionTermAD(dims=[0, 1], coeff=1.0, differentiator=pq)
    pde = PDEResidualLoss(
        terms={
            "diffusion": (-1.0, diffusion),
            "source": (-1.0, SourceTermAD(1.0)),
        },
        reduction="mean",
    )

    def loss_fn(params: Any) -> jnp.ndarray:
        def f(coords: jnp.ndarray) -> jnp.ndarray:
            out = model.apply(params, x=coords[None, :, :])
            return cast(jnp.ndarray, out)[0, :, 0]

        ctx = {"u_fn": f, "coords": pq.coords}
        return pde.abs(ctx, take_root=True)

    step_fn = jax.jit(jax.value_and_grad(loss_fn))

    lr = 1e-2
    losses = []
    for step in range(10):
        loss_val, grads = step_fn(params)
        losses.append(float(loss_val))

        flat_grads = jax.tree_util.tree_leaves(grads)
        assert all(
            bool(jnp.all(jnp.isfinite(g))) for g in flat_grads
        ), f"non-finite gradient at step {step}"
        if step == 0:
            assert any(bool(jnp.any(g != 0)) for g in flat_grads), (
                "all gradients are exactly zero at step 0 -- loss is not "
                "actually differentiable w.r.t. params through this stack"
            )

        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

    assert losses[-1] < losses[0]
