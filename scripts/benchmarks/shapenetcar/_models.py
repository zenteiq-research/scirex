# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architecture registry for this benchmark.

Structured the same way ``lid_driven_cavity``/``darcy``'s ``_models.py`` are --
:data:`SPECS` maps a name to an :class:`ArchitectureSpec`, and
:func:`build_model` is the one place a model class is chosen, as a dict
lookup rather than an ``if``/``elif`` -- so a second architecture can be added
here later without touching ``train.py``/``infer.py``.

Simpler than those two benchmarks in one respect: this one has no
``WindowModel``-style shape adapter for the paper's own architecture
(Transolver is already built to this benchmark's irregular-point-cloud
contract, no time window or grid folding to wrap) and no ``search_grid``
sweep for it -- its hyperparameters are the paper's own fixed choices (see
``_config.py``), not derived from a search.

:class:`GINOCarWrapper` is the one architecture this benchmark adds beyond
the paper, and does need an adapter: GINO maps between point clouds through a
*fixed* latent grid (``_gino_geometry.py``), not through attention over the
points themselves, so it needs the per-sample input/output geometry and
neighbor lists split out of ``x`` and supplied separately -- unlike
``lid_driven_cavity``'s GINO, whose grid is identical for every sample and so
is built and cached once (see ``_gino_geometry.py``'s module docstring for
why a ShapeNetCar sample cannot share that shortcut). Sized to within 0.1% of
Transolver's parameter count, following the same "match the budget, vary the
architecture" rule as :class:`~scripts.benchmarks.lid_driven_cavity._models.
GINOSpec`/``WNOSpec``/``TransolverSpec`` do there -- see ``_config.py``'s
``_GINO_TRANSOLVER_MATCHED`` for the derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from scirex.experimental.giwo import GIWO
from scirex.models import GINO, Transolver
from scripts.benchmarks.shapenetcar import _gino_geometry


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    label: str
    module: type[nn.Module]

    def build(self, arch: dict[str, Any], model_kwargs: dict[str, Any]) -> nn.Module:
        return self.module(**arch, **model_kwargs)


class GINOCarWrapper(nn.Module):
    """Adapts :class:`scirex.models.GINO` to this benchmark's
    ``(x, fx=None, deterministic=False, neighbors_in=None, neighbors_out=None)``
    call contract -- the same one :class:`scirex.models.Transolver` is
    called with in ``_physics.py``/``train.py``, so neither file needs an
    ``if model == ...`` branch.

    ``x`` is ``(1, N, 7)``: the dataset's ``[pos_x, pos_y, pos_z, sdf,
    normal_x, normal_y, normal_z]`` (see ``_data.py``'s ``MODEL_KWARGS``
    comment). The first 3 channels become ``input_geom``/``output_queries``
    (GINO decodes back onto the same points it reads from); the remaining 4
    become the input function ``x=`` GINO itself takes. Position is not
    folded into that input function a second time -- GINO applies its own
    NeRF-style positional embedding inside both GNO kernels, so passing
    coordinates again as channels would only duplicate them (the same
    reasoning :class:`~scripts.benchmarks.lid_driven_cavity._models.
    PointCloudGINO` documents).

    ``latent_grid`` is the one size of the fixed global grid from
    ``_gino_geometry.build_latent_queries``, identical for every sample (see
    that module's docstring); the array itself is rebuilt from this int on
    every call, cheaply, since ``build_latent_queries`` is ``lru_cache``d.
    ``neighbors_in``/
    ``neighbors_out`` are the one part of the geometry that *does* differ
    per sample (a different car's points sit at different distances from
    that fixed grid) -- supplied by the caller (``_gino_geometry.
    attach_neighbors``, run once per split, not per epoch) rather than
    computed here, since a KD-tree search needs concrete coordinate values
    and cannot run inside a jitted forward pass.

    ``fx``/``deterministic`` are accepted and ignored. They exist on this
    signature only so ``_physics.py``'s single ``module.apply(x=x, fx=None,
    deterministic=..., rngs=...)`` call site keeps working unchanged for
    both architectures; GINO here has no dropout to gate and no extra
    conditioning input.

    ``space_dim``/``fun_dim``/``unified_pos``/``geometry`` are likewise
    accepted and ignored: ``_models.build_model`` constructs every
    architecture the same way, ``module(**arch, **model_kwargs)`` (see
    ``ArchitectureSpec.build``), and ``model_kwargs`` is ``_data.py``'s
    ``MODEL_KWARGS`` -- Transolver's geometry contract, not this one's. This
    wrapper takes the 7 input channels apart itself (see above) rather than
    reading these.

    ``in_gno_transform_type="nonlinear_kernelonly"`` (not GINO's "linear"
    default, and the same choice :class:`~scripts.benchmarks.
    lid_driven_cavity._models.PointCloudGINO` makes): the input GNO kernel's
    output feeds the latent FNO directly rather than being multiplied
    elementwise against the input function first. ``modes``, ``width`` and
    ``n_layers`` are what size this model -- see ``_config.py``'s
    ``_GINO_TRANSOLVER_MATCHED`` for the parameter-budget derivation,
    measured the same empirical way the cavity benchmark sizes its own
    GINO/WNO/Transolver runs, by direct ``count_params`` measurement rather
    than a closed-form parameter count.
    """

    latent_grid: int = 10
    out_dim: int = 4
    modes: int = 6
    width: int = 57
    n_layers: int = 4
    gno_mlp_hidden: tuple[int, ...] = (64, 64)
    in_gno_radius: float = 0.14
    out_gno_radius: float = 0.8
    # Transolver's geometry contract (_data.MODEL_KWARGS), unused here.
    space_dim: int = 7
    fun_dim: int = 0
    unified_pos: bool = False
    geometry: str = "irregular"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        fx: Any = None,
        deterministic: bool = False,
        neighbors_in: dict | None = None,
        neighbors_out: dict | None = None,
    ) -> tuple[jnp.ndarray]:
        del fx, deterministic
        pos = x[:, :, :3]
        fn = x[:, :, 3:]
        latent = jnp.asarray(_gino_geometry.build_latent_queries(self.latent_grid))[
            None
        ]
        out = GINO(
            in_channels=fn.shape[-1],
            out_channels=self.out_dim,
            gno_coord_dim=3,
            in_gno_radius=self.in_gno_radius,
            out_gno_radius=self.out_gno_radius,
            in_gno_transform_type="nonlinear_kernelonly",
            in_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            out_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            fno_n_modes=(self.modes, self.modes, self.modes),
            fno_hidden_channels=self.width,
            fno_n_layers=self.n_layers,
            # As lid_driven_cavity's PointCloudGINO: plain Adam/AdamW here
            # (train.py), whose moment estimates would otherwise be
            # complex-valued.
            fno_split_complex_params=True,
        )(
            input_geom=pos,
            latent_queries=latent,
            output_queries=pos,
            x=fn,
            neighbors_in=neighbors_in,
            neighbors_out=neighbors_out,
        )
        return (out[0],)


class GIWOCarWrapper(nn.Module):
    """Adapts :class:`scirex.experimental.giwo.GIWO` ("Geo-WNO": the same GNO-in /
    GNO-out shell as GINO, with its latent FNO swapped for a WNO -- see
    ``src/scirex/experimental/giwo.py``'s module docstring) to this benchmark's
    call contract, exactly as :class:`GINOCarWrapper` adapts :class:`GINO`.

    Every geometry concern (``x``'s layout, the fixed latent grid, per-sample
    ``neighbors_in``/``neighbors_out``, the ignored Transolver-contract
    fields) is identical to :class:`GINOCarWrapper` and documented there, not
    repeated here. ``level``/``wavelet`` are the one genuinely new knob this
    adds over GINO: the latent WNO's own wavelet-decomposition depth and
    family (see ``_config.py``'s ``_GIWO_TRANSOLVER_MATCHED`` for the
    parameter-budget derivation).
    """

    latent_grid: int = 10
    out_dim: int = 4
    width: int = 64
    n_layers: int = 4
    level: int = 2
    wavelet: str = "db4"
    gno_mlp_hidden: tuple[int, ...] = (64, 64)
    in_gno_radius: float = 0.14
    out_gno_radius: float = 0.8
    # Transolver's geometry contract (_data.MODEL_KWARGS), unused here.
    space_dim: int = 7
    fun_dim: int = 0
    unified_pos: bool = False
    geometry: str = "irregular"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        fx: Any = None,
        deterministic: bool = False,
        neighbors_in: dict | None = None,
        neighbors_out: dict | None = None,
    ) -> tuple[jnp.ndarray]:
        del fx, deterministic
        pos = x[:, :, :3]
        fn = x[:, :, 3:]
        latent = jnp.asarray(_gino_geometry.build_latent_queries(self.latent_grid))[
            None
        ]
        out = GIWO(
            in_channels=fn.shape[-1],
            out_channels=self.out_dim,
            gno_coord_dim=3,
            in_gno_radius=self.in_gno_radius,
            out_gno_radius=self.out_gno_radius,
            in_gno_transform_type="nonlinear_kernelonly",
            in_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            out_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            wno_size=(self.latent_grid,) * 3,
            wno_hidden_channels=self.width,
            wno_n_layers=self.n_layers,
            wno_level=self.level,
            wno_wavelet=self.wavelet,
        )(
            input_geom=pos,
            latent_queries=latent,
            output_queries=pos,
            x=fn,
            neighbors_in=neighbors_in,
            neighbors_out=neighbors_out,
        )
        return (out[0],)


_TRANSOLVER_SPEC = ArchitectureSpec(
    name="transolver", label="Transolver", module=Transolver
)
_GINO_SPEC = ArchitectureSpec(name="gino", label="GINO", module=GINOCarWrapper)
_GIWO_SPEC = ArchitectureSpec(
    name="giwo", label="GIWO (Geo-WNO)", module=GIWOCarWrapper
)
SPECS: dict[str, ArchitectureSpec] = {
    spec.name: spec for spec in (_TRANSOLVER_SPEC, _GINO_SPEC, _GIWO_SPEC)
}


def build_model(
    name: str, arch: dict[str, Any], model_kwargs: dict[str, Any]
) -> nn.Module:
    """The model named ``name``, at the size given by ``arch``, wired to this
    dataset's point-cloud geometry via ``model_kwargs`` (``_data.MODEL_KWARGS``)."""
    try:
        spec = SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(SPECS)}"
        ) from None
    return spec.build(arch, model_kwargs)
