# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The architectures of this benchmark, assembled from this repo's own
operators.

Nothing here is a network. Each class below only adapts the benchmark's window
contract, ``(B, n_in, H, W, C) -> (B, n_out, H, W, C)``, to a model that already
lives in ``src/scirex/``:

===========  ===============================================================
FNO          :class:`scirex.models.FNOClassic2d`, with the input window
             folded into its channel axis
GINO         :class:`scirex.models.GINO` over the cell centres as a point
             cloud, with a coarse latent grid carrying the spectral layers
WNO          :class:`scirex.models.WNO`, with the input window folded into
             its channel axis, as for the FNO
Transolver   :class:`scirex.models.Transolver` over the cells as tokens,
             with its physics attention learning the slices it decomposes on
===========  ===============================================================

The FNO is the architecture of Costa Rocha et al. 2023 (Appl. Sci. 13(5),
3165) that this benchmark keeps. The other three are not in that paper. Each is
here to vary one thing about the FNO and hold the rest fixed, at the same
parameter budget, so that a difference in accuracy is a difference in
architecture:

* GINO changes how the model reaches the field, through a point cloud and a
  coarse latent grid instead of the grid itself.
* WNO changes the basis, from Fourier to a wavelet one, which is local in
  space as well as in frequency.
* Transolver changes where the basis comes from, learning a decomposition into
  slices from the data instead of taking a fixed one.

Runs of all three are compared against our own FNO and against the
copy-the-last-frame baseline rather than against a published number, since the
paper reports none.

Multi-frame output is expressed as ``n_out * n_fields`` output channels, which
the adapter splits back into frames. The operators models therefore stay
time-agnostic: how a caller lays time out is the benchmark's business, not
theirs.

:data:`SPECS` maps each name to an :class:`ArchitectureSpec` holding the
paper's search space for that architecture alongside its builder, so adding an
architecture means adding one entry and touching no ``if``/``elif`` anywhere in
the benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from scirex.layers import precompute_neighbors
from scirex.models import GINO, WNO, FNOClassic2d, Transolver
from scripts.benchmarks.cavity._data import FIELDS, cell_centres

N_FIELDS = len(FIELDS)


# ──────────────────────────── window adapters ──────────────────────────────
class WindowModel(nn.Module):
    """Base for the ``(B, n_in, H, W, C) -> (B, n_out, H, W, C)`` contract.

    Subclasses implement :meth:`__call__` and use :meth:`split_frames` to turn
    the wrapped model's ``n_out * out_channels`` output channels back into
    ``n_out`` frames.
    """

    n_out: int
    out_channels: int = N_FIELDS

    @property
    def flat_out_channels(self) -> int:
        return self.n_out * self.out_channels

    def split_frames(self, y: jnp.ndarray) -> jnp.ndarray:
        """``(B, H, W, n_out * C)`` -> ``(B, n_out, H, W, C)``."""
        batch, height, width, _ = y.shape
        return y.reshape(batch, height, width, self.n_out, self.out_channels).transpose(
            0, 3, 1, 2, 4
        )


class ChannelStackedFNO(WindowModel):
    """Stacks the input window along the channel axis and applies the FNO.

    Feeding a time window in as channels is how the reference PyTorch ``FNO2d``
    handles one, and it is why the FNO needs no recurrence here. The x and y
    coordinate channels the original design appends are added by
    ``FNOClassic2d`` itself (``GridEmbedding2D``), not here.

    ``modes``, ``width`` and ``n_layers`` are the paper's Section 2.2.3 names
    for ``FNOClassic2d``'s ``n_modes1``/``n_modes2``, ``hidden_channels`` and
    ``n_layers``. The spectral weights are stored split into real and
    imaginary parts because this benchmark trains with Adam, whose moment
    estimates would otherwise be complex-valued.
    """

    modes: int = 16
    width: int = 32
    n_layers: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, steps, height, width, channels = x.shape
        # (B, T, H, W, C) -> (B, T * C, H, W), time-major, channels-first.
        stacked = x.transpose(0, 1, 4, 2, 3).reshape(
            batch, steps * channels, height, width
        )
        y = FNOClassic2d(
            in_channels=steps * channels,
            out_channels=self.flat_out_channels,
            n_modes1=self.modes,
            n_modes2=self.modes,
            hidden_channels=self.width,
            n_layers=self.n_layers,
            split_complex_params=True,
        )(stacked)
        return self.split_frames(y.transpose(0, 2, 3, 1))


class GridWNO(WindowModel):
    """Stacks the input window along the channel axis and applies the WNO.

    Deliberately the same adapter as :class:`ChannelStackedFNO`: the same
    window, folded into channels the same way, read off the same 128x128 grid.
    The only thing that changes is what the operator does with it. The FNO
    takes a Fourier transform, keeps the lowest ``modes`` frequencies and
    learns a weight per kept frequency; the WNO takes a ``level``-deep wavelet
    decomposition and learns a weight per coefficient of the coarsest
    approximation band and its details. A Fourier mode is global in space, so
    the FNO's weights cannot say "here"; a wavelet coefficient is local in
    both space and frequency, so the WNO's can. Whether that helps on a cavity
    whose interesting structure sits in the corners is the question these runs
    are here to answer.

    ``use_grid`` is left at its default, so the WNO appends its own two
    coordinate channels, matching the ``GridEmbedding2D`` that
    ``FNOClassic2d`` applies for the FNO. Neither model is handed coordinates
    by this adapter.

    ``size`` is the training resolution, which the wavelet transform needs
    ahead of time: the subband weights are shaped from it, and
    :class:`~scirex.layers.WaveletConv` compares it against
    the resolution it is actually given so that evaluating at a different one
    adjusts the decomposition depth instead of silently reshaping. The cavity
    is always 128x128 here, so the two always agree.
    """

    hidden_channels: int = 21
    n_layers: int = 3
    level: int = 5
    wavelet: str = "db6"
    #: Fixed rather than searched, and float32 rather than the model's own
    #: bfloat16 default. Measured on this GPU at the FNO-matched size, eight
    #: epochs each: float32 took 4.08 s per epoch against bfloat16's 4.32 s,
    #: and reached a validation loss of 2.71e-3 against 5.17e-3. bfloat16 buys
    #: nothing here because the subband weights are held at float32 whatever
    #: this is set to (jaxwt's filter banks are float64, and casting them down
    #: twice loses more than the narrower multiply saves), so the low
    #: precision only adds casts around the same amount of work. float32 also
    #: matches the other four architectures, so no dtype difference confounds
    #: the comparison.
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # n_rows/n_cols rather than height/width, for the reason given in
        # PointCloudGINO: several of these models call a channel count
        # "width".
        batch, steps, n_rows, n_cols, channels = x.shape
        # (B, T, H, W, C) -> (B, H, W, T * C). Channels-last throughout, which
        # is what the WNO takes; the FNO adapter transposes to channels-first
        # instead only because FNOClassic2d wants it that way.
        stacked = x.transpose(0, 2, 3, 1, 4).reshape(
            batch, n_rows, n_cols, steps * channels
        )
        y = WNO(
            in_channels=steps * channels,
            out_channels=self.flat_out_channels,
            hidden_channels=self.hidden_channels,
            n_layers=self.n_layers,
            level=self.level,
            size=(n_rows, n_cols),
            wavelet=self.wavelet,
            compute_dtype=self.compute_dtype,
        )(stacked)
        return self.split_frames(y)


class GridTokenTransolver(WindowModel):
    """Reads the grid as one token per cell and applies Transolver.

    The same window as :class:`ChannelStackedFNO` and :class:`GridWNO`, off
    the same 128x128 grid, folded the same way. What changes is that the cells
    become a sequence of 16,384 tokens rather than a field to transform. The
    FNO and the WNO both act through a basis fixed in advance, Fourier or
    wavelet. Transolver learns one instead: its physics attention gives every
    cell a soft membership in ``slice_num`` slices, averages each slice into a
    single token, attends among those tokens, and scatters the result back.
    The decomposition is therefore data-dependent, and the question these runs
    ask is whether a cavity's structure, the corner vortices and the shear
    layer under the lid, is better served by slices the model discovers than
    by frequencies or wavelets chosen ahead of time.

    ``geometry="structured_2d"`` rather than ``"irregular"``: these cells are
    a grid, and the structured path says so, replacing the pointwise
    projections into slice space with ``kernel``-wide convolutions over the
    grid. The irregular path would run, and would throw that neighbourhood
    away.

    ``unified_pos=True`` has Transolver build its own positional encoding, the
    distance from every cell to each point of a ``ref`` by ``ref`` reference
    grid. That is what this repo's Transolver reproduction on structured data
    uses (``scripts/benchmarks/darcy/``, tracking Wu et al.'s Table 8). On the
    structured path the encoding is derived from ``H`` and ``W``, so the
    coordinates passed below are read for their batch size alone. They are
    passed anyway: they are the coordinates these tokens actually stand for,
    and they are what the model would need if ``unified_pos`` were ever
    turned off. No coordinates reach the model as
    channels, matching the FNO and the WNO, both of which append their own.

    ``dropout_rate`` is left at 0. It is wired through every block, but
    nothing in this repo passes ``deterministic=False`` together with a
    dropout key, so it is a no-op end to end and searching it would spend
    configurations on a knob that does nothing.
    """

    n_hidden: int = 128
    n_layers: int = 3
    slice_num: int = 64
    #: Fixed at Wu et al.'s values rather than searched, as ``wavelet`` is for
    #: the WNO. ``n_hidden`` must be at least ``n_head``, because the per-head
    #: width is their integer quotient and a zero width divides by zero; it is
    #: also kept to multiples of ``n_head``, because otherwise the quotient
    #: drops the remainder and part of the width silently never reaches the
    #: attention at all.
    n_head: int = 8
    mlp_ratio: int = 1
    ref: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # n_rows/n_cols for the reason given in PointCloudGINO.
        batch, steps, n_rows, n_cols, channels = x.shape
        # (B, T, H, W, C) -> (B, H * W, T * C), row-major over (H, W), so the
        # tokens line up both with cell_centres() and with the reshape back.
        tokens = x.transpose(0, 2, 3, 1, 4).reshape(
            batch, n_rows * n_cols, steps * channels
        )
        coords = jnp.broadcast_to(
            jnp.asarray(cell_centres(n_rows, n_cols))[None],
            (batch, n_rows * n_cols, 2),
        )
        y = Transolver(
            space_dim=2,
            fun_dim=steps * channels,
            out_dim=self.flat_out_channels,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            n_head=self.n_head,
            slice_num=self.slice_num,
            mlp_ratio=self.mlp_ratio,
            ref=self.ref,
            unified_pos=True,
            geometry="structured_2d",
            H=n_rows,
            W=n_cols,
        )(coords, tokens)
        return self.split_frames(
            y.reshape(batch, n_rows, n_cols, self.flat_out_channels)
        )


# ──────────────────────────── GINO geometry ────────────────────────────────
@dataclass(frozen=True)
class GINOGeometry:
    """The three coordinate sets GINO needs, and the two neighbour lists.

    GINO maps between point clouds, so it takes the query points explicitly
    rather than inferring them from an array's shape: the cell centres it
    reads from (``input_geom``), a coarse regular grid that carries the
    spectral layers (``latent_queries``), and the points it writes to
    (``output_queries``, the same cell centres here).
    """

    input_geom: jnp.ndarray  # (1, H * W, 2)
    latent_queries: jnp.ndarray  # (1, L, L, 2)
    output_queries: jnp.ndarray  # (1, H * W, 2)
    neighbors_in: dict  # cell centres -> latent points
    neighbors_out: dict  # latent points -> cell centres
    in_radius: float
    out_radius: float


@lru_cache(maxsize=16)
def gino_geometry(
    height: int,
    width: int,
    latent_grid: int,
    in_radius_factor: float,
    out_radius_factor: float,
) -> GINOGeometry:
    """Builds :class:`GINOGeometry` for one grid size and latent resolution.

    The cavity mesh never changes between samples or between steps, so both
    neighbour searches are run once here and reused for every forward pass.
    Doing them inside the training step instead would repeat an O(n m) search
    on every batch for a result that is constant.

    Both radii are quoted as multiples of the latent-grid spacing rather than
    as absolute distances, so a coarser or finer latent grid stays connected
    without a second knob to remember. ``out_radius_factor=1.0`` covers every
    cell centre, since the furthest a point in a square cell of side ``s`` can
    sit from the nearest corner is ``s / sqrt(2)``.

    Cached because Flax retraces the module on every ``init`` and every new
    input shape, and the KD-trees cost more than the trace does.
    """
    points = cell_centres(height, width)
    axis = np.linspace(0.0, 1.0, latent_grid, dtype=np.float32)
    # Mirrors cell_centres' row-major (y outer, x inner) layout, so that the
    # flattened latent points line up with the (L, L) grid the FNO sees.
    lat_x, lat_y = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([lat_x.ravel(), lat_y.ravel()], axis=-1)

    spacing = 1.0 / (latent_grid - 1)
    in_radius = in_radius_factor * spacing
    out_radius = out_radius_factor * spacing

    return GINOGeometry(
        input_geom=jnp.asarray(points)[None],
        latent_queries=jnp.asarray(latent).reshape(1, latent_grid, latent_grid, 2),
        output_queries=jnp.asarray(points)[None],
        neighbors_in=precompute_neighbors(points, latent, in_radius),
        neighbors_out=precompute_neighbors(latent, points, out_radius),
        in_radius=in_radius,
        out_radius=out_radius,
    )


class PointCloudGINO(WindowModel):
    """Treats the cell grid as a point cloud and applies GINO to it.

    The window's frames and fields are stacked into the per-point input
    function, exactly as :class:`ChannelStackedFNO` stacks them into channels,
    so the two models see the same information and differ only in how they
    move it around: the FNO transforms the 128x128 grid directly, while GINO
    integrates the points onto a ``latent_grid`` square, runs its spectral
    layers there, and integrates back out to the points.

    The point positions are not folded into the input function. GINO applies
    its own NeRF-style positional embedding inside both GNO kernels, so
    passing coordinates again as channels would only duplicate them.

    ``in_gno_transform_type="nonlinear_kernelonly"`` rather than GINO's default
    ``"linear"``: the default multiplies the kernel's output elementwise
    against the input function, which forces the kernel's output width to equal
    the input function's channel count, here ``n_in * 3``. That would tie the
    GNO's width to the size of the time window. Feeding the input function into
    the kernel instead leaves ``gno_channels`` free, which is what
    this benchmark's earlier GINO cavity trainer did on this same mesh and for
    this same reason. The output GNO keeps the default, where the two widths already
    agree.
    """

    modes: int = 16
    width: int = 32
    n_layers: int = 1
    latent_grid: int = 32
    gno_channels: int = 32
    gno_mlp_hidden: tuple[int, ...] = (64, 64)
    in_radius_factor: float = 1.5
    out_radius_factor: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # n_rows/n_cols rather than height/width: `width` is this class's
        # channel count, and the two must not be confused.
        batch, steps, n_rows, n_cols, channels = x.shape
        geom = gino_geometry(
            n_rows,
            n_cols,
            self.latent_grid,
            self.in_radius_factor,
            self.out_radius_factor,
        )
        # (B, T, H, W, C) -> (B, H * W, T * C), row-major over (H, W) so the
        # points line up with cell_centres().
        f = x.transpose(0, 2, 3, 1, 4).reshape(batch, n_rows * n_cols, steps * channels)
        y = GINO(
            in_channels=steps * channels,
            out_channels=self.flat_out_channels,
            gno_coord_dim=2,
            gno_use_open3d=False,  # open3d's search is 3D only
            in_gno_radius=geom.in_radius,
            out_gno_radius=geom.out_radius,
            in_gno_transform_type="nonlinear_kernelonly",
            in_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            out_gno_channel_mlp_hidden_layers=self.gno_mlp_hidden,
            fno_in_channels=self.gno_channels,
            fno_n_modes=(self.modes, self.modes),
            fno_hidden_channels=self.width,
            fno_n_layers=self.n_layers,
            # As for the FNO above: this benchmark trains with Adam, whose
            # moment estimates would otherwise be complex-valued.
            fno_split_complex_params=True,
        )(
            input_geom=geom.input_geom,
            latent_queries=geom.latent_queries,
            output_queries=geom.output_queries,
            x=f,
            neighbors_in=geom.neighbors_in,
            neighbors_out=geom.neighbors_out,
        )
        return self.split_frames(
            y.reshape(batch, n_rows, n_cols, self.flat_out_channels)
        )


# ─────────────────────── architectures and their search ────────────────────
@dataclass(frozen=True)
class ArchitectureSpec(ABC):
    """One architecture: how to build it, and the sizes the paper searches.

    ``search_grid`` gives the choices the paper states for each knob;
    :meth:`enumerate_configs` expands them into every distinct size, which is
    what the random search samples from without replacement.
    """

    name: str
    label: str
    module: type[WindowModel]

    def build(self, arch: Mapping[str, Any], n_out: int) -> WindowModel:
        return self.module(n_out=n_out, **arch)

    @property
    @abstractmethod
    def search_grid(self) -> Mapping[str, Sequence[Any]]:
        """Choices per knob, in the paper's own terms."""

    @abstractmethod
    def to_arch(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        """One point of :attr:`search_grid` -> keyword arguments for the model."""

    @abstractmethod
    def slug(self, choice: Mapping[str, Any]) -> str:
        """Short, stable name for one point of :attr:`search_grid`."""

    def enumerate_configs(self) -> list[tuple[dict[str, Any], str]]:
        """Every distinct size in :attr:`search_grid`, as ``(arch, slug)``."""
        keys = list(self.search_grid)
        seen: dict[str, dict[str, Any]] = {}
        for values in product(*(self.search_grid[k] for k in keys)):
            choice = dict(zip(keys, values, strict=True))
            seen.setdefault(self.slug(choice), self.to_arch(choice))
        return [(arch, slug) for slug, arch in seen.items()]


@dataclass(frozen=True)
class FNOSpec(ArchitectureSpec):
    """Paper Section 2.2.3: up to 3 layers, modes up to 16 and width up to 32,
    in powers of 2."""

    name: str = "fno"
    label: str = "FNO"
    module: type[WindowModel] = ChannelStackedFNO

    @property
    def search_grid(self) -> Mapping[str, Sequence[Any]]:
        return dict(n_layers=(1, 2, 3), modes=(4, 8, 16), width=(8, 16, 32))

    def to_arch(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            modes=choice["modes"], width=choice["width"], n_layers=choice["n_layers"]
        )

    def slug(self, choice: Mapping[str, Any]) -> str:
        return f"L{choice['n_layers']}_m{choice['modes']}_w{choice['width']}"


@dataclass(frozen=True)
class GINOSpec(ArchitectureSpec):
    """The paper's FNO knobs for the latent backbone, plus the latent grid.

    Reusing the FNO's own search space for the spectral layers keeps the two
    comparable: whatever GINO gives up by going through a point cloud, it does
    not give up by being searched over a smaller set of spectral capacities.
    The latent resolution is the one knob GINO adds, and it is the one that
    decides how much of the 128x128 field survives the trip through the latent
    grid.
    """

    name: str = "gino"
    label: str = "GINO"
    module: type[WindowModel] = PointCloudGINO

    @property
    def search_grid(self) -> Mapping[str, Sequence[Any]]:
        return dict(
            n_layers=(1, 2, 3),
            modes=(4, 8, 16),
            width=(8, 16, 32),
            latent=(16, 32),
        )

    def to_arch(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            modes=min(choice["modes"], choice["latent"] // 2),
            width=choice["width"],
            n_layers=choice["n_layers"],
            latent_grid=choice["latent"],
        )

    def slug(self, choice: Mapping[str, Any]) -> str:
        arch = self.to_arch(choice)
        return (
            f"L{arch['n_layers']}_m{arch['modes']}_w{arch['width']}"
            f"_g{arch['latent_grid']}"
        )


@dataclass(frozen=True)
class WNOSpec(ArchitectureSpec):
    """The paper's layer-count choices, plus the two knobs the wavelet adds.

    ``n_layers`` is the paper's own knob and keeps its choices. ``hidden``
    stands where the FNO's ``width`` stands. ``level`` stands where ``modes``
    stands, and runs the opposite way: a deeper decomposition leaves a smaller
    approximation band, so the weights cover fewer coefficients and the model
    has fewer parameters, exactly as keeping fewer Fourier modes does. Levels
    3 to 5 are the range in which a 128x128 field still has an approximation
    band worth weighting, from 25x25 coefficients down to 14x14.

    The wavelet family is fixed at ``db6`` rather than searched. It is the
    value this repo's own WNO reproduction uses
    (``scripts/benchmarks/darcy/``, tracking Tripura
    and Chakraborty's Table 1), and searching a categorical choice alongside
    three numeric ones would spend the ten-config budget on the wrong
    question.
    """

    name: str = "wno"
    label: str = "WNO"
    module: type[WindowModel] = GridWNO

    @property
    def search_grid(self) -> Mapping[str, Sequence[Any]]:
        return dict(n_layers=(1, 2, 3), level=(3, 4, 5), hidden=(8, 16, 32))

    def to_arch(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            hidden_channels=choice["hidden"],
            n_layers=choice["n_layers"],
            level=choice["level"],
        )

    def slug(self, choice: Mapping[str, Any]) -> str:
        return f"L{choice['n_layers']}_v{choice['level']}_h{choice['hidden']}"


@dataclass(frozen=True)
class TransolverSpec(ArchitectureSpec):
    """The paper's layer-count choices, plus width and the slice count.

    ``n_layers`` is the paper's own knob and keeps its choices. ``hidden``
    stands where the FNO's ``width`` stands, in the same powers of two.
    ``slices`` stands where ``modes`` stands: both say how many basis
    elements the operator gets to work with. The difference is that Fourier
    modes are chosen in advance and slices are learned, so a slice costs
    almost no parameters, only a ``dim_head`` by ``slice_num`` projection per
    layer, while a mode costs a full weight matrix. The parameter count here
    is therefore almost entirely ``n_layers`` and ``hidden``, and ``slices``
    buys expressivity rather than capacity.

    ``n_head``, ``mlp_ratio``, ``ref`` and the 3-wide convolution kernel are
    fixed at Wu et al.'s values rather than searched, for the reason
    :class:`WNOSpec` fixes the wavelet family: spending the ten-configuration
    budget on them would answer a smaller question. The widths are all
    multiples of ``n_head``, which
    :class:`GridTokenTransolver` explains.
    """

    name: str = "transolver"
    label: str = "Transolver"
    module: type[WindowModel] = GridTokenTransolver

    @property
    def search_grid(self) -> Mapping[str, Sequence[Any]]:
        return dict(n_layers=(1, 2, 3), hidden=(32, 64, 128), slices=(16, 32, 64))

    def to_arch(self, choice: Mapping[str, Any]) -> dict[str, Any]:
        return dict(
            n_hidden=choice["hidden"],
            n_layers=choice["n_layers"],
            slice_num=choice["slices"],
        )

    def slug(self, choice: Mapping[str, Any]) -> str:
        return f"L{choice['n_layers']}_h{choice['hidden']}_s{choice['slices']}"


#: Every architecture the benchmark knows, keyed by the name used on the
#: command line, in ``RunConfig.model`` and in the results page.
SPECS: Mapping[str, ArchitectureSpec] = {
    spec.name: spec
    for spec in (
        FNOSpec(),
        GINOSpec(),
        WNOSpec(),
        TransolverSpec(),
    )
}


def build_model(name: str, n_out: int, **arch: Any) -> WindowModel:
    """The window model named ``name``, at the size given by ``arch``."""
    try:
        spec = SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose from {sorted(SPECS)}"
        ) from None
    return spec.build(arch, n_out)
