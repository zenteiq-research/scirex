# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Decoder-only GIWO (Geometry-Informed Wavelet Operator) for this benchmark.

Not a published architecture: GINO with its latent FNO swapped for a
:class:`scirex.models.WNO`, everything else unchanged. The library's own
:class:`scirex.experimental.giwo.GIWO` always runs an input GNO over the point cloud;
this benchmark's GINO run is the paper's decoder-only variant, whose latent
input is the SDF on the latent grid, so GIWO here has to be too.

It is built as a :class:`~scirex.models.GINO` subclass so the
decoder-only shell the ``"paper"`` run uses -- the SDF on the latent grid as
the whole latent input, the output GNO, the projection head, the neighbour
precomputation -- is GINO's own code, not a copy. Only
:meth:`GIWO.latent_embedding` differs: GINO's ``lifting`` + ``fno_blocks``
become one channels-last :class:`WNO` (lift, ``wno_n_layers`` wavelet blocks,
two-layer projection back to ``fno_hidden_channels``). GINO's own
``lifting``/``fno_blocks`` are still declared by ``GINO.setup`` but never
called, so they own no parameters.

``fno_hidden_channels`` keeps its GINO meaning: the latent width the output
GNO and the projection head read.

Importing this module switches WaveletConv's 3-D transform to
:mod:`._fast_dwt`'s separable one (the same transform, ~9x faster on the
64^3 grid): ~0.17 s per training car instead of ~1.1 s.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from scirex.models import GINO, WNO
from scripts.benchmarks.car_cfd._fast_dwt import use_separable_3d_transforms

use_separable_3d_transforms()


# Registered under its own name: the library's GIWO already owns "giwo".
class GIWO(GINO, name="giwo_decoder"):
    """GINO with a WNO latent operator (see the module docstring).

    Parameters (beyond GINO's)
    --------------------------
    wno_size : tuple[int, int, int]
        The latent grid's shape; WNO's subband weights are shaped from it.
    wno_level, wno_wavelet, wno_mode, wno_n_layers : see :class:`WNO`.
    wno_projection_hidden_channels : int
        Hidden width of WNO's own two-layer projection to
        ``fno_hidden_channels``.
    """

    wno_size: Any = (64, 64, 64)
    wno_level: int = 3
    wno_wavelet: str = "db4"
    wno_mode: str = "symmetric"
    wno_n_layers: int = 4
    wno_projection_hidden_channels: int = 128

    def setup(self) -> None:
        super().setup()
        self.wno = WNO(
            in_channels=self._fno_in_channels,
            out_channels=self.fno_hidden_channels,
            hidden_channels=self.fno_hidden_channels,
            n_layers=self.wno_n_layers,
            level=self.wno_level,
            size=tuple(self.wno_size),
            wavelet=self.wno_wavelet,
            mode=self.wno_mode,
            # float32 throughout, as GINO (fno_block_precision="full") and
            # Transolver run, rather than WNO's bfloat16 default.
            compute_dtype=jnp.float32,
            projection_hidden_channels=self.wno_projection_hidden_channels,
        )

    def latent_embedding(self, in_p: Any, ada_in: Any = None) -> Any:
        """``in_p`` (batch, n_1, n_2, n_3, c) -> (batch, hidden, n_1, n_2, n_3),
        the channels-first layout ``GINO.encode`` expects back."""
        del ada_in
        out = self.wno(in_p)
        return jnp.moveaxis(out, -1, 1)
