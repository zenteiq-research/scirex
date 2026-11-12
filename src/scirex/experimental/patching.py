# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""Multi-grid domain decomposition and patching for large-resolution inputs."""

import logging
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from scirex.data.transforms.data_processors import DataProcessor

logger = logging.getLogger(__name__)


def unfold_jax(
    x: jnp.ndarray, dimension: int, kernel_size: int, stride: int
) -> jnp.ndarray:
    """JAX implementation of PyTorch's unfold operation.

    Unfold returns a tensor with a new sliding window dimension.
    For x.unfold(dim, size, step), returns a tensor with:

    - Original dimension `dim` replaced by n_windows
    - New dimension of size `kernel_size` at the end

    Parameters
    ----------
    x : jnp.ndarray
        input tensor
    dimension : int
        dimension along which to unfold
    kernel_size : int
        size of the sliding window
    stride : int
        step size between consecutive windows

    Returns
    -------
    jnp.ndarray
        unfolded tensor
    """
    shape = x.shape
    ndim = x.ndim

    # Handle negative dimensions
    if dimension < 0:
        dimension = ndim + dimension

    dim_size = shape[dimension]
    n_windows = (dim_size - kernel_size) // stride + 1

    # Move target dimension to front for easier processing
    perm_forward = [dimension] + [i for i in range(ndim) if i != dimension]
    x_moved = jnp.transpose(x, perm_forward)

    # Extract sliding windows
    # x_moved shape: (dim_size, ...)
    # We want: (n_windows, kernel_size, ...)
    windows_list = []
    for i in range(n_windows):
        window = jax.lax.slice_in_dim(
            x_moved, i * stride, i * stride + kernel_size, 1, axis=0
        )
        windows_list.append(window)

    windows = jnp.stack(windows_list, axis=0)
    # windows shape: (n_windows, kernel_size, ...)

    # Permute back to match PyTorch unfold output
    # From (n_windows, kernel_size, rest_dims...) to
    # (rest_dims..., n_windows, kernel_size) where rest_dims correspond to
    # original dimensions except the unfolded one
    ndim_windows = windows.ndim
    perm_back = [*list(range(2, ndim_windows)), 0, 1]
    result = jnp.transpose(windows, perm_back)

    return result


class MultigridPatching2D(nn.Module):
    """
    MultigridPatching2D wraps a model in multi-grid domain decomposition and patching.

    Parameters
    ----------
    model : nn.Module
        model to wrap
    levels : int, optional
        number of levels of patching to use, by default 0
    padding_fraction : float, optional
        fraction of input size to add as padding around patches, by default 0
    use_distributed : bool, optional
        whether patching is performed in distributed mode, by default False
    stitching : bool, optional
        whether patches are to be stitched back together
        in spatial dimensions during training, by default True.
        Stitching is always performed during evaluation.
    """

    model: nn.Module | None = None
    levels: int = 0
    padding_fraction: float | list[float] = 0.0
    use_distributed: bool = False
    stitching: bool = True

    padding_height: int = 0
    padding_width: int = 0

    def setup(self) -> None:
        """Initialize MultigridPatching2D.

        If computation is split into distributed data or model parallel, adds parameter
        hooks to account for scattering patches across multiple processes.
        """
        if isinstance(self.padding_fraction, float | int):
            padding_fraction = [self.padding_fraction, self.padding_fraction]
        else:
            padding_fraction = self.padding_fraction
        self.padding_fraction = padding_fraction

        n_patches = 2**self.levels
        if isinstance(n_patches, int):
            n_patches = [n_patches, n_patches]
        self.n_patches = n_patches

        if self.levels > 0:
            logger.info(
                "MGPatching("
                f"n_patches={self.n_patches}, "
                f"padding_fraction={self.padding_fraction}, "
                f"levels={self.levels}, "
                f"use_distributed={self.use_distributed}, "
                f"stitching={self.stitching})"
            )

    def patch(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """use multi-grid domain decomposition to split `x` and `y` into patches.
        If in a distributed scheme, scatters patches across processes.

        Parameters
        ----------
        x : jnp.ndarray
            model input function
        y : jnp.ndarray
            model output function
        """
        # if not stitching in single-device, create patches for y
        if not self.stitching:
            y = make_patches(y, n=self.n_patches, p=0)
        # If not stitching, scatter truth, otherwise keep on every GPU
        if self.use_distributed:
            from scirex.experimental.distributed.mappings import (
                scatter_to_model_parallel_region,
            )

            y = scatter_to_model_parallel_region(y, 0)

        # Create padded patches in batch dimension (identity if levels=0)
        x = self._make_mg_patches(x)
        # Split data across processes
        if self.use_distributed:
            from scirex.experimental.distributed.mappings import (
                scatter_to_model_parallel_region,
            )

            x = scatter_to_model_parallel_region(x, 0)
        return x, y

    def unpatch(
        self, x: jnp.ndarray, y: jnp.ndarray, evaluation: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """unpatch tensors created by `self.patch`. Stitch patches together if in
        evaluation mode, or if stitching is applied.

        Parameters
        ----------
        x : jnp.ndarray
            tensor with patching structure created patching input `x`. May be
            either inputs `x` or raw model outputs (same shape/patching structure)
            Shape (b * n^2, c, h / n + 2 * pad_h, w / n + 2 * pad_w)
        y : jnp.ndarray
            tensor of patched ground-truth `y`.
            Shape (b * n^2, c, h / n, w / n) or (b, c, h, w) when not stitched
        evaluation : bool, optional
            whether in evaluation mode, by default False.
            If True, `x` and `y` are both evaluated after stitching,
            regardless of other settings.
        """
        # Remove padding in the output
        if self.padding_height > 0 or self.padding_width > 0:
            x = self._unpad(x)

        # Gather patches if they are to be stitched back together
        if self.use_distributed and self.stitching:
            from scirex.experimental.distributed.mappings import (
                gather_from_model_parallel_region,
            )

            x = gather_from_model_parallel_region(x, dim=0)

        # Stich patches or patch the truth if output left unstitched
        if self.stitching or evaluation:
            x = self._stitch(x)

        # if x is not stitched during training, y is patched
        # re-stitch y during evaluation only
        if evaluation and not self.stitching:
            y = self._stitch(y)

        return x, y

    def _stitch(self, x: jnp.ndarray) -> jnp.ndarray:
        """Stitch back together multi-grid patches created by `self._make_mg_patches`.

        Small patches are collated along the batch dimension as different
        inputs. Unroll the batch dimension and stick all patches from the
        same input back together in their proper locations.

        For an input shape (n * n * n, c, h / n, w / n),
        produces an output tensor of shape (b, c, h, w)

        Parameters
        ----------
        x : jnp.ndarray
            input tensor, split into patches and collated along batch dim
            shape (batch * n^2, c, h / n, w / n)
        """
        # Only 1D and 2D supported
        assert x.ndim == 4, f"Only 2D patch supported but got input with {x.ndim} dims."

        if self.n_patches[0] <= 1 and self.n_patches[1] <= 1:
            return x

        # Size with padding removed
        size = x.shape

        # if self.mode == "batch-wise":
        B = size[0] // (self.n_patches[0] * self.n_patches[1])
        W = size[3] * self.n_patches[1]

        C = size[1]
        H = size[2] * self.n_patches[0]

        # Reshape
        x = jnp.transpose(x, (0, 3, 2, 1))
        x = jnp.reshape(
            x, (B, self.n_patches[0], self.n_patches[1], size[3], size[2], C)
        )
        x = jnp.transpose(x, (0, 5, 1, 4, 2, 3))
        x = jnp.reshape(x, (B, C, H, W))

        return x

    def _make_mg_patches(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Split a tensor `x` into multi-grid patches. The number of
        patches into which `x` is split is determined by `2 ** self.levels`.

        Steps:

        1. Optionally apply padding if self.padding_fraction > 0

        2. Split each element in the batch into `2**self.levels` patches of equal size

        3. For each level L in `self.levels`, subsample all patches by a
           factor of `2**L`

        4. Pad the subsampled patches so all level tensors are the same size

        5. Concatenate the patches along the channel dim and return

        Parameters
        ----------
        x : jnp.ndarray
            input tensor to split into patches
            shape (batch, c, h, w)
        """
        levels = self.levels
        if levels <= 0:
            return x

        _, _, height, width = x.shape
        # setup() always normalizes padding_fraction to a 2-element list
        # before this method can run.
        assert isinstance(self.padding_fraction, list)
        padding = [
            round(v)
            for v in [
                height * self.padding_fraction[0],
                width * self.padding_fraction[1],
            ]
        ]
        self.padding_height = padding[0]
        self.padding_width = padding[1]

        patched = make_patches(x, n=2**self.levels, p=padding)
        s1_patched = patched.shape[-2] - 2 * padding[0]
        s2_patched = patched.shape[-1] - 2 * padding[1]

        for level in range(1, levels + 1):
            sub_sample = 2**level
            s1_stride = s1_patched // sub_sample
            s2_stride = s2_patched // sub_sample

            x_sub = x[:, :, ::sub_sample, ::sub_sample]

            s2_pad = (
                math.ceil(
                    (s2_patched + (2**levels - 1) * s2_stride - x_sub.shape[-1]) / 2.0
                )
                + padding[1]
            )
            s1_pad = (
                math.ceil(
                    (s1_patched + (2**levels - 1) * s1_stride - x_sub.shape[-2]) / 2.0
                )
                + padding[0]
            )

            if s2_pad > x_sub.shape[-1]:
                diff = s2_pad - x_sub.shape[-1]
                x_sub = jnp.pad(
                    x_sub,
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (x_sub.shape[-1], x_sub.shape[-1]),
                    ),
                    mode="wrap",
                )
                x_sub = jnp.pad(
                    x_sub, pad_width=((0, 0), (0, 0), (0, 0), (diff, diff)), mode="wrap"
                )
            else:
                x_sub = jnp.pad(
                    x_sub,
                    pad_width=((0, 0), (0, 0), (0, 0), (s2_pad, s2_pad)),
                    mode="wrap",
                )

            if s1_pad > x_sub.shape[-2]:
                diff = s1_pad - x_sub.shape[-2]
                x_sub = jnp.pad(
                    x_sub,
                    pad_width=(
                        (0, 0),
                        (0, 0),
                        (x_sub.shape[-2], x_sub.shape[-2]),
                        (0, 0),
                    ),
                    mode="wrap",
                )
                x_sub = jnp.pad(
                    x_sub, pad_width=((0, 0), (0, 0), (diff, diff), (0, 0)), mode="wrap"
                )
            else:
                x_sub = jnp.pad(
                    x_sub,
                    pad_width=((0, 0), (0, 0), (s1_pad, s1_pad), (0, 0)),
                    mode="wrap",
                )

            x_sub = unfold_jax(
                x_sub,
                dimension=-1,
                kernel_size=s2_patched + 2 * padding[1],
                stride=s2_stride,
            )
            x_sub = unfold_jax(
                x_sub,
                dimension=-3,
                kernel_size=s1_patched + 2 * padding[0],
                stride=s1_stride,
            )

            x_sub = jnp.transpose(x_sub, (0, 2, 3, 4, 5, 1))
            x_sub = jnp.reshape(
                x_sub,
                (
                    patched.shape[0],
                    s2_patched + 2 * padding[1],
                    s1_patched + 2 * padding[0],
                    -1,
                ),
            )
            x_sub = jnp.transpose(x_sub, (0, 3, 2, 1))

            patched = jnp.concatenate((patched, x_sub), axis=1)

        return patched

    def _unpad(self, x: jnp.ndarray) -> jnp.ndarray:
        """Remove padding around the edges (`mode=wrap`) of tensor `x`.

        Parameters
        ----------
        x : jnp.ndarray
            padded input tensor, shape
            (b, c, h + self.padding_height * 2, w + self.padding_width * 2)

        Returns
        -------
        x : jnp.ndarray
            unpadded tensor, shape (b, c, h, w)
        """
        return x[
            ...,
            self.padding_height : x.shape[-2] - self.padding_height,
            self.padding_width : x.shape[-1] - self.padding_width,
        ]


def make_patches(
    x: jnp.ndarray, n: int | list[int], p: int | list[int] = 0
) -> jnp.ndarray:
    """make_patches splits `x` into `n` equally-sized patches
    with padding fraction `p`. Stacks patches along the batch dimension.

    Starting with an input tensor of shape (batch, C, s) or (batch, C, h, w),
    returns a corresponding patched output tensor of shape (n * batch, C, s / n + 2p)
    or (n1 * n2 * batch, C, h / n1 + 2 * p1, w / n2 + 2 * p2)

    Parameters
    ----------
    x : jnp.ndarray
        input tensor, before patching
    n : int or list
        number of patches into which to split each example in `x`
    p : int or list, optional
        number of pixels to use when padding `x`, by default 0
    """
    size = x.shape

    # Only 1D and 2D supported
    assert len(size) == 3 or len(size) == 4

    if len(size) == 3:
        d = 1
    else:
        d = 2

    if isinstance(p, int):
        p = [p, p]

    # Pad
    if p[0] > 0 or p[1] > 0:
        if d == 1:
            x = jnp.pad(x, pad_width=((0, 0), (0, 0), (p[0], p[0])), mode="wrap")
        else:
            x = jnp.pad(
                x, pad_width=((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode="wrap"
            )

    if isinstance(n, int):
        n = [n, n]

    if n[0] <= 1 and n[1] <= 1:
        return x

    # Patches must be equally sized
    for j in range(d):
        assert size[-(j + 1)] % n[-(j + 1)] == 0

    # Patch using unfold
    for j in range(d):
        patch_size = size[-(j + 1)] // n[-(j + 1)]
        x = unfold_jax(
            x,
            dimension=-(2 * j + 1),
            kernel_size=patch_size + 2 * p[-(j + 1)],
            stride=patch_size,
        )

    x = jnp.transpose(x, (0, 2, 3, 4, 5, 1))
    x = jnp.reshape(
        x,
        (
            size[0] * n[0] * n[1],
            size[-1] // n[-1] + 2 * p[-1],
            size[-2] // n[-2] + 2 * p[-2],
            size[1],
        ),
    )
    x = jnp.transpose(x, (0, 3, 2, 1))

    return x


class MGPatchingDataProcessor(DataProcessor):
    """Data processor that applies multigrid patching to the inputs of a model.

    The input is cut into overlapping patches before the forward pass and, with
    `stitching`, the patch outputs are joined back into one field afterwards. An optional
    normalizer encodes the inputs and decodes the outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        levels: int,
        padding_fraction: float,
        stitching: bool,
        device: str = "cpu",
        use_distributed: bool = False,
        in_normalizer: Any | None = None,
        out_normalizer: Any | None = None,
    ) -> None:
        """MGPatchingDataProcessor
        Applies multigrid patching to inputs out-of-place
        with an optional output encoder/other data transform

        Parameters
        ----------
        model: nn.Module
            model to wrap in MultigridPatching2D
        levels : int
            number of multi-grid patching levels to use
        padding_fraction : float
            fraction by which to pad inputs in multigrid-patching
        stitching : bool
            whether to stitch back the output from the multi-grid patches
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model inputs, by default None
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model outputs, by default None
        device : str, optional
            device 'cuda' or 'cpu' where computations are performed
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.patcher = MultigridPatching2D(
            model=model,
            levels=self.levels,
            padding_fraction=self.padding_fraction,
            stitching=self.stitching,
            use_distributed=use_distributed,
        )
        self.device = device

        # set normalizers to none by default
        self.in_normalizer, self.out_normalizer = None, None
        if in_normalizer:
            self.in_normalizer = in_normalizer.to(self.device)
        if out_normalizer:
            self.out_normalizer = out_normalizer.to(self.device)
        self.model = model

    def to(self, device: str) -> "MGPatchingDataProcessor":
        self.device = device
        if self.in_normalizer:
            self.in_normalizer = self.in_normalizer.to(self.device)
        if self.out_normalizer:
            self.out_normalizer = self.out_normalizer.to(self.device)
        return self

    def preprocess(self, data_dict: dict, batched: bool = True) -> dict:
        """
        Preprocess data assuming that if encoder exists, it has
        encoded all data during data loading

        Parameters
        ----------
        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        batched: bool
            whether the first dimension of 'x', 'y' represents batching
        """
        data_dict = {
            k: jnp.asarray(v)
            for k, v in data_dict.items()
            if isinstance(v, jnp.ndarray)
        }
        x, y = data_dict["x"], data_dict["y"]
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer:
            y = self.out_normalizer.transform(y)
        data_dict["x"], data_dict["y"] = self.patcher.patch(x, y)

        return data_dict

    def postprocess(
        self, out: jnp.ndarray, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        """
        Postprocess model outputs.

        Parameters
        ----------
        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        out: jnp.ndarray
            model output predictions
        training: bool, optional
            whether in training mode, by default True
        """
        y = data_dict["y"]
        out, y = self.patcher.unpatch(out, y, evaluation=not training)

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)

        data_dict["y"] = y

        return out, data_dict

    def __call__(
        self, data_dict: dict, training: bool = True
    ) -> tuple[jnp.ndarray, dict]:
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict, training=training)
        return output, data_dict
