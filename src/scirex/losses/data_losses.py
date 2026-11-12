# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Lp (relative and absolute) and H1 Sobolev losses for comparing discretized functions.
"""

import math
import warnings
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

# Set warning filter to show each warning only once
warnings.filterwarnings("once", category=UserWarning)


# loss function with rel/abs Lp loss
class LpLoss:
    """LpLoss provides the Lp norm between two discretized d-dimensional functions.

    Note that LpLoss always averages over the spatial dimensions.

    !!! note
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular
        discretization and take `1 / measure` as the quadrature weights.

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: $\\left[\\sum_{i=0}^{n} \\lvert x_i - y_i\\rvert^p\\right]^{1/p}$
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        !!! note
            To perform quadrature, `LpLoss` scales `measure` by the size
            of each spatial dimension of `x`, and multiplies them with
            ||x-y||, such that the final norm is a scaled average over the spatial
            dimensions of `x`.
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        !!! warning
            `LpLoss` always reduces over the spatial dimensions according to
            `self.measure`. `reduction` only applies to the batch and channel
            dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when
        using the relative loss

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scirex.losses import LpLoss
    >>> y = jnp.ones((2, 1, 8, 8))  # (batch, channels, height, width)
    >>> loss = LpLoss(d=2, p=2, reduction="mean")
    >>> round(float(loss.rel(1.1 * y, y)), 3)  # relative L2 error
    0.1
    """

    def __init__(
        self,
        d: int = 1,
        p: int = 2,
        measure: float | Sequence[float] = 1.0,
        reduction: str = "sum",
        eps: float = 1e-8,
    ) -> None:
        """Configure the Lp loss. See the class docstring for parameter details."""
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps

        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions, (
            f"error: expected `reduction` to be one of {allowed_reductions}, "
            f"got {reduction}"
        )
        self.reduction = reduction

        self.measure: Sequence[float]
        if isinstance(measure, int | float):
            self.measure = [measure] * self.d
        else:
            self.measure = measure

    @property
    def name(self) -> str:
        """str: Human-readable identifier, e.g. `"L2_1Dloss"`."""
        return f"L{self.p}_{self.d}Dloss"

    def uniform_quadrature(self, x: jnp.ndarray) -> list[float]:
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of `x` to ensure that
        `LpLoss` computes the average over spatial dims.

        Parameters
        ----------
        x : jnp.ndarray
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0] * self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j] / x.shape[-j]

        return quadrature

    def reduce_all(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        reduce x across the batch according to `self.reduction`

        Parameters
        ----------
        x: jnp.ndarray
            inputs
        """
        if self.reduction == "sum":
            x = jnp.sum(x)
        else:
            x = jnp.mean(x)

        return x

    def abs(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        quadrature: float | Sequence[float] | None = None,
        take_root: bool = True,
    ) -> jnp.ndarray:
        """Absolute Lp-norm

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        # Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, int | float):
                quadrature = [quadrature] * self.d

        # Flatten last d dimensions
        x_shape_prefix = x.shape[: -self.d]
        x_flat = jnp.reshape(x, (*x_shape_prefix, -1))
        y_flat = jnp.reshape(y, (*x_shape_prefix, -1))

        diff_flat = x_flat - y_flat

        if self.p == 1:
            const = math.prod(quadrature)
            diff = const * jnp.sum(jnp.abs(diff_flat), axis=-1, keepdims=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            const = math.prod(quadrature)
            diff = const * jnp.sum(diff_flat**self.p, axis=-1, keepdims=False)
        else:
            const = math.prod(quadrature)
            diff = const * jnp.sum(
                jnp.abs(diff_flat) ** self.p, axis=-1, keepdims=False
            )

        if take_root and self.p != 1:
            diff = diff ** (1.0 / self.p)

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def rel(
        self, x: jnp.ndarray, y: jnp.ndarray, take_root: bool = True
    ) -> jnp.ndarray:
        """
        rel: relative LpLoss
        computes ||x-y||/(||y|| + eps)

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        # Flatten last d dimensions
        x_shape_prefix = x.shape[: -self.d]
        x_flat = jnp.reshape(x, (*x_shape_prefix, -1))
        y_flat = jnp.reshape(y, (*x_shape_prefix, -1))

        diff_flat = x_flat - y_flat

        if self.p == 1:
            diff = jnp.sum(jnp.abs(diff_flat), axis=-1, keepdims=False)
            ynorm = jnp.sum(jnp.abs(y_flat), axis=-1, keepdims=False)
        elif self.p % 2 == 0:  # Even power p: no need for abs() since x^p > 0
            diff = jnp.sum(diff_flat**self.p, axis=-1, keepdims=False)
            ynorm = jnp.sum(y_flat**self.p, axis=-1, keepdims=False)
        else:
            diff = jnp.sum(jnp.abs(diff_flat) ** self.p, axis=-1, keepdims=False)
            ynorm = jnp.sum(jnp.abs(y_flat) ** self.p, axis=-1, keepdims=False)

        if take_root and self.p != 1:
            diff = (diff ** (1.0 / self.p)) / (ynorm ** (1.0 / self.p) + self.eps)
        else:
            diff = diff / (ynorm + self.eps)

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def __call__(
        self, y_pred: jnp.ndarray, y: jnp.ndarray, take_root: bool = True, **kwargs: Any
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        y_pred : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        take_root : bool, optional
            whether to take the p-th root of the norm, by default True
        """
        return self.rel(y_pred, y, take_root=take_root)


class H1Loss:
    """H1 Sobolev norm between two d-dimensional discretized functions.

    !!! note
        In function space, the Sobolev norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to H1Loss, we assume a regular
        discretization and take `1 / measure` as the quadrature weights.

    Parameters
    ----------
    d : int, optional
        dimension of input functions, by default 1
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        !!! note
            To perform quadrature, `H1Loss` scales `measure` by the size
            of each spatial dimension of `x`, and multiplies them with
            ||x-y||, such that the final norm is a scaled average over the spatial
            dimensions of `x`.

    reduction : str, optional
        whether to reduce across the batch and channel dimension
        by summing ('sum') or averaging ('mean')

        !!! warning
            H1Loss always averages over the spatial dimensions.
            `reduction` only applies to the batch and channel dimensions.
    eps : float, optional
        small number added to the denominator for numerical stability when
        using the relative loss
    periodic_in_x : bool, optional
        whether to use periodic boundary conditions in x-direction when
        computing finite differences:

        - True: periodic in x (default)
        - False: non-periodic in x with forward/backward differences at boundaries
        by default True
    periodic_in_y : bool, optional
        whether to use periodic boundary conditions in y-direction when
        computing finite differences:

        - True: periodic in y (default)
        - False: non-periodic in y with forward/backward differences at boundaries
        by default True
    """

    def __init__(
        self,
        d: int = 1,
        measure: float | Sequence[float] = 1.0,
        reduction: str = "sum",
        eps: float = 1e-8,
        periodic_in_x: bool = True,
        periodic_in_y: bool = True,
        periodic_in_z: bool = True,
    ) -> None:
        """Configure the H1 Sobolev loss. See the class docstring for parameters."""
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3D."

        self.d = d
        self.periodic_in_x = periodic_in_x
        self.periodic_in_y = periodic_in_y
        self.periodic_in_z = periodic_in_z

        self.eps = eps

        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions, (
            f"error: expected `reduction` to be one of {allowed_reductions}, "
            f"got {reduction}"
        )
        self.reduction = reduction

        self.measure: Sequence[float]
        if isinstance(measure, int | float):
            self.measure = [measure] * self.d
        else:
            self.measure = measure

    @property
    def name(self) -> str:
        """str: Human-readable identifier, e.g. `"H1_1DLoss"`."""
        return f"H1_{self.d}DLoss"

    def _dx_1d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order x-derivative (1D), periodic or non-periodic."""
        if self.periodic_in_x:
            return (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * h)
        else:
            interior = (u[..., 2:] - u[..., :-2]) / (2.0 * h)
            left = (
                (-11 * u[..., 0] + 18 * u[..., 1] - 9 * u[..., 2] + 2 * u[..., 3])
                / (6.0 * h)
            )[..., jnp.newaxis]
            right = (
                (-2 * u[..., -4] + 9 * u[..., -3] - 18 * u[..., -2] + 11 * u[..., -1])
                / (6.0 * h)
            )[..., jnp.newaxis]
            return jnp.concatenate([left, interior, right], axis=-1)

    def _dx_2d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order x-derivative (2D, operates on axis -2)."""
        if self.periodic_in_x:
            return (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * h)
        else:
            interior = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * h)
            left = (
                -11 * u[..., 0:1, :]
                + 18 * u[..., 1:2, :]
                - 9 * u[..., 2:3, :]
                + 2 * u[..., 3:4, :]
            ) / (6.0 * h)
            right = (
                -2 * u[..., -4:-3, :]
                + 9 * u[..., -3:-2, :]
                - 18 * u[..., -2:-1, :]
                + 11 * u[..., -1:, :]
            ) / (6.0 * h)
            return jnp.concatenate([left, interior, right], axis=-2)

    def _dy_2d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order y-derivative (2D, operates on axis -1)."""
        if self.periodic_in_y:
            return (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * h)
        else:
            interior = (u[..., :, 2:] - u[..., :, :-2]) / (2.0 * h)
            bottom = (
                -11 * u[..., :, 0:1]
                + 18 * u[..., :, 1:2]
                - 9 * u[..., :, 2:3]
                + 2 * u[..., :, 3:4]
            ) / (6.0 * h)
            top = (
                -2 * u[..., :, -4:-3]
                + 9 * u[..., :, -3:-2]
                - 18 * u[..., :, -2:-1]
                + 11 * u[..., :, -1:]
            ) / (6.0 * h)
            return jnp.concatenate([bottom, interior, top], axis=-1)

    def _dx_3d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order x-derivative (3D, operates on axis -3)."""
        if self.periodic_in_x:
            return (jnp.roll(u, -1, axis=-3) - jnp.roll(u, 1, axis=-3)) / (2.0 * h)
        else:
            interior = (u[..., 2:, :, :] - u[..., :-2, :, :]) / (2.0 * h)
            left = (
                -11 * u[..., 0:1, :, :]
                + 18 * u[..., 1:2, :, :]
                - 9 * u[..., 2:3, :, :]
                + 2 * u[..., 3:4, :, :]
            ) / (6.0 * h)
            right = (
                -2 * u[..., -4:-3, :, :]
                + 9 * u[..., -3:-2, :, :]
                - 18 * u[..., -2:-1, :, :]
                + 11 * u[..., -1:, :, :]
            ) / (6.0 * h)
            return jnp.concatenate([left, interior, right], axis=-3)

    def _dy_3d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order y-derivative (3D, operates on axis -2)."""
        if self.periodic_in_y:
            return (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * h)
        else:
            interior = (u[..., :, 2:, :] - u[..., :, :-2, :]) / (2.0 * h)
            left = (
                -11 * u[..., :, 0:1, :]
                + 18 * u[..., :, 1:2, :]
                - 9 * u[..., :, 2:3, :]
                + 2 * u[..., :, 3:4, :]
            ) / (6.0 * h)
            right = (
                -2 * u[..., :, -4:-3, :]
                + 9 * u[..., :, -3:-2, :]
                - 18 * u[..., :, -2:-1, :]
                + 11 * u[..., :, -1:, :]
            ) / (6.0 * h)
            return jnp.concatenate([left, interior, right], axis=-2)

    def _dz_3d(self, u: jnp.ndarray, h: float) -> jnp.ndarray:
        """First-order z-derivative (3D, operates on axis -1)."""
        if self.periodic_in_z:
            return (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2.0 * h)
        else:
            interior = (u[..., :, :, 2:] - u[..., :, :, :-2]) / (2.0 * h)
            bottom = (
                -11 * u[..., :, :, 0:1]
                + 18 * u[..., :, :, 1:2]
                - 9 * u[..., :, :, 2:3]
                + 2 * u[..., :, :, 3:4]
            ) / (6.0 * h)
            top = (
                -2 * u[..., :, :, -4:-3]
                + 9 * u[..., :, :, -3:-2]
                - 18 * u[..., :, :, -2:-1]
                + 11 * u[..., :, :, -1:]
            ) / (6.0 * h)
            return jnp.concatenate([bottom, interior, top], axis=-1)

    def compute_terms(
        self, x: jnp.ndarray, y: jnp.ndarray, quadrature: Sequence[float]
    ) -> tuple[dict[int, jnp.ndarray], dict[int, jnp.ndarray]]:
        """compute_terms computes the necessary
        finite-difference derivative terms for computing
        the H1 norm

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : int or list
            quadrature weights

        """
        dict_x: dict[int, jnp.ndarray] = {}
        dict_y: dict[int, jnp.ndarray] = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = self._dx_1d(x, quadrature[0])
            y_x = self._dx_1d(y, quadrature[0])

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = jnp.reshape(x, (*x.shape[:-2], -1))
            dict_y[0] = jnp.reshape(y, (*y.shape[:-2], -1))

            x_x = self._dx_2d(x, quadrature[0])
            x_y = self._dy_2d(x, quadrature[1])
            y_x = self._dx_2d(y, quadrature[0])
            y_y = self._dy_2d(y, quadrature[1])

            dict_x[1] = jnp.reshape(x_x, (*x_x.shape[:-2], -1))
            dict_x[2] = jnp.reshape(x_y, (*x_y.shape[:-2], -1))

            dict_y[1] = jnp.reshape(y_x, (*y_x.shape[:-2], -1))
            dict_y[2] = jnp.reshape(y_y, (*y_y.shape[:-2], -1))

        else:
            dict_x[0] = jnp.reshape(x, (*x.shape[:-3], -1))
            dict_y[0] = jnp.reshape(y, (*y.shape[:-3], -1))

            x_x = self._dx_3d(x, quadrature[0])
            x_y = self._dy_3d(x, quadrature[1])
            x_z = self._dz_3d(x, quadrature[2])
            y_x = self._dx_3d(y, quadrature[0])
            y_y = self._dy_3d(y, quadrature[1])
            y_z = self._dz_3d(y, quadrature[2])

            dict_x[1] = jnp.reshape(x_x, (*x_x.shape[:-3], -1))
            dict_x[2] = jnp.reshape(x_y, (*x_y.shape[:-3], -1))
            dict_x[3] = jnp.reshape(x_z, (*x_z.shape[:-3], -1))

            dict_y[1] = jnp.reshape(y_x, (*y_x.shape[:-3], -1))
            dict_y[2] = jnp.reshape(y_y, (*y_y.shape[:-3], -1))
            dict_y[3] = jnp.reshape(y_z, (*y_z.shape[:-3], -1))

        return dict_x, dict_y

    def uniform_quadrature(self, x: jnp.ndarray) -> list[float]:
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of `x` to ensure that
        `LpLoss` computes the average over spatial dims.

        Parameters
        ----------
        x : jnp.ndarray
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0] * self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j] / x.shape[-j]

        return quadrature

    def reduce_all(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        reduce x across the batch according to `self.reduction`

        Parameters
        ----------
        x: jnp.ndarray
            inputs
        """
        if self.reduction == "sum":
            x = jnp.sum(x)
        else:
            x = jnp.mean(x)

        return x

    def abs(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        quadrature: float | Sequence[float] | None = None,
        take_root: bool = True,
    ) -> jnp.ndarray:
        """Absolute H1 norm

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True

        Notes
        -----
        The root is taken per sample, *before* `reduce_all` reduces across the
        batch, so `take_root=True` with `reduction="sum"` gives the sum of the
        per-sample norms rather than the norm of the batch. As a consequence
        `abs(take_root=False)` equals `abs(take_root=True) ** 2` only when the
        batch holds a single sample.
        """
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, int | float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const * jnp.sum((dict_x[0] - dict_y[0]) ** 2, axis=-1, keepdims=False)

        for j in range(1, self.d + 1):
            diff += const * jnp.sum(
                (dict_x[j] - dict_y[j]) ** 2, axis=-1, keepdims=False
            )

        if take_root:
            diff = diff**0.5

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def rel(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        quadrature: float | Sequence[float] | None = None,
        take_root: bool = True,
    ) -> jnp.ndarray:
        """Relative H1-norm

        Parameters
        ----------
        x : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True

        Notes
        -----
        The root is taken per sample, *before* `reduce_all` reduces across the
        batch. `take_root=True` with `reduction="sum"` therefore gives the sum
        of the per-sample norms, not the norm of the batch. As a consequence
        `rel(take_root=False)` equals `rel(take_root=True) ** 2` only when the
        batch holds a single sample.
        """
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, int | float):
                quadrature = [quadrature] * self.d

        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = jnp.sum((dict_x[0] - dict_y[0]) ** 2, axis=-1, keepdims=False)
        ynorm = jnp.sum(dict_y[0] ** 2, axis=-1, keepdims=False)

        for j in range(1, self.d + 1):
            diff += jnp.sum((dict_x[j] - dict_y[j]) ** 2, axis=-1, keepdims=False)
            ynorm += jnp.sum(dict_y[j] ** 2, axis=-1, keepdims=False)

        if take_root:
            diff = (diff**0.5) / (ynorm**0.5 + self.eps)
        else:
            diff = diff / (ynorm + self.eps)

        diff = self.reduce_all(diff)
        diff = jnp.squeeze(diff)

        return diff

    def __call__(
        self,
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        quadrature: float | Sequence[float] | None = None,
        take_root: bool = True,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Return the relative H1 loss `rel(y_pred, y)`.

        Parameters
        ----------
        y_pred : jnp.ndarray
            inputs
        y : jnp.ndarray
            targets
        quadrature : float or list, optional
            normalization constant for reduction, by default None
        take_root : bool, optional
            whether to take the square root of the norm, by default True

        Returns
        -------
        jnp.ndarray
            scalar relative H1 loss reduced across the batch and channel dims
        """
        if kwargs:
            warnings.warn(
                f"H1Loss.__call__() received unexpected keyword arguments: "
                f"{list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self.rel(y_pred, y, quadrature=quadrature, take_root=take_root)
