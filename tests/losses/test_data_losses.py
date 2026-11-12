# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the data losses LpLoss and H1Loss (JAX).

Mirrors the ``data_losses`` portion of
``neuralop/losses/tests/test_losses.py``:

* ``test_lploss`` -> ``test_lploss`` (abs is zero on identical inputs; the
  un-normalized L2 error of zeros-vs-ones is 1.0 under mean reduction and 10.0
  under sum reduction; the uniform quadrature weights are ``1 / size`` per
  spatial dim; and both reductions sum-reduce over spatial dims, giving
  ``sqrt(7.5)``);
* ``test_h1loss`` -> ``test_h1loss`` (abs is zero on identical inputs and 1.0 for
  zeros-vs-ones, since the finite-difference derivatives of constants vanish).

The reference also covers ``HdivLoss`` and the ``differentiation`` module
(``FiniteDiff`` / ``FourierDiff`` / ``non_uniform_fd``). ``HdivLoss`` and
``non_uniform_fd`` (meshless/scattered-point finite differences) are not
present in the scirex losses package, so those are reproduced as documented
skips. ``FiniteDiff`` and ``FourierDiff`` (the regular-grid derivative
primitives) *are* ported -- see ``differentiation.py`` and
``tests/test_differentiation.py``. Following the Phase 1 ``layers/tests``
structure, the numeric tests are parametrized over the available CPU/GPU
backends.
"""

import jax
import jax.numpy as jnp
import pytest

from scirex.losses.data_losses import H1Loss, LpLoss


def _available_backends():
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# test_lploss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_lploss(backend) -> None:
    """LpLoss reproduces the reference abs / reduction / quadrature behaviour."""
    with jax.default_device(jax.devices(backend)[0]):
        l2_2d_mean = LpLoss(d=2, p=2, reduction="mean", measure=1.0)
        l2_2d_sum = LpLoss(d=2, p=2, reduction="sum", measure=1.0)
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 1, 4, 4))

        # abs of a tensor against itself is exactly zero
        assert float(l2_2d_mean.abs(x, x)) == 0.0

        zeros = jnp.zeros_like(x)
        ones = jnp.ones_like(x)

        # L2 without normalizing constant: per-element norm is 1, so mean
        # reduction over the batch gives 1.0 and sum reduction gives 10.0
        assert float(l2_2d_mean.abs(zeros, ones)) == pytest.approx(1.0)
        assert float(l2_2d_sum.abs(zeros, ones)) == pytest.approx(10.0)

        # quadrature weights are 1 / size along each spatial dim
        d = 2
        quad = l2_2d_mean.uniform_quadrature(x)
        assert quad[-d:] == [1 / s for s in x.shape[-d:]]

        # both reductions sum-reduce over spatial dims; with batch/channel = 1
        # they agree, and the value is sqrt(7.5)
        base = jnp.arange(1, 5) * jnp.ones((4, 4))
        flipped_x = base.T[None, None]
        x_single = base[None, None]
        zeros_single = jnp.zeros_like(x_single)

        mean_err = l2_2d_mean.abs(flipped_x, zeros_single)
        sum_err = l2_2d_sum.abs(x_single, zeros_single)
        assert float(mean_err) == pytest.approx(float(sum_err))
        assert float(mean_err) == pytest.approx(float(jnp.sqrt(jnp.array(7.5))))


@pytest.mark.parametrize("backend", BACKENDS)
def test_lploss_rel_is_zero_on_identical_and_one_on_zero_pred(backend) -> None:
    """Relative LpLoss is 0 for x==y and ~1 when predicting zeros for nonzero y."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = LpLoss(d=2, p=2, reduction="mean", measure=1.0)
        y = jax.random.normal(jax.random.PRNGKey(2), (8, 1, 4, 4))
        assert float(loss.rel(y, y)) == pytest.approx(0.0, abs=1e-6)
        # ||0 - y|| / ||y|| == 1
        assert float(loss.rel(jnp.zeros_like(y), y)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lploss_p1_reduction(backend) -> None:
    """The p=1 branch sums absolute differences (no p-th root)."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = LpLoss(d=2, p=1, reduction="sum", measure=1.0)
        x = jnp.zeros((3, 1, 4, 4))
        y = jnp.ones((3, 1, 4, 4))
        # const = (1/4)*(1/4) = 1/16; sum |0-1| over 16 = 16; 16/16 = 1 per sample
        assert float(loss.abs(x, y)) == pytest.approx(3.0)


def test_lploss_call_uses_relative_loss() -> None:
    """__call__ delegates to the relative loss (rel)."""
    loss = LpLoss(d=2, p=2, reduction="mean")
    y = jax.random.normal(jax.random.PRNGKey(3), (5, 1, 4, 4))
    assert float(loss(y, y)) == pytest.approx(0.0, abs=1e-6)


def test_lploss_rejects_bad_reduction() -> None:
    """An unsupported reduction is rejected at construction."""
    with pytest.raises(AssertionError):
        LpLoss(d=2, p=2, reduction="median")


# ---------------------------------------------------------------------------
# test_h1loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_h1loss(backend) -> None:
    """H1Loss is zero on identical inputs and 1.0 for zeros-vs-ones."""
    with jax.default_device(jax.devices(backend)[0]):
        h1 = H1Loss(d=2, reduction="mean")
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4, 4))

        assert float(h1.abs(x, x)) == 0.0

        zeros = jnp.zeros_like(x)
        ones = jnp.ones_like(x)
        # finite-difference derivatives of constants vanish, so only the 0th
        # term contributes and the averaged norm is 1.0
        assert float(h1.abs(zeros, ones)) == pytest.approx(1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_h1loss_rel_zero_on_identical(backend) -> None:
    """Relative H1 loss is zero when prediction equals target."""
    with jax.default_device(jax.devices(backend)[0]):
        h1 = H1Loss(d=2, reduction="mean")
        y = jax.random.normal(jax.random.PRNGKey(4), (6, 4, 4))
        assert float(h1.rel(y, y)) == pytest.approx(0.0, abs=1e-6)


def test_h1loss_rejects_dim_out_of_range() -> None:
    """H1Loss only supports d in {1, 2, 3}."""
    with pytest.raises(AssertionError):
        H1Loss(d=4)


def test_h1loss_call_warns_on_unexpected_kwargs() -> None:
    """__call__ warns (and ignores) unexpected keyword arguments."""
    h1 = H1Loss(d=2, reduction="mean")
    y = jax.random.normal(jax.random.PRNGKey(5), (4, 4, 4))
    with pytest.warns(UserWarning):
        h1(y, y, unexpected_kwarg=True)


# ---------------------------------------------------------------------------
# LpLoss: odd p, p=1 relative, explicit quadrature, name, mean reduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_lploss_odd_p_abs(backend) -> None:
    """The odd-p (p>1) branch uses abs() before the power and takes the p-th root."""
    with jax.default_device(jax.devices(backend)[0]):
        loss = LpLoss(d=1, p=3, reduction="mean", measure=1.0)
        x = jnp.ones((2, 5))
        y = jnp.zeros((2, 5))
        # const = 1/5; sum(|1|^3) over 5 = 5; const*5 = 1; 1**(1/3) = 1
        assert float(loss.abs(x, y)) == pytest.approx(1.0)


def test_lploss_odd_p_rel_zero_on_identical() -> None:
    """The odd-p relative branch is zero when prediction equals target."""
    loss = LpLoss(d=1, p=3, reduction="mean")
    y = jax.random.normal(jax.random.PRNGKey(7), (4, 6))
    assert float(loss.rel(y, y)) == pytest.approx(0.0, abs=1e-5)


def test_lploss_p1_rel_zero_on_identical() -> None:
    """The p=1 relative branch divides absolute errors and is zero for x==y."""
    loss = LpLoss(d=1, p=1, reduction="mean")
    y = jax.random.uniform(jax.random.PRNGKey(8), (4, 6)) + 0.5
    assert float(loss.rel(y, y)) == pytest.approx(0.0, abs=1e-6)


def test_lploss_abs_explicit_scalar_quadrature() -> None:
    """A scalar float quadrature is broadcast to every spatial dim."""
    loss = LpLoss(d=1, p=3, reduction="mean")
    x = jnp.ones((2, 5))
    y = jnp.zeros((2, 5))
    # const = 0.5; sum(|1|^3) = 5; 0.5*5 = 2.5; 2.5**(1/3)
    expected = (2.5) ** (1.0 / 3.0)
    assert float(loss.abs(x, y, quadrature=0.5)) == pytest.approx(expected, rel=1e-5)


def test_lploss_name() -> None:
    """The `name` property encodes the order p and dimension d."""
    assert LpLoss(d=2, p=2).name == "L2_2Dloss"
    assert LpLoss(d=3, p=1).name == "L1_3Dloss"


def test_lploss_list_measure_is_kept_per_dim() -> None:
    """A list `measure` is stored as-is (one weight per spatial dim)."""
    loss = LpLoss(d=2, p=2, measure=[0.5, 0.25])
    assert loss.measure == [0.5, 0.25]


def test_h1loss_list_measure_is_kept_per_dim() -> None:
    """A list `measure` is stored as-is for H1Loss too."""
    loss = H1Loss(d=2, measure=[0.5, 0.25])
    assert loss.measure == [0.5, 0.25]


def test_h1loss_rel_with_float_quadrature_and_no_root() -> None:
    """rel() accepts a scalar quadrature and the take_root=False branch."""
    h1 = H1Loss(d=2, reduction="mean")
    y = jax.random.normal(jax.random.PRNGKey(13), (3, 5, 5))
    # zero on identical inputs through both the float-quadrature broadcast and
    # the squared (take_root=False) relative branch
    assert float(h1.rel(y, y, quadrature=0.5)) == pytest.approx(0.0, abs=1e-5)
    assert float(h1.rel(y, y, take_root=False)) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# H1Loss: d=1 and d=3, non-periodic finite differences, name, float quadrature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("d", [1, 3])
def test_h1loss_abs_zeros_vs_ones_dims(backend, d) -> None:
    """For d in {1, 3} the FD derivatives of constants vanish, so abs is 1.0."""
    with jax.default_device(jax.devices(backend)[0]):
        h1 = H1Loss(d=d, reduction="mean")
        shape = (2,) + (4,) * d
        assert float(h1.abs(jnp.zeros(shape), jnp.ones(shape))) == pytest.approx(1.0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_h1loss_non_periodic_zeros_vs_ones(d) -> None:
    """The non-periodic one-sided stencils also annihilate constants (abs == 1.0)."""
    h1 = H1Loss(
        d=d,
        reduction="mean",
        periodic_in_x=False,
        periodic_in_y=False,
        periodic_in_z=False,
    )
    shape = (2,) + (5,) * d
    assert float(h1.abs(jnp.zeros(shape), jnp.ones(shape))) == pytest.approx(1.0)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_h1loss_non_periodic_rel_zero_on_identical(d) -> None:
    """The non-periodic relative H1 loss is zero when prediction equals target."""
    h1 = H1Loss(
        d=d,
        reduction="mean",
        periodic_in_x=False,
        periodic_in_y=False,
        periodic_in_z=False,
    )
    y = jax.random.normal(jax.random.PRNGKey(9), (2,) + (5,) * d)
    assert float(h1.rel(y, y)) == pytest.approx(0.0, abs=1e-5)


def test_h1loss_abs_explicit_scalar_quadrature() -> None:
    """A scalar float quadrature is broadcast across spatial dims in abs()."""
    h1 = H1Loss(d=2, reduction="mean")
    y = jax.random.normal(jax.random.PRNGKey(10), (3, 5, 5))
    # zero on identical inputs regardless of the quadrature weight used
    assert float(h1.abs(y, y, quadrature=0.5)) == pytest.approx(0.0, abs=1e-5)


def test_h1loss_name() -> None:
    """The `name` property encodes the dimension d."""
    assert H1Loss(d=1).name == "H1_1DLoss"
    assert H1Loss(d=3).name == "H1_3DLoss"


def test_h1loss_sum_reduction_matches_mean_for_single_sample() -> None:
    """With a single batch/channel element, sum and mean reduction agree."""
    y = jax.random.normal(jax.random.PRNGKey(11), (1, 6, 6))
    pred = jax.random.normal(jax.random.PRNGKey(12), (1, 6, 6))
    h1_sum = H1Loss(d=2, reduction="sum")
    h1_mean = H1Loss(d=2, reduction="mean")
    assert float(h1_sum.abs(pred, y)) == pytest.approx(float(h1_mean.abs(pred, y)))


# ---------------------------------------------------------------------------
# Reference coverage with no analogue in the JAX port
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Mirrors test_hdivloss. The scirex losses package implements LpLoss and "
    "H1Loss only; HdivLoss is not ported, so there is no analogue."
)
def test_hdivloss() -> None:
    pass


@pytest.mark.skip(
    reason="The reference test_losses.py also covers `non_uniform_fd` via "
    "test_nonuniform_fd -- meshless/scattered-point finite differences for "
    "point-cloud models (GINO/Transolver on irregular geometries). That is not "
    "part of the scirex losses package (deferred to a later phase), so that "
    "test has no analogue here. FiniteDiff/FourierDiff (test_finite_diff_* / "
    "test_fourier_diff_*) *are* ported -- see test_differentiation.py."
)
def test_non_uniform_fd_not_ported() -> None:
    pass
