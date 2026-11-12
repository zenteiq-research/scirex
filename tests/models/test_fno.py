# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
# Portions derived from neuraloperator (https://github.com/neuraloperator/neuraloperator),
# Copyright (c) 2023 NeuralOperator developers, MIT License; see
# THIRD_PARTY_NOTICES.md.

"""
Unit tests for the FNO model (JAX/Flax).

Mirrors ``neuralop/models/tests/test_fno.py``. Each reference test is reproduced
against the Flax port, adapting the PyTorch idioms as follows:

* ``model(x)`` -> ``model.apply(model.init(key, x), x)``;
* the "no unused parameters" check (``param.grad is None``) is mirrored by taking
  ``jax.grad`` of a scalar loss over the params and asserting every leaf has a
  non-zero gradient;
* for complex inputs the loss uses the output modulus (as upstream does);
* tensor sizes follow the reference's CPU branch (s=12, modes=5, width=9,
  n_layers=2) to keep XLA compilation fast.

Following the Phase 1 ``layers/tests`` structure, device-sensitive tests are
parametrized over the available CPU/GPU backends.

Two reference tests target parameters that are **not present** in the JAX port
(``norm_groups`` in ``test_fno_group_norm`` and ``conv_bias_kernel`` in
``test_fno_conv_bias_kernel``); those are reproduced as documented skips, while
the portable part of their coverage (group_norm forward) lives in
``test_fno_norm``.
"""

import warnings
from collections.abc import Callable
from math import prod
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from scirex.layers.common.embeddings import GridEmbedding2D, GridEmbeddingND
from scirex.models.fno import FNO, partialclass

# reference CPU-branch sizes
S = 12
MODES = 5
WIDTH = 9
N_LAYERS = 2
BATCH = 3


def _available_backends() -> list[str]:
    backends = ["cpu"]
    try:
        if jax.devices("gpu"):
            backends.append("gpu")
    except RuntimeError:
        pass
    return backends


BACKENDS = _available_backends()


def _input(
    n_dim: int,
    in_channels: int = 3,
    batch: int = BATCH,
    complex_data: bool = False,
    key: int = 0,
) -> jnp.ndarray:
    """Random (B, C, *size) input, complex64 when complex_data is set."""
    size = (S,) * n_dim
    real = jax.random.normal(jax.random.PRNGKey(key), (batch, in_channels, *size))
    if complex_data:
        imag = jax.random.normal(
            jax.random.PRNGKey(key + 1), (batch, in_channels, *size)
        )
        return (real + 1j * imag).astype(jnp.complex64)
    return real.astype(jnp.float32)


def _count_unused_params(model: FNO, x: jnp.ndarray, **call_kwargs: Any) -> int:
    """Mirror of upstream's unused-parameter check.

    Takes the gradient of a scalar loss (output modulus, summed) w.r.t. the
    params and counts leaves whose gradient is identically zero.
    """
    params = model.init(jax.random.PRNGKey(1), x, **call_kwargs)

    def loss(p: Any) -> jnp.ndarray:
        out = cast(jnp.ndarray, model.apply(p, x, **call_kwargs))
        return jnp.sum(jnp.abs(out))

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    return sum(1 for g in leaves if not bool((g != 0).any()))


def _forward(model: FNO, x: jnp.ndarray, **call_kwargs: Any) -> jnp.ndarray:
    params = model.init(jax.random.PRNGKey(1), x, **call_kwargs)
    return cast(jnp.ndarray, model.apply(params, x, **call_kwargs))


# ---------------------------------------------------------------------------
# test_fno: forward pass, output shape, backward (no unused params)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
@pytest.mark.parametrize("complex_data", [False, True])
def test_fno_forward_and_unused_params(
    n_dim: int, complex_data: bool, backend: str
) -> None:
    """FNO forward gives [B, out, *size] and every parameter receives gradient."""
    with jax.default_device(jax.devices(backend)[0]):
        model = FNO(
            in_channels=3,
            out_channels=1,
            hidden_channels=WIDTH,
            n_modes=(MODES,) * n_dim,
            n_layers=N_LAYERS,
            complex_data=complex_data,
        )
        x = _input(n_dim, complex_data=complex_data)
        out = _forward(model, x)
        assert list(out.shape) == [BATCH, 1, *([S] * n_dim)]
        assert bool(jnp.all(jnp.isfinite(jnp.abs(out))))
        assert _count_unused_params(model, x) == 0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("preactivation", [False, True])
def test_fno_stabilizer_and_preactivation(
    stabilizer: str | None, preactivation: bool, backend: str
) -> None:
    """stabilizer (None/tanh) and preactivation combinations preserve shape."""
    with jax.default_device(jax.devices(backend)[0]):
        model = FNO(
            in_channels=3,
            out_channels=1,
            hidden_channels=WIDTH,
            n_modes=(MODES, MODES),
            n_layers=N_LAYERS,
            stabilizer=stabilizer,
            preactivation=preactivation,
        )
        out = _forward(model, _input(2))
        assert list(out.shape) == [BATCH, 1, S, S]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("lifting_channel_ratio", [1, 2])
def test_fno_lifting_channel_ratio(lifting_channel_ratio: int, backend: str) -> None:
    """Different lifting channel ratios both produce a valid forward pass."""
    with jax.default_device(jax.devices(backend)[0]):
        model = FNO(
            in_channels=3,
            out_channels=1,
            hidden_channels=WIDTH,
            n_modes=(MODES, MODES),
            n_layers=N_LAYERS,
            lifting_channel_ratio=lifting_channel_ratio,
        )
        out = _forward(model, _input(2))
        assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# test_fno_superresolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolution_scaling_factor",
    [[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 2, 2], [1, 0.5, 1]],
)
def test_fno_superresolution(resolution_scaling_factor: list[float]) -> None:
    """resolution_scaling_factor scales spatial dims by prod(factors)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=3,
        resolution_scaling_factor=resolution_scaling_factor,
    )
    out = _forward(model, _input(2))
    factor = prod(resolution_scaling_factor)
    assert list(out.shape) == [BATCH, 1] + [int(round(factor * S))] * 2


# ---------------------------------------------------------------------------
# test_fno_advanced_params (split into norm / skip / channel-mlp coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("norm", [None, "group_norm", "instance_norm"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_fno_norm(norm: str | None, complex_data: bool, backend: str) -> None:
    """Each normalization variant produces the correct output shape."""
    with jax.default_device(jax.devices(backend)[0]):
        model = FNO(
            in_channels=3,
            out_channels=1,
            hidden_channels=16,
            n_modes=(MODES, MODES),
            n_layers=N_LAYERS,
            norm=norm,
            complex_data=complex_data,
        )
        out = _forward(model, _input(2, complex_data=complex_data))
        assert list(out.shape) == [BATCH, 1, S, S]


@pytest.mark.parametrize(
    "channel_mlp_skip", ["linear", "identity", "soft-gating", None]
)
@pytest.mark.parametrize("fno_skip", ["linear", "identity", "soft-gating", None])
def test_fno_skip_options(channel_mlp_skip: str | None, fno_skip: str | None) -> None:
    """All channel_mlp_skip x fno_skip combinations preserve output shape."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=WIDTH,
        n_modes=(MODES, MODES),
        n_layers=N_LAYERS,
        channel_mlp_skip=channel_mlp_skip,
        fno_skip=fno_skip,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


@pytest.mark.parametrize("use_channel_mlp", [True, False])
def test_fno_use_channel_mlp(use_channel_mlp: bool) -> None:
    """Toggling the channel MLP after each FNO block preserves output shape."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=WIDTH,
        n_modes=(MODES, MODES),
        n_layers=N_LAYERS,
        use_channel_mlp=use_channel_mlp,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# test_fno_embedding_and_padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("positional_embedding", ["grid", None])
@pytest.mark.parametrize("domain_padding", [None, 0.1, [0.1, 0.2]])
def test_fno_embedding_and_padding(
    positional_embedding: str | None, domain_padding: float | list[float] | None
) -> None:
    """Positional embedding and domain padding options keep the output shape."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=positional_embedding,
        domain_padding=domain_padding,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# test_fno_channel_mlp_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel_mlp_dropout", [0.0, 0.5])
@pytest.mark.parametrize("channel_mlp_expansion", [0.5, 1.0])
@pytest.mark.parametrize("non_linearity", [jax.nn.gelu, jax.nn.relu, jax.nn.tanh])
def test_fno_channel_mlp_params(
    channel_mlp_dropout: float,
    channel_mlp_expansion: float,
    non_linearity: Callable,
) -> None:
    """Channel-MLP dropout / expansion / non-linearity options preserve shape."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        channel_mlp_dropout=channel_mlp_dropout,
        channel_mlp_expansion=channel_mlp_expansion,
        non_linearity=non_linearity,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# Reference tests targeting parameters not present in the JAX port
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Mirrors test_fno_group_norm, which asserts norm_groups propagation to "
    "torch.nn.GroupNorm layers. The JAX FNO does not expose a `norm_groups` "
    "parameter, so this propagation check has no analogue. The portable part "
    "(norm='group_norm' forward pass) is covered by test_fno_norm."
)
def test_fno_group_norm_norm_groups_not_ported() -> None:
    pass


@pytest.mark.skip(
    reason="Mirrors test_fno_conv_bias_kernel. The JAX FNO does not expose a "
    "`conv_bias_kernel` parameter (no local convolutional bias kernel), so there "
    "is no analogue to port."
)
def test_fno_conv_bias_kernel_not_ported() -> None:
    pass


# ---------------------------------------------------------------------------
# Positional embedding: GridEmbedding2D and GridEmbeddingND instances
# (source lines 227-235)
# ---------------------------------------------------------------------------


def test_fno_gridembedding2d_positional_embedding() -> None:
    """Passing a GridEmbedding2D instance as positional_embedding works
    (source lines 227-229)."""
    pe = GridEmbedding2D(in_channels=3)
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=pe,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


def test_fno_gridembeddingnd_positional_embedding() -> None:
    """Passing a GridEmbeddingND instance as positional_embedding works
    (source lines 234-235)."""
    pe = GridEmbeddingND(in_channels=3, dim=2)
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=pe,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


def test_fno_invalid_positional_embedding_raises() -> None:
    """An unrecognised positional_embedding value raises ValueError
    (source lines 238-241)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding="bad_value",
    )
    with pytest.raises(ValueError, match="positional embedding"):
        _forward(model, _input(2))


# ---------------------------------------------------------------------------
# Domain padding: DomainPadding created and used in __call__
# (source lines 251, 374, 380)
# ---------------------------------------------------------------------------


def test_fno_domain_padding_pad_and_unpad() -> None:
    """FNO with domain_padding > 0 pads before and unpads after FNO blocks
    (source lines 251, 374, 380)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=None,
        domain_padding=0.1,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# resolution_scaling_factor as a scalar float (source lines 261-262)
# ---------------------------------------------------------------------------


def test_fno_scalar_resolution_scaling_factor() -> None:
    """A scalar float resolution_scaling_factor is broadcast to
    [rsf]*n_layers (source lines 261-262)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        resolution_scaling_factor=1.0,
        positional_embedding=None,
    )
    out = _forward(model, _input(2))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# __call__ with unexpected kwargs (source line 355)
# ---------------------------------------------------------------------------


def test_fno_call_unexpected_kwargs_warns() -> None:
    """FNO.__call__ with extra kwargs emits a UserWarning (source line 355)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=None,
    )
    x = _input(2)
    params = model.init(jax.random.PRNGKey(99), x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.apply(params, x, surprise_kwarg="oops")
    assert any("unexpected" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# __call__ with output_shape as tuple (source lines 364-365)
# ---------------------------------------------------------------------------


def test_fno_call_output_shape_as_tuple() -> None:
    """output_shape as a tuple targets only the last FNO block
    (source lines 364-365)."""
    model = FNO(
        in_channels=3,
        out_channels=1,
        n_modes=(MODES, MODES),
        hidden_channels=WIDTH,
        n_layers=N_LAYERS,
        positional_embedding=None,
    )
    x = _input(2)
    params = model.init(jax.random.PRNGKey(1), x)
    out = cast(jnp.ndarray, model.apply(params, x, output_shape=(S, S)))
    assert list(out.shape) == [BATCH, 1, S, S]


# ---------------------------------------------------------------------------
# partialclass helper (source lines 422-423)
# ---------------------------------------------------------------------------


def test_partialclass_creates_subclass() -> None:
    """partialclass returns a class that inherits from the base and has a
    new name (source lines 422-423)."""
    SmallFNO = partialclass(
        "SmallFNO",
        FNO,
        in_channels=3,
        out_channels=1,
        hidden_channels=WIDTH,
        n_modes=(MODES, MODES),
    )
    assert SmallFNO.__name__ == "SmallFNO"
    assert issubclass(SmallFNO, FNO)


def test_fno_applies_after_checkpoint_round_trip(tmp_path) -> None:
    """A loaded FNO can run inference directly (its spectral weights must be JAX arrays)."""
    import jax
    import jax.numpy as jnp

    from scirex.models import FNO

    model = FNO(
        n_modes=(4, 4), in_channels=1, out_channels=1, hidden_channels=8, n_layers=1
    )
    x = jnp.ones((1, 1, 16, 16))
    params = model.init(jax.random.PRNGKey(0), x)
    model.save_checkpoint(tmp_path, "c", params)
    loaded, restored = FNO.from_checkpoint(tmp_path, "c", params)
    assert bool(jnp.allclose(loaded.apply(restored, x), model.apply(params, x)))
