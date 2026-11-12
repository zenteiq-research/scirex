# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Unit tests for the command line of
scripts/benchmarks/pipe/_transolver.py.

The schedule flags --pct-start, --div-factor and --final-div-factor feed
optax.cosine_onecycle_schedule; their defaults must stay the exp_pipe.py
OneCycleLR defaults so existing runs are unchanged.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks.pipe import _transolver as T  # noqa: E402


def test_schedule_defaults_are_the_onecycle_defaults() -> None:
    config = T.build_config(T._parser().parse_args([]))
    assert config["opt"]["pct_start"] == pytest.approx(0.3)
    assert config["opt"]["div_factor"] == pytest.approx(25.0)
    assert config["opt"]["final_div_factor"] == pytest.approx(1e4)


def test_schedule_flags_reach_the_config() -> None:
    args = T._parser().parse_args(
        ["--pct-start", "0.25", "--div-factor", "7.5", "--final-div-factor", "8000"]
    )
    config = T.build_config(args)
    assert config["opt"]["pct_start"] == pytest.approx(0.25)
    assert config["opt"]["div_factor"] == pytest.approx(7.5)
    assert config["opt"]["final_div_factor"] == pytest.approx(8000.0)


def test_published_args_build_the_published_model() -> None:
    """train.py --run transolver_exp_pipe passes _config.TRANSOLVER_ARGS: the
    3,073,985-parameter Transolver of the JAX-vs-PyTorch comparison."""
    import jax
    import jax.numpy as jnp

    from scirex import get_model_jax
    from scirex.utils import count_model_params
    from scripts.benchmarks.pipe._config import TRANSOLVER_ARGS, TRANSOLVER_PARAMS

    args = T._parser().parse_args(TRANSOLVER_ARGS)
    assert (args.epochs, args.seed, args.max_grad_norm) == (500, 2, 0.1)
    config = T.build_config(args)
    config.model.height = config.model.width = 129
    model = get_model_jax(config)
    params = jax.eval_shape(
        lambda: model.init(
            jax.random.PRNGKey(0), x=jnp.zeros((1, 129 * 129, 2)), fx=None
        )
    )
    assert int(count_model_params(params)) == TRANSOLVER_PARAMS
