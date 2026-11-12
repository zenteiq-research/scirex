# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the 1D Poisson FNO worked example on the CPU.

This is the worked example of ``doc/tutorials/NewExample/``. The page checks
are in ``test_tutorial_pages.py``. Here the four tests that the guide requires
of every new example (output shape, finite gradients, a tiny overfit, an
unknown config key refused) run on the example's own functions, followed by
checks of the data against the exact solution, the checkpoint round trip and
the whole ``--smoke`` run, which must stay well inside the two-minute CI
budget. Nothing is downloaded.
"""

import importlib.util
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "poisson1d_fno.py"

# A model small enough that each test takes about a second.
TINY = dict(n_points=32, n_terms=4, modes=8, hidden_channels=8, n_layers=2)


def load_script():
    """Import ``scripts/tutorials/poisson1d_fno.py`` as a module."""
    spec = importlib.util.spec_from_file_location("poisson1d_fno_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The four required tests of the guide ---------------------------------------


def test_output_shape() -> None:
    tutorial = load_script()
    config = tutorial.make_config(**TINY)
    model, params, _ = tutorial.build_model(config, seed=0)
    f = jnp.ones((3, 1, config["n_points"]), jnp.float32)
    out = model.apply(params, f)
    assert out.shape == (3, 1, config["n_points"])
    assert out.dtype == jnp.float32


def test_gradients_are_finite() -> None:
    tutorial = load_script()
    config = tutorial.make_config(n_train=4, n_test=2, **TINY)
    data = tutorial.load_data(config, seed=0)
    model, params, _ = tutorial.build_model(config, seed=0)
    loss_fn = tutorial.make_loss(model)
    grads = jax.grad(loss_fn)(params, data["f_train"], data["u_train"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    for g in leaves:
        assert bool(jnp.all(jnp.isfinite(g)))
    assert any(float(jnp.abs(g).max()) > 0 for g in leaves), "all gradients are zero"


def test_tiny_overfit() -> None:
    """Eight samples, one batch: the training loss must fall well below its start."""
    tutorial = load_script()
    config = tutorial.make_config(
        n_train=8, n_test=8, batch_size=8, epochs=300, learning_rate=1e-2, **TINY
    )
    data = tutorial.load_data(config, seed=0)
    data["f_test"], data["u_test"] = data["f_train"], data["u_train"]
    model, params, key = tutorial.build_model(config, seed=0)
    _, history = tutorial.train(model, params, key, data, config)
    first, last = history[0]["train_loss"], history[-1]["train_loss"]
    # Measured: 5.43 -> 0.035 (CPU, seed 0).
    assert last < 0.1, f"loss {first:.3f} -> {last:.3f}: the model cannot fit 8 samples"
    assert last < first / 20


def test_unknown_config_key_is_rejected() -> None:
    tutorial = load_script()
    with pytest.raises(ValueError, match=r"unknown config key\(s\) \['epoch'\]"):
        tutorial.make_config(epoch=3)
    with pytest.raises(ValueError, match="unknown config key"):
        tutorial.main(["--smoke", "--set", "epoch=3"])
    # The model itself refuses a constructor argument it does not declare.
    with pytest.raises(TypeError, match="unexpected keyword argument 'n_mode'"):
        tutorial.FNO(in_channels=1, out_channels=1, hidden_channels=8, n_mode=(8,))


# The example's own checks ---------------------------------------------------


def test_data_solve_the_poisson_problem() -> None:
    """``-u'' = f`` inside (second-order differences), ``u = 0`` at both ends."""
    tutorial = load_script()
    d = tutorial.poisson1d_dataset(n_samples=3, n_points=401, n_terms=8, seed=0)
    u, f = d["u"][:, 0].astype(np.float64), d["f"][:, 0].astype(np.float64)
    h = 1.0 / 400
    lap = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / h**2
    np.testing.assert_allclose(-lap, f[:, 1:-1], atol=2e-3 * np.abs(f).max())
    np.testing.assert_allclose(u[:, [0, -1]], 0.0, atol=1e-7)


def test_train_and_test_sources_differ() -> None:
    tutorial = load_script()
    data = tutorial.load_data(tutorial.make_config(n_train=4, n_test=4, **TINY), 0)
    assert not np.allclose(data["f_train"], data["f_test"])


def test_rel_l2_matches_definition() -> None:
    tutorial = load_script()
    true = jnp.array([[3.0, 4.0]])
    assert float(tutorial.rel_l2_per_sample(true, true)[0]) == 0.0
    assert float(
        tutorial.rel_l2_per_sample(jnp.zeros((1, 2)), true)[0]
    ) == pytest.approx(1.0)


def test_parse_overrides() -> None:
    tutorial = load_script()
    assert tutorial.parse_overrides(["epochs=5", "learning_rate=1e-3"]) == dict(
        epochs=5, learning_rate=1e-3
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    tutorial = load_script()
    config = tutorial.make_config(n_train=4, n_test=2, **TINY)
    data = tutorial.load_data(config, seed=0)
    model, params, _ = tutorial.build_model(config, seed=3)
    tutorial.save_run(model, params, config, 3, tmp_path)
    loaded, loaded_params, loaded_config, seed = tutorial.load_run(tmp_path)
    assert loaded_config == config and seed == 3
    np.testing.assert_array_equal(
        np.asarray(model.apply(params, data["f_test"])),
        np.asarray(loaded.apply(loaded_params, data["f_test"])),
    )


def test_smoke_run_then_infer_fits_the_ci_budget(tmp_path: Path) -> None:
    """The documented ``--smoke`` run, then ``--infer`` on its checkpoint."""
    tutorial = load_script()
    t0 = perf_counter()
    metrics = tutorial.main(["--smoke", "--out-dir", str(tmp_path)])
    seconds = perf_counter() - t0
    assert seconds < 120, f"the smoke run took {seconds:.0f}s, over the CI budget"
    assert np.isfinite(metrics["test_rel_l2"]) and metrics["test_rel_l2"] < 1.0
    for name in ("fno_params.bytes", "fno_metadata.msgpack", "config.json"):
        assert (tmp_path / name).is_file(), name
    for name in ("loss.png", "prediction.png", "metrics.json"):
        assert (tmp_path / name).is_file(), name
    assert json.loads((tmp_path / "config.json").read_text())["config"]["epochs"] == 10

    again = tutorial.main(["--infer", "--out-dir", str(tmp_path)])
    assert again["test_rel_l2"] == pytest.approx(metrics["test_rel_l2"], rel=1e-6)
