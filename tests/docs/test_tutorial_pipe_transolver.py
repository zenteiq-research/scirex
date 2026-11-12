# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Pipe Transolver tutorial script on the CPU.

The page checks are in ``test_tutorial_pages.py``. Here the script's
functions and its ``main`` run on small synthetic ``Pipe_*.npy`` files (the
same as for the FNO tutorial), so nothing is downloaded. The test that reads
the real dataset runs only when ``PIPE_DATA_DIR`` points at the real files.
"""

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.docs.test_tutorial_pipe_fno import _fake_pipe
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "pipe_transolver.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pipe_transolver_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_size_and_protocol_match_the_benchmark() -> None:
    from scirex.models import Transolver
    from scirex.utils import count_model_params
    from scripts.benchmarks.pipe._config import PROTOCOL as BENCH_PROTOCOL
    from scripts.benchmarks.pipe._config import RUNS

    tutorial = _load_script()
    assert tutorial.SIZES["full"] == RUNS["transolver"].arch
    model = Transolver(
        space_dim=2, fun_dim=0, out_dim=1, unified_pos=False,
        geometry="structured_2d", H=129, W=129, **tutorial.SIZES["full"],
    )  # fmt: skip
    params = jax.eval_shape(
        lambda: model.init(
            jax.random.PRNGKey(0), x=jnp.zeros((1, 129 * 129, 2)), fx=None
        )
    )
    assert count_model_params(params) == 3_073_985
    p = tutorial.PROTOCOL
    for key in ("ntrain", "ntest", "n_total", "epochs", "batch_size", "weight_decay",
                "adam_eps", "pct_start", "div_factor", "final_div_factor",
                "max_grad_norm"):  # fmt: skip
        assert p[key] == BENCH_PROTOCOL[key], key
    assert p["learning_rate"] == BENCH_PROTOCOL["lr"]


def test_init_matches_the_benchmark_bit_for_bit() -> None:
    """The tutorial's model and init equal the benchmark's (same seed)."""
    from scripts.benchmarks.pipe._config import RUNS
    from scripts.benchmarks.pipe._models import get_spec

    tutorial = _load_script()
    grid = (16, 16)
    x = np.random.default_rng(0).normal(size=(grid[0] * grid[1], 2)).astype(np.float32)
    _, ours = tutorial.build_model(tutorial.SIZES["full"], x, grid, seed=3)
    spec = get_spec("transolver")
    arch = RUNS["transolver"].arch
    model = spec.build(arch, grid)
    theirs = model.init(jax.random.PRNGKey(3), x=jnp.asarray(x)[None], fx=None)
    theirs = spec.post_init(theirs, arch, 3)
    same = jax.tree_util.tree_map(
        lambda a, b: bool(jnp.array_equal(a, b)), ours, theirs
    )
    assert all(jax.tree_util.tree_leaves(same))


def test_reference_init_keeps_one_orthogonal_kernel_per_layer() -> None:
    tutorial = _load_script()
    x = np.zeros((64, 2), np.float32)
    size = tutorial.SIZES["smoke"]
    _, params = tutorial.build_model(size, x, (8, 8), seed=0)
    kernels = [
        leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(params)
        if getattr(path[-1], "key", None) == "kernel" and leaf.ndim == 2
    ]
    stds = sorted(float(jnp.std(k)) for k in kernels if min(k.shape) > 1)
    assert len(kernels) > size["n_layers"]
    assert not any(jnp.iscomplexobj(x) for x in jax.tree_util.tree_leaves(params))
    with pytest.raises(RuntimeError, match="orthogonal"):
        tutorial.match_reference_init(params, 0, n_layers=size["n_layers"] + 1)
    assert stds[0] == pytest.approx(0.02, rel=0.5)


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = _load_script()
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(_fake_pipe(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "2",
            "--epochs", "2",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
    assert metrics["worst"] >= metrics["median"]
    assert (out / "prediction.png").stat().st_size > 0
    assert (out / "loss.png").stat().st_size > 0


@pytest.mark.skipif(
    not os.environ.get("PIPE_DATA_DIR"),
    reason="set PIPE_DATA_DIR to the folder with the real Pipe_*.npy files",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    tutorial = _load_script()
    metrics = tutorial.main(
        ["--smoke", "--epochs", "1", "--out-dir", str(tmp_path),
         "--data-dir", os.environ["PIPE_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
