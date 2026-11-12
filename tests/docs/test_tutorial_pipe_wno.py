# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Pipe WNO tutorial script on the CPU.

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

from tests.docs.test_tutorial_pipe_fno import N, _fake_pipe
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "pipe_wno.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pipe_wno_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_size_and_protocol_match_the_benchmark() -> None:
    from scirex.models import WNO
    from scirex.utils import count_model_params
    from scripts.benchmarks.pipe._config import PROTOCOL as BENCH_PROTOCOL
    from scripts.benchmarks.pipe._config import RUNS

    tutorial = _load_script()
    assert tutorial.SIZES["full"] == RUNS["wno"].arch
    model = WNO(
        in_channels=2,
        out_channels=1,
        size=(129, 129),
        use_grid=False,
        compute_dtype=jnp.float32,
        **tutorial.SIZES["full"],
    )
    params = jax.eval_shape(
        lambda: model.init(jax.random.PRNGKey(0), jnp.zeros((1, 129, 129, 2)))
    )
    assert count_model_params(params) == 3_102_157
    p = tutorial.PROTOCOL
    for key in ("ntrain", "ntest", "n_total", "epochs", "batch_size", "weight_decay",
                "adam_eps", "pct_start", "div_factor", "final_div_factor",
                "max_grad_norm"):  # fmt: skip
        assert p[key] == BENCH_PROTOCOL[key], key
    assert p["learning_rate"] == BENCH_PROTOCOL["lr"]


def test_wno_weights_are_real() -> None:
    tutorial = _load_script()
    sample = np.zeros((N, N, 2), dtype=np.float32)
    _, params = tutorial.build_model(tutorial.SIZES["smoke"], sample, seed=0)
    assert not any(jnp.iscomplexobj(x) for x in jax.tree_util.tree_leaves(params))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_smoke_main_on_synthetic_files(tmp_path: Path, dtype: str) -> None:
    tutorial = _load_script()
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(_fake_pipe(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "2",
            "--epochs", "2",
            "--compute-dtype", dtype,
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
