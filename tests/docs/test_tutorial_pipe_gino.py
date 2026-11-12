# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Run the Pipe GINO tutorial script on the CPU.

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

from tests import tolerances
from tests.docs.test_tutorial_pipe_fno import N, _fake_pipe
from tests.public_api_helpers import REPO

SCRIPT = REPO / "scripts" / "tutorials" / "pipe_gino.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pipe_gino_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty_graph(n_nodes: int, max_edges: int) -> dict:
    return dict(
        neighbors_index=np.zeros(max_edges, np.int32),
        segment_ids=np.full(max_edges, n_nodes, np.int32),
        counts=np.zeros(n_nodes, np.int32),
    )


def test_full_size_and_protocol_match_the_benchmark() -> None:
    from scirex.models import GINO
    from scirex.utils import count_model_params
    from scripts.benchmarks.pipe._config import PROTOCOL as BENCH_PROTOCOL
    from scripts.benchmarks.pipe._config import RUNS
    from scripts.benchmarks.pipe._models import GINOSpec

    tutorial = _load_script()
    assert tutorial.SIZES["full"] == RUNS["gino"].arch
    assert tutorial.MAX_NUM_NEIGHBORS == GINOSpec.max_num_neighbors
    n = 129 * 129
    model = GINO(
        in_channels=1, out_channels=1, gno_coord_dim=2, **tutorial.SIZES["full"]
    )
    inputs = tutorial.model_inputs(
        np.zeros((n, 2), np.float32), _empty_graph(n, GINOSpec.max_edges), (129, 129)
    )
    params = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), **inputs))
    assert count_model_params(params) == 3_001_956
    p = tutorial.PROTOCOL
    for key in ("ntrain", "ntest", "n_total", "epochs", "batch_size", "weight_decay",
                "adam_eps", "pct_start", "div_factor", "final_div_factor",
                "max_grad_norm"):  # fmt: skip
        assert p[key] == BENCH_PROTOCOL[key], key
    assert p["learning_rate"] == BENCH_PROTOCOL["lr"]


def test_build_graphs_counts_edges_and_pads_to_the_largest_graph() -> None:
    """No edge is dropped: every graph is padded to the largest one."""
    tutorial = _load_script()
    g = np.stack(np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij"), -1)
    pos = (0.1 * g.reshape(1, 16, 2)).astype(np.float32)  # spacing 0.1
    sparse = pos * 10.0  # no neighbour but itself at radius 0.105
    graphs = tutorial.build_graphs(np.concatenate([pos, sparse]), radius=0.105)
    # every node reaches itself and its 2 to 4 axis neighbours at spacing 0.1
    assert graphs[0]["counts"].min() == 3 and graphs[0]["counts"].max() == 5
    assert [int(g["counts"].sum()) for g in graphs] == [64, 16]
    assert [g["segment_ids"].shape for g in graphs] == [(64,), (64,)]


def test_load_pipe_stride_keeps_every_other_node(tmp_path: Path) -> None:
    tutorial = _load_script()
    folder = _fake_pipe(tmp_path)
    full = tutorial.load_pipe(folder, ntrain=2, ntest=1)
    half = tutorial.load_pipe(folder, ntrain=2, ntest=1, stride=2)
    assert half["x_train"].shape == (2, N // 2, N // 2, 2)
    np.testing.assert_array_equal(half["y_test"], full["y_test"][:, ::2, ::2])
    np.testing.assert_array_equal(half["x_train"], full["x_train"][:, ::2, ::2])


def test_summed_gradients_equal_the_batch_gradient(tmp_path: Path) -> None:
    """Adding per-sample gradients gives the gradient of the summed loss."""
    tutorial = _load_script()
    data = tutorial.load_pipe(_fake_pipe(tmp_path), ntrain=2, ntest=1)
    grid = data["y_train"].shape[1:]
    pos = data["x_train"].reshape(2, -1, 2)
    graphs = tutorial.build_graphs(pos, 0.5)
    y_norm = tutorial.UnitGaussian(data["y_train"], axes=(0, 1, 2))
    model, params = tutorial.build_model(
        tutorial.SIZES["smoke"], pos[0], graphs[0], grid, seed=0
    )
    loss_fn = tutorial.make_loss(model, y_norm, grid)
    per = [
        jax.grad(loss_fn)(
            params, tutorial.model_inputs(pos[i], graphs[i], grid), data["y_train"][i]
        )
        for i in range(2)
    ]
    total = jax.grad(
        lambda p: sum(
            loss_fn(
                p, tutorial.model_inputs(pos[i], graphs[i], grid), data["y_train"][i]
            )
            for i in range(2)
        )
    )(params)
    summed = jax.tree_util.tree_map(jnp.add, *per)
    for a, b in zip(
        jax.tree_util.tree_leaves(summed), jax.tree_util.tree_leaves(total)
    ):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=1e-4, atol=tolerances.atol(1e-6)
        )


def test_smoke_main_on_synthetic_files(tmp_path: Path) -> None:
    tutorial = _load_script()
    out = tmp_path / "out"
    metrics = tutorial.main(
        [
            "--data-dir", str(_fake_pipe(tmp_path / "data")),
            "--smoke",
            "--n-train", "8",
            "--n-test", "2",
            "--epochs", "1",
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
        ["--smoke", "--epochs", "1", "--n-train", "8", "--n-test", "2",
         "--out-dir", str(tmp_path), "--data-dir", os.environ["PIPE_DATA_DIR"]]
    )  # fmt: skip
    assert np.isfinite(metrics["test_rel_l2"])
