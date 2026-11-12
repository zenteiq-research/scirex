# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Darcy benchmark: evaluate a saved checkpoint.

Takes one checkpoint folder -- a Hugging Face download such as
``zenteiq-ai/scirex-fno-darcy-physics-extra-fields``, or a run folder with the
same layout -- holding ``config.json`` and the weights (``model.safetensors``,
or ``params.msgpack``). Everything the prediction needs comes from
``config.json``: the model and its architecture, the input normalisation, the
label-free output scale and offset, and the protocol. The model is rebuilt with
the ``build_model`` and ``make_predict_nodes`` of its ``train_<model>.py``, so
evaluation runs exactly the network that was trained.

Two checks, either or both:

* ``--sample``: predict the test input in the folder's ``sample.safetensors`` and
  compare with the ``expected_output`` stored next to it. Needs no dataset; it
  confirms the download, the install and the weights.
* the test set (default): predict the protocol's test fields of Li et al.'s
  Darcy data and report the benchmark metric, the mean per-sample relative L2
  error at the released sample points (the cell-centre map of the Q4 nodal
  solution), next to the figure recorded in ``config.json``.

Usage (from the repo root)::

    hf download zenteiq-ai/scirex-fno-darcy-physics-extra-fields \\
        --local-dir ckpt/scirex-fno-darcy-physics-extra-fields
    python scripts/benchmarks/darcy_physics/infer.py \\
        --checkpoint ckpt/scirex-fno-darcy-physics-extra-fields --sample
    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth{1,2}.mat
    python scripts/benchmarks/darcy_physics/infer.py \\
        --checkpoint ckpt/scirex-fno-darcy-physics-extra-fields

Runs on a CPU; a GPU only makes the test set faster.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    build_grid_aligned_q4_mesh,
)
from scripts.benchmarks._batch_eval import predict_in_batches  # noqa: E402
from scripts.benchmarks._ckpt import load_params, resolve_weights_path  # noqa: E402
from scripts.benchmarks.darcy._data import load_split, rel_l2  # noqa: E402
from scripts.benchmarks.darcy_physics._config import DEFAULT_DATA_DIR  # noqa: E402
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    cell_centre_interpolation,
    sin_boundary_factor,
    split_coefficient,
)

MODELS = ("fno", "wno", "gino", "transolver")
#: Largest relative difference from ``expected_output`` that ``--sample`` accepts.
SAMPLE_TOLERANCE = 1e-3


def load_config(ckpt: Path) -> dict:
    """The checkpoint's ``config.json``, checked to be a physics-only Darcy run."""
    path = ckpt / "config.json"
    if not path.is_file():
        raise SystemExit(f"no config.json in {ckpt}")
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config.get("run", {}).get("model")
    if model not in MODELS:
        raise SystemExit(f"{path}: run.model is {model!r}, expected one of {MODELS}")
    if not str(config.get("regime", "")).startswith("physics_only"):
        raise SystemExit(
            f"{path}: not a physics-only Darcy checkpoint "
            f"(regime={config.get('regime')!r})"
        )
    return config


def encode_input(a: np.ndarray, normalisation: dict) -> np.ndarray:
    """Raw coefficient fields ``(B, H, W)`` -> normalised ``(B, H, W)``, with the
    statistics stored in ``config.json`` (per pixel, or one global scalar)."""
    spec = normalisation["input"]
    if spec["type"] == "per_pixel":
        mean = np.asarray(spec["mean"], dtype=np.float32).reshape(a.shape[1:])
        std = np.asarray(spec["std"], dtype=np.float32).reshape(a.shape[1:])
        return ((a - mean) / (std + float(spec["eps"]))).astype(np.float32)
    if spec["type"] == "global_scalar":
        return ((a - float(spec["mean"])) / float(spec["std_plus_eps"])).astype(
            np.float32
        )
    raise SystemExit(f"unknown input normalisation type {spec['type']!r}")


def build_predictor(config: dict, n_side: int) -> tuple[Callable, Callable, Any]:
    """``(predict_nodes, to_model_layout, template)`` for the checkpoint's model,
    built with the functions of its ``train_<model>.py``.

    ``to_model_layout`` turns normalised ``(B, H, W)`` fields into the layout the
    model takes; ``predict_nodes`` maps that to the ``(B, H * W)`` nodal solution
    in physical units; ``template`` is a freshly initialised parameter tree.
    """
    model_name = config["run"]["model"]
    arch = dict(config["run"]["arch"])
    out = config["output_transform"]
    scale, offset = float(out["scale_c"]), float(out["offset_m"])
    n_points = n_side * n_side
    xy, _, _ = build_grid_aligned_q4_mesh(n_side - 1)
    boundary = jnp.asarray(sin_boundary_factor(xy), dtype=jnp.float32)
    train = importlib.import_module(
        f"scripts.benchmarks.darcy_physics.train_{model_name}"
    )
    key = jax.random.PRNGKey(0)

    if model_name == "fno":
        model = train.build_model(**arch)

        def layout(a: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(a[:, None])  # (B, 1, H, W)

        template = model.init(key, layout(np.zeros((1, n_side, n_side), np.float32)))
        predict = train.make_predict_nodes(model, scale, offset, boundary)
    elif model_name == "wno":
        model = train.build_model(n_side, **arch)

        def layout(a: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(a[..., None])  # (B, H, W, 1)

        template = model.init(key, layout(np.zeros((1, n_side, n_side), np.float32)))
        predict = train.make_predict_nodes(model, scale, offset, boundary)
    elif model_name == "gino":
        model, gino_kwargs, r_in, r_out = train.build_model(
            n_side, arch["hidden_channels"], arch["modes"], arch["n_layers"]
        )
        for name, built in (("in_gno_radius", r_in), ("out_gno_radius", r_out)):
            if name in arch and not np.isclose(arch[name], built):
                raise SystemExit(
                    f"config.json {name}={arch[name]} differs from the model's {built}"
                )

        def layout(a: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(a.reshape(a.shape[0], n_points))  # (B, N)

        template = model.init(
            key,
            x=layout(np.zeros((1, n_side, n_side), np.float32))[..., None],
            **gino_kwargs,
        )
        predict = train.make_predict_nodes(model, gino_kwargs, scale, offset, boundary)
    else:  # transolver
        model = train.build_model(n_side, **arch)

        def layout(a: np.ndarray) -> jnp.ndarray:
            return jnp.asarray(a.reshape(a.shape[0], n_points, 1))  # (B, N, 1)

        template = model.init(
            key,
            x=jnp.zeros((1, n_points, 2), dtype=jnp.float32),
            fx=layout(np.zeros((1, n_side, n_side), np.float32)),
        )
        predict = train.make_predict_nodes(model, n_points, scale, offset, boundary)
    return predict, layout, template


def to_metric_points(
    u_nodes: np.ndarray, n_side: int, subsample_rate: int
) -> np.ndarray:
    """Nodal solution ``(B, H * W)`` -> ``(B, H, W)`` at the released sample
    points: ``U -> M U M^T`` with the cell-centre map of ``_ritz.py``."""
    m = cell_centre_interpolation(n_side, subsample_rate).astype(np.float64)
    grid = u_nodes.reshape(-1, n_side, n_side).astype(np.float64)
    return np.einsum("ij,bjk,lk->bil", m, grid, m).astype(np.float32)


def check_sample(
    ckpt: Path,
    predict_fn: Callable,
    layout: Callable,
    params: Any,
    config: dict,
    n_side: int,
    subsample_rate: int,
) -> bool:
    """Predict ``sample.safetensors``' input and compare with its expected output."""
    path = ckpt / "sample.safetensors"
    if not path.is_file():
        print(f"[sample] no sample.safetensors in {ckpt}; skipped")
        return True
    sample = load_file(str(path))
    a = sample["input"][None].astype(np.float32)
    u = np.asarray(predict_fn(params, layout(encode_input(a, config["normalisation"]))))
    ok = True
    for name, pred in (
        ("expected_output", u.reshape(1, n_side, n_side)),
        (
            "expected_output_at_metric_points",
            to_metric_points(u, n_side, subsample_rate),
        ),
    ):
        if name not in sample:
            continue
        diff = rel_l2(pred[0], sample[name])
        good = diff <= SAMPLE_TOLERANCE
        ok &= good
        print(
            f"[sample] {name}: relative difference {diff:.2e} "
            f"({'OK' if good else 'FAIL'}, tolerance {SAMPLE_TOLERANCE:g})"
        )
    return ok


def evaluate_test_set(
    predict_fn: Callable,
    layout: Callable,
    params: Any,
    config: dict,
    data_dir: Path,
    batch_size: int,
) -> dict:
    """The benchmark metric on the protocol's test fields."""
    protocol = config["protocol"]
    rate = int(protocol["subsample_rate"])
    split = load_split(data_dir, dict(protocol, n_train=1, subsample_rate=1))
    test_a, _ = split_coefficient(split["test_a"], rate, protocol["cell_mode"])
    test_u = split["test_u"][:, ::rate, ::rate]
    n_side = test_a.shape[1]
    enc = layout(encode_input(test_a.astype(np.float32), config["normalisation"]))
    u_nodes = np.asarray(
        predict_in_batches(jax.jit(predict_fn), params, enc, batch_size)
    )
    mapped = to_metric_points(u_nodes, n_side, rate)
    errs = [rel_l2(mapped[i], test_u[i]) for i in range(test_u.shape[0])]
    errs_nodes = [
        rel_l2(u_nodes[i], test_u[i].reshape(-1)) for i in range(test_u.shape[0])
    ]
    return dict(
        n_test=len(errs),
        test_rel_l2_mean=float(np.mean(errs)),
        test_rel_l2_std=float(np.std(errs)),
        test_rel_l2_mean_at_q4_nodes=float(np.mean(errs_nodes)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="folder with config.json and model.safetensors (or params.msgpack)",
    )
    ap.add_argument(
        "--sample",
        action="store_true",
        help="only check sample.safetensors (no dataset needed)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="folder with piececonst_r421_N1024_smooth{1,2}.mat "
        "(default: $DARCY_DATA_DIR)",
    )
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument(
        "--out", type=Path, help="write the test-set metrics to this JSON file"
    )
    args = ap.parse_args()

    ckpt = args.checkpoint.expanduser()
    config = load_config(ckpt)
    protocol = config["protocol"]
    n_side = int(protocol["resolution"])
    rate = int(protocol["subsample_rate"])
    predict_fn, layout, template = build_predictor(config, n_side)
    weights = resolve_weights_path(ckpt)
    params = load_params(template, weights)
    print(
        f"[{config['run']['name']}] {config.get('regime')}: {weights.name} loaded "
        f"({sum(x.size for x in jax.tree_util.tree_leaves(params)):,} values)"
    )

    ok = check_sample(ckpt, predict_fn, layout, params, config, n_side, rate)
    if args.sample:
        raise SystemExit(0 if ok else 1)

    metrics = evaluate_test_set(
        predict_fn, layout, params, config, args.data_dir, args.batch_size
    )
    recorded = config.get("evaluation", {}).get("test_rel_l2_mean")
    print(
        f"[test] mean relative L2 at the released sample points over "
        f"{metrics['n_test']} fields: {100 * metrics['test_rel_l2_mean']:.4f}%"
        + (
            f" (config.json records {100 * recorded:.4f}%)"
            if recorded is not None
            else ""
        )
    )
    print(
        f"[test] at the Q4 nodes: {100 * metrics['test_rel_l2_mean_at_q4_nodes']:.4f}%"
    )
    if args.out:
        args.out.write_text(
            json.dumps(dict(metrics, checkpoint=str(ckpt)), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[test] metrics written to {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
