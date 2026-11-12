# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Predict with a run trained by scripts/train.py.

Usage:
  python scripts/predict.py --run runs/fno --data new_inputs.npz --out predictions.npz

The data file holds ``x`` (and ``coords`` for a mesh) in the layout used for
training. ``predictions.npz`` gets ``y_pred`` in the units and layout of the
training outputs; if the file also has ``y``, the relative L2 error is
printed. From Python::

    import sys
    sys.path.insert(0, "scripts")
    from predict import load_run

    predict = load_run("runs/fno")
    y_pred = predict(x_new)          # or predict(x_new, coords=coords_new)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from train import (
    DataError,
    Pipeline,
    _as_float,
    configure_devices,
    device_arguments,
    grid_coords,
)


def load_run(run: str | Path, batch_size: int = 16):
    """Load a run directory; return ``predict(x=None, coords=None) -> y_pred``."""
    import jax

    jax.config.update("jax_enable_x64", False)
    run = Path(run)
    if not (run / "config.json").exists():
        raise FileNotFoundError(
            f"{run} has no config.json: not a finished scripts/train.py run"
        )
    cfg = json.loads((run / "config.json").read_text())
    pipe = Pipeline.from_json(cfg)
    info = pipe.info

    # A template of the right structure, then the saved weights into it.
    x0 = (
        None
        if info["c_in"] == 0
        else np.zeros((1, *info["spatial"], info["c_in"]), np.float32)
    )
    c0 = None
    if pipe.coord_lo is not None:
        c0 = pipe.coord_lo + grid_coords((info["spatial"][0],))[:, :1] * (
            pipe.coord_hi - pipe.coord_lo
        )
    template = pipe.init(0, pipe.inputs(x0, c0, 1))
    _, params = type(pipe.net).from_checkpoint(run, "model", template)

    apply = jax.jit(pipe.apply)

    def predict(x=None, coords=None) -> np.ndarray:
        if x is not None:
            x = _as_float("x", x)
        if info["c_in"] and x is None:
            raise DataError("this run needs inputs x")
        if coords is None and info["kind"] == "mesh":
            raise DataError("this run was trained on a mesh: pass coords")
        if coords is not None:
            coords = _as_float("coords", coords)
        n = len(x) if x is not None else (len(coords) if coords.ndim == 3 else 1)
        if pipe.model == "wno" and tuple(x.shape[1:-1]) != tuple(info["spatial"]):
            raise DataError(
                f"WNO was trained on grid {tuple(info['spatial'])}; got {tuple(x.shape[1:-1])}"
            )
        outs = []
        for i in range(0, n, batch_size):
            xb = None if x is None else x[i : i + batch_size]
            cb = (
                coords
                if coords is None or coords.ndim == 2
                else coords[i : i + batch_size]
            )
            m = len(xb) if xb is not None else (len(cb) if cb.ndim == 3 else 1)
            outs.append(np.asarray(apply(params, pipe.inputs(xb, cb, m))))
        return pipe.y_norm.decode(np.concatenate(outs))

    return predict


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Predict with a run trained by scripts/train.py."
    )
    p.add_argument("--run", required=True, help="the run directory written by train.py")
    p.add_argument(
        "--data", required=True, help=".npz file with x (and coords for a mesh)"
    )
    p.add_argument("--out", default="predictions.npz")
    p.add_argument("--batch-size", type=int, default=16)
    device_arguments(p)
    args = p.parse_args(argv)
    configure_devices(
        args.device, args.gpus, not args.no_preallocate, args.deterministic
    )
    try:
        with np.load(args.data, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        predict = load_run(args.run, args.batch_size)
        y_pred = predict(data.get("x"), data.get("coords"))
    except (DataError, FileNotFoundError) as e:
        raise SystemExit(f"Error: {e}") from None
    np.savez(args.out, y_pred=y_pred)
    print(f"Saved {args.out}  y_pred {y_pred.shape}")
    if "y" in data:
        y = data["y"].astype(np.float32)
        b = len(y)
        err = np.linalg.norm((y_pred - y).reshape(b, -1), axis=1) / np.linalg.norm(
            y.reshape(b, -1), axis=1
        )
        print(
            f"relative L2: mean {err.mean():.2%}  median {np.median(err):.2%}  worst {err.max():.2%}"
        )


if __name__ == "__main__":
    main()
