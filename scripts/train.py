# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Train a neural operator on your own data, in one command.

Usage:
  python scripts/train.py --model fno --data my_data.npz
  python scripts/train.py --model transolver --data mesh.npz --size large
  python scripts/train.py --model fno --data my_data.npz --smoke   # CPU check

The data file holds ``x`` (inputs) and ``y`` (outputs), samples first and
channels last: ``(samples, *grid, channels)`` on a regular grid, or
``(samples, points, channels)`` with ``coords`` on a mesh. The script splits
off a test set, normalises, trains on every visible device and saves a run
directory that ``scripts/predict.py`` loads. See
``doc/user_guide/use_on_your_data.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODELS = ("fno", "wno", "gino", "transolver")
GRID_MODELS = ("fno", "wno")
SIZES = ("small", "medium", "large")

# Width, depth and resolution of each model per --size. "smoke" is a tiny
# model for checking a setup on a CPU.
PRESETS = {
    "fno": {
        "smoke": dict(hidden_channels=8, n_modes=4, n_layers=2),
        "small": dict(hidden_channels=24, n_modes=8, n_layers=4),
        "medium": dict(hidden_channels=48, n_modes=12, n_layers=4),
        "large": dict(hidden_channels=96, n_modes=16, n_layers=4),
    },
    "wno": {
        "smoke": dict(hidden_channels=8, n_layers=2, level=1),
        "small": dict(hidden_channels=24, n_layers=4, level=3),
        "medium": dict(hidden_channels=48, n_layers=4, level=3),
        "large": dict(hidden_channels=96, n_layers=4, level=3),
    },
    "transolver": {
        "smoke": dict(n_hidden=16, n_layers=2, n_head=2, slice_num=8),
        "small": dict(n_hidden=64, n_layers=4, n_head=4, slice_num=32),
        "medium": dict(n_hidden=128, n_layers=6, n_head=8, slice_num=64),
        "large": dict(n_hidden=256, n_layers=8, n_head=8, slice_num=64),
    },
    # latent: points per side of GINO's latent grid on the unit box
    "gino": {
        "smoke": dict(latent=8, fno_hidden_channels=8, fno_n_modes=4, fno_n_layers=2),
        "small": dict(latent=16, fno_hidden_channels=24, fno_n_modes=8, fno_n_layers=4),
        "medium": dict(
            latent=24, fno_hidden_channels=48, fno_n_modes=12, fno_n_layers=4
        ),
        "large": dict(
            latent=32, fno_hidden_channels=64, fno_n_modes=16, fno_n_layers=4
        ),
    },
}

# GINO keeps at most this many neighbours per point within the radius.
MAX_NEIGHBORS = 32


class DataError(ValueError):
    """The data file does not have the expected keys, shapes or values."""


# ---------------------------------------------------------------------------
# Devices: must run before JAX is imported
# ---------------------------------------------------------------------------


def configure_devices(
    device: str, gpus: str | None, preallocate: bool, deterministic: bool
) -> None:
    """Set the environment variables JAX reads when it starts."""
    if "jax" in sys.modules:
        raise RuntimeError("configure_devices() must run before JAX is imported")
    if device != "auto":
        os.environ["JAX_PLATFORMS"] = {"gpu": "cuda"}.get(device, device)
        # The older JAX_PLATFORM_NAME would ask for a backend that JAX_PLATFORMS has
        # just excluded (for example "gpu" with --device cpu) and stop the run.
        os.environ.pop("JAX_PLATFORM_NAME", None)
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if not preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if deterministic:
        flags = os.environ.get("XLA_FLAGS", "")
        os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_deterministic_ops=true".strip()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _as_float(name: str, a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == object:
        raise DataError(
            f"'{name}' has samples of different shapes; every sample must have the "
            "same shape (pad meshes to one number of points)"
        )
    if not np.issubdtype(a.dtype, np.number):
        raise DataError(f"'{name}' must be numeric, got {a.dtype}")
    a = a.astype(np.float32)
    if not np.all(np.isfinite(a)):
        raise DataError(f"'{name}' contains NaN or infinite values")
    return a


def _check_channels_last(name: str, a: np.ndarray) -> None:
    if a.ndim < 3:
        raise DataError(
            f"'{name}' has shape {a.shape}; expected (samples, ..., channels). "
            f"For one channel add the axis: {name}[..., None]"
        )
    if a.shape[-1] > 16 and a.shape[1] <= 16:
        raise DataError(
            f"'{name}' has shape {a.shape}: expected channels last, but the last axis "
            f"looks like space. Move the channel axis: np.moveaxis({name}, 1, -1)"
        )


def load_data(path: str | Path, smoke: bool = False) -> dict:
    """Read and check the .npz file; return float32 arrays and a description."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"{path} does not exist")
    with np.load(path, allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    if "y" not in raw:
        raise DataError(f"{path} has no 'y' (the outputs); keys found: {sorted(raw)}")
    if "x" not in raw and "coords" not in raw:
        raise DataError(
            f"{path} needs 'x' (the inputs), or 'coords' if the geometry is the input"
        )
    arrays = {
        k: _as_float(k, v)
        for k, v in raw.items()
        if k in ("x", "y", "coords", "x_test", "y_test")
    }

    y = arrays["y"]
    _check_channels_last("y", y)
    n = len(y)
    mesh = False
    if "x" in arrays:
        x = arrays["x"]
        _check_channels_last("x", x)
        if len(x) != n:
            raise DataError(
                f"x and y have different numbers of samples: {len(x)} and {n}"
            )
        if x.shape[1:-1] != y.shape[1:-1]:
            raise DataError(
                f"x and y have different grids or point counts: {x.shape[1:-1]} and {y.shape[1:-1]}"
            )
    if "coords" in arrays:
        c = arrays["coords"]
        if c.ndim not in (2, 3) or c.shape[-1] not in (1, 2, 3):
            raise DataError(
                f"'coords' has shape {c.shape}; expected (points, d) or (samples, points, d), d = 1, 2 or 3"
            )
        if c.ndim == 3 and len(c) != n:
            raise DataError(
                f"coords and y have different numbers of samples: {len(c)} and {n}"
            )
        if c.shape[-2] != np.prod(y.shape[1:-1]):
            raise DataError(
                f"coords has {c.shape[-2]} points, y has {int(np.prod(y.shape[1:-1]))}"
            )
        mesh = y.ndim == 3
    if ("x_test" in arrays) != ("y_test" in arrays):
        raise DataError("give both 'x_test' and 'y_test', or neither")

    data = dict(
        x=arrays.get("x"),
        y=y,
        coords=arrays.get("coords"),
        x_test=arrays.get("x_test"),
        y_test=arrays.get("y_test"),
        kind="mesh" if mesh else "grid",
        spatial=tuple(y.shape[1:-1]),
        c_in=arrays["x"].shape[-1] if "x" in arrays else 0,
        c_out=y.shape[-1],
    )
    if smoke:
        for k in ("x", "y"):
            if data[k] is not None:
                data[k] = data[k][:32]
        if data["coords"] is not None and data["coords"].ndim == 3:
            data["coords"] = data["coords"][:32]
        for k in ("x_test", "y_test"):
            if data[k] is not None:
                data[k] = data[k][:8]
    return data


def split(data: dict, test_fraction: float, seed: int) -> tuple[dict, dict]:
    """Train and test sets: the given x_test/y_test, or a random share of the samples."""

    def take(idx):
        c = data["coords"]
        return dict(
            x=None if data["x"] is None else data["x"][idx],
            y=data["y"][idx],
            coords=None if c is None else (c[idx] if c.ndim == 3 else c),
        )

    n = len(data["y"])
    if data["y_test"] is not None:
        test = dict(
            x=data["x_test"],
            y=data["y_test"],
            coords=data["coords"]
            if data["coords"] is None or data["coords"].ndim == 2
            else None,
        )
        if data["coords"] is not None and data["coords"].ndim == 3:
            raise DataError(
                "with per-sample coords, put the test samples in x, y and coords and use --test-fraction"
            )
        return take(np.arange(n)), test
    perm = np.random.default_rng(seed).permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    if n - n_test < 1:
        raise DataError(f"{n} samples are too few to split off a test set")
    return take(np.sort(perm[n_test:])), take(np.sort(perm[:n_test]))


class Normalizer:
    """Per-channel mean and standard deviation of the training set."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    @classmethod
    def fit(cls, a: np.ndarray) -> Normalizer:
        axes = tuple(range(a.ndim - 1))
        return cls(a.mean(axes), np.maximum(a.std(axes), 1e-8))

    def encode(self, a):
        return (a - self.mean) / self.std

    def decode(self, a):
        return a * self.std + self.mean

    def to_json(self) -> dict:
        return dict(mean=self.mean.tolist(), std=self.std.tolist())


def grid_coords(spatial: tuple[int, ...]) -> np.ndarray:
    """Points of a uniform grid on [0, 1]^d, flattened: (prod(spatial), d)."""
    axes = [np.linspace(0.0, 1.0, s, dtype=np.float32) for s in spatial]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(spatial))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def parse_set(items: list[str]) -> dict:
    """--set KEY=VALUE pairs; values are ints, floats, booleans, tuples or strings."""

    def value(s: str):
        if "," in s:
            return tuple(value(p) for p in s.split(",") if p)
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        for t in (int, float):
            try:
                return t(s)
            except ValueError:
                pass
        return s

    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = value(v.strip())
    return out


def model_spec(model: str, size: str, info: dict, overrides: dict) -> dict:
    """The constructor arguments of the model for this data, and GINO's graph settings."""
    p = dict(PRESETS[model][size])
    spatial, d = info["spatial"], info["d"]
    c_in, c_out = max(info["c_in"], 1), info["c_out"]
    if model == "fno":
        modes = p.pop("n_modes")
        kw = dict(
            in_channels=c_in,
            out_channels=c_out,
            n_modes=tuple(min(modes, s // 2) for s in spatial),
            **p,
        )
    elif model == "wno":
        # each level halves the grid; keep at least 4 points on the coarsest level
        level = max(
            1,
            min(
                p.pop("level"),
                int(math.log2(min(spatial) / 4)) if min(spatial) >= 8 else 1,
            ),
        )
        # float32, as the benchmarks: WNO's default bfloat16 trains worse
        kw = dict(
            in_channels=c_in,
            out_channels=c_out,
            size=spatial if len(spatial) > 1 else spatial[0],
            level=level,
            compute_dtype="float32",
            **p,
        )
    elif model == "transolver":
        kw = dict(space_dim=d, fun_dim=info["c_in"], out_dim=c_out, **p)
    else:
        latent = p.pop("latent")
        modes = p.pop("fno_n_modes")
        radius = 2.0 / (latent - 1)  # about two latent cells
        kw = dict(
            in_channels=c_in,
            out_channels=c_out,
            gno_coord_dim=d,
            in_gno_radius=radius,
            out_gno_radius=radius,
            fno_n_modes=(min(modes, latent // 2),) * d,
            gno_use_open3d=False,
            **p,
        )
        kw["latent"] = latent
    kw.update(overrides)
    return kw


def build_model(name: str, kw: dict):
    from scirex.models import FNO, GINO, WNO, Transolver

    kw = {k: v for k, v in kw.items() if k != "latent"}
    cls = dict(fno=FNO, wno=WNO, transolver=Transolver, gino=GINO)[name]
    try:
        return cls(**kw)
    except TypeError as e:
        raise SystemExit(
            f"{name}: {e}. Check the --set keys against Models and Losses."
        ) from None


# ---------------------------------------------------------------------------
# Inputs of each model, from normalised arrays in the user's layout
# ---------------------------------------------------------------------------


class Pipeline:
    """Turns arrays in the user's layout into model inputs and model outputs back.

    Everything the run needs to predict new data is in ``to_json()``.
    """

    def __init__(
        self,
        model: str,
        kw: dict,
        info: dict,
        x_norm,
        y_norm,
        coord_lo=None,
        coord_hi=None,
    ):
        self.model, self.kw, self.info = model, kw, info
        self.x_norm, self.y_norm = x_norm, y_norm
        self.coord_lo, self.coord_hi = coord_lo, coord_hi
        self.net = build_model(model, kw)

    # -- coordinates ---------------------------------------------------------

    def points(self, coords, n: int) -> np.ndarray:
        """(n, points, d) coordinates scaled to [0, 1] with the training bounds."""
        if coords is None:
            c = grid_coords(self.info["spatial"])
        else:
            c = (coords - self.coord_lo) / (self.coord_hi - self.coord_lo)
        if c.ndim == 2:
            c = np.broadcast_to(c, (n, *c.shape))
        return np.ascontiguousarray(c, np.float32)

    def latent_grid(self) -> np.ndarray:
        g = self.kw["latent"]
        return grid_coords((g,) * self.info["d"]).reshape(
            *(g,) * self.info["d"], self.info["d"]
        )

    def graphs(self, pts: np.ndarray) -> dict:
        """GINO's encoder and decoder graphs, one per sample, padded to one length."""
        import jax.numpy as jnp

        from scirex.layers import build_neighbor_graph, edge_index_to_neighbors

        lat = self.latent_grid().reshape(-1, self.info["d"])
        enc, dec = [], []
        cache: dict = {}
        for p in pts:
            key = p.tobytes() if len(pts) > 1 else 0
            if key not in cache:
                e_in = build_neighbor_graph(
                    p,
                    lat,
                    self.kw["in_gno_radius"],
                    "radius",
                    max_num_neighbors=MAX_NEIGHBORS,
                )
                e_out = build_neighbor_graph(
                    lat,
                    p,
                    self.kw["out_gno_radius"],
                    "radius",
                    max_num_neighbors=MAX_NEIGHBORS,
                )
                cache[key] = (e_in, e_out)
            e_in, e_out = cache[key]
            enc.append(e_in)
            dec.append(e_out)

        def pad(edges, n_query):
            m = max(e.shape[1] for e in edges)
            out = []
            for e in edges:
                # padding edges: source 0 (any valid point), query one past the
                # last, which the graph layers ignore
                padded = np.zeros((2, m), np.int32)
                padded[1] = n_query
                padded[:, : e.shape[1]] = e
                nb = edge_index_to_neighbors(jnp.asarray(padded), num_query=n_query)
                out.append({k: np.asarray(v) for k, v in nb.items()})
            return {k: np.stack([o[k] for o in out]) for k in out[0]}

        return dict(
            neighbors_in=pad(enc, len(lat)), neighbors_out=pad(dec, pts.shape[1])
        )

    # -- model inputs ----------------------------------------------------------

    def inputs(self, x, coords, n: int) -> dict:
        """Numpy model inputs for n samples; every array has the samples first."""
        xn = None if x is None else self.x_norm.encode(x)
        if self.model in GRID_MODELS:
            return dict(x=xn)
        pts = self.points(coords, n)
        flat = None if xn is None else xn.reshape(n, pts.shape[1], -1)
        if self.model == "transolver":
            return dict(x=pts, fx=flat)
        feats = flat if flat is not None else np.ones((n, pts.shape[1], 1), np.float32)
        return dict(pts=pts, x=feats, **self.graphs(pts))

    def apply(self, params, batch: dict):
        """Normalised predictions in the user's layout, (n, *spatial, c_out)."""
        import jax
        import jax.numpy as jnp

        net = self.net
        n = len(next(v for v in jax.tree_util.tree_leaves(batch)))
        if self.model == "fno":
            x = jnp.moveaxis(batch["x"], -1, 1)  # channels first
            return jnp.moveaxis(net.apply(params, x), 1, -1)
        if self.model == "wno":
            return net.apply(params, batch["x"])
        if self.model == "transolver":
            out = net.apply(params, x=batch["x"], fx=batch["fx"])
            return out.reshape(n, *self.out_spatial(batch), -1)
        lat = jnp.asarray(self.latent_grid())[None]

        def one(pts, x, nb_in, nb_out):
            return net.apply(
                params,
                input_geom=pts[None],
                latent_queries=lat,
                output_queries=pts[None],
                x=x[None],
                neighbors_in=nb_in,
                neighbors_out=nb_out,
            )[0]

        out = jax.vmap(one)(
            batch["pts"], batch["x"], batch["neighbors_in"], batch["neighbors_out"]
        )
        return out.reshape(n, *self.out_spatial(batch), -1)

    def out_spatial(self, batch) -> tuple:
        n_points = (batch["x"] if self.model == "transolver" else batch["pts"]).shape[1]
        spatial = self.info["spatial"]
        return spatial if int(np.prod(spatial)) == n_points else (n_points,)

    def init(self, seed: int, batch: dict):
        import jax

        one = jax.tree_util.tree_map(lambda a: a[:1], batch)
        if self.model == "fno":
            return self.net.init(jax.random.PRNGKey(seed), np.moveaxis(one["x"], -1, 1))
        if self.model == "wno":
            return self.net.init(jax.random.PRNGKey(seed), one["x"])
        if self.model == "transolver":
            return self.net.init(jax.random.PRNGKey(seed), x=one["x"], fx=one["fx"])
        return self.net.init(
            jax.random.PRNGKey(seed),
            input_geom=one["pts"],
            latent_queries=self.latent_grid()[None],
            output_queries=one["pts"],
            x=one["x"],
            neighbors_in=jax.tree_util.tree_map(lambda a: a[0], one["neighbors_in"]),
            neighbors_out=jax.tree_util.tree_map(lambda a: a[0], one["neighbors_out"]),
        )

    def to_json(self) -> dict:
        return dict(
            model=self.model,
            model_kwargs={
                k: list(v) if isinstance(v, tuple) else v for k, v in self.kw.items()
            },
            data=self.info,
            x_norm=None if self.x_norm is None else self.x_norm.to_json(),
            y_norm=self.y_norm.to_json(),
            coord_bounds=None
            if self.coord_lo is None
            else [self.coord_lo.tolist(), self.coord_hi.tolist()],
        )

    @classmethod
    def from_json(cls, cfg: dict) -> Pipeline:
        kw = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in cfg["model_kwargs"].items()
        }
        info = dict(cfg["data"], spatial=tuple(cfg["data"]["spatial"]))
        xn = None if cfg["x_norm"] is None else Normalizer(**cfg["x_norm"])
        lo = hi = None
        if cfg["coord_bounds"] is not None:
            lo, hi = (np.asarray(b, np.float32) for b in cfg["coord_bounds"])
        return cls(cfg["model"], kw, info, xn, Normalizer(**cfg["y_norm"]), lo, hi)


def rel_l2(pred, true):
    """Relative L2 error per sample."""
    import jax.numpy as jnp

    b = pred.shape[0]
    diff = jnp.linalg.norm((pred - true).reshape(b, -1), axis=1)
    return diff / jnp.maximum(jnp.linalg.norm(true.reshape(b, -1), axis=1), 1e-12)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_pipeline(args, train: dict) -> Pipeline:
    info = dict(
        kind=args._kind,
        spatial=args._spatial,
        c_in=0 if train["x"] is None else int(train["x"].shape[-1]),
        c_out=int(train["y"].shape[-1]),
    )
    lo = hi = None
    if train["coords"] is not None:
        c = train["coords"].reshape(-1, train["coords"].shape[-1])
        lo, hi = c.min(0), c.max(0)
        hi = np.where(hi > lo, hi, lo + 1.0)
        info["d"] = int(c.shape[-1])
    else:
        info["d"] = len(args._spatial)
    if args.model in GRID_MODELS and info["kind"] == "mesh":
        raise DataError(
            f"{args.model} needs data on a regular grid; for a mesh use --model transolver or gino"
        )
    if args.model in GRID_MODELS and train["x"] is None:
        raise DataError(f"{args.model} needs an input field 'x'")
    kw = model_spec(
        args.model, "smoke" if args.smoke else args.size, info, parse_set(args.set)
    )
    x_norm = None if train["x"] is None else Normalizer.fit(train["x"])
    return Pipeline(args.model, kw, info, x_norm, Normalizer.fit(train["y"]), lo, hi)


def train_run(args) -> dict:
    """Train, evaluate and save; return the metrics."""
    import jax
    import jax.numpy as jnp
    import optax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    import scirex
    from scirex.training import adamw_cx
    from scirex.utils import count_model_params

    # Keep float32, also if float64 was enabled earlier in the process.
    jax.config.update("jax_enable_x64", False)

    devices = jax.devices()
    print(f"Device: {len(devices)} x {devices[0].device_kind} ({devices[0].platform})")

    data = load_data(args.data, smoke=args.smoke)
    args._kind, args._spatial = data["kind"], data["spatial"]
    train, test = split(data, args.test_fraction, args.seed)
    pipe = make_pipeline(args, train)
    n_train, n_test = len(train["y"]), len(test["y"])
    where = (
        f"grid {data['spatial']}"
        if data["kind"] == "grid"
        else f"mesh {data['spatial'][0]} points"
    )
    print(
        f"Data:   {args.data}  {where}  in {data['c_in']}  out {data['c_out']}  train {n_train}  test {n_test}"
    )

    t_in = pipe.inputs(train["x"], train["coords"], n_train)
    v_in = pipe.inputs(test["x"], test["coords"], n_test)
    t_y = train["y"]
    v_y = test["y"]

    params = pipe.init(args.seed, t_in)
    n_params = count_model_params(params)
    print(
        f"Model:  {args.model}  size {'smoke' if args.smoke else args.size}  {n_params / 1e6:.2f} M parameters"
    )

    # Data parallel: every device holds the weights, each batch is split.
    n_dev = len(devices)
    batch = max(n_dev, (min(args.batch_size, n_train) // n_dev) * n_dev)
    if batch != args.batch_size:
        print(
            f"Batch:  {args.batch_size} -> {batch} ({batch // n_dev} per device on {n_dev} device(s))"
        )
    mesh = Mesh(np.array(devices), ("data",))
    shard = NamedSharding(mesh, PartitionSpec("data"))
    replicate = NamedSharding(mesh, PartitionSpec())

    epochs = 2 if args.smoke else args.epochs
    steps_per_epoch = max(1, n_train // batch)
    schedule = optax.cosine_decay_schedule(
        args.lr, decay_steps=epochs * steps_per_epoch
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(adamw_cx, hyperparam_dtype=jnp.float32)(
            learning_rate=schedule, weight_decay=1e-4
        ),
    )
    params = jax.device_put(params, replicate)
    opt_state = jax.device_put(tx.init(params), replicate)

    def loss_fn(p, b, y):  # relative L2 in the units of the data
        return jnp.mean(rel_l2(pipe.y_norm.decode(pipe.apply(p, b)), y))

    @jax.jit
    def train_step(p, s, b, y):
        loss, g = jax.value_and_grad(loss_fn)(p, b, y)
        u, s = tx.update(g, s, p)
        return optax.apply_updates(p, u), s, loss

    @jax.jit
    def predict_norm(p, b):
        return pipe.apply(p, b)

    def predict(p, inputs, n):
        outs = []
        for i in range(0, n, batch):
            b = jax.tree_util.tree_map(lambda a: a[i : i + batch], inputs)
            m = len(b["x"]) if b.get("x") is not None else len(b["pts"])
            pad = (-m) % n_dev
            if pad:
                b = jax.tree_util.tree_map(
                    lambda a: np.concatenate([a, np.repeat(a[-1:], pad, 0)]), b
                )
            outs.append(np.asarray(predict_norm(p, jax.device_put(b, shard)))[:m])
        return pipe.y_norm.decode(np.concatenate(outs))

    def test_errors(p):
        return np.asarray(
            rel_l2(jnp.asarray(predict(p, v_in, n_test)), jnp.asarray(v_y))
        )

    every = max(1, epochs // 20)
    history = []
    key = jax.random.PRNGKey(args.seed + 1)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        key, sub = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sub, n_train))
        losses = []
        for s in range(steps_per_epoch):
            idx = perm[s * batch : (s + 1) * batch]
            b = jax.tree_util.tree_map(lambda a: a[idx], t_in)
            b, y = jax.device_put((b, t_y[idx]), shard)
            params, opt_state, loss = train_step(params, opt_state, b, y)
            losses.append(loss)
        if epoch % every == 0 or epoch == epochs:
            err = float(test_errors(params).mean())
            train_loss = float(np.mean([float(v) for v in losses]))
            history.append(
                dict(
                    epoch=epoch,
                    train_rel_l2=train_loss,
                    test_rel_l2=err,
                    seconds=time.perf_counter() - t0,
                )
            )
            print(f"epoch {epoch:5d}   train {train_loss:.2e}   test rel. L2 {err:.2%}")
    train_seconds = time.perf_counter() - t0

    errors = test_errors(params)
    out = Path(args.out or f"runs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").unlink(missing_ok=True)  # written last: marks a complete run
    host_params = jax.device_get(params)
    pipe.net.save_checkpoint(out, "model", host_params)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    metrics = dict(
        model=args.model,
        size="smoke" if args.smoke else args.size,
        n_params=int(n_params),
        n_train=n_train,
        n_test=n_test,
        test_rel_l2_mean=float(errors.mean()),
        test_rel_l2_median=float(np.median(errors)),
        test_rel_l2_p90=float(np.percentile(errors, 90)),
        test_rel_l2_worst=float(errors.max()),
        epochs=epochs,
        train_seconds=train_seconds,
        provenance=dict(
            command=" ".join([Path(sys.executable).name, *sys.argv]),
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            python=platform.python_version(),
            jax=jax.__version__,
            scirex=scirex.__version__,
            devices=[f"{d.platform}:{d.device_kind}" for d in devices],
        ),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    options = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    (out / "config.json").write_text(
        json.dumps(dict(pipe.to_json(), options=options), indent=2)
    )
    print(f"Saved {out}  (test rel. L2 {errors.mean():.2%})")
    return metrics


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def device_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("devices")
    g.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
        help="force a backend (default: the best one found)",
    )
    g.add_argument("--gpus", help="use only these GPUs, e.g. 0 or 0,1")
    g.add_argument(
        "--no-preallocate",
        action="store_true",
        help="do not reserve 75%% of the GPU memory at start",
    )
    g.add_argument(
        "--deterministic",
        action="store_true",
        help="bit-identical runs (GINO on a GPU becomes much slower)",
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument(
        "--data", required=True, help="the .npz file with x, y (and coords for a mesh)"
    )
    p.add_argument(
        "--size",
        choices=SIZES,
        default="medium",
        help="model width and depth (default: medium)",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--batch-size", type=int, default=16, help="samples per step, over all devices"
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="peak learning rate (AdamW, cosine decay)",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="share held out when there is no x_test",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="run directory (default: runs/<model>)")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="any model argument; repeat for several",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model, 2 epochs on 32 samples: checks the setup",
    )
    device_arguments(p)
    args = p.parse_args(argv)
    if not 0 < args.test_fraction < 1:
        p.error("--test-fraction must be between 0 and 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_devices(
        args.device, args.gpus, not args.no_preallocate, args.deterministic
    )
    try:
        train_run(args)
    except DataError as e:
        raise SystemExit(f"Data error: {e}") from None
    except Exception as e:  # turn an XLA out-of-memory traceback into a hint
        if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
            raise SystemExit(
                "Out of memory: lower --batch-size, or use --size small."
            ) from None
        raise


if __name__ == "__main__":
    main()
