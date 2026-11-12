# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The models and their adapters for this benchmark.

Three architectures, chosen by a run's ``architecture`` (:data:`ARCHITECTURES`):

* ``"gino"``, the paper's own, built by ``scirex.get_model_jax`` from the
  zencfg config ``_config.build_config`` produces;
* ``"transolver"``, :class:`scirex.models.Transolver` on the car's surface
  mesh (vertex position + vertex normal per point), at GINO's parameter
  budget;
* ``"giwo"``, :class:`_giwo.GIWO` (GINO with a WNO latent operator), on
  GINO's inputs and neighbour lists, at GINO's parameter budget.

What this module adds around them:

* :class:`CarCFDModelWrapper` holds the params and optimiser and exposes
  the ``train_step``/``__call__`` contract
  :class:`scirex.experimental.training.Trainer` drives. The generic
  ``_model_wrapper.FlaxModelWrapper`` routes only an ``x`` key; GINO also
  needs its geometry (``input_geom``/``latent_queries``/``output_queries``),
  its latent SDF (``latent_features``) and its neighbour lists.
* :class:`GINOCFDDataProcessor` turns a Car-CFD batch into GINO's inputs,
  :class:`TransolverCFDDataProcessor` into Transolver's ``x``. In eval mode
  both de-normalise prediction and truth, so the test metric is the paper's
  *de-normalised* relative L2 for either model.
* :func:`build` is the one place an architecture is chosen.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from scirex import get_model_jax
from scirex.data.transforms.data_processors import DataProcessor
from scirex.models import Transolver
from scirex.training import adamw_cx
from scripts.benchmarks.car_cfd._config import lr_schedule
from scripts.benchmarks.car_cfd._data import attach_neighbors
from scripts.benchmarks.car_cfd._giwo import GIWO

#: Every keyword GINO's ``__call__`` takes (Transolver's ``x`` among them);
#: anything else in a batch (``y``, raw mesh fields) is dropped before
#: ``module.apply``.
MODEL_KEYS = frozenset(
    {
        "input_geom",
        "latent_queries",
        "output_queries",
        "x",
        "latent_features",
        "ada_in",
        "neighbors_in",
        "neighbors_out",
    }
)


def model_inputs(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if k in MODEL_KEYS and v is not None}


class CarCFDModelWrapper:
    """A Flax module (GINO or Transolver) plus its params and optax optimiser.

    Single-device (``jax.jit``) only -- the benchmark pins to one GPU via
    ``CUDA_VISIBLE_DEVICES``. ``optax.inject_hyperparams`` makes the learning
    rate part of the optimiser state, so :meth:`set_lr` can apply StepLR
    without recompiling the step.
    ``learning_rate`` may also be an optax schedule (per step);
    ``inject_hyperparams`` then evaluates it every step and the current value
    sits in ``opt_state.hyperparams``. ``grad_clip`` clips the global
    gradient norm before the Adam update.
    """

    def __init__(
        self,
        flax_module,
        learning_rate: float,
        weight_decay: float,
        seed: int = 0,
        grad_clip: float | None = None,
        b2: float = 0.999,
        eps: float = 1e-6,
    ):
        self.module = flax_module
        self.learning_rate = learning_rate
        self.params = None
        self.opt_state = None
        self.key = jax.random.PRNGKey(seed)

        def make_tx(learning_rate, weight_decay):
            adam = adamw_cx(learning_rate, weight_decay=weight_decay, b2=b2, eps=eps)
            if grad_clip is None:
                return adam
            return optax.chain(optax.clip_by_global_norm(grad_clip), adam)

        self.tx = optax.inject_hyperparams(make_tx)(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self._jit_fn = None
        # The eval forward pass, jitted once per input shape.
        self._apply = jax.jit(flax_module.apply)

    @property
    def initialized(self) -> bool:
        return self.params is not None

    def init(self, batch: dict) -> None:
        self.params = self.module.init(self.key, **model_inputs(batch))
        self.opt_state = self.tx.init(self.params)

    def set_lr(self, new_lr: float) -> None:
        self.learning_rate = new_lr
        self.opt_state.hyperparams["learning_rate"] = jnp.array(new_lr)

    def _build_jit_fn(self, loss_fn):
        module = self.module
        tx = self.tx

        @jax.jit
        def _step(params, opt_state, inputs, y):
            def forward_loss(p):
                return loss_fn(module.apply(p, **inputs), y=y)

            loss, grads = jax.value_and_grad(forward_loss)(params)
            updates, new_opt = tx.update(grads, opt_state, params)
            return loss, optax.apply_updates(params, updates), new_opt

        return _step

    def __call__(self, **kwargs):
        if not self.initialized:
            self.init(kwargs)
        return self._apply(self.params, **model_inputs(kwargs))

    def train_step(self, batch: dict, loss_fn):
        if not self.initialized:
            self.init(batch)
        if self._jit_fn is None:
            self._jit_fn = self._build_jit_fn(loss_fn)
        loss, self.params, self.opt_state = self._jit_fn(
            self.params, self.opt_state, model_inputs(batch), batch["y"]
        )
        return loss

    def precompute_neighbors(self, *args, **kwargs):
        return self.module.precompute_neighbors(*args, **kwargs)


class GINOCFDDataProcessor(DataProcessor):
    """Converts Car-CFD batches into GINO inputs (see the module docstring)."""

    def __init__(self, normalizer, device: str = "cuda"):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None
        self.training = True

    def preprocess(self, sample: dict) -> dict:
        vertices = np.asarray(sample["vertices"])
        press = np.asarray(sample["press"])
        # (1, 1, n_press) or (1, n_press) -> (1, n_press, 1)
        truth = np.expand_dims(np.squeeze(press, axis=0), axis=-1)

        # The dataset drops vertices 16..111 from the pressure field (see
        # CarCFDDatasetjax), so the output queries are cut to match.
        out_p = vertices[:, : truth.shape[1], :]

        batch = dict(
            input_geom=vertices,
            latent_queries=np.asarray(sample["query_points"]),
            output_queries=out_p,
            latent_features=np.asarray(sample["distance"]),
            y=truth,
            x=None,
        )
        for key in ("neighbors_in", "neighbors_out"):
            if sample.get(key) is not None:
                batch[key] = {k: np.asarray(v) for k, v in sample[key].items()}
        return batch

    def postprocess(self, out, sample: dict):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = jnp.asarray(sample["y"])
            if y.ndim > 1:
                y = jnp.squeeze(y, axis=0)
            sample["y"] = self.normalizer.inverse_transform(y)
        return out, sample

    def to(self, device: str):
        self.device = device
        if hasattr(self.normalizer, "to"):
            self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model
        return self

    def __call__(self, sample: dict, training: bool = True):
        self.training = training
        sample = self.preprocess(sample)
        out = self.model(**model_inputs(sample))
        return self.postprocess(out, sample)


class TransolverCFDDataProcessor(GINOCFDDataProcessor):
    """Converts Car-CFD batches into Transolver's ``x``: per surface point,
    the range-normalised vertex position and the unit vertex normal,
    ``(1, n_press, 6)``.

    This is the Transolver paper's Car-Design-ShapeNetCar input (position,
    SDF, normal) less the SDF, which is identically zero on the surface the
    pressure lives on. Post-processing (de-normalisation) is GINO's.
    """

    # DataProcessor is a Flax dataclass, whose transform would otherwise give
    # this subclass a generated __init__ in place of the parent's.
    def __init__(self, normalizer, device: str = "cuda"):
        super().__init__(normalizer, device)

    def preprocess(self, sample: dict) -> dict:
        # Every car has the same vertex count, so a batch of B cars stacks:
        # press (B, 1, n_press) -> (B, n_press, 1).
        vertices = np.asarray(sample["vertices"])
        truth = np.asarray(sample["press"]).reshape(vertices.shape[0], -1, 1)
        n_press = truth.shape[1]
        # Cut to the pressure field's vertices, as GINO's output queries are.
        pos = vertices[:, :n_press, :]
        normals = np.asarray(sample["vertex_normals"])[:, :n_press, :]
        return dict(x=np.concatenate([pos, normals], axis=-1), y=truth)


#: architecture -> (label, data processor class).
ARCHITECTURES: dict[str, tuple[str, type[GINOCFDDataProcessor]]] = {
    "gino": ("GINO", GINOCFDDataProcessor),
    "transolver": ("Transolver", TransolverCFDDataProcessor),
    "giwo": ("GIWO", GINOCFDDataProcessor),
}

#: Architectures that read GINO's geometry and need its neighbour lists.
GNO_ARCHITECTURES = frozenset({"gino", "giwo"})


def build_module(run, config: Any):
    """The bare Flax module for ``run``: GINO from ``config``
    (``_config.build_config``'s output), Transolver from ``run.model``, GIWO
    from ``config`` plus ``run.model``'s ``wno_*`` fields."""
    if run.architecture == "gino":
        return get_model_jax(config)
    if run.architecture == "giwo":
        kwargs = {k: v for k, v in config.model.items() if not k.startswith("_")}
        kwargs.pop("model_arch")
        kwargs["in_channels"] = kwargs.pop("data_channels")
        wno = {k: v for k, v in run.model.items() if k.startswith("wno_")}
        return GIWO(**kwargs, wno_size=(run.sdf_query_resolution,) * 3, **wno)
    if run.architecture == "transolver":
        return Transolver(
            space_dim=6, fun_dim=0, out_dim=1, geometry="irregular", **run.model
        )
    raise ValueError(
        f"run {run.name!r}: unknown architecture {run.architecture!r}; "
        f"choose from {sorted(ARCHITECTURES)}"
    )


def build(
    run, config: Any, data_module, log_dir, seed: int = 0
) -> tuple[CarCFDModelWrapper, GINOCFDDataProcessor]:
    """The wrapped model and its data processor for ``run``, on
    ``data_module``. GINO's neighbour lists are precomputed into the dataset
    here for GINO and GIWO (Transolver needs none)."""
    learning_rate = config.opt.learning_rate
    if run.schedule is not None:
        steps_per_epoch = -(-len(data_module.train_data) // run.batch_size)
        learning_rate = lr_schedule(run, steps_per_epoch)
    model = CarCFDModelWrapper(
        build_module(run, config),
        learning_rate=learning_rate,
        weight_decay=config.opt.weight_decay,
        seed=seed,
        grad_clip=run.grad_clip,
        b2=run.adam_b2,
        eps=run.adam_eps,
    )
    if run.architecture in GNO_ARCHITECTURES:
        attach_neighbors(model, data_module, log_dir)
    processor_cls = ARCHITECTURES[run.architecture][1]
    return model, processor_cls(deepcopy(data_module.normalizers["press"]))


def predict_denormalised(
    model: CarCFDModelWrapper, processor: GINOCFDDataProcessor, sample: dict
) -> tuple[np.ndarray, np.ndarray]:
    """``(truth, pred)`` surface pressure for one loader batch, in physical
    units, flattened to ``(n_press,)``."""
    batch = processor.preprocess(sample)
    pred = np.asarray(model(**model_inputs(batch))).squeeze(-1)  # (1, n_press)
    truth = np.squeeze(np.asarray(sample["press"]), axis=0)  # (1, n_press)
    inv = processor.normalizer.inverse_transform
    return (
        np.asarray(inv(jnp.asarray(truth))).ravel(),
        np.asarray(inv(jnp.asarray(pred))).ravel(),
    )
