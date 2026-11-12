<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
SPDX-License-Identifier: CC-BY-4.0
-->

# scirex.experimental

Code in `scirex.experimental` has **no stability guarantee**. Any name here can change
its signature or behaviour, move, or disappear in any release, including a patch
release, without a deprecation period and without a CHANGELOG entry under
*Deprecated*. Notable changes are listed in the CHANGELOG on a best-effort basis.

Use it if you accept that. Pin the exact ScireX version when you do, and expect to adapt
on upgrade. Code moves out of `scirex.experimental` into a stable package once its
interface has settled; the move is announced in the CHANGELOG.

Everything is imported from its module, for example
`from scirex.experimental.giwo import GIWO`. There is no package-level export list.

## What lives here

| Module | What it provides | Status |
|---|---|---|
| `scirex.experimental.giwo` | `GIWO`, a geometry-informed wavelet neural operator (GINO's outer graph layers around a WNO latent operator) | Not a published architecture; the interface is not settled. `get_model_jax` builds it under the name `"giwo"`. |
| `scirex.experimental.patching` | `MultigridPatching2D`, `make_patches`, `MGPatchingDataProcessor`: multigrid domain decomposition for high-resolution inputs | 2D only; the interface is not settled |
| `scirex.experimental.distributed` | Model-parallel communication, tensor splitting and gather helpers (`comm`, `helpers`, `mappings`) | Ported from a PyTorch implementation; the model-parallel interface is not settled |
| `scirex.experimental.losses` | `PDEResidualLoss` and the `AdvectionTerm`, `DivergenceTerm`, `GradientTerm`, `ReactionTerm`, `TransientTerm` terms; `NavierStokesEqnLoss`; `T3Galerkin`, `T6Galerkin`; `FourierDiff`; the automatic-differentiation residuals (`ADDiff`, `*TermAD`) | Not used by the released benchmarks; the interface is not settled |
| `scirex.experimental.training` | `Trainer`, `SimpleDataLoader`, `save_training_state`, `load_training_state`: a generic training loop for any object with the `train_step` / `__call__` contract | Only the Car-CFD benchmark uses it; the other benchmarks have their own loops. The interface is not settled |
| `scirex.experimental.tuning` | `tune`, `suggest_params`, `TrialContext`: Optuna hyperparameter tuning tracked in MLflow | Needs the `hpo` extra (`pip install "scirex[hpo]"`); the interface is not settled |
| `scirex.experimental.distributed.config` | `DistributedConfig`, the multi-GPU / model-parallel config class | Used by the Car-CFD experiment config; not part of `scirex.config` |
| `scirex.experimental.geometry` | Mesh I/O (`mesh_io`) and Gmsh mesh generation (`gmsh_mesh`) for the Galerkin losses | Needs optional packages (`meshio`, `gmsh`); the interface is not settled |
| `scirex.experimental.analysis` | `spectrum_2d`, `compute_rank`, `compute_stable_rank`, `compute_explained_variance` | Research helpers; their semantics may change |

## GIWO

::: scirex.experimental.giwo.GIWO

## Patching

::: scirex.experimental.patching.MGPatchingDataProcessor

::: scirex.experimental.patching.MultigridPatching2D

::: scirex.experimental.patching.make_patches

## Tuning

`import scirex.experimental.tuning` always works. Using a name without the `hpo` extra
raises an `ImportError` that names the extra.

::: scirex.experimental.tuning.tune

::: scirex.experimental.tuning.suggest_params

::: scirex.experimental.tuning.TrialContext

## Distributed configuration

::: scirex.experimental.distributed.config.DistributedConfig

## Experimental losses

::: scirex.experimental.losses.PDEResidualLoss

::: scirex.experimental.losses.AdvectionTerm

::: scirex.experimental.losses.DivergenceTerm

::: scirex.experimental.losses.GradientTerm

::: scirex.experimental.losses.ReactionTerm

::: scirex.experimental.losses.TransientTerm

::: scirex.experimental.losses.FourierDiff

::: scirex.experimental.losses.NavierStokesEqnLoss

::: scirex.experimental.losses.DarcyHybridLossGINO

::: scirex.experimental.losses.BatchedVariableCoefficientGalerkinResidualLossT6

::: scirex.experimental.losses.ADDiff

::: scirex.experimental.losses.AdvectionTermAD

::: scirex.experimental.losses.DiffusionTermAD

::: scirex.experimental.losses.ReactionTermAD

::: scirex.experimental.losses.SourceTermAD

::: scirex.experimental.losses.T3Galerkin

::: scirex.experimental.losses.T6Galerkin

::: scirex.experimental.losses.BatchedVariableCoefficientGalerkinResidualLossT3

## Experimental training loop

`Trainer` needs a model wrapper that holds the parameters and optimiser and provides
`train_step`; the Car-CFD benchmark (`scripts/benchmarks/car_cfd/_models.py`) shows one.
The library models (`FNO`, `GINO`, ...) are plain Flax modules and cannot be passed to
`Trainer` directly.

::: scirex.experimental.training.Trainer

::: scirex.experimental.training.SimpleDataLoader

::: scirex.experimental.training.save_training_state

::: scirex.experimental.training.load_training_state

## Analysis

::: scirex.experimental.analysis.spectrum_2d

::: scirex.experimental.analysis.compute_rank

::: scirex.experimental.analysis.compute_stable_rank

::: scirex.experimental.analysis.compute_explained_variance
