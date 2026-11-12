# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Model configuration presets.

Ready-to-use `ConfigBase` definitions for the neural operators (FNO, GINO,
Transolver) at a range of sizes, referenced by the experiment configs.
"""

from typing import Any

from zencfg import ConfigBase


class ModelConfig(ConfigBase):
    """Base class of all model configurations.

    Subclasses (`FNOConfig`, `GINOConfig`, `TransolverConfig`)
    add the hyperparameters of one architecture and select it through their
    `model_arch` field. Presets such as `FNOSmall2d` can be selected by
    class name wherever zencfg resolves a `ModelConfig`.
    """

    arch: str
    data_channels: int
    out_channels: int


class FNOConfig(ModelConfig):
    """Full set of hyperparameters of the Fourier neural operator (`FNO`).

    Most users start from `SimpleFNOConfig` or one of the size presets
    (`FNOSmall2d`, `FNOMedium2d`, ...), which expose the few
    parameters that matter most.
    """

    model_arch: str = "fno"
    data_channels: int
    out_channels: int
    n_modes: list[int]
    hidden_channels: int
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 4
    n_layers: int = 4
    domain_padding: float = 0.0
    norm: str = "None"
    fno_skip: str = "linear"
    implementation: str = "factorized"
    use_channel_mlp: bool = True
    channel_mlp_expansion: float = 0.5
    channel_mlp_dropout: float = 0
    separable: bool = False
    factorization: str = "None"
    rank: float = 1.0
    fixed_rank_modes: bool = False
    stabilizer: str = "None"


class SimpleFNOConfig(FNOConfig):
    """
    SimpleFNOConfig: a basic FNO config that provides access to only
    the most important FNO parameters.
    """

    data_channels: int
    out_channels: int
    n_modes: list[int]
    hidden_channels: int
    projection_channel_ratio: int


class FNOSmall2d(SimpleFNOConfig):
    """FNOSmall2d: a basic, small FNO for 2d problems."""

    data_channels: int = 1
    out_channels: int = 1
    n_modes: list[int] = [16, 16]
    hidden_channels: int = 24
    projection_channel_ratio: int = 2


class FNOMedium2d(SimpleFNOConfig):
    """FNOMedium2d: a basic, medium-sized FNO for 2d problems."""

    data_channels: int = 1
    out_channels: int = 1
    n_modes: list[int] = [64, 64]
    hidden_channels: int = 64
    projection_channel_ratio: int = 4


class FNOLarge2d(SimpleFNOConfig):
    """FNOLarge2d: a large FNO for 2d problems."""

    data_channels: int = 1
    out_channels: int = 1
    n_modes: list[int] = [64, 64]
    hidden_channels: int = 128
    projection_channel_ratio: int = 2


class FNOHuge2d(SimpleFNOConfig):
    """
    FNOHuge2d: a giant FNO for 2d problems.

    Note that this will likely be too large for most single-GPU local settings,
    and should be paired with tensor factorization, low-rank/sparse optimization
    via TensorGRaD, or some other means of reducing memory overhead.
    """

    data_channels: int = 1
    out_channels: int = 1
    n_modes: list[int] = [128, 128]
    hidden_channels: int = 128
    projection_channel_ratio: int = 2


class FNOMedium3d(SimpleFNOConfig):
    """FNOMedium3d: a medium FNO for 3d problems."""

    data_channels: int = 1
    out_channels: int = 1
    n_modes: list[int] = [32, 32, 32]
    hidden_channels: int = 64
    projection_channel_ratio: int = 4


class GINOConfig(ModelConfig):
    """Hyperparameters of the geometry-informed neural operator (`GINO`).

    Covers the graph neural operator encoder/decoder (`gno_*` keys) and the
    latent FNO (`fno_*` keys). Use one of the presets (`GINOSmall3d`,
    `GINOPoisson2d`, `GINODarcy2d`) as a starting point.
    """

    model_arch: str = "gino"  # all GINO configs must use this
    data_channels: int
    out_channels: int
    latent_feature_channels: int | None = None
    # False selects the GINO paper's decoder-only variant (no input GNO).
    use_in_gno: bool = True
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    gno_radius: float = 0.033
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_pos_embed_type: str = "nerf"
    gno_weighting_function: str | None = None
    gno_weight_function_scale: float | None = None
    fno_n_modes: list[int] = [16, 16, 16]
    fno_hidden_channels: int = 64
    fno_use_channel_mlp: bool = True
    fno_norm: str | None = "instance_norm"
    fno_ada_in_features: int = 32
    fno_factorization: str | None = "tucker"
    fno_rank: float = 0.4
    fno_domain_padding: float = 0.125
    fno_channel_mlp_expansion: float = 1.0
    fno_resolution_scaling_factor: int = 1


class GINOSmall3d(GINOConfig):
    """GINO preset for 3d problems with one scalar output field."""

    data_channels: int = 0
    out_channels: int = 1
    latent_feature_channels: int | None = 1
    gno_coord_dim: int = 3
    gno_coord_embed_dim: int = 16
    gno_radius: float = 0.033


class GINOPoisson2d(GINOConfig):
    """GINO preset for the 2d Poisson problem on the unit square."""

    data_channels: int = 3
    out_channels: int = 1
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 2
    in_gno_pos_embed_type: Any | None = None
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 16
    gno_embed_max_positions: int = 600
    gno_use_torch_scatter: bool = True
    in_gno_radius: float = 0.16
    out_gno_radius: float = 0.175
    in_gno_transform_type: str = "linear"
    out_gno_transform_type: str = "linear"
    gno_reduction: str = "mean"
    gno_weighting_function: str = "half_cos"
    gno_weight_function_scale: float = 0.030625
    gno_use_open3d: bool = False
    in_gno_channel_mlp_hidden_layers: list[int] = [256, 512, 256]
    out_gno_channel_mlp_hidden_layers: list[int] = [512, 1024, 512]
    in_gno_tanh: Any | None = None
    out_gno_tanh: Any | None = None
    fno_n_modes: list[int] = [20, 20]
    fno_hidden_channels: int = 64
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4
    fno_use_channel_mlp: bool = True
    fno_channel_mlp_expansion: float = 0.5
    fno_norm: str = "group_norm"
    fno_ada_in_features: int = 8
    fno_factorization: Any | None = None
    fno_rank: float = 0.8
    fno_domain_padding: float = 0.0


class GINODarcy2d(GINOConfig):
    """GINO preset for the 2D Darcy-flow benchmark: a single scalar input
    field (the permeability) and a single scalar output field (the
    pressure), both living on the same fixed regular grid -- see
    `doc/theory_guide/operator_learning.md` (GINO section).

    Radius/embedding settings reuse `GINOPoisson2d`'s own 2D
    unit-square tuning (Poisson is Darcy with a constant permeability
    `a == 1`, so the same coordinate scale applies). Only fields that
    `GINO` (`src/scirex/models/gino.py`) actually declares as
    constructor parameters are carried over -- `GINOPoisson2d` also sets
    `gno_reduction` and `in_gno_tanh`, neither of which `GINO` has a
    matching field for (its reduction mode is hardcoded per-block in
    `GINO.setup()`, and it only has `out_gno_tanh`), so those two are
    deliberately not repeated here.
    """

    data_channels: int = 1
    out_channels: int = 1
    projection_channel_ratio: int = 4
    gno_coord_dim: int = 2
    in_gno_pos_embed_type: Any | None = None
    out_gno_pos_embed_type: str = "transformer"
    gno_embed_channels: int = 16
    gno_embed_max_positions: int = 600
    gno_use_torch_scatter: bool = True
    in_gno_radius: float = 0.16
    out_gno_radius: float = 0.175
    gno_use_open3d: bool = False
    in_gno_channel_mlp_hidden_layers: list[int] = [256, 512, 256]
    out_gno_channel_mlp_hidden_layers: list[int] = [512, 1024, 512]
    fno_n_modes: list[int] = [16, 16]
    fno_hidden_channels: int = 32
    fno_lifting_channel_ratio: int = 4
    fno_n_layers: int = 4
    fno_channel_mlp_expansion: float = 0.5
    fno_ada_in_features: int = 8
    fno_norm: str = "group_norm"
    fno_factorization: Any | None = None
    fno_rank: float = 0.8
    fno_domain_padding: float = 0.0


class TransolverConfig(ModelConfig):
    """Config for the unified Transolver model (Physics Attention over
    slice tokens), covering the Irregular Mesh, Structured Mesh 2D, and
    Structured Mesh 3D geometries from a single class keyed by `geometry`.

    Note: unlike FNO/GINO, Transolver has no `in_channels`/`out_channels`
    fields -- its inputs/outputs are parameterized by `space_dim`/`fun_dim`/
    `out_dim` instead. `data_channels`/`out_channels` are kept only for
    interface parity with `ModelConfig` (`data_channels` mirrors `fun_dim`,
    or `space_dim` when `fun_dim` is 0; `out_channels` mirrors `out_dim`).
    `get_model_jax` detects (via the target model class's dataclass fields)
    that Transolver has no `in_channels`/`out_channels` and drops
    `data_channels`/`out_channels` instead of injecting them, so
    Transolver's own fields pass through untouched -- Transolver can be built
    through `get_model_jax()` like any other registered model.

    The structured-mesh sizes are config keys `height`/`width`/`depth`;
    `get_model_jax` forwards them to the model's `H`/`W`/`D` fields
    (the model's constructor names are kept so existing checkpoints load).
    """

    model_arch: str = "transolver"
    data_channels: int = 1
    out_channels: int = 1
    space_dim: int = 2
    fun_dim: int = 1
    out_dim: int = 1
    n_layers: int = 3
    n_hidden: int = 64
    # 0.0 matches the reference Transolver's own defaults (its ShapeNetCar
    # main.py never enables dropout). The wiring exists (every
    # TransolverBlock/MLP/attention Dropout layer reads this), so
    # setting it > 0 works for callers that also pass deterministic=False
    # with a dropout rng.
    dropout_rate: float = 0.0
    n_head: int = 4
    time_input: bool = False
    mlp_ratio: int = 1
    slice_num: int = 32
    ref: int = 8
    unified_pos: bool = False
    geometry: str = "irregular"  # "irregular" | "structured_2d" | "structured_3d"
    concat_pos: bool = False
    grid_range: Any | None = None
    height: int | None = None
    width: int | None = None
    depth: int | None = None
    kernel: int = 3


class TransolverShapeNetCar(TransolverConfig):
    """Irregular Mesh preset matching `main.py`'s hardcoded `Model(...)`
    call in the Transolver paper's own Car-Design-ShapeNetCar benchmark
    (`space_dim=7, fun_dim=0, out_dim=4, unified_pos=0` -- the actually
    reported experiment does NOT use the unified positional grid embedding).

    `space_dim=7`: per-point feature vector is
    `[pos_x, pos_y, pos_z, sdf, normal_x, normal_y, normal_z]` (already
    built by `load_shapenet_car` -- see
    `src/scirex/data/datasets/shapenet_car.py`), passed as the
    model's `x` directly since `fun_dim=0`/`unified_pos=False`.
    `out_dim=4`: `[velo_x, velo_y, velo_z, press]`.
    """

    data_channels: int = 7
    out_channels: int = 4
    space_dim: int = 7
    fun_dim: int = 0
    out_dim: int = 4
    n_hidden: int = 256
    n_layers: int = 8
    n_head: int = 8
    mlp_ratio: int = 2
    slice_num: int = 32
    unified_pos: bool = False
    ref: int = 8
    geometry: str = "irregular"


class TransolverAirfRANS(TransolverConfig):
    """Irregular Mesh preset for the AirfRANS 2D airfoil RANS-CFD benchmark
    (SST turbulence closure), sized the same way as `TransolverShapeNetCar`
    above but for a 2-D point cloud instead of 3-D.

    `space_dim=7`: per-point feature vector is ``[pos_x, pos_y,
    implicit_distance, dir_x, dir_y, u_inf_x, u_inf_y]`` (already built by
    `load_airfrans` -- see
    `src/scirex/data/datasets/airfrans.py`), passed as the
    model's `x` directly since `fun_dim=0`/`unified_pos=False`. These
    defaults are meant to be overwritten at runtime by `load_airfrans`'s
    own `model_kwargs` -- kept in sync here only for
    documentation/standalone-construction purposes.
    `out_dim=4`: `[Ux, Uy, p, nut]`.

    NOTE: the official `thuml/Transolver` `Airfoil-Design-AirfRANS`
    recipe actually runs with `unified_pos=1`/`concat_pos=1` (an 8x8
    reference-grid positional embedding) -- deliberately left off here so
    this preset stays the plain/baseline architecture. The AirfRANS
    benchmark (`scripts/benchmarks/airfrans/`) sets its own architecture.
    """

    data_channels: int = 7
    out_channels: int = 4
    space_dim: int = 7
    fun_dim: int = 0
    out_dim: int = 4
    n_hidden: int = 256
    n_layers: int = 8
    n_head: int = 8
    mlp_ratio: int = 2
    slice_num: int = 32
    unified_pos: bool = False
    ref: int = 8
    geometry: str = "irregular"


class TransolverDarcy2dAligned(TransolverConfig):
    """Transolver preset for the 2D Darcy-flow benchmark (pure-data,
    supervised-only training -- no physics-loss term), aligned with the
    Transolver paper's own published Darcy recipe (`Transolver_Darcy.sh`
    and `exp_darcy.py` in the reference implementation) rather than this
    codebase's earlier, unaligned defaults.

    An earlier attempt with unaligned settings (`unified_pos=False`,
    `slice_num=32`, `n_layers=4`, no gradient clipping, reduce-on-plateau
    LR) reached only ~10.4% relative L2 error on this dataset, far short of
    the paper's own reported 0.57%. Comparing directly against the reference
    training script found several concrete gaps -- `unified_pos` disabled
    (raw (x,y) coordinates went straight into attention with no positional
    encoding), half the reference's `slice_num`/`n_layers`, no gradient
    clipping, and no LR schedule. With this preset's settings plus gradient
    clipping and a OneCycle LR schedule (both set in the training script,
    not here), a manufactured-solution sanity check first confirmed the
    optimization behavior itself was fixed (0.15% on a clean Poisson
    problem, matching a plain MLP baseline), and the real Zenodo Darcy
    dataset at resolution 64 (rather than this codebase's usual
    resolution-16 default) then reached 3.34% relative L2 -- both
    ingredients (recipe alignment and resolution) mattered; resolution 64
    was the larger factor of the two.

    `geometry="structured_2d"` (rather than `TransolverDarcy2d`'s
    original `"irregular"`) with `height=width=64` matches the reference's own
    grid-based positional handling; `fun_dim=1` carries the permeability
    field as a per-point feature (the point cloud itself is the same fixed
    grid every sample, so geometry alone carries no information -- only the
    permeability value at each point does); `out_dim=1` is the scalar
    pressure field.
    """

    data_channels: int = 1
    out_channels: int = 1
    space_dim: int = 2
    fun_dim: int = 1
    out_dim: int = 1
    n_hidden: int = 128
    n_layers: int = 8
    n_head: int = 8
    slice_num: int = 64
    unified_pos: bool = True
    geometry: str = "structured_2d"
    height: int = 64
    width: int = 64


class TransolverLaplace2d(TransolverConfig):
    """Transolver preset for fitting a closed-form analytical solution of
    Laplace's or Poisson's equation on a 2D domain, queried at a fixed set
    of points every step.

    `fun_dim=0`: there is no per-sample input field -- the model is a
    pure coordinates-to-solution map. `out_dim=1`: the scalar solution
    field `u`. Deliberately small (`n_hidden=64`, `n_layers=3`, the
    `TransolverConfig` base defaults) since this fits one fixed problem
    instance, not a dataset-scale operator-learning problem.
    """

    data_channels: int = 2
    out_channels: int = 1
    space_dim: int = 2
    fun_dim: int = 0
    out_dim: int = 1
    unified_pos: bool = True
    geometry: str = "irregular"


class TransolverKappaOperator2d(TransolverLaplace2d):
    """`TransolverLaplace2d`, but configured for a per-instance
    coefficient field passed alongside `[x, y]` as `fx` at call time
    (see `Transolver.__call__`'s `fx` argument) -- what actually turns
    Transolver from a per-instance solution-fitter into an operator:
    trained across a batch of sampled coefficient-field instances, a
    single model is then queried zero-shot on unseen coefficient fields
    just by changing `fx`, no retraining, unlike
    `TransolverLaplace2d`, which is never called with an `fx`
    and only ever fits one fixed problem instance from scratch.

    `fun_dim=1` and `data_channels=1` (mirroring it, per
    `TransolverConfig`'s own stated convention) record that intent for
    callers reading the config, but note the model itself does not read
    `fun_dim` at call time -- whether the extra channel is actually used
    is decided purely by whether `fx` is passed to `__call__`, not by
    this config value. Passing `fx=None` here would silently fall back
    to the non-operator, coordinates-only behavior despite this config
    still claiming `fun_dim=1`.
    """

    fun_dim: int = 1
    data_channels: int = 1


class TransolverPipe(TransolverConfig):
    """Structured Mesh 2D preset matching `exp_pipe.py`'s argparse
    defaults in the Transolver paper's own PDE-Solving-StandardBenchmark
    Pipe benchmark
    (https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/exp_pipe.py),
    2D Navier-Stokes pipe flow on a structured 129x129 mesh.

    `space_dim=2`: per-point feature is the raw `(pos_x, pos_y)`
    coordinate pair (`fun_dim=0`, `unified_pos=0` in the reference CLI --
    passed as the model's `x` directly, `fx=None`). `out_dim=1`: the
    scalar output field (`Pipe_Q.npy`'s channel 0). `height`/`width`
    default to 129 (`s1`/`s2` in `exp_pipe.py` with the default
    `--downsamplex`/`--downsampley` of 1, i.e. no downsampling of the
    raw 129x129 mesh) -- overridden per-run if downsampling is requested,
    see `scripts/benchmarks/pipe/_transolver.py`.

    Every other field below is `exp_pipe.py`'s own **argparse default**:
    ``--n-hidden 64 --n-layers 3 --n-heads 4 --mlp_ratio 1 --slice_num 32
    --ref 8 --dropout 0.0 --unified_pos 0``. These already happen to equal
    `TransolverConfig`'s own base defaults (n_hidden=64, n_layers=3,
    n_head=4, mlp_ratio=1, slice_num=32, ref=8) -- kept explicit here rather
    than relying on that coincidence, so this preset stays correct even if
    the base class's defaults ever change for an unrelated benchmark.

    Note this is the **argparse-default** configuration, not the one that
    produced the paper's own published Pipe number -- see
    `TransolverPipeOfficial` for that, and
    `doc/benchmarks/pipe/results/transolver.md` for the
    full story of the gap between them.
    """

    data_channels: int = 2
    out_channels: int = 1
    space_dim: int = 2
    fun_dim: int = 0
    out_dim: int = 1
    n_hidden: int = 64
    n_layers: int = 3
    n_head: int = 4
    mlp_ratio: int = 1
    slice_num: int = 32
    ref: int = 8
    dropout_rate: float = 0.0
    unified_pos: bool = False
    geometry: str = "structured_2d"
    height: int = 129
    width: int = 129


class TransolverPipeOfficial(TransolverPipe):
    """`TransolverPipe`, but with the hyperparameters the Transolver
    paper's reference repo ACTUALLY used to produce its published Pipe
    result -- its own launch script,
    `Transolver_Pipe.sh`, overrides nearly every architecture flag
    away from `exp_pipe.py`'s argparse defaults:

    ``--n-hidden 64->128 --n-layers 3->8 --n-heads 4->8 --mlp_ratio 1->2
    --slice_num 32->64``, plus gradient clipping at `--max_grad_norm 0.1`
    (`exp_pipe.py`'s own default: no clipping -- set via
    `--max_grad_norm` on `scripts/benchmarks/pipe/_transolver.py`,
    not a model-config field). `TransolverPipe`'s own docstring
    documents only the argparse defaults, which is a substantially smaller
    model (272,621 params here vs. 3,073,985 with this preset -- an 11.3x
    difference) and was the single largest driver of a 2.1x test-error gap
    against the paper's own reported number, closed to ~1.4-1.5x by matching
    these hyperparameters alone. Full investigation, seed-variance
    measurements (JAX and PyTorch), and a cross-check against the paper's
    own unmodified PyTorch code:
    `doc/benchmarks/pipe/results/transolver.md`.
    """

    n_hidden: int = 128
    n_layers: int = 8
    n_head: int = 8
    mlp_ratio: int = 2
    slice_num: int = 64
