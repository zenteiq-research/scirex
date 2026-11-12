#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""End-to-end check that ScireX works on this operating system.

It is run by ``scripts/dev/check_platform.sh`` and prints what it does as it goes.

1. Platform: operating system, CPU architecture, Python and library versions, the JAX
   devices, and whether this combination is supported.
2. Imports: every module of ``scirex`` is imported; a missing optional package is a
   warning, anything else is a failure.
3. Public API: every name a stable package exports is looked up and checked to be a
   class or a callable.
4. Workflow, for Transolver, FNO, GINO and WNO: dummy data, a few training steps, save
   a checkpoint, load it into a new model, and run inference; the loaded model must give
   the same output as the trained one.

Usage::

    python scripts/dev/check_platform.py [--only fno,wno,gino,transolver]
        [--steps 2] [--backend auto|cpu|gpu|tpu] [--keep DIR] [--skip-imports]

Exit status 1 when a check fails. Nothing is written outside a temporary folder (or the
folder given with ``--keep``).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pkgutil
import platform
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path

SUPPORTED_PYTHON = ((3, 10), (3, 12))
STABLE_PACKAGES = (
    "scirex",
    "scirex.models",
    "scirex.layers",
    "scirex.losses",
    "scirex.data",
    "scirex.training",
    "scirex.config",
    "scirex.utils",
)
# A module that fails to import only because one of these is missing is a warning.
OPTIONAL_PACKAGES = {
    "wandb",
    "open3d",
    "optuna",
    "mlflow",
    "meshio",
    "gmsh",
    "vtk",
    "pandas",
    "docx",
    "pyvista",
    "torch",
    "torch_geometric",
}

RESULTS: list[tuple[str, str, str]] = []  # (status, name, detail)
REQUESTED_PYTHON = ""  # "3.11" and so on: what --python-version asked for
REQUESTED = "auto"  # "auto", "cpu", "gpu" or "tpu": what --backend asked for
JAX_PLATFORM_NAMES = {"cpu": "cpu", "gpu": "cuda", "tpu": "tpu"}


def normalise_platform(name: str) -> str:
    """JAX calls an NVIDIA or AMD GPU "cuda", "rocm" or "gpu"; call all of them "gpu"."""
    return "gpu" if name in {"cuda", "rocm", "gpu"} else name


def platform_of(tree) -> set[str]:
    """The kinds of device on which the arrays of a pytree live."""
    import jax

    kinds: set[str] = set()
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "devices"):
            kinds |= {normalise_platform(d.platform) for d in leaf.devices()}
    return kinds


def backend_hint(requested: str) -> str:
    if requested == "gpu":
        found = shutil.which("nvidia-smi")
        return (
            "      hint: JAX has no GPU. "
            + (
                "nvidia-smi is present; "
                if found
                else "nvidia-smi was not found (no NVIDIA driver); "
            )
            + 'install the CUDA wheel: pip install "scirex[cuda12]", and check that no CPU-only '
            "jaxlib is installed. See doc/install/troubleshooting.md."
        )
    if requested == "tpu":
        return '      hint: JAX has no TPU. Run on a TPU VM and install: pip install "scirex[tpu]".'
    return "      hint: JAX could not start the CPU backend."


def say(text: str = "") -> None:
    print(text, flush=True)


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    say(f"      -> {status}  {name}" + (f"  ({detail})" if detail else ""))


def section(number: int, title: str) -> None:
    say()
    say(f"=== [{number}] {title} " + "=" * max(3, 66 - len(title)))


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


# ---------------------------------------------------------------------------
# 1. Platform
# ---------------------------------------------------------------------------
def check_platform() -> None:
    section(1, "Platform and compatibility")
    system = platform.system()
    machine = platform.machine()
    say(f"  operating system : {system} {platform.release()} ({platform.platform()})")
    say(f"  CPU architecture : {machine}")
    say(
        f"  Python           : {sys.version.split()[0]} ({platform.python_implementation()}, "
        f"{'64' if sys.maxsize > 2**32 else '32'}-bit) at {sys.executable}"
    )
    in_wsl = "microsoft" in platform.release().lower()
    if in_wsl:
        say("  environment      : Windows Subsystem for Linux (WSL2)")

    low, high = SUPPORTED_PYTHON
    here = sys.version_info[:2]
    if low <= here <= high:
        record(
            "PASS",
            "python version",
            f"{here[0]}.{here[1]} is in the supported range 3.10 to 3.12",
        )
    else:
        record(
            "FAIL",
            "python version",
            f"{here[0]}.{here[1]} is outside the supported range 3.10 to 3.12",
        )
    if REQUESTED_PYTHON:
        want = tuple(int(part) for part in REQUESTED_PYTHON.split(".")[:2])
        if here == want:
            record("PASS", "requested python", f"running Python {REQUESTED_PYTHON}")
        else:
            record(
                "FAIL",
                "requested python",
                f"requested {REQUESTED_PYTHON} but running {here[0]}.{here[1]}",
            )
    if sys.maxsize <= 2**32:
        record("FAIL", "64-bit python", "JAX needs a 64-bit Python")

    if system == "Linux":
        record(
            "PASS",
            "operating system",
            f"Linux {machine}: CPU and NVIDIA GPU (CUDA 12) are supported",
        )
    elif system == "Darwin":
        note = "Apple Silicon" if machine == "arm64" else "Intel Mac"
        record(
            "WARN",
            "operating system",
            f"macOS, {note}: CPU only. There is no supported GPU path on a Mac",
        )
    elif system == "Windows":
        record(
            "WARN",
            "operating system",
            "native Windows: CPU only. For an NVIDIA GPU use WSL2 (Linux)",
        )
    else:
        record("WARN", "operating system", f"{system} is not one of the tested systems")

    say("  libraries        :")
    from importlib import metadata

    for name in (
        "scirex",
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "numpy",
        "scipy",
        "jaxwt",
    ):
        try:
            say(f"    {name:16s} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            say(f"    {name:16s} not installed")
            if name != "scirex":
                record(
                    "FAIL", f"dependency {name}", "not installed; run: pip install -e ."
                )
    try:
        import jax

        devices = jax.devices()
        say(f"  JAX backend      : {jax.default_backend()}")
        say(f"  JAX devices      : {devices}")
        record(
            "PASS",
            "jax runs",
            f"backend {jax.default_backend()}, {len(devices)} device(s)",
        )
        actual = normalise_platform(jax.default_backend())
        if REQUESTED == "auto":
            record("PASS", "backend", f"auto: JAX chose {actual} ({devices[0]})")
            if actual == "cpu":
                say(
                    "      note: JAX is on the CPU. To test a GPU or a TPU use --backend gpu or --backend tpu."
                )
        elif actual == REQUESTED:
            record("PASS", "backend", f"requested {REQUESTED}, running on {devices[0]}")
        else:
            record(
                "FAIL",
                "backend",
                f"requested {REQUESTED} but JAX is running on {actual}",
            )
            say(backend_hint(REQUESTED))
        value = float(jax.jit(lambda x: (x * x).sum())(jax.numpy.arange(4.0)))
        if value == 14.0:
            record("PASS", "jit computation", "a jitted sum of squares gave 14.0")
        else:
            record("FAIL", "jit computation", f"expected 14.0, got {value}")
    except Exception as exc:  # - report any failure of the platform
        text = str(exc).strip()
        if text:
            reason = text.splitlines()[0]
        elif REQUESTED != "auto":
            reason = f"JAX could not initialise the {REQUESTED} backend (no usable device or plugin)"
        else:
            reason = "no message"
        record("FAIL", "jax runs", f"{type(exc).__name__}: {reason}")
        if REQUESTED != "auto":
            record(
                "FAIL", "backend", f"requested {REQUESTED}, but JAX could not start it"
            )
            say(backend_hint(REQUESTED))


# ---------------------------------------------------------------------------
# 2. Imports
# ---------------------------------------------------------------------------
def library_hint(package: str, message: str) -> str:
    """How to fix an optional package whose native library does not load."""
    if "libusb" in message:
        return (
            "Its wheel links libusb. On macOS run `pip install 'open3d>=0.19,<0.20'`, which needs no system library; "
            "on Debian and Ubuntu run `sudo apt-get install libusb-1.0-0`. See doc/install/troubleshooting.md."
        )
    return f"Reinstall {package}, or see doc/install/troubleshooting.md."


def check_imports() -> None:
    section(2, "Import every module")
    import scirex

    say(f"  scirex {scirex.__version__} from {Path(scirex.__file__).parent}")
    ok = 0
    optional: list[str] = []
    broken_optional: dict[str, str] = {}
    failed: list[str] = []
    names = [m.name for m in pkgutil.walk_packages(scirex.__path__, "scirex.")]
    names.insert(0, "scirex")
    say(f"  found {len(names)} modules; importing each one")
    for name in names:
        if name.endswith("__main__"):
            continue
        try:
            importlib.import_module(name)
            ok += 1
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing in OPTIONAL_PACKAGES:
                optional.append(f"{name} (needs {missing})")
                say(f"    optional  {name}: needs {missing}")
            else:
                failed.append(f"{name}: {exc}")
                say(f"    FAILED    {name}: {exc}")
        except Exception as exc:
            text = str(exc)
            broken = next(
                (
                    pkg
                    for pkg in sorted(OPTIONAL_PACKAGES)
                    if isinstance(exc, ImportError)
                    and f"site-packages/{pkg}/" in text.replace("\\", "/")
                ),
                None,
            )
            if broken:
                # An optional package is installed but its native library does not load.
                optional.append(f"{name} ({broken} is installed but does not load)")
                broken_optional.setdefault(broken, text)
                say(f"    optional  {name}: {broken} is installed but cannot be loaded")
            else:
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
                say(f"    FAILED    {name}: {type(exc).__name__}: {exc}")
    record(
        "PASS" if not failed else "FAIL",
        "import all modules",
        f"{ok} imported, {len(optional)} need or cannot load an optional package, {len(failed)} failed",
    )
    for pkg, text in broken_optional.items():
        record(
            "WARN",
            f"optional package {pkg}",
            f"installed but cannot load. {library_hint(pkg, text)}",
        )
    if optional:
        record(
            "WARN",
            "optional packages",
            f"{len(optional)} module(s) skipped; install the extras to test them: pip install -e '.[all]'",
        )


# ---------------------------------------------------------------------------
# 3. Public API
# ---------------------------------------------------------------------------
def check_public_api() -> None:
    section(3, "Public classes and functions")
    total = classes = functions = 0
    problems: list[str] = []
    for package in STABLE_PACKAGES:
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            problems.append(f"{package}: {exc}")
            continue
        names = list(getattr(module, "__all__", []))
        say(f"  {package}: {len(names)} exported names")
        for name in names:
            total += 1
            try:
                obj = getattr(module, name)
            except (
                Exception
            ) as exc:  # - lazy exports raise ImportError for a missing extra
                problems.append(f"{package}.{name}: {type(exc).__name__}: {exc}")
                continue
            if inspect.isclass(obj):
                classes += 1
            elif callable(obj):
                functions += 1
            elif name.startswith("__"):
                pass
            else:
                problems.append(
                    f"{package}.{name}: neither a class nor a callable ({type(obj).__name__})"
                )
    for line in problems:
        say(f"    PROBLEM {line}")
    record(
        "PASS" if not problems else "FAIL",
        "public API",
        f"{total} names looked up: {classes} classes, {functions} functions, {len(problems)} problems",
    )


# ---------------------------------------------------------------------------
# 4. Workflow per model
# ---------------------------------------------------------------------------
def _train_and_roundtrip(
    label: str,
    model,
    apply_inputs: dict,
    target,
    root: Path,
    steps: int,
    optimizer_name: str = "adam",
) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    from scirex.training import adamw_cx

    say(f"  [{label}] initialising parameters")
    (params, took) = timed(lambda: model.init(jax.random.PRNGKey(0), **apply_inputs))
    from scirex.utils import count_model_params

    say(
        f"  [{label}] {count_model_params(params):,} parameters, initialised in {took:.1f}s"
    )

    optimizer = adamw_cx(1e-3) if optimizer_name == "adamw_cx" else optax.adam(1e-3)
    say(f"  [{label}] optimizer: {optimizer_name}")
    state = optimizer.init(params)

    def loss_fn(p):
        out = model.apply(p, **apply_inputs)
        return jnp.mean((out - target) ** 2)

    # GINO searches neighbours with data-dependent shapes, so it runs without jit.
    value_and_grad = jax.value_and_grad(loss_fn)
    if label != "GINO":
        value_and_grad = jax.jit(value_and_grad)
    say(
        f"  [{label}] {'not jitted (data-dependent neighbour search)' if label == 'GINO' else 'jitted'}"
    )
    losses = []
    for step in range(1, steps + 1):
        (result, took) = timed(lambda: value_and_grad(params))
        loss, grads = result
        loss = float(loss)
        grad_norm = float(optax.global_norm(grads))
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        losses.append(loss)
        say(
            f"  [{label}] training step {step}/{steps}: loss {loss:.6f}, gradient norm {grad_norm:.4e} ({took:.1f}s)"
        )
        if not (jnp.isfinite(loss) and jnp.isfinite(grad_norm)):
            raise RuntimeError(f"non-finite loss or gradient at step {step}")
    final = float(loss_fn(params))
    say(
        f"  [{label}] loss after {steps} steps: {final:.6f} (first step {losses[0]:.6f})"
    )
    if not jnp.isfinite(final):
        raise RuntimeError("non-finite loss after training")

    folder = root / label.lower()
    say(f"  [{label}] saving checkpoint to {folder}")
    model.save_checkpoint(folder, "dummy", params)
    for path in sorted(folder.iterdir()):
        say(f"           {path.name}: {path.stat().st_size:,} bytes")

    say(f"  [{label}] loading the checkpoint into a new model")
    template = model.init(jax.random.PRNGKey(1), **apply_inputs)
    restored_model, restored = type(model).from_checkpoint(folder, "dummy", template)
    same = jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda a, b: bool(jnp.all(a == b)), params, restored)
    )
    say(f"  [{label}] restored parameters identical to the trained ones: {same}")
    if not same:
        raise RuntimeError("restored parameters differ from the saved ones")

    say(f"  [{label}] running inference with the loaded model")
    expected = model.apply(params, **apply_inputs)
    (got, took) = timed(lambda: restored_model.apply(restored, **apply_inputs))
    where = platform_of(params) | platform_of(restored) | platform_of(got)
    say(
        f"  [{label}] trained parameters, loaded parameters and output live on: "
        f"{', '.join(sorted(where)) or 'unknown'}"
    )
    if REQUESTED != "auto" and where != {REQUESTED}:
        raise RuntimeError(
            f"requested {REQUESTED} but the arrays are on {sorted(where)}"
        )
    diff = float(jnp.max(jnp.abs(jnp.asarray(got) - jnp.asarray(expected))))
    say(
        f"  [{label}] output shape {tuple(got.shape)}, max difference to the trained model {diff:.2e} ({took:.1f}s)"
    )
    if not diff <= 1e-5:
        raise RuntimeError(f"loaded model output differs by {diff:.2e}")


def _fno_case():
    import jax
    import jax.numpy as jnp

    from scirex.models import FNO

    model = FNO(
        n_modes=(6, 6), in_channels=1, out_channels=1, hidden_channels=8, n_layers=2
    )
    key = jax.random.PRNGKey(10)
    x = jax.random.normal(key, (4, 1, 16, 16))
    y = jnp.sin(x)
    say("  [FNO] dummy data: input (4, 1, 16, 16) channels-first, target = sin(input)")
    return model, {"x": x}, y, "adamw_cx"


def _wno_case():
    import jax
    import jax.numpy as jnp

    from scirex.models import WNO

    model = WNO(
        in_channels=2,
        out_channels=1,
        hidden_channels=8,
        n_layers=2,
        level=2,
        size=(16, 16),
        wavelet="db2",
    )
    x = jax.random.normal(jax.random.PRNGKey(11), (2, 16, 16, 2))
    y = jnp.sum(x, axis=-1, keepdims=True)
    say(
        "  [WNO] dummy data: input (2, 16, 16, 2) channels-last, target = sum of channels"
    )
    return model, {"x": x}, y, "adam"


def _gino_case():
    import jax
    import jax.numpy as jnp

    from scirex.models import GINO

    n_points, dim, density = 30, 3, 5
    model = GINO(
        in_channels=3,
        out_channels=1,
        in_gno_radius=0.5,
        out_gno_radius=0.5,
        projection_channel_ratio=4,
        gno_coord_dim=dim,
        in_gno_channel_mlp_hidden_layers=[8, 8],
        out_gno_channel_mlp_hidden_layers=[8, 8],
        fno_n_modes=(4, 4, 4),
        fno_hidden_channels=8,
        fno_n_layers=2,
        fno_lifting_channel_ratio=2,
        gno_use_open3d=False,
    )
    axes = [jnp.linspace(0, 1, density)] * dim
    mesh = jnp.stack(jnp.meshgrid(*axes, indexing="xy"))
    latent = jnp.transpose(mesh, (*range(1, dim + 1), 0))[None]
    points = jax.random.uniform(jax.random.PRNGKey(12), (1, n_points, dim))
    inputs = {
        "input_geom": points,
        "latent_queries": latent,
        "output_queries": points,
        "x": jax.random.normal(jax.random.PRNGKey(13), (2, n_points, 3)),
        "latent_features": None,
        "ada_in": None,
    }
    y = jnp.sum(inputs["x"], axis=-1, keepdims=True)
    say(
        f"  [GINO] dummy data: {n_points} random 3D points, 2 samples, latent grid {density}^3, target = sum of channels"
    )
    return model, inputs, y, "adam"


def _transolver_case():
    import jax
    import jax.numpy as jnp

    from scirex.models import Transolver

    model = Transolver(
        space_dim=2,
        fun_dim=1,
        out_dim=1,
        n_hidden=16,
        n_layers=2,
        n_head=2,
        slice_num=4,
        geometry="irregular",
    )
    x = jax.random.uniform(jax.random.PRNGKey(14), (2, 50, 2))
    fx = jnp.sin(3.0 * x[..., :1])
    y = fx**2
    say(
        "  [Transolver] dummy data: 2 samples of 50 random 2D points with 1 feature, target = feature squared"
    )
    return model, {"x": x, "fx": fx}, y, "adam"


CASES = {
    "transolver": ("Transolver", _transolver_case),
    "fno": ("FNO", _fno_case),
    "gino": ("GINO", _gino_case),
    "wno": ("WNO", _wno_case),
}


def check_workflows(only: list[str], steps: int, root: Path) -> None:
    section(4, f"End-to-end workflow ({steps} training steps, checkpoint, inference)")
    for key in only:
        label, build = CASES[key]
        say()
        say(f"  --- {label} " + "-" * max(3, 60 - len(label)))
        try:
            (case, took) = timed(build)
            model, inputs, target, optimizer = case  # type: ignore[misc]
            say(f"  [{label}] model built in {took:.1f}s")
            _train_and_roundtrip(label, model, inputs, target, root, steps, optimizer)
            record("PASS", f"{label} workflow", "train, save, load, infer")
        except ModuleNotFoundError as exc:
            record("WARN", f"{label} workflow", f"skipped: missing package {exc.name}")
        except Exception as exc:
            say(traceback.format_exc(limit=6))
            record("FAIL", f"{label} workflow", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--only",
        default="transolver,fno,gino,wno",
        help="comma-separated models to run",
    )
    parser.add_argument(
        "--steps", type=int, default=2, help="training steps per model (default 2)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "gpu", "tpu"],
        default="auto",
        help="device to test: auto (what JAX picks), cpu, gpu or tpu. A named backend must be "
        "available; otherwise the check fails instead of falling back to the CPU",
    )
    parser.add_argument("--cpu", action="store_true", help="same as --backend cpu")
    parser.add_argument(
        "--python-version",
        metavar="X.Y",
        default="",
        help="fail unless this script runs on Python X.Y (the shell wrapper also looks for that interpreter)",
    )
    parser.add_argument(
        "--keep", metavar="DIR", help="write the checkpoints to DIR and keep them"
    )
    parser.add_argument(
        "--skip-imports", action="store_true", help="skip the import and API checks"
    )
    args = parser.parse_args()

    global REQUESTED, REQUESTED_PYTHON
    REQUESTED_PYTHON = args.python_version
    REQUESTED = "cpu" if args.cpu else args.backend
    if REQUESTED != "auto":
        os.environ["JAX_PLATFORMS"] = JAX_PLATFORM_NAMES[REQUESTED]
    only = [name.strip().lower() for name in args.only.split(",") if name.strip()]
    unknown = [name for name in only if name not in CASES]
    if unknown:
        say(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(CASES)}")
        return 2

    say("ScireX platform check")
    check_platform()
    if not any(
        status == "FAIL" and name in {"jax runs", "python version", "requested python"}
        for status, name, _ in RESULTS
    ):
        if not args.skip_imports:
            check_imports()
            check_public_api()
        if args.keep:
            root = Path(args.keep)
            root.mkdir(parents=True, exist_ok=True)
            check_workflows(only, args.steps, root)
        else:
            tmp = Path(tempfile.mkdtemp(prefix="scirex-platform-"))
            try:
                check_workflows(only, args.steps, tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    else:
        say("\nThe platform check failed, so the other checks are skipped.")

    section(5, "Summary")
    width = max((len(n) for _, n, _ in RESULTS), default=10)
    for status, name, detail in RESULTS:
        say(f"  {status:4s}  {name:{width}s}  {detail}")
    failed = [r for r in RESULTS if r[0] == "FAIL"]
    warned = [r for r in RESULTS if r[0] == "WARN"]
    say()
    if failed:
        say(f"RESULT: FAILED ({len(failed)} failed, {len(warned)} warnings)")
        return 1
    say(
        f"RESULT: OK on {platform.system()} {platform.machine()} ({len(warned)} warnings)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
