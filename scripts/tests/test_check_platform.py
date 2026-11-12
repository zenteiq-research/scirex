# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The platform check script is valid, documented, and its workflow runs end to end."""

import os
import subprocess
import sys
from pathlib import Path


def _cpu_env() -> dict[str, str]:
    """Select the CPU even when the suite runs with a GPU requested (verify_all.sh --gpu)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("JAX_PLATFORM_NAME", None)
    return env


ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "scripts" / "dev" / "check_platform.sh"
PY = ROOT / "scripts" / "dev" / "check_platform.py"


def test_script_is_valid_bash_and_executable() -> None:
    assert subprocess.run(["bash", "-n", str(SH)], check=False).returncode == 0
    assert SH.stat().st_mode & 0o111


def test_help_lists_the_options() -> None:
    done = subprocess.run(
        ["bash", str(SH), "--help"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0
    for option in (
        "--backend",
        "--cpu",
        "--only",
        "--steps",
        "--python",
        "--keep",
        "--light",
    ):
        assert option in done.stdout


def test_unknown_model_is_refused() -> None:
    done = subprocess.run(
        [sys.executable, str(PY), "--only", "nonsense", "--skip-imports"],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 2
    assert "unknown model" in done.stdout


def test_transolver_workflow_runs_end_to_end() -> None:
    done = subprocess.run(
        [
            "bash",
            str(SH),
            "--cpu",
            "--light",
            "--only",
            "transolver",
            "--steps",
            "2",
            "--skip-imports",
            "--python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
        env=_cpu_env(),
    )
    out = done.stdout
    assert done.returncode == 0, out[-2000:]
    for phrase in (
        "detected:",
        "training step 2/2",
        "saving checkpoint",
        "loading the checkpoint",
        "running inference",
        "RESULT: OK",
    ):
        assert phrase in out


def test_unknown_backend_is_refused() -> None:
    done = subprocess.run(
        [sys.executable, str(PY), "--backend", "quantum"],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 2


def test_wrong_python_version_is_a_failure() -> None:
    done = subprocess.run(
        [sys.executable, str(PY), "--python-version", "2.7", "--skip-imports"],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 1
    assert "requested 2.7" in done.stdout


def _load_platform_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_platform_under_test", PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_libusb_failure_of_open3d_gets_an_install_hint() -> None:
    module = _load_platform_module()
    message = (
        "dlopen(/x/.venv/lib/python3.11/site-packages/open3d/pybind.so): Library not loaded: "
        "/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib"
    )
    hint = module.library_hint("open3d", message)
    assert "open3d>=0.19,<0.20" in hint
    assert "apt-get install libusb-1.0-0" in hint
    assert "open3d" in module.OPTIONAL_PACKAGES
