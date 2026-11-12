# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The install pages cover every install path and their commands are real."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "doc" / "install"
PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
MKDOCS = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

# page, H1 title, H2 sections it must have
PAGES = {
    "index.md": (
        "Install ScireX",
        [
            "System requirements",
            "Install with pip",
            "NVIDIA GPU",
            "TPU",
            "Environment variables",
            "Other ways to install",
        ],
    ),
    "docker.md": (
        "Use Docker",
        ["Requirements", "Build the images", "Run", "Docker Compose"],
    ),
    "source.md": (
        "Build from source",
        ["Get the code", "Install", "Exact tested versions"],
    ),
    "verify.md": ("Verify the install", []),
    "uninstall.md": ("Uninstall", []),
    "troubleshooting.md": ("Troubleshooting", []),
}
TROUBLESHOOTING = [
    "`LD_LIBRARY_PATH` override",
    "Old driver",
    "Wrong CUDA wheel",
    "x64 flag",
    "WSL2",
    "macOS: no GPU",
]


def text(page: str) -> str:
    return (INSTALL / page).read_text(encoding="utf-8")


def headings(page: str, level: str) -> list[str]:
    no_code = re.sub(r"```.*?```", "", text(page), flags=re.S)
    return re.findall(rf"^{level} (.+)$", no_code, re.M)


ALL = "\n".join(text(p) for p in PAGES)


@pytest.mark.parametrize("page", PAGES)
def test_page_has_its_title_and_sections(page) -> None:
    title, sections = PAGES[page]
    assert headings(page, "#") == [title]
    for section in sections:
        assert any(section in h for h in headings(page, "##")), (page, section)


def test_pages_are_in_the_nav_in_order() -> None:
    nav = MKDOCS[MKDOCS.index("  - Install:") :]
    listed = re.findall(r"install/([\w.]+\.md)", nav.split("\n  - Learn:")[0])
    # the install logs are records of a run, not guide pages
    listed = [p for p in listed if not p.startswith("logs")]
    assert listed == list(PAGES)


@pytest.mark.parametrize("title", TROUBLESHOOTING)
def test_troubleshooting_topics(title) -> None:
    assert any(title in h for h in headings("troubleshooting.md", "##")), title


def test_every_extra_named_exists_in_pyproject() -> None:
    names = set(re.findall(r"scirex\[([a-z0-9_,-]+)\]", ALL))
    names |= set(re.findall(r'-e "\.\[([a-z0-9_,-]+)\]"', ALL))
    assert {"cuda12", "tpu", "dev", "all"} <= names
    for name in names:
        for extra in name.split(","):
            assert re.search(rf"^{extra}\s*=", PYPROJECT, re.M), f"extra {extra!r}"


def test_cuda13_is_stated_as_unsupported_because_of_the_jax_range() -> None:
    section = text("index.md").split("CUDA 13 wheels")[1].split("\n## ")[0]
    assert (
        "not supported" in section.lower() or "outside the supported range" in section
    )
    assert re.search(r'"jax>=0\.6[^"]*<0\.7"', PYPROJECT)


def test_scripts_named_in_the_pages_exist() -> None:
    for path in re.findall(r"python (scripts/[\w/.-]+\.py)", ALL):
        assert (ROOT / path).is_file(), path


def test_docker_commands_use_real_targets() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    for target in re.findall(r"--target (\w+)", ALL):
        assert f"AS {target}" in dockerfile


def test_internal_links_resolve() -> None:
    for page in PAGES:
        for target in re.findall(r"\]\(([\w.-]+\.md)(?:#[\w-]+)?\)", text(page)):
            assert (INSTALL / target).is_file(), (page, target)


def test_windows_steps_are_in_the_windows_tab() -> None:
    page = text("index.md")
    tab = page.split('=== "Windows"', 1)[1].split("\n**Linux install log", 1)[0]
    for step in [
        "**1. Enable long paths.**",
        "LongPathsEnabled",
        "**2. Install Python.**",
        "**3. Create a virtual environment.**",
        "**4. Install.**",
        "**5. Check.**",
        "**What does not work on Windows**",
    ]:
        assert step in tab, step
    assert not (INSTALL / "windows.md").exists()


def test_the_old_single_page_is_gone() -> None:
    assert not (ROOT / "doc" / "user_guide" / "installation.md").exists()


# --- scripts/dev/verify_install.py ---------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "verify_install", ROOT / "scripts" / "dev" / "verify_install.py"
)
vi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vi)


def _cpu_env() -> dict[str, str]:
    """An environment that selects the CPU, even when the suite runs with a GPU requested."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("JAX_PLATFORM_NAME", None)
    return env


def test_verify_install_passes_on_cpu() -> None:
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dev" / "verify_install.py"),
            "--require",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify_install: ok" in done.stdout


def test_verify_install_fails_without_the_required_device() -> None:
    # On the CPU no TPU exists, so requiring one must fail and say so.
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "dev" / "verify_install.py"),
            "--require",
            "tpu",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 1
    assert "FAIL" in done.stdout


def test_verify_install_checks_float64() -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dev" / "verify_install.py"), "--x64"],
        capture_output=True,
        text=True,
        check=False,
        env=_cpu_env(),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "x64 True" in done.stdout
