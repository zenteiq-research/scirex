#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
"""Build the sdist and wheel and check that they are fit to publish. Never uploads.

Steps
-----
1. ``python -m build`` into a folder outside the repository (or ``--dist DIR`` to check
   files that are already built);
2. ``python -m twine check`` on both files;
3. metadata of the wheel: name, version equal to ``pyproject.toml``, summary, README
   (markdown, no relative links), SPDX licence expression, licence files, authors,
   project URLs, classifiers, ``Requires-Python``, dependencies and every extra;
4. contents of both files are listed (``--list``) and checked: the legal files and
   ``py.typed`` are present, and no checkpoint, dataset, notebook, ``.env`` file,
   bytecode, or oversized file is shipped, and no text file contains a personal path,
   internal host or secret (the patterns of ``check_leaks.py``);
5. the sdist is installed once into a scratch folder with ``--no-deps`` and every
   ``scirex`` module is imported from there, to catch files missing from the sdist.

Usage::

    python scripts/dev/check_package.py [--dist DIR] [--list] [--no-install]
        [--runtime-python PYTHON] [--strict]

``--runtime-python`` is the interpreter (Python 3.10 to 3.12, with JAX, Flax, ... and pip)
for step 5; the default is the one running this script. Exit status 1 with
``--strict`` when a problem is found.
"""

from __future__ import annotations

import argparse
import email
import importlib.util
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_leaks", ROOT / "scripts" / "dev" / "check_leaks.py"
)
leaks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaks)

MAX_FILE_BYTES = 1_000_000
MAX_ARCHIVE_BYTES = 5_000_000
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy", ".h5", ".hdf5",
    ".pkl", ".pickle", ".ipynb", ".pyc", ".pyo", ".csv", ".gif", ".png", ".mp4",
}  # fmt: skip
# Top-level folders that must not be shipped (scirex/data is a source package, fine).
FORBIDDEN_TOP = {"checkpoints", "data", "doc", "docs", ".git", ".github", "docker"}
REQUIRED_URLS = ("Homepage", "Documentation", "Repository", "Issues", "Changelog")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".cff", ".txt", ".ini", ".json", ".yml"}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def pyproject_text(root: Path) -> str:
    return (root / "pyproject.toml").read_text(encoding="utf-8")


def declared_version(root: Path) -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text(root), re.M)
    return m.group(1) if m else ""


def declared_extras(root: Path) -> list[str]:
    block = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)",
        pyproject_text(root),
        re.M | re.S,
    )
    return re.findall(r"^([A-Za-z0-9_-]+)\s*=", block.group(1), re.M) if block else []


def build(root: Path, out: Path) -> list[str]:
    done = run([sys.executable, "-m", "build", "--outdir", str(out), str(root)])
    if done.returncode != 0:
        return [f"python -m build failed:\n{done.stdout[-800:]}{done.stderr[-800:]}"]
    return []


def find_files(dist: Path) -> tuple[Path | None, Path | None]:
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    return (sdists[-1] if sdists else None, wheels[-1] if wheels else None)


def twine_check(files: list[Path]) -> list[str]:
    done = run([sys.executable, "-m", "twine", "check", "--strict", *map(str, files)])
    if done.returncode != 0:
        return [f"twine check failed:\n{done.stdout[-800:]}{done.stderr[-400:]}"]
    return []


def wheel_metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_string(z.read(name).decode("utf-8"))


def check_metadata(meta: email.message.Message, root: Path) -> list[str]:
    problems: list[str] = []

    def need(ok: bool, what: str) -> None:
        if not ok:
            problems.append(f"metadata: {what}")

    need(meta["Name"] == "scirex", f"Name is {meta['Name']!r}")
    need(
        meta["Version"] == declared_version(root),
        f"Version {meta['Version']!r} differs from pyproject.toml",
    )
    need(bool(meta["Summary"]), "no Summary (description)")
    need(
        meta["Description-Content-Type"] == "text/markdown",
        "README content type is not text/markdown",
    )
    body = meta.get_payload() or ""
    need(len(body) > 500, "README (long description) is missing or short")
    relative = re.findall(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)", body)
    need(not relative, f"README has relative links: {sorted(set(relative))[:3]}")
    need(
        meta["License-Expression"] == "LGPL-3.0-only",
        f"License-Expression is {meta['License-Expression']!r}",
    )
    files = meta.get_all("License-File") or []
    for lic in ("LICENSE", "COPYING"):
        need(any(f.endswith(lic) for f in files), f"License-File {lic} missing")
    need(
        bool(meta["Author-email"] or meta["Author"])
        and bool(meta["Maintainer-email"] or meta["Maintainer"]),
        "authors or maintainers missing",
    )
    urls = " ".join(meta.get_all("Project-URL") or [])
    for label in REQUIRED_URLS:
        need(f"{label}," in urls, f"Project-URL {label} missing")
    classifiers = meta.get_all("Classifier") or []
    need(
        any(c.startswith("License :: OSI Approved") for c in classifiers),
        "no licence classifier",
    )
    need(
        any(c.startswith("Programming Language :: Python :: 3.") for c in classifiers),
        "no Python version classifiers",
    )
    need(bool(meta["Requires-Python"]), "Requires-Python missing")
    requires = meta.get_all("Requires-Dist") or []
    need(any("extra ==" not in r for r in requires), "no runtime dependencies")
    provided = set(meta.get_all("Provides-Extra") or [])
    for extra in declared_extras(root):
        need(extra in provided, f"extra {extra!r} missing from the wheel metadata")
    return problems


def members(archive: Path) -> list[tuple[str, int, bytes | None]]:
    """``(name, size, text or None)`` for every file in a wheel or sdist."""
    out: list[tuple[str, int, bytes | None]] = []
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                suffix = Path(info.filename).suffix.lower()
                data = z.read(info) if suffix in TEXT_SUFFIXES else None
                out.append((info.filename, info.file_size, data))
    else:
        with tarfile.open(archive) as t:
            for info in t.getmembers():
                if not info.isfile():
                    continue
                suffix = Path(info.name).suffix.lower()
                data = t.extractfile(info).read() if suffix in TEXT_SUFFIXES else None
                out.append((info.name.split("/", 1)[-1], info.size, data))
    return out


def check_contents(archive: Path, listed: list[tuple[str, int, bytes | None]]):
    problems: list[str] = []
    label = archive.name
    names = [n for n, _, _ in listed]
    required = (
        ["src/scirex/__init__.py", "src/scirex/py.typed", "LICENSE", "COPYING"]
        + ["NOTICE", "README.md", "pyproject.toml"]
        if archive.suffix != ".whl"
        else ["scirex/__init__.py", "scirex/py.typed"]
    )
    for need in required:
        if need not in names:
            problems.append(f"{label}: {need} is missing")
    if archive.suffix == ".whl" and not any("licenses/LICENSE" in n for n in names):
        problems.append(f"{label}: licence files are not in dist-info/licenses")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        problems.append(f"{label}: {archive.stat().st_size} bytes, over the limit")
    for name, size, data in listed:
        path = Path(name)
        base = path.name
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{label}: {name}: file type must not be shipped")
        if base == ".env" or base.startswith(".env."):
            problems.append(f"{label}: {name}: environment file")
        if path.parts[0] in FORBIDDEN_TOP or "__pycache__" in path.parts:
            problems.append(f"{label}: {name}: folder must not be shipped")
        if size > MAX_FILE_BYTES:
            problems.append(f"{label}: {name}: {size} bytes, over the file limit")
        if data is not None and name != "pyproject.toml":
            for no, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                for category, hit in leaks.scan_line(line):
                    problems.append(f"{label}: {name}:{no}: {category}: {hit}")
    return problems


IMPORT_ALL = """
import importlib, pkgutil, sys
import scirex
bad = []
for mod in pkgutil.walk_packages(scirex.__path__, "scirex."):
    try:
        importlib.import_module(mod.name)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "scirex":
            bad.append(f"{mod.name}: {e}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"{mod.name}: {type(e).__name__}: {e}")
print("VERSION", scirex.__version__)
print("FILE", scirex.__file__)
for b in bad:
    print("BAD", b)
"""


def install_check(sdist: Path, runtime: str, root: Path) -> list[str]:
    """Install the sdist alone, then import every module from the install."""
    with tempfile.TemporaryDirectory(prefix="scirex-sdist-") as tmp:
        target = Path(tmp) / "site"
        done = run(
            [runtime, "-m", "pip", "install", "--no-deps", "--quiet"]
            + ["--target", str(target), str(sdist)]
        )
        if done.returncode != 0:
            return [f"install from sdist failed:\n{done.stderr[-800:]}"]
        code = f"import sys; sys.path.insert(0, {str(target)!r})\n" + IMPORT_ALL
        probe = run([runtime, "-c", code], cwd=tmp)
    if probe.returncode != 0:
        return [f"import from the sdist install failed:\n{probe.stderr[-800:]}"]
    problems = []
    version = re.search(r"^VERSION (.+)$", probe.stdout, re.M)
    file = re.search(r"^FILE (.+)$", probe.stdout, re.M)
    if not version or version.group(1) != declared_version(root):
        problems.append(f"installed version {version and version.group(1)!r} differs")
    if not file or "scirex-sdist-" not in file.group(1):
        problems.append(
            "scirex was imported from somewhere other than the sdist install"
        )
    problems += [
        f"sdist install: {b}" for b in re.findall(r"^BAD (.+)$", probe.stdout, re.M)
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--dist", type=Path, help="check these files instead of building")
    ap.add_argument("--list", action="store_true", help="print the contents")
    ap.add_argument("--no-install", action="store_true", help="skip step 5")
    ap.add_argument("--runtime-python", default=sys.executable)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any problem")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    problems: list[str] = []
    scratch = None
    dist = args.dist
    if dist is None:
        scratch = tempfile.TemporaryDirectory(prefix="scirex-dist-")
        dist = Path(scratch.name)
        problems += build(root, dist)
    sdist, wheel = find_files(dist)
    if not sdist or not wheel:
        problems.append(f"no sdist and wheel in {dist}")
    else:
        problems += twine_check([sdist, wheel])
        problems += check_metadata(wheel_metadata(wheel), root)
        for archive in (sdist, wheel):
            listed = members(archive)
            if args.list:
                print(f"--- {archive.name}: {len(listed)} files")
                for name, size, _ in listed:
                    print(f"{size:>9d}  {name}")
            problems += check_contents(archive, listed)
        if not args.no_install:
            problems += install_check(sdist, args.runtime_python, root)
    for p in problems:
        print("PROBLEM", p)
    print(f"check_package: {len(problems)} problem(s); nothing was uploaded")
    if scratch:
        scratch.cleanup()
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
