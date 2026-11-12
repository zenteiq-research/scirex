# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Cross-cutting enforcement tests for the stable public API of ScireX.

The per-package tests (``tests/*/test_init.py`` and friends) pin each package on
its own. The tests here check what must hold across all of them: the surface
matches the committed snapshot, every public name is documented, nothing private
or experimental leaks into a stable package, every package has an API page, and
internal code does not use deprecated import paths.
"""

import ast
import inspect
import json
import re

import pytest

from tests.public_api_helpers import (
    PUBLIC_PACKAGES,
    REPO,
    SNAPSHOT_PATH,
    SRC,
    live_surface,
    public_items,
    resolve_public,
)

_PUBLIC_ITEMS = public_items()
_ITEM_IDS = [f"{package}.{name}" for package, name in _PUBLIC_ITEMS]


def _resolve_or_skip(package: str, name: str):
    """Resolve a public name, skipping when an optional dependency is missing."""
    try:
        return resolve_public(package, name)
    except ImportError as err:
        pytest.skip(f"{package}.{name} needs an optional dependency: {err}")


# --------------------------------------------------------------------------- #
# 1. Snapshot
# --------------------------------------------------------------------------- #
def _diff_surfaces(old: dict[str, list[str]], new: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for package in sorted(set(old) | set(new)):
        before, after = set(old.get(package, [])), set(new.get(package, []))
        if package not in old:
            lines.append(f"  {package}: package ADDED to the public API")
        if package not in new:
            lines.append(f"  {package}: package REMOVED from the public API")
        for name in sorted(after - before):
            lines.append(f"  {package}: + {name} (added)")
        for name in sorted(before - after):
            lines.append(f"  {package}: - {name} (removed)")
    return "\n".join(lines)


def test_public_surface_matches_the_committed_snapshot() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    live = live_surface()
    if snapshot != live:
        pytest.fail(
            "The public API differs from tests/api_snapshot.json "
            "(+ = in the code but not in the snapshot, - = the reverse):\n"
            f"{_diff_surfaces(snapshot, live)}\n\n"
            "If the change is intentional, regenerate the snapshot with\n"
            "    python scripts/dev/update_api_snapshot.py\n"
            "and commit it. Removing a name needs a deprecation cycle first.",
            pytrace=False,
        )


def test_snapshot_file_is_canonical() -> None:
    from tests.public_api_helpers import render_snapshot

    text = SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text == render_snapshot(json.loads(text)), (
        "tests/api_snapshot.json is not in canonical form; "
        "run python scripts/dev/update_api_snapshot.py"
    )


def test_public_packages_list_has_no_duplicates() -> None:
    assert len(PUBLIC_PACKAGES) == len(set(PUBLIC_PACKAGES))


# --------------------------------------------------------------------------- #
# 2. Docstrings
# --------------------------------------------------------------------------- #
def _has_doc(obj: object, name: str) -> bool:
    """True if ``obj`` has a hand-written docstring (generated ones don't count)."""
    doc = inspect.getdoc(obj) or ""
    return bool(doc) and not doc.startswith(f"{name}(")


def _public_methods(cls: type) -> list[tuple[str, object]]:
    """Public callables/properties defined in ``cls`` or its scirex base classes."""
    found: dict[str, object] = {}
    for klass in reversed(cls.__mro__):
        if not klass.__module__.startswith("scirex"):
            continue
        for attr, member in vars(klass).items():
            if attr.startswith("_"):
                continue
            if isinstance(member, (staticmethod, classmethod)):
                member = member.__func__
            if inspect.isfunction(member) or isinstance(member, property):
                found[attr] = member
    return sorted(found.items())


@pytest.mark.parametrize(("package", "name"), _PUBLIC_ITEMS, ids=_ITEM_IDS)
def test_public_name_has_a_docstring(package: str, name: str) -> None:
    obj = _resolve_or_skip(package, name)
    if not (inspect.isclass(obj) or callable(obj)):
        pytest.skip("not a class or callable")
    assert _has_doc(obj, name), f"{package}.{name} has no docstring"


@pytest.mark.parametrize(
    ("package", "name"),
    _PUBLIC_ITEMS,
    ids=_ITEM_IDS,
)
def test_public_methods_have_docstrings(package: str, name: str) -> None:
    obj = _resolve_or_skip(package, name)
    if not inspect.isclass(obj):
        pytest.skip("not a class")
    missing = [
        method
        for method, member in _public_methods(obj)
        if not _has_doc(member, method)
    ]
    assert not missing, f"{package}.{name}: undocumented public methods {missing}"


# --------------------------------------------------------------------------- #
# 3. Private / experimental isolation
# --------------------------------------------------------------------------- #
_FORBIDDEN_PREFIXES = ("scirex._internal", "scirex.experimental")


def test_private_and_experimental_packages_are_not_public() -> None:
    for package in PUBLIC_PACKAGES:
        assert not package.startswith(_FORBIDDEN_PREFIXES), package
    assert "scirex.experimental" not in PUBLIC_PACKAGES
    assert "scirex._internal" not in PUBLIC_PACKAGES


@pytest.mark.parametrize(("package", "name"), _PUBLIC_ITEMS, ids=_ITEM_IDS)
def test_public_name_is_not_defined_in_internal_or_experimental(
    package: str, name: str
) -> None:
    obj = _resolve_or_skip(package, name)
    module = getattr(obj, "__module__", None)
    if module is None:
        pytest.skip("object has no __module__")
    assert not module.startswith(
        _FORBIDDEN_PREFIXES
    ), f"{package}.{name} is defined in {module}, which is not stable"


# --------------------------------------------------------------------------- #
# 4. Documentation coverage
# --------------------------------------------------------------------------- #
def _api_page(package: str) -> str:
    parts = package.split(".")
    return "index" if len(parts) == 1 else parts[1]


_API_PAGES = sorted({_api_page(p) for p in PUBLIC_PACKAGES} | {"experimental"})


@pytest.mark.parametrize("page", _API_PAGES)
def test_api_page_exists_and_is_in_the_nav(page: str) -> None:
    path = REPO / "doc" / "api" / f"{page}.md"
    assert path.is_file(), f"missing API reference page {path.relative_to(REPO)}"
    nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert f"api/{page}.md" in nav, f"mkdocs.yml does not mention api/{page}.md"


# --------------------------------------------------------------------------- #
# 5. Deprecation hygiene
# --------------------------------------------------------------------------- #
def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


def _deprecated_training_names() -> frozenset[str]:
    import scirex.training

    return frozenset(scirex.training._DEPRECATED_REEXPORTS)


def test_deprecated_training_names_exist() -> None:
    assert _deprecated_training_names(), "the deprecated shim list is empty"


def test_internal_code_does_not_use_deprecated_training_paths() -> None:
    deprecated = _deprecated_training_names()
    offences: list[str] = []
    for path in sorted((SRC / "scirex").rglob("*.py")):
        rel = path.relative_to(SRC)
        module = ".".join(rel.with_suffix("").parts)
        is_package = path.name == "__init__.py"
        if is_package:
            module = module.removesuffix(".__init__")
        if module == "scirex.training":
            continue  # the shim itself
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if _resolve_from(module, is_package, node) == "scirex.training":
                    bad = sorted(a.name for a in node.names if a.name in deprecated)
                    if bad:
                        offences.append(f"{rel}:{node.lineno} imports {bad}")
            elif isinstance(node, ast.Attribute) and node.attr in deprecated:
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "training"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "scirex"
                ):
                    offences.append(
                        f"{rel}:{node.lineno} uses scirex.training.{node.attr}"
                    )
    assert not offences, (
        "internal code must use scirex._internal.wno_train_utils, not the "
        "deprecated scirex.training names:\n  " + "\n  ".join(offences)
    )


# --------------------------------------------------------------------------- #
# 5b. Deprecations are documented (policy table, CHANGELOG) and consistent
# --------------------------------------------------------------------------- #
# Files that may call the helpers in ``scirex._internal.deprecation``. A new file
# that does must document its deprecation (policy table + CHANGELOG), then be added
# here. See "Deprecating a public name" in doc/dev_guide/api_policy.md.
DEPRECATION_SITES = frozenset(
    {
        "scirex/_internal/aliases.py",
        "scirex/losses/__init__.py",
        "scirex/training/__init__.py",
    }
)
_DEPRECATION_HELPERS = frozenset({"warn_deprecated", "deprecated"})


def _deprecated_names() -> list[tuple[str, str]]:
    """(package, name) for every name that is deprecated through a package shim."""
    import scirex.losses
    import scirex.training

    names = [("scirex.training", n) for n in scirex.training._DEPRECATED_REEXPORTS]
    names += [("scirex.training", n) for n in scirex.training._MOVED_TO_EXPERIMENTAL]
    names += [("scirex.losses", n) for n in scirex.losses._DEPRECATED_EXPORTS]
    return sorted(names)


def _policy_table_rows() -> list[str]:
    text = (REPO / "doc" / "dev_guide" / "api_policy.md").read_text(encoding="utf-8")
    section = text.split("## Currently deprecated names", 1)[1]
    return [line for line in section.splitlines() if line.startswith("|")]


def _changelog_deprecated_text() -> str:
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    chunks = re.findall(r"^### Deprecated\n(.*?)(?=^#{1,3} |\Z)", text, re.M | re.S)
    return "\n".join(chunks)


_DEPRECATED = _deprecated_names()
_DEPRECATED_IDS = [f"{package}.{name}" for package, name in _DEPRECATED]


def test_the_deprecated_registry_is_not_empty() -> None:
    assert _DEPRECATED, "no deprecated names found; did the registries move?"


@pytest.mark.parametrize(("package", "name"), _DEPRECATED, ids=_DEPRECATED_IDS)
def test_deprecated_name_is_in_the_policy_table_with_the_warned_versions(
    package: str, name: str
) -> None:
    import importlib

    with pytest.warns(DeprecationWarning) as record:
        getattr(importlib.import_module(package), name)
    message = str(record[0].message)
    since = re.search(r"since ScireX (\S+) ", message)
    removal = re.search(r"removed in (\S+) or later", message)
    assert since and removal, f"unexpected warning text: {message}"

    rows = [
        row
        for row in _policy_table_rows()
        if f"`{package}.{name}`" in row or (f"`{name}`" in row and package in row)
    ]
    assert rows, f"{package}.{name} is deprecated in code but not in the policy table"
    assert any(
        since.group(1) in row and removal.group(1).rstrip(".") in row for row in rows
    ), f"policy table versions for {package}.{name} differ from the warning: {message}"


@pytest.mark.parametrize(("package", "name"), _DEPRECATED, ids=_DEPRECATED_IDS)
def test_deprecated_name_is_in_the_changelog(package: str, name: str) -> None:
    text = _changelog_deprecated_text()
    assert text, "the CHANGELOG has no 'Deprecated' section"
    assert (
        name in text
    ), f"{package}.{name} is not under a CHANGELOG 'Deprecated' heading"


def test_deprecation_helpers_are_only_used_at_registered_sites() -> None:
    found: set[str] = set()
    for path in sorted((SRC / "scirex").rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "scirex/_internal/deprecation.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    called = func.id
                else:
                    called = getattr(func, "attr", "")
                if called in _DEPRECATION_HELPERS:
                    found.add(rel)
    unregistered = sorted(found - DEPRECATION_SITES)
    stale = sorted(DEPRECATION_SITES - found)
    assert not unregistered, (
        f"{unregistered} deprecate something. Document it in the policy table and the "
        "CHANGELOG 'Deprecated' section, then add the file to DEPRECATION_SITES."
    )
    assert (
        not stale
    ), f"{stale} no longer deprecate anything; remove them from DEPRECATION_SITES."
