# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Check every Problem_Model tutorial page against its script.

Each page ``doc/tutorials/<Folder>/<Problem>_<Model>.md`` (``<Folder>`` is the
model for its two main tutorials, ``Others`` for the rest) copies its code from
``scripts/tutorials/<problem>_<model>.py``. These tests keep the two in step,
check that the script only uses public API, that the page has the required
headings in order, that its images and links exist and that it is in the
navigation. Running each script on synthetic data is left to the tutorial's
own test module (``test_tutorial_<problem>_<model>.py``).
"""

import ast
import re
from pathlib import Path

import pytest

from tests.public_api_helpers import PUBLIC_PACKAGES, REPO

PAGES = REPO / "doc" / "tutorials"
SCRIPTS = REPO / "scripts" / "tutorials"
MKDOCS = REPO / "mkdocs.yml"

# Marker that puts a fenced block under the verbatim check.
SNIPPET = re.compile(r"<!-- tutorial-snippet -->\n```python\n(.*?)\n```", re.DOTALL)

REQUIRED_HEADINGS = [
    "## Summary",
    "## Setup",
    "## The problem",
    "## Step-by-step walkthrough",
    "## Run it",
    "## Results",
    "## Next steps",
    "## References",
]

THIRD_PARTY = {
    "flax",
    "jax",
    "matplotlib",
    "mpl_toolkits",  # matplotlib's 3-D plotting
    "numpy",
    "open3d",  # reads the Car-CFD meshes
    "optax",
    "scipy",
    "vtk",
}
STDLIB = {
    "__future__",
    "argparse",
    "functools",
    "json",
    "math",
    "os",
    "pathlib",
    "platform",
    "re",
    "time",
}

# <Problem>_<Model>.md, e.g. Pipe_FNO.md or ShapeNetCar_GINO.md
PAGE_NAME = re.compile(
    r"^[A-Z][A-Za-z]*(?:_[A-Z][A-Za-z]*)*_(FNO|WNO|GINO|GIWO|Transolver)\.md$"
)

TUTORIALS = sorted(p for p in PAGES.glob("*/*.md") if PAGE_NAME.match(p.name))


def _script_for(page: Path) -> Path:
    return SCRIPTS / f"{page.stem.lower()}.py"


def _imported_modules(script: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
    return names


def test_there_are_tutorials() -> None:
    assert TUTORIALS, "no doc/tutorials/<Folder>/<Problem>_<Model>.md page found"


# The worked example of the "Add a new example" guide sits with the guide.
GUIDE_FOLDER = "NewExample"
# Problems whose pages keep a folder of their own instead of their model's.
PROBLEM_FOLDERS = {"DarcyPhysics"}
# Every tutorial that is not one of a model's two main ones.
OTHERS_FOLDER = "Others"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_is_in_its_model_folder(page: Path) -> None:
    problem, model = page.stem.rsplit("_", 1)
    folder = page.parent.name
    assert folder in (model, GUIDE_FOLDER, OTHERS_FOLDER) or (
        folder == problem and problem in PROBLEM_FOLDERS
    ), page


@pytest.mark.parametrize("model", ["FNO", "WNO", "GINO", "Transolver"])
def test_each_model_folder_holds_two_tutorials(model: str) -> None:
    pages = sorted(p.name for p in (PAGES / model).glob("*.md"))
    assert len(pages) == 2, pages


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_has_a_script(page: Path) -> None:
    assert _script_for(page).exists(), f"{page.name} has no {_script_for(page).name}"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_script_compiles_and_runs_as_main(page: Path) -> None:
    text = _script_for(page).read_text()
    compile(text, str(_script_for(page)), "exec")
    assert 'if __name__ == "__main__":' in text
    assert text.count("\n# %%\n") >= 8


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_script_imports_only_public_modules(page: Path) -> None:
    for name in _imported_modules(_script_for(page)):
        root = name.split(".")[0]
        if root == "scirex":
            assert name in PUBLIC_PACKAGES, f"{name} is not a public package"
        else:
            assert root in THIRD_PARTY | STDLIB, f"unexpected import {name}"
        assert "scripts" not in name.split("."), name
        assert not any(part.startswith("_") for part in name.split(".")[1:]), name


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_has_required_headings(page: Path) -> None:
    headings = [
        line for line in page.read_text().splitlines() if line.startswith("## ")
    ]
    for heading in REQUIRED_HEADINGS:
        assert heading in headings, f"missing heading {heading!r}"
    positions = [headings.index(h) for h in REQUIRED_HEADINGS]
    assert positions == sorted(positions), "headings are out of order"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_snippets_appear_verbatim_in_script(page: Path) -> None:
    script = _script_for(page).read_text()
    snippets = SNIPPET.findall(page.read_text())
    assert len(snippets) >= 10
    for snippet in snippets:
        assert snippet in script, f"not found in the script:\n{snippet[:200]}"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_covers_every_function_and_class_of_the_script(page: Path) -> None:
    text = page.read_text()
    for node in ast.parse(_script_for(page).read_text()).body:
        # The page ends with a pointer to the complete script, which holds main().
        if isinstance(node, ast.FunctionDef) and node.name != "main":
            assert f"def {node.name}(" in text, f"{node.name} is not on the page"
        elif isinstance(node, ast.ClassDef):
            assert f"class {node.name}" in text, f"{node.name} is not on the page"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_images_and_links_exist(page: Path) -> None:
    targets = re.findall(r"!?\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", page.read_text())
    local = [t for t in targets if not re.match(r"[a-z]+:", t)]
    assert any(t.endswith(".png") for t in local)
    for target in local:
        assert (page.parent / target).resolve().exists(), f"missing {target}"


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_has_no_unfinished_markers(page: Path) -> None:
    text = page.read_text()
    assert not re.search(r"\b(TODO|TBD|FIXME)\b", text)
    assert f"`scripts/tutorials/{page.stem.lower()}.py`" in text


@pytest.mark.parametrize("page", TUTORIALS, ids=lambda p: p.stem)
def test_page_is_in_mkdocs_nav(page: Path) -> None:
    nav = MKDOCS.read_text()
    entry = f"tutorials/{page.parent.name}/{page.name}"
    assert entry in nav
    if page.stem.endswith("_GIWO"):
        # GIWO is experimental: its tutorials sit in the Experimental section of Benchmarks
        assert nav.index(entry) > nav.index("- Experimental:")
    else:
        assert nav.index(entry) < nav.index("- Benchmarks:")
