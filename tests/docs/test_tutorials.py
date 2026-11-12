# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Check the ShapeNetCar tutorial page against its script.

The page ``doc/tutorials/Transolver/ShapeNetCar_Transolver.md`` copies its code from
``scripts/tutorials/shapenetcar_transolver.py``. These tests keep the two in
step, check that the script only uses public API, that the page has the
required headings, that its images and links exist, and that it is in the
navigation. A CPU smoke test runs the script's functions on synthetic data, so
nothing is downloaded. The one test that reads the real dataset runs only when
``vtk`` is installed and ``SHAPENETCAR_DATA`` points at ``training_data``.
"""

import ast
import importlib.util
import os
import re
from pathlib import Path

import numpy as np
import pytest

from tests.public_api_helpers import PUBLIC_PACKAGES, REPO

SCRIPT = REPO / "scripts" / "tutorials" / "shapenetcar_transolver.py"
PAGE = REPO / "doc" / "tutorials" / "Transolver" / "ShapeNetCar_Transolver.md"
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

THIRD_PARTY = {"flax", "jax", "matplotlib", "numpy", "optax"}
STDLIB = {"__future__", "argparse", "os", "platform", "pathlib", "time"}


def _load_script():
    spec = importlib.util.spec_from_file_location("shapenetcar_tutorial", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _imported_modules() -> list[str]:
    names = []
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
    return names


# ── static checks ────────────────────────────────────────────────────────────


def test_script_compiles() -> None:
    compile(SCRIPT.read_text(), str(SCRIPT), "exec")


def test_script_imports_only_public_modules() -> None:
    for name in _imported_modules():
        root = name.split(".")[0]
        if root == "scirex":
            assert name in PUBLIC_PACKAGES, f"{name} is not a public package"
        else:
            assert root in THIRD_PARTY | STDLIB, f"unexpected import {name}"
        assert "scripts" not in name.split("."), name
        assert not any(part.startswith("_") for part in name.split(".")[1:]), name


def test_script_has_sphinx_gallery_cells_and_entry_points() -> None:
    text = SCRIPT.read_text()
    assert text.count("\n# %%\n") >= 8
    tree = ast.parse(text)
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {
        "build_subset",
        "train",
        "evaluate",
        "plot_surface_pressure",
        "main",
    } <= defs
    assert 'if __name__ == "__main__":' in text


def test_page_has_required_headings() -> None:
    text = PAGE.read_text()
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    for heading in REQUIRED_HEADINGS:
        assert heading in headings, f"missing heading {heading!r}"
    positions = [headings.index(h) for h in REQUIRED_HEADINGS]
    assert positions == sorted(positions), "headings are out of order"


def test_page_snippets_appear_verbatim_in_script() -> None:
    script = SCRIPT.read_text()
    snippets = SNIPPET.findall(PAGE.read_text())
    assert len(snippets) >= 10
    for snippet in snippets:
        assert snippet in script, f"not found in the script:\n{snippet[:200]}"


def test_page_covers_every_function_of_the_script() -> None:
    page = PAGE.read_text()
    tree = ast.parse(SCRIPT.read_text())
    for node in tree.body:
        # The page ends with a pointer to the complete script, which holds main().
        if isinstance(node, ast.FunctionDef) and node.name != "main":
            assert f"def {node.name}(" in page, f"{node.name} is not on the page"


def test_page_images_and_links_exist() -> None:
    text = PAGE.read_text()
    targets = re.findall(r"!?\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
    local = [t for t in targets if not re.match(r"[a-z]+:", t)]
    assert any(t.endswith(".png") for t in local)
    for target in local:
        assert (PAGE.parent / target).resolve().exists(), f"missing {target}"


def test_page_has_no_unfinished_markers() -> None:
    text = PAGE.read_text()
    assert not re.search(r"\b(TODO|TBD|FIXME)\b", text)


def test_page_is_in_mkdocs_nav() -> None:
    nav = MKDOCS.read_text()
    assert "tutorials/Transolver/ShapeNetCar_Transolver.md" in nav
    assert nav.index("tutorials/Transolver/ShapeNetCar_Transolver.md") < nav.index(
        "- Benchmarks:"
    )


# ── CPU smoke tests on synthetic data ────────────────────────────────────────


def _fake_cars(n_cars: int, n_points: int = 40, n_surf: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    cars = []
    for _ in range(n_cars):
        surf = np.zeros(n_points, dtype=bool)
        surf[:n_surf] = True
        cars.append(
            dict(
                x=rng.normal(size=(n_points, 7)).astype(np.float32),
                y=rng.normal(size=(n_points, 4)).astype(np.float32),
                pos=rng.normal(size=(n_points, 3)).astype(np.float32),
                surf=surf,
            )
        )
    return cars


def test_train_evaluate_plot_on_synthetic_cars(tmp_path: Path) -> None:
    tutorial = _load_script()
    train_cars, val_cars = _fake_cars(3), _fake_cars(2, seed=1)
    model_kwargs = dict(
        space_dim=7, fun_dim=0, out_dim=4, unified_pos=False, geometry="irregular"
    )
    model, params = tutorial.build_model(
        model_kwargs, tutorial.SIZES["smoke"], train_cars[0], seed=0
    )
    params, history = tutorial.train(model, params, train_cars, val_cars, 2, seed=0)
    assert [h["epoch"] for h in history] == [0, 1]
    assert all(
        np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in history
    )

    coef_norm = (np.zeros(7), np.ones(7), np.zeros(4), np.ones(4))
    metrics = tutorial.evaluate(model, params, val_cars, coef_norm)
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    assert len(metrics["predictions"]) == 2

    tutorial.plot_surface_pressure(
        val_cars[0], metrics["predictions"][0], coef_norm, tmp_path / "p.png"
    )
    tutorial.plot_loss(history, tmp_path / "loss.png")
    assert (tmp_path / "p.png").stat().st_size > 0
    assert (tmp_path / "loss.png").stat().st_size > 0


def test_rel_l2_matches_definition() -> None:
    tutorial = _load_script()
    truth = np.array([3.0, 4.0])
    assert tutorial.rel_l2(truth, truth) == 0.0
    assert tutorial.rel_l2(np.zeros(2), truth) == pytest.approx(1.0, abs=1e-6)


def test_build_subset_links_complete_cars_only(tmp_path: Path) -> None:
    tutorial = _load_script()
    src = tmp_path / "src"
    for fold in ("param0", "param1"):
        for name in ("a", "b", "c", "d"):
            car = src / fold / name
            car.mkdir(parents=True)
            if name != "a":  # "a" is an incomplete car: no VTK files
                (car / "quadpress_smpl.vtk").touch()
                (car / "hexvelo_smpl.vtk").touch()
    picked = tutorial.build_subset(src, tmp_path / "sub", n_train=2, n_val=1)
    assert picked == {"param0": ["b"], "param1": ["b", "c"]}
    assert sorted(p.name for p in (tmp_path / "sub" / "param1").iterdir()) == ["b", "c"]
    # A second call with other counts drops the links of the first call.
    tutorial.build_subset(src, tmp_path / "sub", n_train=1, n_val=1)
    assert [p.name for p in (tmp_path / "sub" / "param1").iterdir()] == ["b"]
    with pytest.raises(ValueError, match="complete cars"):
        tutorial.build_subset(src, tmp_path / "sub", n_train=9, n_val=1)


def test_smoke_main_on_synthetic_vtk_cars(tmp_path: Path) -> None:
    pytest.importorskip("vtk")
    pytest.importorskip("sklearn")
    from tests.data.datasets.test_shapenet_car_dataset import _make_sample

    src = tmp_path / "raw"
    for fold in (0, 1):
        for s in range(3):
            _make_sample(src / f"param{fold}" / f"car_{s}", seed=fold * 10 + s)
    out = tmp_path / "out"
    tutorial = _load_script()
    metrics = tutorial.main(
        [
            "--data-dir", str(src),
            "--smoke",
            "--n-train", "2",
            "--n-val", "1",
            "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
    assert (out / "surface_pressure.png").exists()
    assert (out / "loss.png").exists()
    assert len(list((out / "subset" / "param1").iterdir())) == 2


@pytest.mark.skipif(
    not os.environ.get("SHAPENETCAR_DATA"),
    reason="set SHAPENETCAR_DATA to the real training_data folder to run this test",
)
def test_smoke_main_on_real_data(tmp_path: Path) -> None:
    pytest.importorskip("vtk")
    tutorial = _load_script()
    metrics = tutorial.main(
        [
            "--smoke",
            "--out-dir",
            str(tmp_path),
            "--data-dir",
            os.environ["SHAPENETCAR_DATA"],
        ]
    )
    assert np.isfinite(metrics["rel_l2_volume"])
    assert np.isfinite(metrics["rel_l2_surface"])
