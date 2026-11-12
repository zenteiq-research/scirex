# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Darcy benchmark: the unlabelled-scaling figures, built from
the committed ``metrics.json`` files only (no training, no GPU).

For each model, the final test relative L2 against the number of coefficient
fields the physics loss sees (the 1000 released ones, then 1000 + the first
K fields of the sampled pool), with that model's data-driven reference and
the formulation floor; written to
``doc/assets/benchmarks/darcy_physics_<model>/scaling.png``.
Also all four models in one figure,
``doc/assets/benchmarks/darcy_physics_comparison/scaling.png``.

Usage (from the repo root)::

    python scripts/benchmarks/darcy_physics/plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmarks.darcy._config import MODEL_LABELS  # noqa: E402
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    COMPARISON_OUT_DIR,
    DEFAULT_OUT_DIR,
    FIELDS,
    FLOOR_OUT_DIR,
    MODELS,
    run_dir,
)

BLUE, RED, ORANGE = "#256abf", "crimson", "#c97b00"
MODEL_COLORS = dict(transolver="#256abf", fno="#2a9d5c", gino="#8a4fbf", wno="#c97b00")


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _data_driven(model: str) -> float:
    """This repository's data-driven run of ``model`` (1000 labels), in %."""
    path = DEFAULT_OUT_DIR / f"darcy_{model}" / "metrics.json"
    return 100 * _load(path)["final_test_rel_l2_mean"]


def references(model: str) -> list[tuple[str, float, str]]:
    """(label, value in %, colour) of each data-driven reference line."""
    if model == "transolver":
        return [("Wu et al. (2024) Transolver, pure data, 1000 labels", 0.57, RED)]
    if model == "fno":
        return [("Li et al. (2021) FNO, pure data, 1000 labels", 1.08, RED)]
    if model == "gino":
        return [("ZenteiQ GINO, pure data, 1000 labels", _data_driven("gino"), ORANGE)]
    return [
        ("ZenteiQ WNO, pure data, 1000 labels", _data_driven("wno"), ORANGE),
        ("Tripura & Chakraborty (2023) WNO, pure data, 1000 labels", 0.84, RED),
    ]


def load_runs(model: str) -> list[dict]:
    """The committed metrics of ``model``'s runs, one per entry of FIELDS."""
    return [_load(run_dir(model, n) / "metrics.json") for n in FIELDS]


def floor_percent(runs: list[dict]) -> float:
    """Exact Q4 solve vs the released solutions, for the runs' cell rule."""
    cell_modes = {r["cell_mode"] for r in runs}
    assert len(cell_modes) == 1, cell_modes
    key = f"res{runs[0]['resolution'] - 1}_{cell_modes.pop()}"
    spec = _load(FLOOR_OUT_DIR / "metrics.json")
    return 100 * spec["results"][key]["mapped"]["rel_l2_mean"]


def _budget(runs: list[dict]) -> str:
    steps = sorted(r["steps_run"] for r in runs)
    lo, hi = round(steps[0] / 1000), round(steps[-1] / 1000)
    return f"{hi}k" if lo == hi else f"~{hi}k"


def _finish(ax: plt.Axes, ys: list[float], out: Path) -> None:
    ax.set_xscale("log")
    ax.set_xticks(FIELDS, [f"{n / 1000:g}k" for n in FIELDS])
    ax.minorticks_off()
    ax.set_xlabel("coefficient fields seen by the physics loss (released 1k + sampled)")
    ax.set_ylabel("test rel-L2 [%]")
    ax.set_ylim(0, max(ys) * 1.2)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig = ax.figure
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_model(model: str) -> None:
    runs = load_runs(model)
    ys = [100 * r["final_test_rel_l2_mean"] for r in runs]
    floor = floor_percent(runs)
    label = MODEL_LABELS[model]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        FIELDS, ys, color=BLUE, marker="o", lw=2,
        label=f"ZenteiQ {label}, pure physics (no labels)",
    )  # fmt: skip
    for x, y in zip(FIELDS, ys):
        ax.annotate(
            f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9
        )
    refs = references(model)
    for ref_label, value, color in refs:
        ax.axhline(
            value, color=color, ls="--", lw=1.5, label=f"{ref_label} ({value:.3f}%)"
        )
    ax.axhline(
        floor, color="k", ls=":", lw=1.2,
        label=f"Formulation floor, exact Q4 solve ({floor:.3f}%)",
    )  # fmt: skip
    ax.set_title(f"Unlabelled scaling, {label} ({_budget(runs)} steps per run, seed 0)")
    _finish(
        ax,
        ys + [v for _, v, _ in refs],
        run_dir(model, FIELDS[-1]).parent / "scaling.png",
    )


def plot_comparison() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    all_ys: list[float] = []
    for model in MODELS:
        runs = load_runs(model)
        ys = [100 * r["final_test_rel_l2_mean"] for r in runs]
        all_ys += ys
        ax.plot(
            FIELDS, ys, color=MODEL_COLORS[model], marker="o", lw=2,
            label=f"{MODEL_LABELS[model]} ({_budget(runs)} steps)",
        )  # fmt: skip
    floor = floor_percent(load_runs(MODELS[0]))
    ax.axhline(
        0.57, color=RED, ls="--", lw=1.5,
        label="Wu et al. (2024) Transolver, pure data, 1000 labels (0.57%)",
    )  # fmt: skip
    ax.axhline(
        floor, color="k", ls=":", lw=1.2,
        label=f"Formulation floor, exact Q4 solve ({floor:.3f}%)",
    )  # fmt: skip
    ax.set_title("Unlabelled scaling, pure physics (no labels), seed 0")
    _finish(ax, all_ys, COMPARISON_OUT_DIR / "scaling.png")


def main() -> None:
    for model in MODELS:
        plot_model(model)
    plot_comparison()


if __name__ == "__main__":
    main()
