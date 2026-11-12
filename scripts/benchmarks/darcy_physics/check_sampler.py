# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Physics-only Darcy benchmark, sampler check: do the extra, unlabelled
coefficient fields (the fixed pool of ``_ritz.py``) follow the same law as
Li et al.'s released training fields?

Compares, on the native 421 x 421 grid, the first ``--n`` released training
coefficients (``piececonst_r421_N1024_smooth1.mat``) with the first ``--n``
fields of the pool
(:func:`~scirex.losses.darcy_piecewise_coefficient.sample_piecewise_coefficient`):

- the coefficient values;
- the area fraction of the high phase (mean and spread over fields);
- the interface density (fraction of neighbouring node pairs with different
  values), a proxy for the correlation length;
- the autocorrelation of the high-phase indicator at several lags
  (also drawn in ``sampler.png``).

No solutions are involved.

Usage (from the repo root)::

    export DARCY_DATA_DIR=/path/to/dir  # piececonst_r421_N1024_smooth1.mat
    python scripts/benchmarks/darcy_physics/check_sampler.py

Written to ``doc/assets/benchmarks/darcy_coefficient_sampler_check/``
(``metrics.json`` and ``sampler.png``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", False)

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from scirex.losses.darcy_piecewise_coefficient import (  # noqa: E402
    sample_piecewise_coefficient,
)
from scripts.benchmarks._plot_utils import save_metrics_json  # noqa: E402
from scripts.benchmarks.darcy._data import load_split  # noqa: E402
from scripts.benchmarks.darcy_physics._config import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PROTOCOL,
    SAMPLER_OUT_DIR,
)
from scripts.benchmarks.darcy_physics._ritz import (  # noqa: E402
    extra_coefficient_keys,
)

LAGS = (1, 5, 10, 20, 40, 80)


def field_statistics(a: np.ndarray) -> dict:
    """Per-field statistics of ``(N, n, n)`` two-valued coefficients,
    summarised as mean and standard deviation over the fields."""
    hi = (a > 0.5 * (a.min() + a.max())).astype(np.float32)
    area = hi.mean(axis=(1, 2))
    interface = 0.5 * (
        (hi[:, 1:, :] != hi[:, :-1, :]).mean(axis=(1, 2))
        + (hi[:, :, 1:] != hi[:, :, :-1]).mean(axis=(1, 2))
    )
    centred = hi - hi.mean(axis=(1, 2), keepdims=True)
    var = np.clip(centred.var(axis=(1, 2)), 1e-9, None)
    autocorr = {}
    for lag in LAGS:
        cx = (centred[:, lag:, :] * centred[:, :-lag, :]).mean(axis=(1, 2))
        cy = (centred[:, :, lag:] * centred[:, :, :-lag]).mean(axis=(1, 2))
        autocorr[str(lag)] = float(np.mean(0.5 * (cx + cy) / var))
    return {
        "values": sorted(np.unique(a).tolist()),
        "high_area_fraction_mean": float(area.mean()),
        "high_area_fraction_std": float(area.std()),
        "interface_density_mean": float(interface.mean()),
        "interface_density_std": float(interface.std()),
        "autocorrelation_by_lag_nodes": autocorr,
    }


def plot_autocorrelation(results: dict, out_path: Path) -> None:
    """High-phase autocorrelation by lag, released vs sampled fields."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, color, lw, ms, label in (
        ("released_smooth1", "#9aa7b8", 6, 10, "released (smooth1, first fields)"),
        ("sampled_pool", "#256abf", 2, 5, "sampled pool (first fields)"),
    ):
        ac = results[key]["autocorrelation_by_lag_nodes"]
        ax.plot(
            [int(k) for k in ac],
            list(ac.values()),
            marker="o",
            ms=ms,
            color=color,
            lw=lw,
            label=label,
        )
    r, s = results["released_smooth1"], results["sampled_pool"]
    ax.text(
        0.98,
        0.95,
        f"high-phase area: {r['high_area_fraction_mean']:.3f} vs "
        f"{s['high_area_fraction_mean']:.3f}\n"
        f"interface density: {r['interface_density_mean']:.4f} vs "
        f"{s['interface_density_mean']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax.set_xlabel("lag [native nodes]")
    ax.set_ylabel("autocorrelation of the high-phase indicator")
    ax.set_title("Sampled vs released coefficient fields (native 421 grid)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=PROTOCOL["n_train"])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=SAMPLER_OUT_DIR)
    args = ap.parse_args()

    # load_split always reads the test file too; one sample is enough.
    released = load_split(
        args.data_dir, dict(PROTOCOL, n_train=args.n, n_test=1, subsample_rate=1)
    )["train_a"]
    n_native = released.shape[1]
    sampled = np.stack(
        [
            sample_piecewise_coefficient(k, n_native)
            for k in extra_coefficient_keys(args.n)
        ]
    )

    results = {
        "released_smooth1": field_statistics(released),
        "sampled_pool": field_statistics(sampled),
    }
    for name, r in results.items():
        print(
            f"[check_sampler] {name:<17} values {r['values']}  "
            f"high-phase area {r['high_area_fraction_mean']:.3f} "
            f"(std {r['high_area_fraction_std']:.3f})  interface density "
            f"{r['interface_density_mean']:.4f} (std {r['interface_density_std']:.4f})",
            flush=True,
        )
        print(
            "[check_sampler]   autocorrelation "
            + "  ".join(
                f"lag {k}: {v:.3f}"
                for k, v in r["autocorrelation_by_lag_nodes"].items()
            ),
            flush=True,
        )
    metrics = {
        "problem": "darcy_coefficient_sampler_check",
        "description": (
            "Statistics of Li et al.'s released training coefficients vs the "
            "fixed extra-coefficient pool of darcy_physics/_ritz.py "
            "(native 421 grid, no solutions)."
        ),
        "n_fields": args.n,
        "results": results,
    }
    plot_autocorrelation(results, args.out_dir / "sampler.png")
    out = save_metrics_json(metrics, args.out_dir / "metrics.json")
    print(f"[check_sampler] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
