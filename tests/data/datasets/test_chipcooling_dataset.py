# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""
Unit tests for the two-chip electronics-cooling dataset loader
(src/scirex/data/datasets/chipcooling.py).

Covers:
  - _sim_sort_key() numeric (not lexicographic) ordering of sim_<N>.npz names
  - load_data() end-to-end on a small synthetic dataset, asserting each field
    is paired with the matching DOE parameter row even when the filename
    indices are not zero-padded (regression test for the lexicographic-sort
    misalignment bug), plus the file-count-mismatch error path
  - build_domain_mask() shape and coverage
  - build_input_channels() shape/content and the wrong-parameter-count error
  - normalize_inputs() z-scoring of the physical channels with the mask
    channel left untouched
  - normalize_targets() normalisation on valid cells and zeroing of invalid
    cells
  - evaluate_model() exactness on a perfect prediction

Pure NumPy -- no dataset files or JAX needed.

Run with:
    pytest tests/data/datasets/test_chipcooling_dataset.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scirex.data.datasets import chipcooling as cc
from scirex.data.datasets.chipcooling import (
    build_domain_mask,
    build_input_channels,
    evaluate_model,
    load_data,
    normalize_inputs,
    normalize_targets,
)
from tests import tolerances

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_dataset(root: Path, n_sim: int, n_points: int = 8):
    """Write a minimal model_setup.json + non-zero-padded sim_<i>.npz files.

    Parameter row ``i`` and simulation file ``sim_i`` both carry the value
    ``i`` (in the first parameter column and as the constant temperature
    field, respectively), so a correct loader lines them up index-for-index.
    """
    doe = {"caseA": {"a_idx": [float(i) for i in range(n_sim)]}}
    (root / "model_setup.json").write_text(json.dumps({"doe_configuration": doe}))

    coords = np.zeros((n_points, 3), dtype=np.float32)
    coords[:, 1] = np.linspace(0.0, 1.0, n_points)  # y
    coords[:, 2] = np.linspace(0.0, 2.0, n_points)  # z
    for i in range(n_sim):
        np.savez(
            root / f"sim_{i}.npz",
            **{
                cc.FIELD_KEY: np.full(n_points, float(i), dtype=np.float32),
                cc.COORD_KEY: coords,
            },
        )


# ---------------------------------------------------------------------------
# _sim_sort_key
# ---------------------------------------------------------------------------


def test_sim_sort_key_orders_numerically_not_lexicographically():
    names = [Path(f"sim_{i}.npz") for i in (10, 2, 1, 11, 100, 3)]
    ordered = [p.name for p in sorted(names, key=cc._sim_sort_key)]
    assert ordered == [
        "sim_1.npz",
        "sim_2.npz",
        "sim_3.npz",
        "sim_10.npz",
        "sim_11.npz",
        "sim_100.npz",
    ]


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


def test_load_data_pairs_each_field_with_its_parameter_row(tmp_path):
    # 12 sims => lexicographic order (sim_0, sim_1, sim_10, sim_11, sim_2, ...)
    # differs from numeric order, so this exercises the sort fix.
    n_sim = 12
    _write_synthetic_dataset(tmp_path, n_sim)

    parameters, fields, coords, param_names = load_data(dataset_dir=tmp_path)

    assert parameters.shape == (n_sim, 1)
    assert fields.shape[0] == n_sim
    assert coords.shape[1] == 3
    assert param_names == ["caseA.a_idx"]

    # Row i must carry value i in both the parameter column and the field.
    np.testing.assert_allclose(parameters[:, 0], np.arange(n_sim))
    field_means = fields.mean(axis=1)
    np.testing.assert_allclose(field_means, np.arange(n_sim))


def test_load_data_refuses_pickled_object_arrays(tmp_path):
    """A .npz holding a pickled object array is rejected, never unpickled."""
    _write_synthetic_dataset(tmp_path, n_sim=2)

    class _Payload:
        def __reduce__(self):  # would run on unpickling
            return (Path(tmp_path / "pwned").touch, ())

    np.savez(
        tmp_path / "sim_1.npz",
        **{
            cc.FIELD_KEY: np.array([_Payload()], dtype=object),
            cc.COORD_KEY: np.zeros((1, 3), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_data(dataset_dir=tmp_path)
    assert not (tmp_path / "pwned").exists()


def test_load_data_raises_on_file_count_mismatch(tmp_path):
    _write_synthetic_dataset(tmp_path, n_sim=4)
    (tmp_path / "sim_0.npz").unlink()  # 3 files now, 4 parameter rows
    with pytest.raises(RuntimeError, match="simulation files"):
        load_data(dataset_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_domain_mask
# ---------------------------------------------------------------------------


def test_build_domain_mask_shape_and_coverage():
    coords = np.zeros((5, 3), dtype=np.float32)
    coords[:, 1] = [0.0, 0.25, 0.5, 0.75, 1.0]  # y
    coords[:, 2] = [0.0, 0.5, 1.0, 1.5, 2.0]  # z
    y_grid = np.linspace(0.0, 1.0, 4)
    z_grid = np.linspace(0.0, 2.0, 6)

    mask = build_domain_mask(coords, y_grid, z_grid)

    assert mask.shape == (4, 6)
    assert mask.dtype == bool
    assert mask.any()  # at least one node landed in a cell


# ---------------------------------------------------------------------------
# build_input_channels
# ---------------------------------------------------------------------------


def test_build_input_channels_shape_and_content():
    H, W = 3, 4
    params = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    mask = np.zeros((H, W), dtype=bool)
    mask[0, 0] = True

    X = build_input_channels(params, mask, H=H, W=W)

    assert X.shape == (2, 7, H, W)
    # channels 0..5 are the parameters broadcast over the grid
    for i in range(6):
        assert np.allclose(X[:, i], params[:, i, None, None])
    # channel 6 is the 0/1 domain mask
    assert np.array_equal(X[0, 6], mask.astype(np.float32))


def test_build_input_channels_rejects_wrong_param_count():
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="6 parameters"):
        build_input_channels(np.zeros((3, 5), dtype=np.float32), mask, H=2, W=2)


# ---------------------------------------------------------------------------
# normalize_inputs
# ---------------------------------------------------------------------------


def test_normalize_inputs_zscores_params_and_preserves_mask_channel():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(10, 7, 3, 3)).astype(np.float32)
    X_test = rng.normal(size=(4, 7, 3, 3)).astype(np.float32)
    X_train[:, 6] = 1.0  # mask channel
    X_test[:, 6] = 1.0

    X_tr_n, X_te_n, mean, std = normalize_inputs(X_train, X_test)

    assert mean.shape == (1, 6, 1, 1)
    assert std.shape == (1, 6, 1, 1)
    # normalised train params have ~zero mean per channel
    np.testing.assert_allclose(
        X_tr_n[:, :6].mean(axis=(0, 2, 3)), np.zeros(6), atol=tolerances.atol(1e-5)
    )
    # mask channel is passed through untouched
    assert np.array_equal(X_tr_n[:, 6], X_train[:, 6])
    assert np.array_equal(X_te_n[:, 6], X_test[:, 6])


# ---------------------------------------------------------------------------
# normalize_targets
# ---------------------------------------------------------------------------


def test_normalize_targets_normalises_valid_and_zeros_invalid():
    mask = np.array([[True, True], [False, True]])
    Y_train = np.arange(2 * 1 * 2 * 2, dtype=np.float32).reshape(2, 1, 2, 2)
    Y_test = Y_train + 3.0

    Y_tr_n, Y_te_n, y_mean, y_std = normalize_targets(Y_train, Y_test, mask)

    # invalid cell (row 1, col 0) is zeroed in both splits
    assert np.all(Y_tr_n[:, 0, ~mask] == 0.0)
    assert np.all(Y_te_n[:, 0, ~mask] == 0.0)
    # valid training cells are standardised: (x - mean) / std
    expected = (Y_train[:, 0][:, mask] - y_mean) / y_std
    np.testing.assert_allclose(
        Y_tr_n[:, 0][:, mask], expected, rtol=tolerances.rtol(1e-6)
    )


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_perfect_prediction():
    mask = np.array([[True, False], [True, True]])
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(5, 1, 2, 2)).astype(np.float32)

    m = evaluate_model(Y_true=Y, Y_pred=Y.copy(), mask=mask)

    assert m["R2"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(0.0, abs=1e-6)
    assert m["MAE"] == pytest.approx(0.0, abs=1e-6)
    assert m["R2_per_sample"].shape == (5,)
