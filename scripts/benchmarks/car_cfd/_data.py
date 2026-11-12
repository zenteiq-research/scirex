# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Data loading for the Car-CFD benchmark.

Reads neuraloperator's processed Car-CFD mirror (Zenodo record 13936501,
CC BY 4.0) through :class:`scirex.data.datasets.car_cfd.
CarCFDDatasetjax`, which downloads it on first use into
``DEFAULT_DATA_ROOT``. Each sample is one car: its surface mesh vertices,
the SDF of the car on a regular ``query_res^3`` latent grid, and the surface
pressure (normalised; ``data_module.normalizers["press"]`` inverts it).
Vertices and the grid are scaled into [0, 1] with one dataset-wide bounding
box, and the pressure mean/std is fitted on all loaded cars, train and test
together.

GINO's graph is per-sample (a different car sits at different distances from
the fixed latent grid) and needs a concrete KD-tree search that cannot run
inside a jitted step, so :func:`attach_neighbors` runs it once per split,
then pads every sample's edge list to one length so ``jax.jit`` compiles
the train/eval step once rather than once per car.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from scirex.data.datasets.car_cfd import CarCFDDatasetjax

DEFAULT_DATA_ROOT = Path("~/data/car-pressure-data/processed-car-pressure-data")

#: The dataset's own sub-directory under the root (``train.txt``,
#: ``test.txt``, ``data/<id>/tri_mesh.ply``).
ITEM_DIR = "processed-car-pressure-data"


def load_data_module(
    data_root: Path,
    query_res: int,
    n_train: int,
    n_test: int,
    download: bool = True,
) -> CarCFDDatasetjax:
    return CarCFDDatasetjax(
        root_dir=str(Path(data_root).expanduser()),
        query_res=[query_res] * 3,
        n_train=n_train,
        n_test=n_test,
        download=download,
    )


def attach_neighbors(model, data_module: CarCFDDatasetjax, log_dir: Path) -> None:
    """Precomputes GINO's neighbour lists for both splits, in place, and pads
    them to a shared length (see this module's docstring)."""
    output_n_points = data_module.train_data.data_list[0]["press"].shape[1]
    for split, dataset in (
        ("train", data_module.train_data),
        ("test", data_module.test_data),
    ):
        print(f"[car_cfd] precomputing neighbours ({split}) ...", flush=True)
        model.precompute_neighbors(
            dataset, output_n_points=output_n_points, log_dir=str(log_dir), split=split
        )
    pad_neighbor_data(
        [data_module.train_data.data_list, data_module.test_data.data_list]
    )


def pad_neighbor_data(data_lists: list) -> tuple[int, int]:
    """Pads neighbour index arrays to a fixed maximum length for ``jax.jit``.

    Padded edges point at index 0 and carry segment id ``n_out`` (one past the
    last real query), so a segment-sum over ``n_out`` segments drops them.
    """
    all_samples = [s for dl in data_lists for s in dl]

    # The decoder-only variant builds no input graph, so "neighbors_in" is
    # absent and there is nothing to pad for it.
    has_in = all("neighbors_in" in s for s in all_samples)

    max_in = (
        max(s["neighbors_in"]["neighbors_index"].shape[0] for s in all_samples)
        if has_in
        else 0
    )
    max_out = max(s["neighbors_out"]["neighbors_index"].shape[0] for s in all_samples)

    to_pad = [("neighbors_out", max_out)]
    if has_in:
        to_pad.insert(0, ("neighbors_in", max_in))

    for s in all_samples:
        for key, max_edges in to_pad:
            nbrs = s[key]
            idx = np.asarray(nbrs["neighbors_index"])
            splits = np.asarray(nbrs["neighbors_row_splits"])
            n_real = idx.shape[0]
            n_out = splits.shape[0] - 1

            counts = splits[1:] - splits[:-1]
            seg_ids = np.repeat(np.arange(n_out), counts)
            pad_len = max_edges - n_real

            s[key] = {
                "neighbors_index": np.concatenate(
                    [idx, np.zeros(pad_len, dtype=idx.dtype)]
                ),
                "segment_ids": np.concatenate(
                    [seg_ids, np.full(pad_len, n_out, dtype=seg_ids.dtype)]
                ),
                "counts": counts,
            }

    return max_in, max_out


def load_test_mesh(data_root: Path, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, triangles)`` of test car ``index`` (``test.txt`` order,
    which is the test loader's order), for the surface figures and the VTK
    export. Its vertices line up with the dataset's ``press`` field."""
    item_dir = Path(data_root).expanduser() / ITEM_DIR
    car_id = (item_dir / "test.txt").read_text().split(",")[index].strip()
    data_subdir = item_dir / "data"
    sample_dir = data_subdir / car_id
    if not sample_dir.exists():
        sample_dir = data_subdir / "data" / car_id
    mesh = o3d.io.read_triangle_mesh(str(sample_dir / "tri_mesh.ply"))
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
