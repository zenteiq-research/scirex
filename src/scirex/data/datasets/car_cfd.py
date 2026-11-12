# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""JAX/Flax version of Car CFD Dataset"""

import logging
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from scirex._internal.paths import get_project_root
from scirex._internal.web_utils import download_from_zenodo_record

from ..containers.pt_dataset import _load_pt
from .mesh_datamodule import MeshDataModule

logger = logging.getLogger(__name__)


class CarCFDDatasetjax(MeshDataModule):
    """Processed version of the Car-CFD dataset for JAX/Flax.

    CarCFDDataset is a processed version of the dataset introduced in
    [1], which encodes a triangular mesh over the surface of a 3D model car
    and provides the air pressure at each centroid and vertex of the mesh when
    the car is placed in a simulated wind tunnel with a recorded inlet velocity.

    In our case the inlet velocity is **not** loaded or used: `attributes` is
    set to `["press"]`, so each sample carries only the surface mesh
    (`vertices`, scaled into [0, 1] with one dataset-wide bounding box
    shared by all cars, not a per-car box), the signed distance function
    evaluated over the regular 3D query cube (`distance`), the query grid
    itself (`query_points`), the triangle areas, and the pressure targets.
    The pressure targets are standardised with one scalar mean and std that
    `MeshDataModule` fits on all loaded cars, train and test together.
    Downstream, `GINOCFDDataProcessor` passes `x=None` - there is no input
    function beyond the geometry and its SDF.

    Pressure is loaded per mesh vertex and sliced on load to
    `press[:, 0:16]` concatenated with `press[:, 112:]`, dropping the 96
    entries at indices 16-111 so the target width matches the vertex count.
    The reason those particular indices are dropped is not recorded anywhere in
    this repository.

    Data is also stored on Zenodo: https://zenodo.org/records/13936501
    ("Three-dimensional flow dataset over ShapeNet-Car", CC-BY-4.0). Note this
    is the ShapeNet-Car dataset, not the parametric Ahmed-body dataset also used
    in the GINO paper.

    Parameters
    ----------
    root_dir : Union[str, Path]
        root directory at which data is stored.
    n_train : int, optional
        Number of training instances to load, by default 1
    n_test : int, optional
        Number of testing instances to load, by default 1
    query_res : sequence of int, optional
        Dimension-wise resolution of signed distance function
        (SDF) query cube, by default [32,32,32]
    download : bool, optional
        Whether to download data from Zenodo, by default True

    Attributes
    ----------
    train_data: dict
        dictionary of training examples
    test_data: dict
        dictionary of testing examples
    normalizers: dict
        normalizers for data

    References
    ----------
    [1] : Umetani, N. and Bickel, B. (2018). "Learning three-dimensional
        flow for interactive aerodynamic design". ACM Transactions on
        Graphics, 2018. https://dl.acm.org/doi/10.1145/3197517.3201325.
    """

    def __init__(
        self,
        root_dir: str | Path,
        n_train: int = 1,
        n_test: int = 1,
        query_res: Sequence[int] = (32, 32, 32),
        download: bool = True,
    ) -> None:
        """Initialize the CarCFDDataset."""
        self.zenodo_record_id = "13936501"

        if isinstance(root_dir, str):
            root_dir = Path(root_dir).expanduser().resolve()
        logger.debug("root_dir", root_dir)

        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        if download:
            download_from_zenodo_record(record_id=self.zenodo_record_id, root=root_dir)

        # Initialize mesh datamodule
        super().__init__(
            root_dir=root_dir,
            item_dir_name="processed-car-pressure-data",
            n_train=n_train,
            n_test=n_test,
            query_res=list(query_res),
            attributes=["press"],
        )

        # process data list to remove specific vertices from pressure to
        # match number of vertices
        for i, data in enumerate(self.train_data.data_list):
            press = data["press"]
            self.train_data.data_list[i]["press"] = jnp.concatenate(
                (press[:, 0:16], press[:, 112:]), axis=1
            )
        for i, data in enumerate(self.test_data.data_list):
            press = data["press"]
            self.test_data.data_list[i]["press"] = jnp.concatenate(
                (press[:, 0:16], press[:, 112:]), axis=1
            )


def load_mini_car() -> dict:
    """Load the 3-example mini Car-CFD dataset we package along with our module.

    See `neuralop.data.datasets.CarCFDDataset` for more detailed references
    """
    return _load_pt(
        (get_project_root() / "scirex/data/datasets/data/mini_car.pt").as_posix()
    )
