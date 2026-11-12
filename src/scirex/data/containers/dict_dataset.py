# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Dictionary-valued dataset that stores each sample as a dict of named fields."""


class DictDataset:
    """DictDataset is a basic dataset form that stores each batch
    as a dictionary of tensors or other data structures


    """

    def __init__(
        self,
        data_list: list[dict],
        constant: dict | None = None,
    ) -> None:
        """Create the dataset.

        Parameters
        ----------
        data_list : List[dict]
            list of individual batch dictionaries
        constant : dict, optional
            if each data batch shares some constant valued key/val pairs,
            they can be stored in constant for simplicity
        """
        self.data_list = data_list
        self.constant = constant

    def __getitem__(self, index: int) -> dict:
        """Return a copy of sample `index` plus the shared `constant` entries."""
        return_dict = self.data_list[
            index
        ].copy()  # Make a copy to avoid modifying the original

        if self.constant is not None:
            return_dict.update(self.constant)

        return return_dict

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data_list)
