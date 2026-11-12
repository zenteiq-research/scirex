# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The torch-free ``.pt`` loader must refuse pickles that name arbitrary globals."""

import io
import os
import pickle
import zipfile
from pathlib import Path

import pytest

from scirex.data.containers.pt_dataset import _load_pt, _PtUnpickler


class _Touch:
    """Pickles to a call that creates ``path`` if it is ever executed."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self) -> tuple:
        return (os.system, (f"touch {self.path}",))


def _write_malicious_pt(pt_path: Path, payload: object) -> None:
    with zipfile.ZipFile(pt_path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))


def test_load_pt_blocks_code_execution(tmp_path: Path) -> None:
    sentinel = tmp_path / "pwned"
    pt = tmp_path / "evil.pt"
    _write_malicious_pt(pt, _Touch(sentinel.as_posix()))

    with pytest.raises(pickle.UnpicklingError):
        _load_pt(pt.as_posix())
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("os", "system"),
        ("posix", "system"),
        ("builtins", "eval"),
        ("subprocess", "Popen"),
    ],
)
def test_find_class_rejects_unlisted_globals(module: str, name: str) -> None:
    up = _PtUnpickler(io.BytesIO(b""), {})
    with pytest.raises(pickle.UnpicklingError):
        up.find_class(module, name)


def test_find_class_allows_ordered_dict() -> None:
    import collections

    up = _PtUnpickler(io.BytesIO(b""), {})
    assert up.find_class("collections", "OrderedDict") is collections.OrderedDict
