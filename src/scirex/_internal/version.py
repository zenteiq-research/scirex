# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Where `scirex.__version__` comes from.

The version is written once, as `version` in the `[project]` table of
`pyproject.toml`. The build backend copies it into the package metadata, and
`scirex.__version__` reads it back from there, so the two cannot disagree. After
changing the version in `pyproject.toml` of a development install, run
`pip install -e .` again so the metadata catches up.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

#: name of the distribution on PyPI (the import package is `scirex`)
DISTRIBUTION = "scirex"

#: reported when the version cannot be determined at all
UNKNOWN = "0+unknown"

_VERSION_LINE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
_NAME_LINE = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)


def version_from_pyproject(path: Path) -> str | None:
    """The `version` of the `scirex` project in `path`, or `None`.

    Only used when the package is not installed (a source checkout run through
    `PYTHONPATH=src`); `None` if the file is missing or belongs to another project.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    name, version = _NAME_LINE.search(text), _VERSION_LINE.search(text)
    if name is None or version is None or name.group(1) != DISTRIBUTION:
        return None
    return version.group(1)


def package_version() -> str:
    """The installed version of ScireX, from its package metadata.

    Falls back to `pyproject.toml` next to `src/` when the package is not installed,
    and to `UNKNOWN` when that is not there either.
    """
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        pass
    checkout = Path(__file__).resolve().parents[3] / "pyproject.toml"
    return version_from_pyproject(checkout) or UNKNOWN
