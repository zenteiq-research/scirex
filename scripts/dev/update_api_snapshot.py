# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Regenerate ``tests/api_snapshot.json`` from the live public API.

Run this after an intentional change to the ``__all__`` of a public package::

    python scripts/dev/update_api_snapshot.py

and commit the updated snapshot together with the change.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from tests.public_api_helpers import (  # noqa: E402
    SNAPSHOT_PATH,
    live_surface,
    render_snapshot,
)


def main() -> None:
    """Write the snapshot file."""
    SNAPSHOT_PATH.write_text(render_snapshot(live_surface()), encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
