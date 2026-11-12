# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

import sys
from pathlib import Path

REQUIRED_HEADER = "SPDX-License-Identifier: LGPL-3.0-only"


def check_file(path: Path) -> bool:
    first_lines = path.read_text().splitlines()[:5]
    return any(REQUIRED_HEADER in line for line in first_lines)


def main() -> None:
    files = [Path(f) for f in sys.argv[1:] if f.endswith(".py")]
    failures = [f for f in files if not check_file(f)]
    if failures:
        print(f"Missing SPDX licence header (expected '{REQUIRED_HEADER}') in:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
