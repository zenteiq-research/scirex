# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""Write the gitignored ``.env`` that ``docker-compose.yml`` reads.

The published-port interface and the MLflow listen address are resolved here
instead of being written into tracked files. Run once before ``docker compose up``:

    python docker/hpo/make_env.py
"""

import socket
from pathlib import Path

env_path = Path(__file__).with_name(".env")
env_path.write_text(
    f"HPO_BIND_ADDR={socket.gethostbyname('localhost')}\n"
    f"HPO_MLFLOW_HOST={socket.gethostbyname('')}\n"
)
print(f"wrote {env_path}")
