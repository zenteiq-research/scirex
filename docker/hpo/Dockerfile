# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

# MLflow tracking server image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep the minimum in step with the `hpo` extra in pyproject.toml.
RUN pip install "mlflow>=2.10" "psycopg[binary]>=3.1"

# No git in the image; stop MLflow warning about it on every run.
ENV GIT_PYTHON_REFRESH=quiet

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

# Matches the compose healthcheck; assumes the server listens on port 5000.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
