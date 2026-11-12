# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

# Selects which JAX accelerator variant the dev/prod stages below build on.
# One Dockerfile covers cpu/cuda12 (they differ only in one RUN line) instead
# of near-duplicate files that could drift apart; must be declared before the
# first FROM to be usable inside a later FROM.
# cuda13 is deferred: jax==0.6.2, the pinned version everywhere else in this
# release, has no "cuda13" extra (added in jax 0.7.0) -- pip installs it
# silently as a no-op CPU fallback instead of failing, so it would ship as a
# mislabeled image. Revisit once the release's jax pin moves past 0.6.2.
ARG VARIANT=cuda12

# ── base: Python 3.10, system deps, non-root user, accelerator-independent
# runtime deps. Base image pinned by digest (multi-arch index: resolves to
# amd64 or arm64 automatically) rather than a mutable tag. ──
FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Non-root user shared by the dev and prod stages (VA findings F-01 and the
# Oct release missing-USER finding). UID/GID default to 1000, the first user
# on most Linux hosts; pass HOST_UID/HOST_GID (see docker-compose.yml) so files
# written to the bind-mounted checkout in dev stay owned by the host user.
# --create-home gives wandb/jax/matplotlib a real $HOME to cache into instead
# of failing on /nonexistent.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} scirex \
    && useradd --create-home --shell /bin/bash --uid ${UID} --gid ${GID} scirex

# Debian mirrors can answer plain-HTTP apt requests with 403 on some networks
# (seen on the Spark); HTTPS works everywhere, so switch the sources first.
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
# requirements.lock is generated from requirements.txt via pip-compile
# (--generate-hashes -c requirements.txt, run against a 3.10 image so the
# resolution matches this interpreter):
#   docker run --rm -v "$PWD":/workspace -w /workspace python:3.10-slim \
#     bash -c 'pip install pip-tools && pip-compile --generate-hashes \
#       --allow-unsafe -c requirements.txt -o requirements.lock \
#       <(grep -Ev "^(jax|jaxlib|open3d)==" requirements.txt)'
# --require-hashes verifies every downloaded wheel against a pinned hash
# instead of trusting the index at build time. jax/jaxlib land here too
# (pulled in transitively by flax/jaxwt/optax/orbax-checkpoint, pinned to
# 0.6.2 by the requirements.txt constraint) -- leave them in rather than
# filtering them out as before: --require-hashes mode refuses to resolve
# anything not explicitly listed, so removing them would break every
# package here that depends on jax. The accelerator stages below then
# find jax/jaxlib already satisfied and only add what's missing (cuda12's
# plugin) or nothing (cpu).
# open3d has no hash-locked entry -- its pin needs a *range*
# (>=0.19,<0.21), not an exact version: 0.19.0 has no linux-aarch64 wheel
# (0.20.0 does), so the range is what resolves on both x86_64 and aarch64,
# and --require-hashes can't be combined with a range in the same install.
RUN pip install --require-hashes -r requirements.lock && \
    pip install "open3d>=0.19,<0.21"

# Benchmark extras (pyproject `benchmark` extra): vtk (Car-CFD, AirfRANS, ShapeNetCar),
# meshio and pandas (Cavity). open3d needs libgfortran5, libgomp1 and OpenGL/EGL
# (libgl1, libegl1) at import time; the x86_64 open3d 0.20 wheel also links
# libusb-1.0 (RealSense support), so it fails to import without libusb-1.0-0.
# A separate layer keeps the large JAX/requirements layer above cached;
# numpy is re-pinned so none of these can move it off the tested 1.26.4.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgfortran5 \
        libgomp1 \
        libgl1 \
        libegl1 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "vtk>=9.0" "meshio>=5.0" "pandas>=2.0" "numpy==1.26.4"

# ── accelerator variants: the only thing that differs between cpu/cuda12 ──
FROM base AS cpu
RUN pip install "jax==0.6.2" "jaxlib==0.6.2"

FROM base AS cuda12
RUN pip install "jax[cuda12]==0.6.2"

# ── dev: editable install + dev tools ────────────────────────────────────
# Resolves to whichever accelerator stage above VARIANT names, e.g.
# --build-arg VARIANT=cpu --target dev.
FROM ${VARIANT} AS dev

COPY requirements_dev.txt ./
# requirements_dev.txt starts with "-r requirements.txt" which was already
# handled in the base stage; skip that line and install only the dev extras.
RUN grep -Ev "^(-r|jax==|jaxlib==)" requirements_dev.txt | pip install -r /dev/stdin

# --chown at copy time avoids a recursive chown that would duplicate every
# copied file in a second layer.
COPY --chown=scirex:scirex . .
# --no-deps: all deps were installed above; editable link points to /workspace
# so host-mounted source is picked up automatically at runtime.
RUN pip install -e . --no-deps

USER scirex
ENV HOME=/home/scirex

# OCI image metadata, applied last so VERSION/REVISION -- which differ on
# every real build -- don't invalidate the cached layers above (jax, apt,
# vtk/meshio/pandas) that a LABEL earlier in the file would bust. No
# sensible default -- pass them at build time, e.g.:
#   --build-arg VERSION=1.0.0 --build-arg REVISION=$(git rev-parse HEAD)
ARG VERSION=unknown
ARG REVISION=unknown
LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/zenteiq-research/scirex" \
      org.opencontainers.image.licenses="LGPL-3.0-only" \
      org.opencontainers.image.authors="research@zenteiq.ai"

CMD ["bash"]

# ── prod: non-editable install ────────────────────────────────────────────
# Same VARIANT selection as dev, e.g. --build-arg VARIANT=cpu --target prod.
FROM ${VARIANT} AS prod

COPY --chown=scirex:scirex . .
# Installs the wheel built just before `docker build`
# (`rm -rf dist && python -m build --wheel`) instead of building scirex
# from the copied source tree, so prod runs the exact artifact that was
# built and can be hashed/verified, not a rebuild of it. The rest of the
# tree (scripts/, config/) still needs to be copied above -- scripts/ in
# particular isn't part of the installed package (see pyproject.toml's
# packages = ["src/scirex"]) but is how the release-day smoke tests and
# the paper-number regeneration are actually run inside this image.
RUN pip install dist/*.whl --no-deps

# The compilers from the base stage are needed to install the dependencies, not to run
# ScireX: remove them from the runtime image. (The dev stage keeps them.)
RUN apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

# Run as the non-root user created in the base stage (VA finding F-01).
USER scirex
ENV HOME=/home/scirex

# VA finding F-02: no liveness signal existed. CMD is an interactive/batch
# bash shell, not an HTTP service, so there's no endpoint to poll -- this
# checks that the installed package still imports, which catches a
# corrupted environment (the failure mode actually possible here).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import scirex" || exit 1

# OCI image metadata, applied last so VERSION/REVISION -- which differ on
# every real build -- don't invalidate the cached layers above (jax, apt,
# vtk/meshio/pandas) that a LABEL earlier in the file would bust. No
# sensible default -- pass them at build time, e.g.:
#   --build-arg VERSION=1.0.0 --build-arg REVISION=$(git rev-parse HEAD)
ARG VERSION=unknown
ARG REVISION=unknown
LABEL org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.source="https://github.com/zenteiq-research/scirex" \
      org.opencontainers.image.licenses="LGPL-3.0-only" \
      org.opencontainers.image.authors="research@zenteiq.ai"

CMD ["bash"]
