# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

.PHONY: install lint format type-check test test-unit test-integration clean docker-dev docker-prod docker-buildx-cpu docker-buildx-cpu-local docker-build-cpu docker-build-cuda12

IMAGE ?= scirex
VERSION := $(shell grep -m1 '^version = ' pyproject.toml | sed -E 's/^version = "([^"]+)"/\1/')
REVISION := $(shell git rev-parse HEAD)

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

type-check:
	mypy src/scirex/

# Package tests live under tests/ (mirroring src/scirex/); the benchmark-script
# tests live in scripts/tests/. There is no per-test unit vs. integration
# marker. "unit" here means the real distinction the pytest.ini markers
# describe: JIT disabled (fast, catches logic bugs) vs. JIT-compiled (slower,
# also exercises the XLA-compiled path). scripts/tests/ runs in its own
# process (see ci.yml): several scripts flip jax_enable_x64 at import, which
# would corrupt scirex's float64 tests if collected together.
test-unit:
	JAX_DISABLE_JIT=1 pytest tests/ -v --tb=short -m "not gpu"

test-integration:
	pytest tests/ -v --tb=short -m "not gpu"

test:
	pytest tests/ -v --tb=short -m "not gpu" --cov=scirex --cov-report=term-missing
	pytest scripts/tests/ -v --tb=short -m "not gpu"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type d -name htmlcov      -exec rm -rf {} +
	rm -f .coverage coverage.xml

docker-dev:
	HOST_UID=$$(id -u) HOST_GID=$$(id -g) docker compose up --build dev

docker-prod:
	HOST_UID=$$(id -u) HOST_GID=$$(id -g) docker compose up --build prod

# amd64+arm64 cpu image, needs push: buildx can't --load a multi-platform
# result into the local daemon, only a single platform at a time. Override
# IMAGE with a real registry path (not decided yet -- see PKG-11) before
# running, e.g. make docker-buildx-cpu IMAGE=ghcr.io/zenteiq/scirex.
# Requires a builder with both platforms registered (QEMU/binfmt for
# whichever arch isn't this host's own) -- the default docker-driver
# builder only has the native platform, so this also needs:
#   docker buildx create --use --name scirex-multiarch
docker-buildx-cpu:
	docker buildx build --platform linux/amd64,linux/arm64 \
		--build-arg VARIANT=cpu --build-arg VERSION=$(VERSION) --build-arg REVISION=$(REVISION) \
		--target prod -f docker/Dockerfile -t $(IMAGE):$(VERSION)-cpu --push .

# Native-arch-only cpu build, loaded into the local daemon for testing --
# what to use on a machine without the other platform's QEMU/binfmt set up.
docker-buildx-cpu-local:
	docker buildx build \
		--build-arg VARIANT=cpu --build-arg VERSION=$(VERSION) --build-arg REVISION=$(REVISION) \
		--target prod -f docker/Dockerfile -t $(IMAGE):$(VERSION)-cpu --load .

# Single-arch release tags for this host -- $(IMAGE):$(VERSION)-cpu and
# $(IMAGE):$(VERSION)-cuda12, e.g. scirex:1.0.0-cpu. No "latest"
# tag: a release candidate shouldn't become the implicit default pull.
docker-build-cpu:
	docker build --build-arg VARIANT=cpu --build-arg VERSION=$(VERSION) --build-arg REVISION=$(REVISION) \
		--target prod -f docker/Dockerfile -t $(IMAGE):$(VERSION)-cpu .

docker-build-cuda12:
	docker build --build-arg VARIANT=cuda12 --build-arg VERSION=$(VERSION) --build-arg REVISION=$(REVISION) \
		--target prod -f docker/Dockerfile -t $(IMAGE):$(VERSION)-cuda12 .
