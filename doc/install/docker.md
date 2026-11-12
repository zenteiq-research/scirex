<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Use Docker

Docker gives a ready environment with every dependency installed. The repository has two
images: `dev` (editable install and development tools) and `prod` (non-editable install,
runs as a non-root user).

## Requirements

- Docker 24 or newer.
- For NVIDIA GPUs: the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  on the host, with a driver that supports CUDA 12 (525 or newer).

## Build the images

From the repository root. The default image uses JAX with CUDA 12 (`VARIANT=cuda12`):

```bash
docker build --target dev -t scirex-dev -f docker/Dockerfile .
```

The `prod` image installs ScireX from a wheel, so build the wheel first (`dist/` is not in a
fresh clone, and the `prod` build fails without it):

```bash
pip install build
rm -rf dist && python -m build --wheel
docker build --target prod -t scirex-prod -f docker/Dockerfile .
```

For a machine without an NVIDIA GPU, build the CPU variant instead. It uses plain `jax` and
is smaller than the CUDA image:

```bash
docker build --target dev  --build-arg VARIANT=cpu -t scirex-dev-cpu  -f docker/Dockerfile .
docker build --target prod --build-arg VARIANT=cpu -t scirex-prod-cpu -f docker/Dockerfile .
```

The container user has UID and GID 1000. If your host user differs, add
`--build-arg UID=$(id -u) --build-arg GID=$(id -g)` so files written to a mounted folder keep
your ownership.

## Run

A shell with the GPU, your data (read-only) and a folder for checkpoints:

```bash
docker run --rm -it --gpus all \
    -v "$PWD/data":/workspace/data:ro \
    -v "$PWD/checkpoints":/workspace/checkpoints \
    scirex-prod
```

CPU only: use an image built with `VARIANT=cpu` (see above) and leave out `--gpus all`.

```bash
docker run --rm -it scirex-prod-cpu python -c "import scirex; print(scirex.__version__)"
```

For development, mount the source so edits on the host are used in the container:

```bash
docker run --rm -it --gpus all -v "$PWD":/workspace scirex-dev
```

## Docker Compose

`docker compose run --rm dev` and `docker compose run --rm prod` do the same with the
mounts from `docker-compose.yml`. Both services request all NVIDIA GPUs, so on a host
without the NVIDIA Container Toolkit use `docker run` as above. Set `HOST_UID` and `HOST_GID` so files written to the
mounts keep your ownership:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
```

## Next

- [Verify the install](verify.md)
- [Troubleshooting](troubleshooting.md)
