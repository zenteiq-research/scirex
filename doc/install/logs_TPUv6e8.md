<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Install log: TPU v6e-8

The platform check on a TPU VM, run with `scripts/dev/check_platform.py --backend tpu`.

| Item | Value |
| :---- | :---- |
| Date | 2026-11-06 |
| Repository | scirex-zenteiq |
| Host | `<host>` (TPU VM, zone asia-south1-c, accelerator v6e-8) |
| Script | `scripts/dev/check_platform.py --backend tpu` |
| Raw log on the VM | `~/scirex-zenteiq/check_tpu.log` |

## 1. Result

| Item | Outcome |
| :---- | :---- |
| TPU detected by JAX | PASS: backend tpu, 8 devices |
| Run really on TPU | PASS: requested tpu, running on TPU_0; arrays on tpu for every model |
| Transolver train / save / load / infer | PASS |
| FNO train / save / load / infer | PASS |
| GINO train / save / load / infer | PASS |
| WNO train / save / load / infer | PASS |
| Public API lookup (73 names) | PASS: 0 problems |

**Summary.** All four model workflows passed on the TPU. The module import check is not included in this log.

What this does not show:

* It used one chip (TPU_0). The script has no multi-device sharding, so use of all 8 chips is untested.
* It is a smoke test (dummy data, 2 training steps). Speed and accuracy on real data are untested.

## 2. Environment

| Item | Value |
| :---- | :---- |
| OS | Linux 6.8.0-1015-gcp, x86_64, glibc 2.35 |
| Python | 3.10.12 (repo supports 3.10 to 3.12) |
| Virtual environment | `~/tpu-env` |
| scirex-zenteiq | 1.0.0rc1 |
| jax / jaxlib | 0.6.2 / 0.6.2 |
| flax | 0.10.7 |
| optax | 0.2.8 |
| numpy / scipy | 1.26.4 / 1.15.3 |
| jaxwt | 0.1.1 |
| libtpu (from tpu-info) | 0.0.17 |
| TPU | 8 x TPU v6e, about 31 GiB HBM per chip |

Install used:

```bash
pip install -e ".[dev,tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## 3. Check output (key parts of check_tpu.log)

```text
=== [1] Platform and compatibility
  Python      : 3.10.12 (CPython, 64-bit) at ~/tpu-env/bin/python
  JAX backend : tpu
  JAX devices : 8 x TpuDevice (id 0..7, coords (0,0,0) .. (1,3,0), core_on_chip=0)
  -> PASS  jax runs         (backend tpu, 8 device(s))
  -> PASS  backend          (requested tpu, running on TPU_0(process=0,(0,0,0,0)))
  -> PASS  jit computation  (a jitted sum of squares gave 14.0)

=== [3] Public classes and functions
  -> PASS  public API (73 names: 53 classes, 20 functions, 0 problems)
```

Section 4 of the script (end-to-end workflow, 2 training steps, checkpoint, reload, inference):

| Model | Params | Loss, step 1 to final | Restored params identical | Arrays on | Max diff |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Transolver | 4,029 | 0.686541 to 0.433204 | True | tpu | 0.00e+00 |
| FNO | 6,817 | 0.383778 to 0.376362 | True | tpu | 0.00e+00 |
| GINO | 19,452 | 4.584443 to 3.690735 | True | tpu | 0.00e+00 |
| WNO | 19,897 | 1.986885 to 1.874952 | True | tpu | 0.00e+00 |

GINO runs without jit (data-dependent neighbour search): step 1 took 13.7 s, step 2 took 1.0 s. It also printed expected warnings that int64 was truncated to int32 because `jax_enable_x64` is off (`neighbor_search.py:248`).

Summary section of the script:

```text
  PASS  python version       3.10 is in the supported range 3.10 to 3.12
  PASS  operating system     Linux x86_64
  PASS  jax runs             backend tpu, 8 device(s)
  PASS  backend              requested tpu, running on TPU_0(process=0,(0,0,0,0))
  PASS  jit computation      a jitted sum of squares gave 14.0
  PASS  public API           73 names looked up, 0 problems
  PASS  Transolver workflow  train, save, load, infer
  PASS  FNO workflow         train, save, load, infer
  PASS  GINO workflow        train, save, load, infer
  PASS  WNO workflow         train, save, load, infer
```

## 4. Commands to reproduce

```bash
source ~/tpu-env/bin/activate && cd ~/scirex-zenteiq
mkdir -p ~/tpu_logs && export TPU_LOG_DIR=~/tpu_logs TPU_MIN_LOG_LEVEL=3 TF_CPP_MIN_LOG_LEVEL=3
python -c "import jax; print(jax.default_backend(), jax.device_count())"   # expect: tpu 8
```
