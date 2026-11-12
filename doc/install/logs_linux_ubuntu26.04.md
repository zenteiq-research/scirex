<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Install log: Ubuntu 26.04.1 LTS

The output of the [Linux install](index.md) on one machine. The package is installed from a wheel built from the
checkout, the same file that pip installs from PyPI after the release.

| | |
| --- | --- |
| Operating system | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, x86_64 |
| CPU | Intel(R) Core(TM) i5-14600K, 20 threads |
| Memory | 15 GB |
| GPU | none (nvidia-smi not found) |
| Python | 3.12.16 |
| Install | `pip install dist/scirex-1.0.0rc1-py3-none-any.whl` |
| Date | 2026-11-11 08:37 UTC |
| Result | every step passed |

## Python

```bash
python3.12 --version
```

```text
Python 3.12.16
```

Exit status 0, 0s.

## Create the virtual environment

```bash
python3.12 -m venv scirex-venv
```

```text
(no output)
```

Exit status 0, 1s.

## Upgrade pip

```bash
source scirex-venv/bin/activate && python -m pip install --upgrade pip
```

```text
Requirement already satisfied: pip in ./scirex-venv/lib/python3.12/site-packages (25.0.1)
Collecting pip
  Using cached pip-26.3-py3-none-any.whl.metadata (4.6 kB)
Using cached pip-26.3-py3-none-any.whl (2.0 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 25.0.1
    Uninstalling pip-25.0.1:
      Successfully uninstalled pip-25.0.1
Successfully installed pip-26.3
```

Exit status 0, 1s.

## Build the wheel from the checkout

```bash
source scirex-venv/bin/activate && pip wheel --no-deps --progress-bar off -w dist <repo> && ls dist
```

```text
Processing <repo>
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Building wheels for collected packages: scirex
  Building wheel for scirex (pyproject.toml): started
  Building wheel for scirex (pyproject.toml): finished with status 'done'
  Created wheel for scirex: filename=scirex-1.0.0rc1-py3-none-any.whl size=362245 sha256=1f2b4a1261c7f43ce6794e039d60952b7f5819c602b3721346b6b8ea68e4ff94
  Stored in directory: /tmp/pip-ephem-wheel-cache-XXXXXXXX/wheels/c7/b6/9a/f93b95af8d7da9603e9801ff473c9cb8ab6b7e2e746bf300ae
Successfully built scirex
scirex-1.0.0rc1-py3-none-any.whl
```

Exit status 0, 1s.

## Install ScireX

```bash
source scirex-venv/bin/activate && WHEEL=$(ls dist/scirex-*.whl) && pip install --progress-bar off $WHEEL
```

```text
Processing ./dist/scirex-1.0.0rc1-py3-none-any.whl
Collecting einops<0.9,>=0.8.2 (from scirex==1.0.0rc1)
  Using cached einops-0.8.2-py3-none-any.whl.metadata (13 kB)
Collecting flax<0.11,>=0.10.7 (from scirex==1.0.0rc1)
  Using cached flax-0.10.7-py3-none-any.whl.metadata (11 kB)
Collecting jax<0.7,>=0.6.2 (from scirex==1.0.0rc1)
  Using cached jax-0.6.2-py3-none-any.whl.metadata (13 kB)
Collecting jaxlib<0.7,>=0.6.2 (from scirex==1.0.0rc1)
  Using cached jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl.metadata (1.3 kB)
Collecting jaxwt<0.2,>=0.1.1 (from scirex==1.0.0rc1)
  Using cached jaxwt-0.1.1-py3-none-any.whl.metadata (7.2 kB)
Collecting matplotlib<4,>=3.9.4 (from scirex==1.0.0rc1)
  Using cached matplotlib-3.11.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (80 kB)
Collecting numpy<2,>=1.26.4 (from scirex==1.0.0rc1)
  Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Collecting opt-einsum<4,>=3.4 (from scirex==1.0.0rc1)
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting optax<0.3,>=0.2.8 (from scirex==1.0.0rc1)
  Using cached optax-0.2.8-py3-none-any.whl.metadata (7.9 kB)
Collecting requests<3,>=2.34.2 (from scirex==1.0.0rc1)
  Using cached requests-2.34.2-py3-none-any.whl.metadata (4.8 kB)
Collecting scikit-learn<2,>=1.7.2 (from scirex==1.0.0rc1)
  Using cached scikit_learn-1.9.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.3 kB)
Collecting scipy<2,>=1.15.3 (from scirex==1.0.0rc1)
  Using cached scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting tensorly<0.10,>=0.9 (from scirex==1.0.0rc1)
  Using cached tensorly-0.9.0-py3-none-any.whl.metadata (8.6 kB)
Collecting zencfg<0.8,>=0.7 (from scirex==1.0.0rc1)
  Using cached zencfg-0.7.0-py3-none-any.whl.metadata (9.9 kB)
Collecting msgpack (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached msgpack-1.2.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.3 kB)
Collecting orbax-checkpoint (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached orbax_checkpoint-0.12.9-py3-none-any.whl.metadata (3.3 kB)
Collecting tensorstore (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached tensorstore-0.1.86-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (3.3 kB)
Collecting rich>=11.1 (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached rich-15.0.0-py3-none-any.whl.metadata (18 kB)
Collecting typing_extensions>=4.2 (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached typing_extensions-4.16.0-py3-none-any.whl.metadata (3.3 kB)
Collecting PyYAML>=5.4.1 (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting treescope>=0.1.7 (from flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached treescope-0.1.11-py3-none-any.whl.metadata (6.7 kB)
Collecting ml_dtypes>=0.5.0 (from jax<0.7,>=0.6.2->scirex==1.0.0rc1)
  Using cached ml_dtypes-0.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting PyWavelets (from jaxwt<0.2,>=0.1.1->scirex==1.0.0rc1)
  Using cached pywavelets-1.10.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (7.6 kB)
Collecting contourpy>=1.0.1 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached contourpy-1.4.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (3.8 kB)
Collecting cycler>=0.10 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.28.2 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached fonttools-4.68.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (139 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached kiwisolver-1.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (5.2 kB)
Collecting packaging>=20.0 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached packaging-26.3-py3-none-any.whl.metadata (3.5 kB)
Collecting pillow>=9 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.1 kB)
Collecting pyparsing>=3 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached pyparsing-3.3.3-py3-none-any.whl.metadata (5.9 kB)
Collecting python-dateutil>=2.7 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached python_dateutil-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting absl-py>=0.7.1 (from optax<0.3,>=0.2.8->scirex==1.0.0rc1)
  Using cached absl_py-2.5.1-py3-none-any.whl.metadata (3.3 kB)
Collecting charset_normalizer<4,>=2 (from requests<3,>=2.34.2->scirex==1.0.0rc1)
  Using cached charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (46 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.34.2->scirex==1.0.0rc1)
  Using cached idna-3.20-py3-none-any.whl.metadata (7.2 kB)
Collecting urllib3<3,>=1.26 (from requests<3,>=2.34.2->scirex==1.0.0rc1)
  Using cached urllib3-2.8.0-py3-none-any.whl.metadata (7.4 kB)
Collecting certifi>=2023.5.7 (from requests<3,>=2.34.2->scirex==1.0.0rc1)
  Using cached certifi-2026.10.23-py3-none-any.whl.metadata (2.5 kB)
Collecting joblib>=1.4.0 (from scikit-learn<2,>=1.7.2->scirex==1.0.0rc1)
  Using cached joblib-1.6.0-py3-none-any.whl.metadata (6.5 kB)
Collecting narwhals>=2.0.1 (from scikit-learn<2,>=1.7.2->scirex==1.0.0rc1)
  Using cached narwhals-2.28.0-py3-none-any.whl.metadata (15 kB)
Collecting threadpoolctl>=3.5.0 (from scikit-learn<2,>=1.7.2->scirex==1.0.0rc1)
  Using cached threadpoolctl-3.8.0-py3-none-any.whl.metadata (29 kB)
Collecting scipy<2,>=1.15.3 (from scirex==1.0.0rc1)
  Using cached scipy-1.18.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
INFO: pip is looking at multiple versions of scipy to determine which version is compatible with other requirements. This could take a while.
  Using cached scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting pydantic (from zencfg<0.8,>=0.7->scirex==1.0.0rc1)
  Using cached pydantic-2.14.3-py3-none-any.whl.metadata (141 kB)
Collecting contourpy>=1.0.1 (from matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cloudpickle>=3.0 (from joblib>=1.4.0->scikit-learn<2,>=1.7.2->scirex==1.0.0rc1)
  Using cached cloudpickle-3.1.2-py3-none-any.whl.metadata (7.1 kB)
Collecting ml_dtypes>=0.5.0 (from jax<0.7,>=0.6.2->scirex==1.0.0rc1)
  Using cached ml_dtypes-0.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.9 kB)
Collecting tzdata (from python-dateutil>=2.7->matplotlib<4,>=3.9.4->scirex==1.0.0rc1)
  Using cached tzdata-2026.6-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=11.1->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached markdown_it_py-4.3.0-py3-none-any.whl.metadata (7.9 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich>=11.1->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached pygments-2.21.0-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=11.1->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Collecting etils>=1.15.0 (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached etils-1.15.0-py3-none-any.whl.metadata (6.5 kB)
Collecting prometheus-client>=0.20.0 (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached prometheus_client-0.26.0-py3-none-any.whl.metadata (2.1 kB)
Collecting aiofiles (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached aiofiles-25.1.0-py3-none-any.whl.metadata (6.3 kB)
Collecting protobuf (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached protobuf-7.36.2-cp310-abi3-manylinux2014_x86_64.whl.metadata (595 bytes)
Collecting humanize (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached humanize-4.16.0-py3-none-any.whl.metadata (8.0 kB)
Collecting simplejson>=3.16.0 (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached simplejson-4.3.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (3.8 kB)
Collecting psutil (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl.metadata (22 kB)
Collecting uvloop (from orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached uvloop-0.23.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.1 kB)
Collecting fsspec (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached fsspec-2026.9.0-py3-none-any.whl.metadata (10 kB)
Collecting zipp (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached zipp-4.1.2-py3-none-any.whl.metadata (3.6 kB)
Collecting gcsfs>=2026.4.0 (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached gcsfs-2026.10.1-py3-none-any.whl.metadata (4.6 kB)
Collecting aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached aiohttp-3.14.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.3 kB)
Collecting decorator>4.1.2 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached decorator-5.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting google-auth-oauthlib (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_auth_oauthlib-1.5.0-py3-none-any.whl.metadata (2.6 kB)
Collecting google-auth>=1.2 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_auth-2.62.2-py3-none-any.whl.metadata (6.0 kB)
Collecting google-cloud-storage-control (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_cloud_storage_control-1.17.0-py3-none-any.whl.metadata (9.8 kB)
Collecting google-cloud-storage>=3.14.1 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_cloud_storage-3.17.1-py3-none-any.whl.metadata (14 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached aiohappyeyeballs-2.7.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached attrs-26.1.0-py3-none-any.whl.metadata (8.8 kB)
Collecting frozenlist>=1.1.1 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (20 kB)
Collecting multidict<8.0,>=4.5 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached multidict-7.2.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (6.1 kB)
Collecting propcache>=0.2.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached propcache-0.5.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (25 kB)
Collecting yarl<2.0,>=1.25.1 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached yarl-1.25.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (111 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached pyasn1_modules-0.4.2-py3-none-any.whl.metadata (3.5 kB)
Collecting cryptography>=38.0.3 (from google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached cryptography-50.0.2-cp311-abi3-manylinux_2_34_x86_64.whl.metadata (4.4 kB)
Collecting cffi>=2.0.0 (from cryptography>=38.0.3->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached cffi-2.1.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.5 kB)
Collecting pycparser (from cffi>=2.0.0->cryptography>=38.0.3->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached pycparser-3.12-py3-none-any.whl.metadata (9.7 kB)
Collecting google-api-core<3.0.0,>=2.27.0 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_api_core-2.44.0-py3-none-any.whl.metadata (3.2 kB)
Collecting google-cloud-core<3.0.0,>=2.4.2 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_cloud_core-2.8.0-py3-none-any.whl.metadata (2.9 kB)
Collecting google-resumable-media<3.0.0,>=2.7.2 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_resumable_media-2.11.0-py3-none-any.whl.metadata (2.3 kB)
Collecting google-crc32c<2.0.0,>=1.6.0 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached google_crc32c-1.9.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.2 kB)
Collecting googleapis-common-protos<2.0.0,>=1.69.2 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached googleapis_common_protos-1.76.0-py3-none-any.whl.metadata (8.5 kB)
Collecting proto-plus<2.0.0,>=1.26.1 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached proto_plus-1.29.0-py3-none-any.whl.metadata (2.2 kB)
Collecting opentelemetry-api<2.0.0,>=1.44.0 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached opentelemetry_api-1.45.2-py3-none-any.whl.metadata (1.4 kB)
Collecting pyasn1<0.7.0,>=0.6.1 (from pyasn1-modules>=0.2.1->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached pyasn1-0.6.5-py3-none-any.whl.metadata (8.4 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached oauthlib-4.0.1-py3-none-any.whl.metadata (8.6 kB)
Collecting grpcio<2.0.0,>=1.59.0 (from google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached grpcio-1.85.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (3.8 kB)
Collecting grpc-google-iam-v1<1.0.0,>=0.14.2 (from google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached grpc_google_iam_v1-0.14.6-py3-none-any.whl.metadata (8.9 kB)
Collecting grpcio-status<2.0.0,>=1.59.0 (from google-api-core[grpc]<3.0.0,>=2.28.0->google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax<0.11,>=0.10.7->scirex==1.0.0rc1)
  Using cached grpcio_status-1.85.1-py3-none-any.whl.metadata (1.3 kB)
Collecting annotated-types>=0.6.0 (from pydantic->zencfg<0.8,>=0.7->scirex==1.0.0rc1)
  Using cached annotated_types-0.8.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.50.4 (from pydantic->zencfg<0.8,>=0.7->scirex==1.0.0rc1)
  Using cached pydantic_core-2.50.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.6 kB)
Collecting typing-inspection>=0.4.4 (from pydantic->zencfg<0.8,>=0.7->scirex==1.0.0rc1)
  Using cached typing_inspection-0.4.5-py3-none-any.whl.metadata (2.6 kB)
Using cached einops-0.8.2-py3-none-any.whl (65 kB)
Using cached flax-0.10.7-py3-none-any.whl (456 kB)
Using cached jax-0.6.2-py3-none-any.whl (2.7 MB)
Using cached jaxlib-0.6.2-cp312-cp312-manylinux2014_x86_64.whl (89.9 MB)
Using cached jaxwt-0.1.1-py3-none-any.whl (26 kB)
Using cached matplotlib-3.11.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.9 MB)
Using cached numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
Using cached opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Using cached optax-0.2.8-py3-none-any.whl (402 kB)
Using cached requests-2.34.2-py3-none-any.whl (73 kB)
Using cached charset_normalizer-3.5.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (257 kB)
Using cached idna-3.20-py3-none-any.whl (69 kB)
Using cached scikit_learn-1.9.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (9.2 MB)
Using cached scipy-1.17.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (35.2 MB)
Using cached tensorly-0.9.0-py3-none-any.whl (7.4 MB)
Using cached urllib3-2.8.0-py3-none-any.whl (135 kB)
Using cached zencfg-0.7.0-py3-none-any.whl (31 kB)
Using cached absl_py-2.5.1-py3-none-any.whl (137 kB)
Using cached certifi-2026.10.23-py3-none-any.whl (135 kB)
Using cached contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Using cached fonttools-4.68.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (5.4 MB)
Using cached joblib-1.6.0-py3-none-any.whl (306 kB)
Using cached cloudpickle-3.1.2-py3-none-any.whl (22 kB)
Using cached kiwisolver-1.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
Using cached ml_dtypes-0.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (5.0 MB)
Using cached narwhals-2.28.0-py3-none-any.whl (490 kB)
Using cached packaging-26.3-py3-none-any.whl (129 kB)
Using cached pillow-13.0.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (7.1 MB)
Using cached pyparsing-3.3.3-py3-none-any.whl (126 kB)
Using cached python_dateutil-4.0.0-py3-none-any.whl (92 kB)
Using cached pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (807 kB)
Using cached rich-15.0.0-py3-none-any.whl (310 kB)
Using cached pygments-2.21.0-py3-none-any.whl (1.3 MB)
Using cached markdown_it_py-4.3.0-py3-none-any.whl (96 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Using cached threadpoolctl-3.8.0-py3-none-any.whl (32 kB)
Using cached treescope-0.1.11-py3-none-any.whl (186 kB)
Using cached typing_extensions-4.16.0-py3-none-any.whl (45 kB)
Using cached msgpack-1.2.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (475 kB)
Using cached orbax_checkpoint-0.12.9-py3-none-any.whl (1.4 MB)
Using cached etils-1.15.0-py3-none-any.whl (177 kB)
Using cached gcsfs-2026.10.1-py3-none-any.whl (88 kB)
Using cached aiohttp-3.14.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.9 MB)
Using cached multidict-7.2.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (516 kB)
Using cached yarl-1.25.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (117 kB)
Using cached aiohappyeyeballs-2.7.1-py3-none-any.whl (15 kB)
Using cached aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Using cached attrs-26.1.0-py3-none-any.whl (67 kB)
Using cached decorator-5.3.1-py3-none-any.whl (10 kB)
Using cached frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (242 kB)
Using cached fsspec-2026.9.0-py3-none-any.whl (221 kB)
Using cached google_auth-2.62.2-py3-none-any.whl (270 kB)
Using cached cryptography-50.0.2-cp311-abi3-manylinux_2_34_x86_64.whl (4.8 MB)
Using cached cffi-2.1.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (221 kB)
Using cached google_cloud_storage-3.17.1-py3-none-any.whl (345 kB)
Using cached google_api_core-2.44.0-py3-none-any.whl (225 kB)
Using cached google_cloud_core-2.8.0-py3-none-any.whl (31 kB)
Using cached google_crc32c-1.9.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (36 kB)
Using cached google_resumable_media-2.11.0-py3-none-any.whl (81 kB)
Using cached googleapis_common_protos-1.76.0-py3-none-any.whl (307 kB)
Using cached opentelemetry_api-1.45.2-py3-none-any.whl (60 kB)
Using cached proto_plus-1.29.0-py3-none-any.whl (50 kB)
Using cached protobuf-7.36.2-cp310-abi3-manylinux2014_x86_64.whl (343 kB)
Using cached prometheus_client-0.26.0-py3-none-any.whl (64 kB)
Using cached propcache-0.5.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (250 kB)
Using cached pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)
Using cached pyasn1-0.6.5-py3-none-any.whl (84 kB)
Using cached simplejson-4.3.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (213 kB)
Using cached tensorstore-0.1.86-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (21.8 MB)
Using cached aiofiles-25.1.0-py3-none-any.whl (14 kB)
Using cached google_auth_oauthlib-1.5.0-py3-none-any.whl (20 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached oauthlib-4.0.1-py3-none-any.whl (161 kB)
Using cached google_cloud_storage_control-1.17.0-py3-none-any.whl (115 kB)
Using cached grpc_google_iam_v1-0.14.6-py3-none-any.whl (33 kB)
Using cached grpcio-1.85.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (7.4 MB)
Using cached grpcio_status-1.85.1-py3-none-any.whl (14 kB)
Using cached humanize-4.16.0-py3-none-any.whl (137 kB)
Using cached psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl (155 kB)
Using cached pycparser-3.12-py3-none-any.whl (62 kB)
Using cached pydantic-2.14.3-py3-none-any.whl (494 kB)
Using cached pydantic_core-2.50.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (2.3 MB)
Using cached annotated_types-0.8.0-py3-none-any.whl (13 kB)
Using cached typing_inspection-0.4.5-py3-none-any.whl (14 kB)
Using cached pywavelets-1.10.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (4.5 MB)
Using cached tzdata-2026.6-py2.py3-none-any.whl (348 kB)
Using cached uvloop-0.23.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (4.5 MB)
Using cached zipp-4.1.2-py3-none-any.whl (10 kB)
Installing collected packages: zipp, uvloop, urllib3, tzdata, typing_extensions, threadpoolctl, simplejson, PyYAML, pyparsing, pygments, pycparser, pyasn1, psutil, protobuf, propcache, prometheus-client, pillow, packaging, opt-einsum, oauthlib, numpy, narwhals, multidict, msgpack, mdurl, kiwisolver, idna, humanize, google-crc32c, fsspec, frozenlist, fonttools, etils, einops, decorator, cycler, cloudpickle, charset_normalizer, certifi, attrs, annotated-types, aiohappyeyeballs, aiofiles, absl-py, yarl, typing-inspection, treescope, scipy, requests, PyWavelets, python-dateutil, pydantic-core, pyasn1-modules, proto-plus, opentelemetry-api, ml_dtypes, markdown-it-py, joblib, grpcio, googleapis-common-protos, google-resumable-media, contourpy, cffi, aiosignal, tensorstore, tensorly, scikit-learn, rich, requests-oauthlib, pydantic, matplotlib, jaxlib, grpcio-status, cryptography, aiohttp, zencfg, jax, grpc-google-iam-v1, google-auth, optax, jaxwt, google-auth-oauthlib, google-api-core, google-cloud-core, google-cloud-storage-control, google-cloud-storage, gcsfs, orbax-checkpoint, flax, scirex
Successfully installed PyWavelets-1.10.0 PyYAML-6.0.3 absl-py-2.5.1 aiofiles-25.1.0 aiohappyeyeballs-2.7.1 aiohttp-3.14.5 aiosignal-1.4.0 annotated-types-0.8.0 attrs-26.1.0 certifi-2026.10.23 cffi-2.1.1 charset_normalizer-3.5.2 cloudpickle-3.1.2 contourpy-1.3.3 cryptography-50.0.2 cycler-0.12.1 decorator-5.3.1 einops-0.8.2 etils-1.15.0 flax-0.10.7 fonttools-4.68.0 frozenlist-1.8.0 fsspec-2026.9.0 gcsfs-2026.10.1 google-api-core-2.44.0 google-auth-2.62.2 google-auth-oauthlib-1.5.0 google-cloud-core-2.8.0 google-cloud-storage-3.17.1 google-cloud-storage-control-1.17.0 google-crc32c-1.9.0 google-resumable-media-2.11.0 googleapis-common-protos-1.76.0 grpc-google-iam-v1-0.14.6 grpcio-1.85.1 grpcio-status-1.85.1 humanize-4.16.0 idna-3.20 jax-0.6.2 jaxlib-0.6.2 jaxwt-0.1.1 joblib-1.6.0 kiwisolver-1.5.1 markdown-it-py-4.3.0 matplotlib-3.11.2 mdurl-0.1.2 ml_dtypes-0.5.4 msgpack-1.2.4 multidict-7.2.1 narwhals-2.28.0 numpy-1.26.4 oauthlib-4.0.1 opentelemetry-api-1.45.2 opt-einsum-3.4.0 optax-0.2.8 orbax-checkpoint-0.12.9 packaging-26.3 pillow-13.0.0 prometheus-client-0.26.0 propcache-0.5.4 proto-plus-1.29.0 protobuf-7.36.2 psutil-7.2.2 pyasn1-0.6.5 pyasn1-modules-0.4.2 pycparser-3.12 pydantic-2.14.3 pydantic-core-2.50.4 pygments-2.21.0 pyparsing-3.3.3 python-dateutil-4.0.0 requests-2.34.2 requests-oauthlib-2.0.0 rich-15.0.0 scikit-learn-1.9.1 scipy-1.17.1 scirex-1.0.0rc1 simplejson-4.3.0 tensorly-0.9.0 tensorstore-0.1.86 threadpoolctl-3.8.0 treescope-0.1.11 typing-inspection-0.4.5 typing_extensions-4.16.0 tzdata-2026.6 urllib3-2.8.0 uvloop-0.23.0 yarl-1.25.1 zencfg-0.7.0 zipp-4.1.2
```

Exit status 0, 22s.

## Check the installed packages

```bash
source scirex-venv/bin/activate && pip check
```

```text
No broken requirements found.
```

Exit status 0, 0s.

## Import scirex

```bash
source scirex-venv/bin/activate && python -c "import scirex; print(scirex.__version__)"
```

```text
1.0.0rc1
```

Exit status 0, 0s.

## Versions and devices

```bash
source scirex-venv/bin/activate && python -m scirex.info
```

```text
ScireX:  1.0.0rc1
OS:      Linux 7.0.0-38-generic x86_64
backend: cpu
devices: [CpuDevice(id=0)]
x64:     False
flax: 0.10.7
optax: 0.2.8
jax:    0.6.2
jaxlib: 0.6.2
numpy:  1.26.4
python: 3.12.16 (main, Oct 26 2026, 15:28:01) [GCC 15.2.0]
device info: cpu-1, 1 local devices
process_count: 1
platform: uname_result(system='Linux', node='<host>', release='7.0.0-38-generic', version='#38-Ubuntu SMP PREEMPT_DYNAMIC Fri Sep  4 09:10:14 UTC 2026', machine='x86_64')
```

Exit status 0, 1s.

## Verify the install

```bash
source scirex-venv/bin/activate && python <repo>/scripts/dev/verify_install.py
```

```text
python   3.12.16 on Linux-7.0.0-38-generic-x86_64-with-glibc2.43
scirex   1.0.0rc1  ($WORK/scirex-venv/lib/python3.12/site-packages/scirex/__init__.py)
jax      0.6.2, backend cpu, x64 False
device   TFRT_CPU_0
wheels   flax==0.10.7, jax==0.6.2, jaxlib==0.6.2, jaxwt==0.1.1, optax==0.2.8, scirex==1.0.0rc1
compute  jit matmul on {CpuDevice(id=0)} -> 83.0
verify_install: ok
```

Exit status 0, 0s.

## Check the whole workflow

```bash
source scirex-venv/bin/activate && bash <repo>/scripts/dev/check_platform.sh --backend auto --python $(which python)
```

```text
ScireX platform check
------------------------------------------------------------------
step 1: operating system
  detected: Linux, x86_64
  NVIDIA GPU: none found (nvidia-smi is missing); the CPU will be used
  cores: 20
  memory: 14.6 GB total, 6.0 GB available

step 2: python interpreter
  using: $WORK/scirex-venv/bin/python 3.12.16

step 3: library checks (imports, public API, train / save / load / infer)
------------------------------------------------------------------
ScireX platform check

=== [1] Platform and compatibility ========================================
  operating system : Linux 7.0.0-38-generic (Linux-7.0.0-38-generic-x86_64-with-glibc2.43)
  CPU architecture : x86_64
  Python           : 3.12.16 (CPython, 64-bit) at $WORK/scirex-venv/bin/python
      -> PASS  python version  (3.12 is in the supported range 3.10 to 3.12)
      -> PASS  operating system  (Linux x86_64: CPU and NVIDIA GPU (CUDA 12) are supported)
  libraries        :
    scirex           1.0.0rc1
    jax              0.6.2
    jaxlib           0.6.2
    flax             0.10.7
    optax            0.2.8
    numpy            1.26.4
    scipy            1.17.1
    jaxwt            0.1.1
  JAX backend      : cpu
  JAX devices      : [CpuDevice(id=0)]
      -> PASS  jax runs  (backend cpu, 1 device(s))
      -> PASS  backend  (auto: JAX chose cpu (TFRT_CPU_0))
      note: JAX is on the CPU. To test a GPU or a TPU use --backend gpu or --backend tpu.
      -> PASS  jit computation  (a jitted sum of squares gave 14.0)

=== [2] Import every module ===============================================
  scirex 1.0.0rc1 from <repo>/src/scirex
  found 114 modules; importing each one
    optional  scirex.experimental.tuning.tuner: needs mlflow
      -> PASS  import all modules  (113 imported, 1 need or cannot load an optional package, 0 failed)
      -> WARN  optional packages  (1 module(s) skipped; install the extras to test them: pip install -e '.[all]')

=== [3] Public classes and functions ======================================
  scirex: 5 exported names
  scirex.models: 8 exported names
  scirex.layers: 17 exported names
  scirex.losses: 8 exported names
  scirex.data: 10 exported names
  scirex.training: 1 exported names
  scirex.config: 22 exported names
  scirex.utils: 2 exported names
      -> PASS  public API  (73 names looked up: 53 classes, 20 functions, 0 problems)

=== [4] End-to-end workflow (2 training steps, checkpoint, inference) =====

  --- Transolver --------------------------------------------------
  [Transolver] dummy data: 2 samples of 50 random 2D points with 1 feature, target = feature squared
  [Transolver] model built in 0.2s
  [Transolver] initialising parameters
  [Transolver] 4,029 parameters, initialised in 1.4s
  [Transolver] optimizer: adam
  [Transolver] jitted
  [Transolver] training step 1/2: loss 0.686805, gradient norm 1.1588e+01 (0.4s)
  [Transolver] training step 2/2: loss 0.493627, gradient norm 4.5075e+00 (0.0s)
  [Transolver] loss after 2 steps: 0.427427 (first step 0.686805)
  [Transolver] saving checkpoint to /tmp/scirex-platform-XXXXXXXX/transolver
           dummy_metadata.msgpack: 339 bytes
           dummy_params.bytes: 17,597 bytes
  [Transolver] loading the checkpoint into a new model
  [Transolver] restored parameters identical to the trained ones: True
  [Transolver] running inference with the loaded model
  [Transolver] trained parameters, loaded parameters and output live on: cpu
  [Transolver] output shape (2, 50, 1), max difference to the trained model 0.00e+00 (0.0s)
      -> PASS  Transolver workflow  (train, save, load, infer)

  --- FNO ---------------------------------------------------------
  [FNO] dummy data: input (4, 1, 16, 16) channels-first, target = sin(input)
  [FNO] model built in 0.1s
  [FNO] initialising parameters
  [FNO] 6,817 parameters, initialised in 1.9s
  [FNO] optimizer: adamw_cx
  [FNO] jitted
  [FNO] training step 1/2: loss 0.383859, gradient norm 3.1054e-01 (0.3s)
  [FNO] training step 2/2: loss 0.379999, gradient norm 3.0488e-01 (0.0s)
  [FNO] loss after 2 steps: 0.376228 (first step 0.383859)
  [FNO] saving checkpoint to /tmp/scirex-platform-XXXXXXXX/fno
           dummy_metadata.msgpack: 725 bytes
           dummy_params.bytes: 28,070 bytes
  [FNO] loading the checkpoint into a new model
  [FNO] restored parameters identical to the trained ones: True
  [FNO] running inference with the loaded model
  [FNO] trained parameters, loaded parameters and output live on: cpu
  [FNO] output shape (4, 1, 16, 16), max difference to the trained model 0.00e+00 (0.0s)
      -> PASS  FNO workflow  (train, save, load, infer)

  --- GINO --------------------------------------------------------
  [GINO] dummy data: 30 random 3D points, 2 samples, latent grid 5^3, target = sum of channels
  [GINO] model built in 0.3s
  [GINO] initialising parameters
$WORK/scirex-venv/lib/python3.12/site-packages/jax/_src/numpy/array_methods.py:122: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
<repo>/src/scirex/layers/blocks/neighbor_search.py:248: UserWarning: Explicitly requested dtype <class 'jax.numpy.int64'> requested in array is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  splits = jnp.concatenate([jnp.array([0]), jnp.array(counts, dtype=jnp.int64)])
  [GINO] 19,452 parameters, initialised in 4.2s
  [GINO] optimizer: adam
  [GINO] not jitted (data-dependent neighbour search)
  [GINO] training step 1/2: loss 4.578099, gradient norm 2.1468e+01 (4.6s)
  [GINO] training step 2/2: loss 4.071411, gradient norm 1.6897e+01 (0.4s)
  [GINO] loss after 2 steps: 3.687968 (first step 4.578099)
  [GINO] saving checkpoint to /tmp/scirex-platform-XXXXXXXX/gino
           dummy_metadata.msgpack: 1,472 bytes
           dummy_params.bytes: 78,966 bytes
  [GINO] loading the checkpoint into a new model
  [GINO] restored parameters identical to the trained ones: True
  [GINO] running inference with the loaded model
  [GINO] trained parameters, loaded parameters and output live on: cpu
  [GINO] output shape (2, 30, 1), max difference to the trained model 0.00e+00 (0.1s)
      -> PASS  GINO workflow  (train, save, load, infer)

  --- WNO ---------------------------------------------------------
  [WNO] dummy data: input (2, 16, 16, 2) channels-last, target = sum of channels
  [WNO] model built in 0.1s
  [WNO] initialising parameters
  [WNO] 19,897 parameters, initialised in 2.0s
  [WNO] optimizer: adam
  [WNO] jitted
  [WNO] training step 1/2: loss 1.987134, gradient norm 3.1468e+00 (0.3s)
  [WNO] training step 2/2: loss 1.930696, gradient norm 3.0076e+00 (0.0s)
  [WNO] loss after 2 steps: 1.874705 (first step 1.987134)
  [WNO] saving checkpoint to /tmp/scirex-platform-XXXXXXXX/wno
           dummy_metadata.msgpack: 537 bytes
           dummy_params.bytes: 80,138 bytes
  [WNO] loading the checkpoint into a new model
  [WNO] restored parameters identical to the trained ones: True
  [WNO] running inference with the loaded model
  [WNO] trained parameters, loaded parameters and output live on: cpu
  [WNO] output shape (2, 16, 16, 1), max difference to the trained model 0.00e+00 (0.0s)
      -> PASS  WNO workflow  (train, save, load, infer)

=== [5] Summary ===========================================================
  PASS  python version       3.12 is in the supported range 3.10 to 3.12
  PASS  operating system     Linux x86_64: CPU and NVIDIA GPU (CUDA 12) are supported
  PASS  jax runs             backend cpu, 1 device(s)
  PASS  backend              auto: JAX chose cpu (TFRT_CPU_0)
  PASS  jit computation      a jitted sum of squares gave 14.0
  PASS  import all modules   113 imported, 1 need or cannot load an optional package, 0 failed
  WARN  optional packages    1 module(s) skipped; install the extras to test them: pip install -e '.[all]'
  PASS  public API           73 names looked up: 53 classes, 20 functions, 0 problems
  PASS  Transolver workflow  train, save, load, infer
  PASS  FNO workflow         train, save, load, infer
  PASS  GINO workflow        train, save, load, infer
  PASS  WNO workflow         train, save, load, infer

RESULT: OK on Linux x86_64 (1 warnings)

Platform check finished on Linux x86_64: OK
```

Exit status 0, 25s.
