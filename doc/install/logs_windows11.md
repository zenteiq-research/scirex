<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Install log: Windows 11

The output of the Windows tab of [Install with pip](index.md#install-with-pip) on one machine, step by step: the checks before the install,
the install from a source checkout, and the checks after it. The user name and the folders of
this machine are replaced: the checkout by `<repo>`, the temporary work folder by `%WORK%`, the home folder by
`%USERPROFILE%`.

| | |
| --- | --- |
| Operating system | Windows 11 Home Single Language 10.0.26100, 64-bit |
| Hardware | CPU only (native Windows has no JAX GPU build) |
| Python | 3.12.10 (64-bit) |
| Install | `pip install -r requirements.txt`, then `pip install -e . --no-deps` (source checkout) |
| Package | scirex-zenteiq 1.0.0rc1, JAX 0.6.2 |
| Source | a copy of the 1.0.0rc1 source checkout. `src/`, `tests/`, `requirements.txt` and [`scripts/dev/verify_install.py`](../../scripts/dev/verify_install.py) are identical to the release (line endings aside); only the contact addresses in `pyproject.toml` and some documentation pages differ |
| Date | 2026-11-10 |
| Result | every step passed |

The virtual environment was created in a temporary folder (`%WORK%\winvenv`) so that the checkout stays clean;
`python` in the commands below is the `python.exe` of that environment. The install page uses `.venv`.

## Operating system

```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture | Format-List
```

```text
Caption        : Microsoft Windows 11 Home Single Language
Version        : 10.0.26100
OSArchitecture : 64-bit
```

## Step 1: long paths

```powershell
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled
```

```text
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
    LongPathsEnabled    REG_DWORD    0x1
```

## Step 2: Python

```powershell
python --version
python -c "import sys, platform; print(sys.executable); print(platform.architecture())"
```

```text
Python 3.12.10
%LOCALAPPDATA%\Programs\Python\Python312\python.exe
('64bit', 'WindowsPE')
```

## PowerShell execution policy

```powershell
Get-ExecutionPolicy -List
```

```text
        Scope ExecutionPolicy
        ----- ---------------
MachinePolicy       Undefined
   UserPolicy       Undefined
      Process          Bypass
  CurrentUser    RemoteSigned
 LocalMachine       Undefined
```

## git

```powershell
git --version
```

```text
git version 2.45.1.windows.1
```

## Step 3: create the virtual environment

```powershell
python -m venv %WORK%\winvenv
```

```text
Actual environment location may have moved due to redirects, links or junctions.
  Requested location: %WORK%\winvenv\Scripts\python.exe
  Actual location:    %WORK%\winvenv\Scripts\python.exe
(The two paths differ only in the spelling of the user folder: short name and long name.)
```

Exit status 0.

## Step 4: upgrade pip

```powershell
python -m pip install --upgrade pip
```

```text
Requirement already satisfied: pip in %WORK%\winvenv\lib\site-packages (25.0.1)
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

Exit status 0.

## Step 4: install the requirements

```powershell
python -m pip install -r requirements.txt
```

```text
Collecting jax==0.6.2 (from -r requirements.txt (line 1))
  Using cached jax-0.6.2-py3-none-any.whl.metadata (13 kB)
Collecting jaxlib==0.6.2 (from -r requirements.txt (line 2))
  Using cached jaxlib-0.6.2-cp312-cp312-win_amd64.whl.metadata (1.4 kB)
Collecting flax==0.10.7 (from -r requirements.txt (line 3))
  Using cached flax-0.10.7-py3-none-any.whl.metadata (11 kB)
Collecting optax==0.2.8 (from -r requirements.txt (line 4))
  Using cached optax-0.2.8-py3-none-any.whl.metadata (7.9 kB)
Collecting numpy==1.26.4 (from -r requirements.txt (line 5))
  Using cached numpy-1.26.4-cp312-cp312-win_amd64.whl.metadata (61 kB)
Collecting opt-einsum==3.4.0 (from -r requirements.txt (line 6))
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting einops==0.8.2 (from -r requirements.txt (line 7))
  Using cached einops-0.8.2-py3-none-any.whl.metadata (13 kB)
Collecting tensorly==0.9.0 (from -r requirements.txt (line 8))
  Using cached tensorly-0.9.0-py3-none-any.whl.metadata (8.6 kB)
Collecting wandb==0.26.0 (from -r requirements.txt (line 9))
  Using cached wandb-0.26.0-py3-none-win_amd64.whl.metadata (11 kB)
Collecting zencfg==0.7.0 (from -r requirements.txt (line 10))
  Using cached zencfg-0.7.0-py3-none-any.whl.metadata (9.9 kB)
Collecting matplotlib==3.9.4 (from -r requirements.txt (line 11))
  Using cached matplotlib-3.9.4-cp312-cp312-win_amd64.whl.metadata (11 kB)
Collecting open3d==0.19.0 (from -r requirements.txt (line 12))
  Using cached open3d-0.19.0-cp312-cp312-win_amd64.whl.metadata (4.2 kB)
Collecting jaxwt==0.1.1 (from -r requirements.txt (line 13))
  Using cached jaxwt-0.1.1-py3-none-any.whl.metadata (7.2 kB)
Collecting scipy==1.15.3 (from -r requirements.txt (line 14))
  Using cached scipy-1.15.3-cp312-cp312-win_amd64.whl.metadata (60 kB)
Collecting requests==2.34.2 (from -r requirements.txt (line 15))
  Using cached requests-2.34.2-py3-none-any.whl.metadata (4.8 kB)
Collecting scikit-learn==1.7.2 (from -r requirements.txt (line 16))
  Using cached scikit_learn-1.7.2-cp312-cp312-win_amd64.whl.metadata (11 kB)
Collecting ml_dtypes>=0.5.0 (from jax==0.6.2->-r requirements.txt (line 1))
  Using cached ml_dtypes-0.6.0-cp312-cp312-win_amd64.whl.metadata (8.8 kB)
Collecting msgpack (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached msgpack-1.2.4-cp312-cp312-win_amd64.whl.metadata (8.5 kB)
Collecting orbax-checkpoint (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached orbax_checkpoint-0.12.9-py3-none-any.whl.metadata (3.3 kB)
Collecting tensorstore (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached tensorstore-0.1.86-cp312-cp312-win_amd64.whl.metadata (3.3 kB)
Collecting rich>=11.1 (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached rich-15.0.0-py3-none-any.whl.metadata (18 kB)
Collecting typing_extensions>=4.2 (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached typing_extensions-4.16.0-py3-none-any.whl.metadata (3.3 kB)
Collecting PyYAML>=5.4.1 (from flax==0.10.7->-r requirements.txt (line 3))
  Using cached pyyaml-6.0.3-cp312-cp312-win_amd64.whl.metadata (2.4 kB)
Collecting treescope>=0.1.7 (from flax==0.10.7->-r requirements.txt (line 3))
  Downloading treescope-0.1.11-py3-none-any.whl.metadata (6.7 kB)
Collecting absl-py>=0.7.1 (from optax==0.2.8->-r requirements.txt (line 4))
  Using cached absl_py-2.5.1-py3-none-any.whl.metadata (3.3 kB)
Collecting click>=8.0.1 (from wandb==0.26.0->-r requirements.txt (line 9))
  Downloading click-8.5.1-py3-none-any.whl.metadata (2.6 kB)
Collecting gitpython!=3.1.29,>=1.0.0 (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached gitpython-3.2.1-py3-none-any.whl.metadata (13 kB)
Collecting packaging (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached packaging-26.3-py3-none-any.whl.metadata (3.5 kB)
Collecting platformdirs (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached platformdirs-4.14.5-py3-none-any.whl.metadata (5.7 kB)
Collecting protobuf!=5.28.0,!=5.29.0,<8,>4.21.0 (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached protobuf-7.36.2-cp310-abi3-win_amd64.whl.metadata (595 bytes)
Collecting pydantic<3 (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached pydantic-2.14.3-py3-none-any.whl.metadata (141 kB)
Collecting sentry-sdk>=2.0.0 (from wandb==0.26.0->-r requirements.txt (line 9))
  Using cached sentry_sdk-2.74.0-py3-none-any.whl.metadata (11 kB)
Collecting charset_normalizer<4,>=2 (from requests==2.34.2->-r requirements.txt (line 15))
  Using cached charset_normalizer-3.5.2-cp312-cp312-win_amd64.whl.metadata (47 kB)
Collecting idna<4,>=2.5 (from requests==2.34.2->-r requirements.txt (line 15))
  Using cached idna-3.20-py3-none-any.whl.metadata (7.2 kB)
Collecting urllib3<3,>=1.26 (from requests==2.34.2->-r requirements.txt (line 15))
  Using cached urllib3-2.8.0-py3-none-any.whl.metadata (7.4 kB)
Collecting certifi>=2023.5.7 (from requests==2.34.2->-r requirements.txt (line 15))
  Using cached certifi-2026.10.23-py3-none-any.whl.metadata (2.5 kB)
Collecting contourpy>=1.0.1 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached contourpy-1.4.0-cp312-cp312-win_amd64.whl.metadata (3.8 kB)
Collecting cycler>=0.10 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached fonttools-4.68.0-cp312-cp312-win_amd64.whl.metadata (142 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached kiwisolver-1.5.1-cp312-cp312-win_amd64.whl.metadata (5.2 kB)
Collecting pillow>=8 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached pillow-13.0.0-cp312-cp312-win_amd64.whl.metadata (9.3 kB)
Collecting pyparsing>=2.3.1 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached pyparsing-3.3.3-py3-none-any.whl.metadata (5.9 kB)
Collecting python-dateutil>=2.7 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached python_dateutil-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting dash>=2.6.0 (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached dash-4.5.0-py3-none-any.whl.metadata (11 kB)
Collecting werkzeug>=3.0.0 (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached werkzeug-3.1.9-py3-none-any.whl.metadata (4.1 kB)
Collecting flask>=3.0.0 (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached flask-3.1.3-py3-none-any.whl.metadata (3.2 kB)
Collecting nbformat>=5.7.0 (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached nbformat-5.11.1-py3-none-any.whl.metadata (3.7 kB)
Collecting configargparse (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached configargparse-1.9.0-py3-none-any.whl.metadata (24 kB)
Collecting ipywidgets>=8.0.4 (from open3d==0.19.0->-r requirements.txt (line 12))
  Using cached ipywidgets-8.1.9-py3-none-any.whl.metadata (2.4 kB)
Collecting PyWavelets (from jaxwt==0.1.1->-r requirements.txt (line 13))
  Using cached pywavelets-1.10.0-cp312-cp312-win_amd64.whl.metadata (7.6 kB)
Collecting joblib>=1.2.0 (from scikit-learn==1.7.2->-r requirements.txt (line 16))
  Using cached joblib-1.6.0-py3-none-any.whl.metadata (6.5 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn==1.7.2->-r requirements.txt (line 16))
  Using cached threadpoolctl-3.8.0-py3-none-any.whl.metadata (29 kB)
Collecting annotated-types>=0.6.0 (from pydantic<3->wandb==0.26.0->-r requirements.txt (line 9))
  Using cached annotated_types-0.8.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.50.4 (from pydantic<3->wandb==0.26.0->-r requirements.txt (line 9))
  Using cached pydantic_core-2.50.4-cp312-cp312-win_amd64.whl.metadata (6.7 kB)
Collecting typing-inspection>=0.4.4 (from pydantic<3->wandb==0.26.0->-r requirements.txt (line 9))
  Using cached typing_inspection-0.4.5-py3-none-any.whl.metadata (2.6 kB)
Collecting contourpy>=1.0.1 (from matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached contourpy-1.3.3-cp312-cp312-win_amd64.whl.metadata (5.5 kB)
Collecting plotly>=5.0.0 (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached plotly-7.1.0-py3-none-any.whl.metadata (8.8 kB)
Collecting importlib-metadata (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Downloading importlib_metadata-10.2.0-py3-none-any.whl.metadata (4.6 kB)
Collecting retrying (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached retrying-1.4.2-py3-none-any.whl.metadata (5.5 kB)
Collecting nest-asyncio (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached nest_asyncio-1.6.0-py3-none-any.whl.metadata (2.8 kB)
Collecting setuptools (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached setuptools-84.2.1-py3-none-any.whl.metadata (6.6 kB)
Collecting janus>=1.0.0 (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached janus-2.0.0-py3-none-any.whl.metadata (5.3 kB)
Collecting comm (from dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached comm-0.2.3-py3-none-any.whl.metadata (3.7 kB)
Collecting blinker>=1.9.0 (from flask>=3.0.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
Collecting itsdangerous>=2.2.0 (from flask>=3.0.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
Collecting jinja2>=3.1.2 (from flask>=3.0.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting markupsafe>=2.1.1 (from flask>=3.0.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached markupsafe-3.0.4-cp312-cp312-win_amd64.whl.metadata (2.8 kB)
Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.29,>=1.0.0->wandb==0.26.0->-r requirements.txt (line 9))
  Using cached gitdb-4.0.12-py3-none-any.whl.metadata (1.2 kB)
Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb==0.26.0->-r requirements.txt (line 9))
  Using cached smmap-5.0.3-py3-none-any.whl.metadata (4.6 kB)
Collecting ipython>=6.1.0 (from ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached ipython-9.18.1-py3-none-any.whl.metadata (4.6 kB)
Collecting traitlets>=4.3.1 (from ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached traitlets-5.16.1-py3-none-any.whl.metadata (10 kB)
Collecting widgetsnbextension~=4.0.16 (from ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached widgetsnbextension-4.0.16-py3-none-any.whl.metadata (1.6 kB)
Collecting jupyterlab_widgets~=3.0.17 (from ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jupyterlab_widgets-3.0.17-py3-none-any.whl.metadata (20 kB)
Collecting colorama>=0.4.4 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting ipython-pygments-lexers>=1.0.0 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl.metadata (1.1 kB)
Collecting jedi>=0.18.2 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jedi-0.20.1-py2.py3-none-any.whl.metadata (23 kB)
Collecting matplotlib-inline>=0.1.6 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached matplotlib_inline-0.2.2-py3-none-any.whl.metadata (2.4 kB)
Collecting prompt_toolkit<3.1.0,>=3.0.41 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached prompt_toolkit-3.0.53-py3-none-any.whl.metadata (6.4 kB)
Collecting psutil>=7 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached psutil-7.2.2-cp37-abi3-win_amd64.whl.metadata (22 kB)
Collecting pygments>=2.14.0 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached pygments-2.21.0-py3-none-any.whl.metadata (2.5 kB)
Collecting stack_data>=0.6.0 (from ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached stack_data-0.6.3-py3-none-any.whl.metadata (18 kB)
Collecting wcwidth>=0.1.4 (from prompt_toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached wcwidth-0.9.4-cp310-abi3-win_amd64.whl.metadata (24 kB)
Collecting parso<0.9.0,>=0.8.7 (from jedi>=0.18.2->ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached parso-0.8.7-py2.py3-none-any.whl.metadata (8.2 kB)
Collecting cloudpickle>=3.0 (from joblib>=1.2.0->scikit-learn==1.7.2->-r requirements.txt (line 16))
  Using cached cloudpickle-3.1.2-py3-none-any.whl.metadata (7.1 kB)
Collecting ml_dtypes>=0.5.0 (from jax==0.6.2->-r requirements.txt (line 1))
  Using cached ml_dtypes-0.5.4-cp312-cp312-win_amd64.whl.metadata (9.2 kB)
Collecting fastjsonschema>=2.15 (from nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached fastjsonschema-2.22.3-py3-none-any.whl.metadata (2.1 kB)
Collecting jsonschema>=2.6 (from nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jsonschema-4.26.0-py3-none-any.whl.metadata (7.6 kB)
Collecting jupyter-core!=5.0.*,>=4.12 (from nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jupyter_core-5.9.1-py3-none-any.whl.metadata (1.5 kB)
Collecting attrs>=22.2.0 (from jsonschema>=2.6->nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached attrs-26.1.0-py3-none-any.whl.metadata (8.8 kB)
Collecting jsonschema-specifications>=2023.03.6 (from jsonschema>=2.6->nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached jsonschema_specifications-2025.9.1-py3-none-any.whl.metadata (2.9 kB)
Collecting referencing>=0.28.4 (from jsonschema>=2.6->nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached referencing-0.37.0-py3-none-any.whl.metadata (2.8 kB)
Collecting rpds-py>=0.25.0 (from jsonschema>=2.6->nbformat>=5.7.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached rpds_py-2026.9.1-cp312-cp312-win_amd64.whl.metadata (4.2 kB)
Collecting narwhals>=1.15.1 (from plotly>=5.0.0->dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached narwhals-2.28.0-py3-none-any.whl.metadata (15 kB)
Collecting tzdata (from python-dateutil>=2.7->matplotlib==3.9.4->-r requirements.txt (line 11))
  Using cached tzdata-2026.6-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting markdown-it-py>=2.2.0 (from rich>=11.1->flax==0.10.7->-r requirements.txt (line 3))
  Using cached markdown_it_py-4.2.0-py3-none-any.whl.metadata (7.4 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=11.1->flax==0.10.7->-r requirements.txt (line 3))
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Collecting executing>=1.2.0 (from stack_data>=0.6.0->ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached executing-2.3.0-py3-none-any.whl.metadata (9.1 kB)
Collecting asttokens>=2.1.0 (from stack_data>=0.6.0->ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached asttokens-3.0.2-py3-none-any.whl.metadata (5.7 kB)
Collecting pure-eval (from stack_data>=0.6.0->ipython>=6.1.0->ipywidgets>=8.0.4->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached pure_eval-0.2.4-py3-none-any.whl.metadata (6.4 kB)
Collecting zipp>=3.20 (from importlib-metadata->dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Using cached zipp-4.1.2-py3-none-any.whl.metadata (3.6 kB)
Collecting lazy-importer>=0.1 (from importlib-metadata->dash>=2.6.0->open3d==0.19.0->-r requirements.txt (line 12))
  Downloading lazy_importer-0.1.0-py3-none-any.whl.metadata (7.3 kB)
Collecting etils>=1.15.0 (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached etils-1.15.0-py3-none-any.whl.metadata (6.5 kB)
Collecting prometheus-client>=0.20.0 (from orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached prometheus_client-0.26.0-py3-none-any.whl.metadata (2.1 kB)
Collecting aiofiles (from orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached aiofiles-25.1.0-py3-none-any.whl.metadata (6.3 kB)
Collecting humanize (from orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached humanize-4.16.0-py3-none-any.whl.metadata (8.0 kB)
Collecting simplejson>=3.16.0 (from orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached simplejson-4.2.0-cp312-cp312-win_amd64.whl.metadata (3.9 kB)
Collecting fsspec (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached fsspec-2026.9.0-py3-none-any.whl.metadata (10 kB)
Collecting gcsfs>=2026.4.0 (from etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached gcsfs-2026.10.1-py3-none-any.whl.metadata (4.6 kB)
Collecting aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached aiohttp-3.14.5-cp312-cp312-win_amd64.whl.metadata (8.5 kB)
Collecting decorator>4.1.2 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached decorator-5.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting google-auth-oauthlib (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_auth_oauthlib-1.5.0-py3-none-any.whl.metadata (2.6 kB)
Collecting google-auth>=1.2 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_auth-2.62.1-py3-none-any.whl.metadata (6.0 kB)
Collecting google-cloud-storage-control (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_cloud_storage_control-1.17.0-py3-none-any.whl.metadata (9.8 kB)
Collecting google-cloud-storage>=3.14.1 (from gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_cloud_storage-3.17.1-py3-none-any.whl.metadata (14 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached aiohappyeyeballs-2.7.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting frozenlist>=1.1.1 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached frozenlist-1.8.0-cp312-cp312-win_amd64.whl.metadata (21 kB)
Collecting multidict<8.0,>=4.5 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached multidict-7.2.1-cp312-cp312-win_amd64.whl.metadata (6.3 kB)
Collecting propcache>=0.2.0 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached propcache-0.5.4-cp312-cp312-win_amd64.whl.metadata (25 kB)
Collecting yarl<2.0,>=1.25.1 (from aiohttp!=4.0.0a0,!=4.0.0a1,>=3.9.0->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached yarl-1.25.1-cp312-cp312-win_amd64.whl.metadata (115 kB)
Collecting pyasn1-modules>=0.2.1 (from google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached pyasn1_modules-0.4.2-py3-none-any.whl.metadata (3.5 kB)
Collecting cryptography>=38.0.3 (from google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached cryptography-50.0.2-cp311-abi3-win_amd64.whl.metadata (4.4 kB)
Collecting cffi>=2.0.0 (from cryptography>=38.0.3->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached cffi-2.1.1-cp312-cp312-win_amd64.whl.metadata (2.6 kB)
Collecting pycparser (from cffi>=2.0.0->cryptography>=38.0.3->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached pycparser-3.12-py3-none-any.whl.metadata (9.7 kB)
Collecting google-api-core<3.0.0,>=2.27.0 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_api_core-2.44.0-py3-none-any.whl.metadata (3.2 kB)
Collecting google-cloud-core<3.0.0,>=2.4.2 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_cloud_core-2.8.0-py3-none-any.whl.metadata (2.9 kB)
Collecting google-resumable-media<3.0.0,>=2.7.2 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_resumable_media-2.11.0-py3-none-any.whl.metadata (2.3 kB)
Collecting google-crc32c<2.0.0,>=1.6.0 (from google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached google_crc32c-1.9.0-cp312-cp312-win_amd64.whl.metadata (2.2 kB)
Collecting googleapis-common-protos<2.0.0,>=1.69.2 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached googleapis_common_protos-1.76.0-py3-none-any.whl.metadata (8.5 kB)
Collecting proto-plus<2.0.0,>=1.26.1 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached proto_plus-1.29.0-py3-none-any.whl.metadata (2.2 kB)
Collecting opentelemetry-api<2.0.0,>=1.44.0 (from google-api-core<3.0.0,>=2.27.0->google-cloud-storage>=3.14.1->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached opentelemetry_api-1.45.2-py3-none-any.whl.metadata (1.4 kB)
Collecting pyasn1<0.7.0,>=0.6.1 (from pyasn1-modules>=0.2.1->google-auth>=1.2->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached pyasn1-0.6.5-py3-none-any.whl.metadata (8.4 kB)
Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached oauthlib-4.0.1-py3-none-any.whl.metadata (8.6 kB)
Collecting grpcio<2.0.0,>=1.59.0 (from google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached grpcio-1.85.1-cp312-cp312-win_amd64.whl.metadata (3.9 kB)
Collecting grpc-google-iam-v1<1.0.0,>=0.14.2 (from google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached grpc_google_iam_v1-0.14.6-py3-none-any.whl.metadata (8.9 kB)
Collecting grpcio-status<2.0.0,>=1.59.0 (from google-api-core[grpc]<3.0.0,>=2.28.0->google-cloud-storage-control->gcsfs>=2026.4.0->etils[epath,epath-gcs,epy]>=1.15.0->orbax-checkpoint->flax==0.10.7->-r requirements.txt (line 3))
  Using cached grpcio_status-1.85.1-py3-none-any.whl.metadata (1.3 kB)
Using cached jax-0.6.2-py3-none-any.whl (2.7 MB)
Using cached jaxlib-0.6.2-cp312-cp312-win_amd64.whl (57.9 MB)
Using cached flax-0.10.7-py3-none-any.whl (456 kB)
Using cached optax-0.2.8-py3-none-any.whl (402 kB)
Using cached numpy-1.26.4-cp312-cp312-win_amd64.whl (15.5 MB)
Using cached opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Using cached einops-0.8.2-py3-none-any.whl (65 kB)
Using cached tensorly-0.9.0-py3-none-any.whl (7.4 MB)
Using cached wandb-0.26.0-py3-none-win_amd64.whl (24.9 MB)
Using cached requests-2.34.2-py3-none-any.whl (73 kB)
Using cached zencfg-0.7.0-py3-none-any.whl (31 kB)
Using cached matplotlib-3.9.4-cp312-cp312-win_amd64.whl (7.8 MB)
Using cached open3d-0.19.0-cp312-cp312-win_amd64.whl (69.2 MB)
Using cached jaxwt-0.1.1-py3-none-any.whl (26 kB)
Using cached scipy-1.15.3-cp312-cp312-win_amd64.whl (41.0 MB)
Using cached scikit_learn-1.7.2-cp312-cp312-win_amd64.whl (8.7 MB)
Using cached charset_normalizer-3.5.2-cp312-cp312-win_amd64.whl (207 kB)
Using cached idna-3.20-py3-none-any.whl (69 kB)
Using cached protobuf-7.36.2-cp310-abi3-win_amd64.whl (456 kB)
Using cached pydantic-2.14.3-py3-none-any.whl (494 kB)
Using cached pydantic_core-2.50.4-cp312-cp312-win_amd64.whl (2.3 MB)
Using cached typing_extensions-4.16.0-py3-none-any.whl (45 kB)
Using cached urllib3-2.8.0-py3-none-any.whl (135 kB)
Using cached absl_py-2.5.1-py3-none-any.whl (137 kB)
Using cached annotated_types-0.8.0-py3-none-any.whl (13 kB)
Using cached certifi-2026.10.23-py3-none-any.whl (135 kB)
Downloading click-8.5.1-py3-none-any.whl (126 kB)
Using cached contourpy-1.3.3-cp312-cp312-win_amd64.whl (226 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Using cached dash-4.5.0-py3-none-any.whl (9.3 MB)
Using cached flask-3.1.3-py3-none-any.whl (103 kB)
Using cached werkzeug-3.1.9-py3-none-any.whl (228 kB)
Using cached blinker-1.9.0-py3-none-any.whl (8.5 kB)
Using cached fonttools-4.68.0-cp312-cp312-win_amd64.whl (2.5 MB)
Using cached gitpython-3.2.1-py3-none-any.whl (231 kB)
Using cached gitdb-4.0.12-py3-none-any.whl (62 kB)
Using cached smmap-5.0.3-py3-none-any.whl (24 kB)
Using cached ipywidgets-8.1.9-py3-none-any.whl (140 kB)
Using cached jupyterlab_widgets-3.0.17-py3-none-any.whl (217 kB)
Using cached widgetsnbextension-4.0.16-py3-none-any.whl (2.2 MB)
Using cached comm-0.2.3-py3-none-any.whl (7.3 kB)
Using cached ipython-9.18.1-py3-none-any.whl (649 kB)
Using cached prompt_toolkit-3.0.53-py3-none-any.whl (392 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached ipython_pygments_lexers-1.1.1-py3-none-any.whl (8.1 kB)
Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Using cached janus-2.0.0-py3-none-any.whl (12 kB)
Using cached jedi-0.20.1-py2.py3-none-any.whl (4.9 MB)
Using cached parso-0.8.7-py2.py3-none-any.whl (107 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Using cached joblib-1.6.0-py3-none-any.whl (306 kB)
Using cached cloudpickle-3.1.2-py3-none-any.whl (22 kB)
Using cached kiwisolver-1.5.1-cp312-cp312-win_amd64.whl (70 kB)
Using cached markupsafe-3.0.4-cp312-cp312-win_amd64.whl (14 kB)
Using cached matplotlib_inline-0.2.2-py3-none-any.whl (9.5 kB)
Using cached ml_dtypes-0.5.4-cp312-cp312-win_amd64.whl (212 kB)
Using cached nbformat-5.11.1-py3-none-any.whl (79 kB)
Using cached fastjsonschema-2.22.3-py3-none-any.whl (28 kB)
Using cached jsonschema-4.26.0-py3-none-any.whl (90 kB)
Using cached attrs-26.1.0-py3-none-any.whl (67 kB)
Using cached jsonschema_specifications-2025.9.1-py3-none-any.whl (18 kB)
Using cached jupyter_core-5.9.1-py3-none-any.whl (29 kB)
Using cached packaging-26.3-py3-none-any.whl (129 kB)
Using cached pillow-13.0.0-cp312-cp312-win_amd64.whl (7.3 MB)
Using cached platformdirs-4.14.5-py3-none-any.whl (35 kB)
Using cached plotly-7.1.0-py3-none-any.whl (9.7 MB)
Using cached narwhals-2.28.0-py3-none-any.whl (490 kB)
Using cached psutil-7.2.2-cp37-abi3-win_amd64.whl (137 kB)
Using cached pygments-2.21.0-py3-none-any.whl (1.3 MB)
Using cached pyparsing-3.3.3-py3-none-any.whl (126 kB)
Using cached python_dateutil-4.0.0-py3-none-any.whl (92 kB)
Using cached pyyaml-6.0.3-cp312-cp312-win_amd64.whl (154 kB)
Using cached referencing-0.37.0-py3-none-any.whl (26 kB)
Using cached rich-15.0.0-py3-none-any.whl (310 kB)
Using cached markdown_it_py-4.2.0-py3-none-any.whl (91 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Using cached rpds_py-2026.9.1-cp312-cp312-win_amd64.whl (222 kB)
Using cached sentry_sdk-2.74.0-py3-none-any.whl (549 kB)
Using cached stack_data-0.6.3-py3-none-any.whl (24 kB)
Using cached asttokens-3.0.2-py3-none-any.whl (28 kB)
Using cached executing-2.3.0-py3-none-any.whl (29 kB)
Using cached threadpoolctl-3.8.0-py3-none-any.whl (32 kB)
Using cached traitlets-5.16.1-py3-none-any.whl (86 kB)
Downloading treescope-0.1.11-py3-none-any.whl (186 kB)
Using cached typing_inspection-0.4.5-py3-none-any.whl (14 kB)
Using cached wcwidth-0.9.4-cp310-abi3-win_amd64.whl (597 kB)
Using cached configargparse-1.9.0-py3-none-any.whl (37 kB)
Downloading importlib_metadata-10.2.0-py3-none-any.whl (34 kB)
Downloading lazy_importer-0.1.0-py3-none-any.whl (12 kB)
Using cached zipp-4.1.2-py3-none-any.whl (10 kB)
Using cached msgpack-1.2.4-cp312-cp312-win_amd64.whl (77 kB)
Using cached nest_asyncio-1.6.0-py3-none-any.whl (5.2 kB)
Using cached orbax_checkpoint-0.12.9-py3-none-any.whl (1.4 MB)
Using cached etils-1.15.0-py3-none-any.whl (177 kB)
Using cached gcsfs-2026.10.1-py3-none-any.whl (88 kB)
Using cached aiohttp-3.14.5-cp312-cp312-win_amd64.whl (506 kB)
Using cached multidict-7.2.1-cp312-cp312-win_amd64.whl (95 kB)
Using cached yarl-1.25.1-cp312-cp312-win_amd64.whl (102 kB)
Using cached aiohappyeyeballs-2.7.1-py3-none-any.whl (15 kB)
Using cached aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Using cached decorator-5.3.1-py3-none-any.whl (10 kB)
Using cached frozenlist-1.8.0-cp312-cp312-win_amd64.whl (44 kB)
Using cached fsspec-2026.9.0-py3-none-any.whl (221 kB)
Using cached google_auth-2.62.1-py3-none-any.whl (267 kB)
Using cached cryptography-50.0.2-cp311-abi3-win_amd64.whl (3.8 MB)
Using cached cffi-2.1.1-cp312-cp312-win_amd64.whl (185 kB)
Using cached google_cloud_storage-3.17.1-py3-none-any.whl (345 kB)
Using cached google_api_core-2.44.0-py3-none-any.whl (225 kB)
Using cached google_cloud_core-2.8.0-py3-none-any.whl (31 kB)
Using cached google_crc32c-1.9.0-cp312-cp312-win_amd64.whl (35 kB)
Using cached google_resumable_media-2.11.0-py3-none-any.whl (81 kB)
Using cached googleapis_common_protos-1.76.0-py3-none-any.whl (307 kB)
Using cached opentelemetry_api-1.45.2-py3-none-any.whl (60 kB)
Using cached proto_plus-1.29.0-py3-none-any.whl (50 kB)
Using cached prometheus_client-0.26.0-py3-none-any.whl (64 kB)
Using cached propcache-0.5.4-cp312-cp312-win_amd64.whl (46 kB)
Using cached pyasn1_modules-0.4.2-py3-none-any.whl (181 kB)
Using cached pyasn1-0.6.5-py3-none-any.whl (84 kB)
Using cached simplejson-4.2.0-cp312-cp312-win_amd64.whl (93 kB)
Using cached tensorstore-0.1.86-cp312-cp312-win_amd64.whl (14.3 MB)
Using cached aiofiles-25.1.0-py3-none-any.whl (14 kB)
Using cached google_auth_oauthlib-1.5.0-py3-none-any.whl (20 kB)
Using cached requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
Using cached oauthlib-4.0.1-py3-none-any.whl (161 kB)
Using cached google_cloud_storage_control-1.17.0-py3-none-any.whl (115 kB)
Using cached grpc_google_iam_v1-0.14.6-py3-none-any.whl (33 kB)
Using cached grpcio-1.85.1-cp312-cp312-win_amd64.whl (5.6 MB)
Using cached grpcio_status-1.85.1-py3-none-any.whl (14 kB)
Using cached humanize-4.16.0-py3-none-any.whl (137 kB)
Using cached pure_eval-0.2.4-py3-none-any.whl (11 kB)
Using cached pycparser-3.12-py3-none-any.whl (62 kB)
Using cached pywavelets-1.10.0-cp312-cp312-win_amd64.whl (4.3 MB)
Using cached retrying-1.4.2-py3-none-any.whl (10 kB)
Using cached setuptools-84.2.1-py3-none-any.whl (876 kB)
Using cached tzdata-2026.6-py2.py3-none-any.whl (348 kB)
Installing collected packages: pure-eval, zipp, widgetsnbextension, wcwidth, urllib3, tzdata, typing_extensions, traitlets, threadpoolctl, smmap, simplejson, setuptools, rpds-py, retrying, PyYAML, pyparsing, pygments, pycparser, pyasn1, psutil, protobuf, propcache, prometheus-client, platformdirs, pillow, parso, packaging, opt-einsum, oauthlib, numpy, nest-asyncio, narwhals, multidict, msgpack, mdurl, markupsafe, lazy-importer, kiwisolver, jupyterlab_widgets, janus, itsdangerous, idna, humanize, google-crc32c, fsspec, frozenlist, fonttools, fastjsonschema, executing, etils, einops, decorator, cycler, configargparse, comm, colorama, cloudpickle, click, charset_normalizer, certifi, blinker, attrs, asttokens, annotated-types, aiohappyeyeballs, aiofiles, absl-py, yarl, werkzeug, typing-inspection, treescope, stack_data, sentry-sdk, scipy, requests, referencing, PyWavelets, python-dateutil, pydantic-core, pyasn1-modules, proto-plus, prompt_toolkit, plotly, opentelemetry-api, ml_dtypes, matplotlib-inline, markdown-it-py, jupyter-core, joblib, jinja2, jedi, ipython-pygments-lexers, importlib-metadata, grpcio, googleapis-common-protos, google-resumable-media, gitdb, contourpy, cffi, aiosignal, tensorstore, tensorly, scikit-learn, rich, requests-oauthlib, pydantic, matplotlib, jsonschema-specifications, jaxlib, ipython, grpcio-status, gitpython, flask, cryptography, aiohttp, zencfg, wandb, jsonschema, jax, ipywidgets, grpc-google-iam-v1, google-auth, dash, optax, nbformat, jaxwt, google-auth-oauthlib, google-api-core, open3d, google-cloud-core, google-cloud-storage-control, google-cloud-storage, gcsfs, orbax-checkpoint, flax

Successfully installed PyWavelets-1.10.0 PyYAML-6.0.3 absl-py-2.5.1 aiofiles-25.1.0 aiohappyeyeballs-2.7.1 aiohttp-3.14.5 aiosignal-1.4.0 annotated-types-0.8.0 asttokens-3.0.2 attrs-26.1.0 blinker-1.9.0 certifi-2026.10.23 cffi-2.1.1 charset_normalizer-3.5.2 click-8.5.1 cloudpickle-3.1.2 colorama-0.4.6 comm-0.2.3 configargparse-1.9.0 contourpy-1.3.3 cryptography-50.0.2 cycler-0.12.1 dash-4.5.0 decorator-5.3.1 einops-0.8.2 etils-1.15.0 executing-2.3.0 fastjsonschema-2.22.3 flask-3.1.3 flax-0.10.7 fonttools-4.68.0 frozenlist-1.8.0 fsspec-2026.9.0 gcsfs-2026.10.1 gitdb-4.0.12 gitpython-3.2.1 google-api-core-2.44.0 google-auth-2.62.1 google-auth-oauthlib-1.5.0 google-cloud-core-2.8.0 google-cloud-storage-3.17.1 google-cloud-storage-control-1.17.0 google-crc32c-1.9.0 google-resumable-media-2.11.0 googleapis-common-protos-1.76.0 grpc-google-iam-v1-0.14.6 grpcio-1.85.1 grpcio-status-1.85.1 humanize-4.16.0 idna-3.20 importlib-metadata-10.2.0 ipython-9.18.1 ipython-pygments-lexers-1.1.1 ipywidgets-8.1.9 itsdangerous-2.2.0 janus-2.0.0 jax-0.6.2 jaxlib-0.6.2 jaxwt-0.1.1 jedi-0.20.1 jinja2-3.1.6 joblib-1.6.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 jupyter-core-5.9.1 jupyterlab_widgets-3.0.17 kiwisolver-1.5.1 lazy-importer-0.1.0 markdown-it-py-4.2.0 markupsafe-3.0.4 matplotlib-3.9.4 matplotlib-inline-0.2.2 mdurl-0.1.2 ml_dtypes-0.5.4 msgpack-1.2.4 multidict-7.2.1 narwhals-2.28.0 nbformat-5.11.1 nest-asyncio-1.6.0 numpy-1.26.4 oauthlib-4.0.1 open3d-0.19.0 opentelemetry-api-1.45.2 opt-einsum-3.4.0 optax-0.2.8 orbax-checkpoint-0.12.9 packaging-26.3 parso-0.8.7 pillow-13.0.0 platformdirs-4.14.5 plotly-7.1.0 prometheus-client-0.26.0 prompt_toolkit-3.0.53 propcache-0.5.4 proto-plus-1.29.0 protobuf-7.36.2 psutil-7.2.2 pure-eval-0.2.4 pyasn1-0.6.5 pyasn1-modules-0.4.2 pycparser-3.12 pydantic-2.14.3 pydantic-core-2.50.4 pygments-2.21.0 pyparsing-3.3.3 python-dateutil-4.0.0 referencing-0.37.0 requests-2.34.2 requests-oauthlib-2.0.0 retrying-1.4.2 rich-15.0.0 rpds-py-2026.9.1 scikit-learn-1.7.2 scipy-1.15.3 sentry-sdk-2.74.0 setuptools-84.2.1 simplejson-4.2.0 smmap-5.0.3 stack_data-0.6.3 tensorly-0.9.0 tensorstore-0.1.86 threadpoolctl-3.8.0 traitlets-5.16.1 treescope-0.1.11 typing-inspection-0.4.5 typing_extensions-4.16.0 tzdata-2026.6 urllib3-2.8.0 wandb-0.26.0 wcwidth-0.9.4 werkzeug-3.1.9 widgetsnbextension-4.0.16 yarl-1.25.1 zencfg-0.7.0 zipp-4.1.2
```

Exit status 0.

## Step 4: install scirex-zenteiq

```powershell
python -m pip install -e . --no-deps
```

```text
Obtaining file:///<repo>
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Installing backend dependencies: started
  Installing backend dependencies: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Building wheels for collected packages: scirex-zenteiq
  Building editable for scirex-zenteiq (pyproject.toml): started
  Building editable for scirex-zenteiq (pyproject.toml): finished with status 'done'
  Created wheel for scirex-zenteiq: filename=scirex_zenteiq-1.0.0rc1-py3-none-any.whl size=27140 sha256=fae38d5f47e737438d3dc8249e0d00500c2093c8cf858d9a6ce2be3c92bbf451
  Stored in directory: %LOCALAPPDATA%\Temp\pip-ephem-wheel-cache-3krdjqk1\wheels\4f\e3\a5\2abbc93c07b0d6ae9fe52c868bd43bc04980c12f0be68c2258
Successfully built scirex-zenteiq
Installing collected packages: scirex-zenteiq
Successfully installed scirex-zenteiq-1.0.0rc1
```

Exit status 0.

## Step 5: pip check

```powershell
python -m pip check
```

```text
No broken requirements found.
```

Exit status 0.

## Step 5: verify_install.py

```powershell
python scripts/dev/verify_install.py
```

```text
python   3.12.10 on Windows-11-10.0.26100-SP0
scirex   1.0.0rc1  (<repo>\src\scirex\__init__.py)
jax      0.6.2, backend cpu, x64 False
device   TFRT_CPU_0
wheels   flax==0.10.7, jax==0.6.2, jaxlib==0.6.2, jaxwt==0.1.1, optax==0.2.8, scirex-zenteiq==1.0.0rc1
compute  jit matmul on {CpuDevice(id=0)} -> 83.0
verify_install: ok
```

Exit status 0.
