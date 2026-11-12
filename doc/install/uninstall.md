<!--
SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
License notice: Documentation text and diagrams on this page are licensed
under CC BY 4.0. Source-code examples are licensed under LGPL-3.0-only
unless the example states otherwise.
-->

# Uninstall

```bash
pip uninstall scirex
```

This leaves JAX and the other dependencies in place. To remove the CUDA wheels too:

```bash
pip uninstall jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
pip freeze | grep -i nvidia | cut -d= -f1 | xargs pip uninstall -y
```

To remove everything, delete the virtual environment (`rm -rf scirex-venv`). For Docker,
`docker rmi scirex-dev scirex-prod`.
