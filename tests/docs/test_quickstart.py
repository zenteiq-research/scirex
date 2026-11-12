# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""The Quickstart is one script: its code blocks must parse, import only public names
and generate data that really solves the Poisson problem it describes."""

import ast
import re
from pathlib import Path

import numpy as np

PAGE = Path(__file__).resolve().parents[2] / "doc" / "user_guide" / "quickstart.md"
BLOCKS = re.findall(r"```python\n(.*?)```", PAGE.read_text(encoding="utf-8"), re.S)


def test_the_code_blocks_form_one_valid_script() -> None:
    ast.parse("\n\n".join(BLOCKS))


def test_the_page_imports_only_public_names() -> None:
    for node in ast.walk(ast.parse("\n".join(BLOCKS))):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("scirex")
        ):
            module = __import__(node.module, fromlist=["x"])
            for alias in node.names:
                assert alias.name in module.__all__, (node.module, alias.name)


def test_the_generated_data_solves_the_poisson_equation() -> None:
    namespace: dict = {}
    source = BLOCKS[0].split("x_train, y_train")[0]
    exec(source, namespace)  # - the documented function, run in isolation
    a, u = namespace["make_dataset"](2, 32, seed=0)
    a, u = np.asarray(a)[:, 0], np.asarray(u)[:, 0]
    k = np.fft.fftfreq(32, d=1.0 / 32) * 2 * np.pi
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    lap_u = -np.fft.ifft2(k2 * np.fft.fft2(u)).real  # -laplace(u), spectrally
    # each u was rescaled to unit standard deviation, so compare up to that factor per example
    for lap, src in zip(lap_u, a):
        scale = (lap * src).sum() / (src * src).sum()
        assert np.linalg.norm(lap - scale * src) / np.linalg.norm(lap) < 1e-4
