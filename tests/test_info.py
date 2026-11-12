# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""``python -m scirex.info`` prints the environment report that bug reports ask for.

The tests do not assume a device: they pass on a CPU-only host and on a GPU host.
"""

import socket
import subprocess
import sys

import jax

import scirex
from scirex.info import environment_report

FIELDS = (
    "ScireX:",
    "OS:",
    "backend:",
    "devices:",
    "x64:",
    "jax:",
    "jaxlib:",
    "python:",
)


def test_report_has_every_field() -> None:
    report = environment_report()
    for field in FIELDS:
        assert field in report, field


def test_report_matches_the_running_environment() -> None:
    report = environment_report()
    assert f"ScireX:  {scirex.__version__}" in report
    assert f"backend: {jax.default_backend()}" in report
    assert f"jax:    {jax.__version__}" in report
    assert repr(jax.devices()[0]) in report


def test_report_does_not_name_the_machine() -> None:
    assert socket.gethostname() not in environment_report()


def test_report_has_no_stray_quote() -> None:
    assert 'local devices"' not in environment_report()
    assert 'local device"' not in environment_report()


def test_module_runs_from_the_command_line() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "scirex.info"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    for field in FIELDS:
        assert field in done.stdout, field
