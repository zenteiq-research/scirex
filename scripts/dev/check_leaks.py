#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
r"""Look for things that must not ship in a public release.

Searches every tracked text file for:

* **personal paths**: ``/home/<user>``, ``/Users/<user>``, ``C:\Users\<user>``;
* **hostnames and internal links**: Azure DevOps, and URLs or ``user@host``
  addresses whose host ends in ``.internal``, ``.corp``, ``.lan``, ``.local`` or
  ``.intranet``, plus any host given with ``--host``;
* **IP addresses**: IPv4 addresses other than loopback, ``0.0.0.0`` and the
  documentation ranges (``192.0.2.x``, ``198.51.100.x``, ``203.0.113.x``);
* **secrets**: private keys, GitHub, AWS, Slack, Google and OpenAI tokens,
  and ``api_key = "<long literal>"`` assignments;
* **profanity**: a short built-in word list (whole words, case-insensitive);
* **names**: customer or personal names given with ``--term`` (never stored in the repo).

Usage::

    python scripts/dev/check_leaks.py [--root REPO] [--term NAME ...]
        [--host HOST ...] [--strict]

Each finding is printed as ``file:line: category: text``. A line that ends with
``# leaks: ok`` (or contains ``leaks: ok`` in a comment) is skipped.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from pathlib import Path

# The checkers and their tests contain deliberately bad strings as fixtures.
SELF = {
    "scripts/dev/check_leaks.py",
    "scripts/tests/test_check_leaks.py",
    "scripts/tests/test_check_notebooks.py",
    "scripts/tests/test_check_package.py",
    "scripts/tests/test_check_no_committers.py",
    "scripts/tests/test_check_contacts.py",
}
TEXT_SUFFIXES = {
    ".py", ".md", ".ipynb", ".toml", ".cff", ".txt", ".yml", ".yaml", ".cfg", ".tex",
    ".rst", ".json", ".sh", ".ini", ".html", ".css", ".js", ".env", ".properties", "",
}  # fmt: skip
MAX_BYTES = 2_000_000

PERSONAL_PATH = re.compile(
    r"(?:/home/(?!runner\b|scirex\b|user\b|\$|<)[a-z_][a-z0-9_.-]*"
    r"|/Users/(?!Shared\b|runner\b)[A-Za-z][\w.-]*"
    r"|[A-Za-z]:\\Users\\(?!Public\b)[\w.-]+)"
)
HOST_IN_URL = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\b[\w.-]+@)([A-Za-z0-9][A-Za-z0-9.-]*)"
)
PRIVATE_SUFFIXES = (".internal", ".corp", ".lan", ".local", ".intranet", ".home.arpa")
PRIVATE_DOMAINS = ("dev.azure.com", "visualstudio.com")
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r"|\bghp_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"
    r"|\bAKIA[0-9A-Z]{16}\b|\bxox[baprs]-[A-Za-z0-9-]{10,}"
    r"|\bAIza[0-9A-Za-z_-]{35}\b|\bsk-[A-Za-z0-9]{32,}"
    r"|\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"'][A-Za-z0-9_\-/+=]{20,}[\"'])",
    re.I,
)
# A short list; extend it with --profanity-file if you need more.
PROFANITY = (
    "fuck", "fucking", "shit", "bullshit", "bitch", "bastard", "asshole", "dick",
    "cunt", "piss", "crap", "damn", "wtf", "stfu",
)  # fmt: skip
PROFANITY_RE = re.compile(r"\b(?:" + "|".join(PROFANITY) + r")\b", re.I)
ALLOW_IPS = {"0.0.0.0", "127.0.0.1", "255.255.255.255"}
DOC_NETS = [
    ipaddress.ip_network(n)
    for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")
]
SKIP_MARK = "leaks: ok"


def _tracked(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [root / f for f in out if f and f not in SELF]


def scan_line(
    line: str, terms: tuple[str, ...] = (), hosts: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    """Return ``(category, matched text)`` for one line."""
    if SKIP_MARK in line:
        return []
    found: list[tuple[str, str]] = []
    for m in PERSONAL_PATH.finditer(line):
        found.append(("personal path", m.group(0)))
    for m in HOST_IN_URL.finditer(line):
        host = m.group(1).lower().rstrip(".")
        if host.endswith(PRIVATE_SUFFIXES) or any(
            host == d or host.endswith("." + d) for d in PRIVATE_DOMAINS
        ):
            found.append(("internal host or link", host))
    for d in PRIVATE_DOMAINS:
        if d in line.lower() and not any(
            c == "internal host or link" for c, _ in found
        ):
            found.append(("internal host or link", d))
    for host in hosts:
        if host.lower() in line.lower():
            found.append(("internal host or link", host))
    for m in IPV4.finditer(line):
        text = m.group(0)
        try:
            ip = ipaddress.ip_address(text)
        except ValueError:
            continue  # e.g. 999.1.1.1 or a version number
        if text in ALLOW_IPS or any(ip in net for net in DOC_NETS):
            continue
        if not (ip.is_private or ip.is_global or ip.is_link_local):
            continue
        # skip dotted version strings written right after "==", "v" or a package
        # name ("nvidia_cudnn_cu12-9.28.1.4")
        start = m.start()
        if start > 0 and line[start - 1] in "=v<>~-_":
            continue
        # "nvidia-cudnn-cu12: 9.28.1.4" in the platform check's package list
        if re.search(r"nvidia[-_\w]*:\s*$", line[:start]):
            continue
        found.append(("IP address", text))
    for m in SECRET.finditer(line):
        found.append(("secret", m.group(0)[:12] + "..."))
    for m in PROFANITY_RE.finditer(line):
        found.append(("profanity", m.group(0)))
    for term in terms:
        if term.lower() in line.lower():
            found.append(("listed term", term))
    return found


def check(
    root: Path, terms: tuple[str, ...] = (), hosts: tuple[str, ...] = ()
) -> list[str]:
    problems: list[str] = []
    for path in _tracked(root):
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root)
        for line_no, line in enumerate(text.splitlines(), 1):
            for category, hit in scan_line(line, terms, hosts):
                problems.append(f"{rel}:{line_no}: {category}: {hit}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument(
        "--term", action="append", default=[], help="name that must not appear"
    )
    ap.add_argument(
        "--host", action="append", default=[], help="host that must not appear"
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on findings")
    args = ap.parse_args()
    problems = check(args.root, tuple(args.term), tuple(args.host))
    for p in problems:
        print(p)
    by_cat: dict[str, int] = {}
    for p in problems:
        cat = p.split(": ", 2)[1]
        by_cat[cat] = by_cat.get(cat, 0) + 1
    print(
        f"\n{len(problems)} finding(s)."
        + "".join(f"  {k}: {v}" for k, v in by_cat.items())
    )
    return 1 if (problems and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
