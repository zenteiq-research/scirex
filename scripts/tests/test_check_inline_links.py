# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only

"""check_inline_links.py: which inline code becomes a link, and which does not."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "check_inline_links", ROOT / "scripts" / "dev" / "check_inline_links.py"
)
cil = importlib.util.module_from_spec(spec)
sys.modules["check_inline_links"] = cil
spec.loader.exec_module(cil)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "doc" / "api").mkdir(parents=True)
    (tmp_path / "doc" / "api" / "models.md").write_text(
        "::: scirex.models.FNO\n\n::: scirex.models.get_model_jax\n"
    )
    (tmp_path / "doc" / "api" / "training.md").write_text(
        "::: scirex.training.adamw_cx\n"
    )
    (tmp_path / "scripts" / "dev").mkdir(parents=True)
    (tmp_path / "scripts" / "reproduce.py").write_text("x = 1\n")
    (tmp_path / "scripts" / "secret.py").write_text("x = 1\n")
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "PUBLIC_EXCLUDE.txt").write_text("scripts/secret.py\n")
    return tmp_path


def run(repo, rel, text):
    idx = cil.Index(repo)
    return cil.fix(text, rel, idx), cil.scan(text, rel, idx)


def test_an_api_symbol_links_to_its_entry_at_every_mention(repo):
    out, found = run(
        repo, "doc/user_guide/a.md", "Use `FNO` and `FNO(...)` or `scirex.models.FNO`."
    )
    assert out.count("](../api/models.md#scirex.models.FNO)") == 3
    assert len(found) == 3


def test_a_module_links_to_its_page_and_a_name_like_dtype_does_not(repo):
    out, _ = run(repo, "doc/a.md", "Pass `dtype` to `scirex.models` or use `adamw_cx`.")
    assert "[`scirex.models`](api/models.md)" in out
    assert "[`adamw_cx`](api/training.md#scirex.training.adamw_cx)" in out
    assert "`dtype`" in out and "[`dtype`]" not in out


def test_a_stable_subpackage_links_to_its_parents_page(repo):
    out, _ = run(repo, "doc/a.md", "Import from `scirex.training.optim`.")
    assert "[`scirex.training.optim`](api/training.md)" in out


def test_experimental_modules_listed_on_the_experimental_page_link_to_it(repo):
    (repo / "doc" / "api" / "experimental.md").write_text(
        "| Module | Holds |\n|---|---|\n| `scirex.experimental.tuning` | tune |\n"
    )
    out, _ = run(
        repo,
        "doc/a.md",
        "Use `scirex.experimental.tuning`, not `scirex.experimental.other`.",
    )
    assert "[`scirex.experimental.tuning`](api/experimental.md)" in out
    assert "[`scirex.experimental.other`]" not in out


def test_flags_env_vars_and_bare_file_names_are_left_alone(repo):
    text = (
        "Run `--device cpu` with `SCIREX_CKPT_ROOT`; read `infer.py`, `metrics.json`."
    )
    out, found = run(repo, "doc/a.md", text)
    assert out == text and not found


def test_a_full_path_links_to_the_file_unless_it_stays_private(repo):
    text = "See `scripts/reproduce.py`, `scripts/secret.py` and `scripts/missing.py`."
    out, _ = run(repo, "doc/a.md", text)
    # relative to the page; the docs build turns it into a link to the repository
    assert "[`scripts/reproduce.py`](../scripts/reproduce.py)" in out
    assert (
        "`scripts/secret.py`" in out and "[`scripts/secret.py`]" not in out
    )  # private
    assert (
        "`scripts/missing.py`" in out and "[`scripts/missing.py`]" not in out
    )  # no file


def test_root_pages_link_api_names_to_the_site_and_paths_relatively(repo):
    out, _ = run(repo, "README.md", "`FNO` is in `scripts/reproduce.py`.")
    assert f"]({cil.SITE_URL}/api/models/#scirex.models.FNO)" in out
    assert "](scripts/reproduce.py)" in out


def test_headings_code_blocks_comments_and_existing_links_are_untouched(repo):
    text = "\n".join(
        [
            "# The `FNO` model",
            "```python",
            "x = `FNO`",
            "```",
            "<!-- `FNO` -->",
            "Already [`FNO`](elsewhere.md) here.",
        ]
    )
    out, found = run(repo, "doc/a.md", text)
    assert out == text and not found


def test_a_four_backtick_fence_holds_a_three_backtick_fence(repo):
    text = "````markdown\n```bash\n`FNO`\n```\n`FNO`\n````\nafter `FNO`"
    out, found = run(repo, "doc/a.md", text)
    assert [f[0] for f in found] == [7]  # only the line after the fence


def test_fix_is_idempotent(repo):
    once, _ = run(repo, "doc/a.md", "Use `FNO` in `scripts/reproduce.py`.")
    twice, found = run(repo, "doc/a.md", once)
    assert twice == once and not found
