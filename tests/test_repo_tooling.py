"""The repository's tooling rules: one lint/format toolchain, the CI triggers.

- Ruff only (CQ-02). black and isort used to be configured and documented alongside ruff,
  while pre-commit ran ruff + ruff-format. The two formatters disagree on some lines (e.g.
  assert messages) and isort silently skipped non-cp1252 files on Windows, so the contract
  is: ruff is the only linter, import sorter ("I" rules) and formatter, and every place that
  tells a contributor or agent what to run says the same thing.
- The CI workflow runs on feature and fix branches too (X7).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _section(text: str, name: str) -> str:
    """Body of one TOML table (a regex: the test needs no TOML parser)."""
    match = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert match, f"[{name}] missing from pyproject.toml"
    return match.group(1)


def test_pyproject_configures_only_ruff():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not re.search(r"^\[tool\.black\]", text, re.M), "[tool.black] competes with ruff format"
    assert not re.search(r"^\[tool\.isort\]", text, re.M), (
        "[tool.isort] competes with ruff's I rules"
    )
    ruff = _section(text, "tool.ruff")
    assert re.search(r"^line-length\s*=\s*100$", ruff, re.M)
    assert re.search(r'^target-version\s*=\s*"py314"$', ruff, re.M)
    select = re.search(r"^select\s*=\s*\[(.*?)\]", _section(text, "tool.ruff.lint"), re.M)
    assert select and '"I"' in select.group(1), "ruff must own import sorting"


def test_pre_commit_runs_ruff_lint_and_format_only():
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook_ids = re.findall(r"-\s*id:\s*([\w-]+)", text)
    assert "ruff" in hook_ids or "ruff-check" in hook_ids
    assert "ruff-format" in hook_ids
    assert not {"black", "isort"} & set(hook_ids)
    assert "psf/black" not in text and "pycqa/isort" not in text.lower()


def test_claude_md_commands_match_pre_commit():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "-m ruff format ." in text
    assert "-m ruff check --fix ." in text
    assert not re.search(r"-m\s+(black|isort)\b", text), "CLAUDE.md still says run black/isort"
    assert "</invoke>" not in text, "stray tool-call markup at the end of CLAUDE.md"


def test_ci_runs_on_feature_and_fix_branches():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = doc.get("on", doc.get(True))
    assert {"feat/**", "fix/**"} <= set(on["push"]["branches"])
    assert "pull_request" in on


# --------------------------------------------------------- requirements stay UTF-8 (CI)


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
def test_the_requirements_files_are_utf8(name):
    """pip in CI died on a UTF-16 requirements.txt (memory ci-deps-install-gotchas)."""
    raw = (ROOT / name).read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), f"{name} is UTF-16"
    assert b"\x00" not in raw, f"{name} holds NUL bytes (UTF-16?)"
    raw.decode("utf-8")  # strict: raises on anything else


# --------------------------------------------- no test module imports another (wave 0)

#: Imports between test modules that wave 2 (generation, class labels, dashboards) still has
#: to remove; this list may only shrink.
WAVE_2_LEFT = {
    "test_viz_dash_view_names": {"test_viz_data_cache"},
    "test_viz_data_faceted": {"test_viz_data_cache"},
    "test_viz_data_growth": {"test_viz_data_cache"},
    "test_viz_data_metric_charts": {"test_viz_data_cache"},
    "test_viz_data_theme": {"test_viz_data_cache"},
    "test_viz_data_time_view": {"test_viz_data_cache"},
    "test_viz_theme_bar_label_contrast": {"test_viz_data_cache"},
}  # fmt: skip


def imported_test_modules(source: str, own: str) -> set[str]:
    """The other ``tests.test_*`` modules ``source`` imports (``own`` is its own name)."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            parts = name.split(".")
            if len(parts) > 1 and parts[0] == "tests" and parts[1].startswith("test_"):
                found.add(parts[1])
    return found - {own}


def test_the_import_reader_finds_a_test_module_import():
    source = (
        "from tests.test_a import helper\n"
        "import tests.test_b as b\n"
        "from tests import test_c, window_support\n"
        "from tests.test_self import itself\n"
        "def f():\n    from tests.test_d import late\n"
        "from tests.correct_support import paint\n"
    )
    assert imported_test_modules(source, "test_self") == {"test_a", "test_b", "test_c", "test_d"}


def test_no_test_module_imports_another_test_module():
    found = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        imported = imported_test_modules(path.read_text(encoding="utf-8"), path.stem)
        if extra := imported - WAVE_2_LEFT.get(path.stem, set()):
            found[path.name] = sorted(extra)
    assert found == {}, "move shared helpers into a tests/<area>_support.py module"
