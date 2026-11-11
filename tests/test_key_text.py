"""Platform key text (O21): Cmd / Option on macOS, ``app.gui.key_text`` Qt-free, and no
raw "Ctrl+" / "Alt+" text left in ``app/gui`` outside it. The shortcut registry's macOS
rows (sheet, tooltips, stage keys, Pinch row): ``tests/test_shortcuts.py``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.gui.key_text import platform_keys

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "app" / "gui"
KEY_TOKEN = re.compile(r"\b(Ctrl|Alt)\+")
NOT_WRAPPED_FILES = {"shortcuts.py", "key_text.py"}  # the registry is data; key_text is the map


# ------------------------------------------------------------------ platform_keys
def test_ctrl_and_alt_become_cmd_and_option_on_macos():
    assert platform_keys("Ctrl+Shift+Z", "darwin") == "Cmd+Shift+Z"
    assert platform_keys("Alt+2", "darwin") == "Option+2"
    assert platform_keys("Ctrl+Alt+2", "darwin") == "Cmd+Option+2"
    assert platform_keys("Ctrl+= / Ctrl+- zoom · Ctrl+wheel", "darwin") == (
        "Cmd+= / Cmd+- zoom · Cmd+wheel"
    )


@pytest.mark.parametrize("platform", ["win32", "linux", "cygwin"])
def test_other_platforms_keep_the_text(platform):
    assert platform_keys("Ctrl+wheel", platform) == "Ctrl+wheel"
    assert platform_keys("Alt+F4", platform) == "Alt+F4"


def test_only_whole_tokens_before_a_plus_are_replaced():
    assert platform_keys("Control panel", "darwin") == "Control panel"
    assert platform_keys("Ctrl is a key", "darwin") == "Ctrl is a key"
    assert platform_keys("Alternative+1 Ctrls+1 MyCtrl+1", "darwin") == (
        "Alternative+1 Ctrls+1 MyCtrl+1"
    )


def test_the_platform_defaults_to_sys_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert platform_keys("Ctrl+Z") == "Cmd+Z"
    monkeypatch.setattr(sys, "platform", "win32")
    assert platform_keys("Ctrl+Z") == "Ctrl+Z"


def test_key_text_is_qt_free():
    code = "import sys, app.gui.key_text; print('PyQt6' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------ ratchet
def _docstring_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _wrapped_ids(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "id", getattr(func, "attr", None)) == "platform_keys":
            ids.update(id(c) for a in node.args for c in ast.walk(a) if isinstance(c, ast.Constant))
    return ids


def _raw_key_texts() -> list[str]:
    found = []
    for path in sorted(GUI.rglob("*.py")):
        if path.name in NOT_WRAPPED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstring_ids(tree) | _wrapped_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip
                and KEY_TOKEN.search(node.value)
            ):
                found.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: {node.value!r}")
    return found


def test_no_raw_ctrl_text_outside_key_text():
    assert _raw_key_texts() == []
